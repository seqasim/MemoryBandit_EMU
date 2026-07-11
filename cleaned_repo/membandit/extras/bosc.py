import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

# LFPAnalysis.eBOSC_wrapper imported lazily inside compute_exact_bosc_channel
from membandit.behavior import (
    load_behavior_tables,
    preprocess_combined_df,
    preprocess_learn_df,
    resolve_trial_for_event,
)
from membandit.behavior import resolve_trial_for_event as _resolve_trial_col_for_event


RPE_CATEGORY_ORDER = ["negative", "neutral", "positive"]
RPE_LABEL_MAP = {
    "neg": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def normalize_rpe_cat_labels(series: pd.Series) -> pd.Categorical:
    normalized = series.astype(str).str.strip().str.lower().map(RPE_LABEL_MAP)
    if normalized.isna().any():
        bad_values = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unrecognized rpe_cat labels: {bad_values}")
    return pd.Categorical(normalized, categories=RPE_CATEGORY_ORDER, ordered=True)


def load_and_preprocess_behavior(load_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    learn_df, combined_df = load_behavior_tables(load_dir)
    learn_df = preprocess_learn_df(learn_df)
    combined_df = preprocess_combined_df(combined_df)

    validate_required_columns(learn_df, ["participant", "rpe", "rpe_cat"], "learn_df")
    validate_required_columns(combined_df, ["participant", "condition", "rpe", "rpe_cat", "trials_mem"], "combined_df")

    return learn_df, combined_df


def assign_exact_band(freq: float) -> str | float:
    if 4.0 <= float(freq) < 8.0:
        return "theta"
    if 8.0 <= float(freq) < 13.0:
        return "alpha"
    return np.nan


def prepare_bosc_channel_dataframe(
    epochs: mne.Epochs,
    chan_name: str,
    subj_id: str,
    event: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data_df = epochs.copy().pick([chan_name]).to_data_frame(time_format=None)
    validate_required_columns(data_df, ["epoch", "time", chan_name], f"BOSC dataframe for {subj_id} {event} {chan_name}")

    data_df = data_df.sort_values(["epoch", "time"]).reset_index(drop=True)
    raw_epoch_labels = data_df["epoch"].drop_duplicates().tolist()
    if len(raw_epoch_labels) == 0:
        raise ValueError(f"No epoch rows found for {subj_id} {event} {chan_name}")

    epoch_map = {old_epoch: new_epoch for new_epoch, old_epoch in enumerate(raw_epoch_labels)}
    data_df["epoch_original"] = data_df["epoch"]
    data_df["epoch"] = data_df["epoch_original"].map(epoch_map).astype(int)

    counts_per_trial = data_df.groupby("epoch").size().sort_index()
    n_trials = int(len(counts_per_trial))
    if n_trials == 0:
        raise ValueError(f"No BOSC trials remain after epoch normalization for {subj_id} {event} {chan_name}")

    expected_trials = list(range(n_trials))
    if counts_per_trial.index.tolist() != expected_trials:
        raise ValueError(
            f"Normalized BOSC epoch labels are not contiguous for {subj_id} {event} {chan_name}: "
            f"{counts_per_trial.index.tolist()}"
        )

    empty_trials = [trial_id for trial_id, count in counts_per_trial.items() if int(count) == 0]
    if empty_trials:
        raise ValueError(
            f"Empty BOSC trials after epoch normalization for {subj_id} {event} {chan_name}: {empty_trials}"
        )

    n_timepoints_per_trial = data_df.groupby("epoch")["time"].nunique().sort_index()
    n_timepoints = int(n_timepoints_per_trial.iloc[0])
    if n_timepoints == 0:
        raise ValueError(f"No BOSC timepoints remain after epoch normalization for {subj_id} {event} {chan_name}")

    ragged_trials = n_timepoints_per_trial[n_timepoints_per_trial != n_timepoints]
    if len(ragged_trials) > 0:
        raise ValueError(
            f"Ragged BOSC trial lengths for {subj_id} {event} {chan_name}: "
            f"{ragged_trials.to_dict()}"
        )

    epoch_labels_contiguous = raw_epoch_labels == expected_trials
    diagnostics = {
        "n_bosc_trials": n_trials,
        "n_bosc_timepoints": n_timepoints,
        "epoch_labels_contiguous": bool(epoch_labels_contiguous),
        "raw_epoch_labels_preview": raw_epoch_labels[:10],
    }
    return data_df, diagnostics


def _prepare_subject_behavior(
    behavior_df: pd.DataFrame,
    subj_id: str,
    day: str,
    df_name: str,
) -> pd.DataFrame:
    validate_required_columns(behavior_df, ["participant"], df_name)
    subj_df = behavior_df[behavior_df["participant"] == subj_id].copy()
    if len(subj_df) == 0:
        raise ValueError(f"{df_name}: no rows found for participant {subj_id}")
    if "condition" in subj_df.columns:
        subj_df = subj_df[subj_df["condition"] == day].copy()
    if len(subj_df) == 0:
        raise ValueError(f"{df_name}: no {day} rows found for participant {subj_id}")
    return subj_df


def get_aligned_behavior_from_learn_df(
    learn_df: pd.DataFrame,
    subj_id: str,
    event: str,
    n_epochs: int,
    day: str = "Day1",
) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    trial_col = _resolve_trial_col_for_event(learn_df, event)
    required = ["participant", "rpe", "rpe_cat", trial_col]
    if "condition" in learn_df.columns:
        required.append("condition")
    validate_required_columns(learn_df, required, "learn_df")

    subj_df = _prepare_subject_behavior(learn_df, subj_id, day=day, df_name="learn_df")
    n_day1_rows = int(len(subj_df))

    subj_df = subj_df.dropna(subset=[trial_col]).copy()
    n_trial_rows = int(len(subj_df))
    if n_trial_rows == 0:
        raise ValueError(f"learn_df: no valid {trial_col} rows for {subj_id} {event}")

    subj_df = subj_df.dropna(subset=["rpe", "rpe_cat"]).copy()
    n_rpe_valid_rows = int(len(subj_df))
    if n_rpe_valid_rows == 0:
        raise ValueError(f"learn_df: no valid rpe/rpe_cat rows for {subj_id} {event}")

    subj_df = subj_df.copy()
    subj_df["trial_from_event"] = pd.to_numeric(subj_df[trial_col], errors="coerce")
    if subj_df["trial_from_event"].isna().any():
        raise ValueError(f"learn_df: non-numeric {trial_col} values for {subj_id} {event}")
    if not np.allclose(subj_df["trial_from_event"], np.round(subj_df["trial_from_event"])):
        raise ValueError(f"learn_df: non-integer {trial_col} values for {subj_id} {event}")

    subj_df["trial_from_event"] = subj_df["trial_from_event"].astype(int) - 1
    if subj_df["trial_from_event"].duplicated().any():
        dupes = subj_df.loc[subj_df["trial_from_event"].duplicated(), trial_col].tolist()
        raise ValueError(f"learn_df: duplicate {trial_col} values for {subj_id} {event}: {dupes}")
    if not subj_df["trial_from_event"].between(0, n_epochs - 1).all():
        bad = subj_df.loc[~subj_df["trial_from_event"].between(0, n_epochs - 1), trial_col].tolist()
        raise ValueError(f"learn_df: out-of-range {trial_col} values for {subj_id} {event}: {bad}")

    subj_df = subj_df.sort_values(by=trial_col).reset_index(drop=True)
    subj_df["trial_zero"] = np.arange(len(subj_df), dtype=int)
    subj_df["rpe_cat"] = normalize_rpe_cat_labels(subj_df["rpe_cat"])
    audit = {
        "n_day1_rows": n_day1_rows,
        "n_trial_rows": n_trial_rows,
        "n_rpe_valid_rows": n_rpe_valid_rows,
    }
    return subj_df, trial_col, audit


def get_aligned_behavior_from_combined_df(
    combined_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    day: str = "Day1",
) -> Tuple[pd.DataFrame, str, Dict[str, int]]:
    trial_col = "trials_mem"
    required = ["participant", "condition", "rpe", "rpe_cat", trial_col]
    validate_required_columns(combined_df, required, "combined_df")

    subj_df = _prepare_subject_behavior(combined_df, subj_id, day=day, df_name="combined_df")
    n_day1_rows = int(len(subj_df))

    subj_df = subj_df.dropna(subset=[trial_col]).copy()
    n_trial_rows = int(len(subj_df))
    if n_trial_rows == 0:
        raise ValueError(f"combined_df: no valid {trial_col} rows for {subj_id} cue_on")

    subj_df = subj_df.sort_values(by=trial_col).reset_index(drop=True)
    subj_df = subj_df.dropna(subset=["rpe", "rpe_cat"]).copy()
    n_rpe_valid_rows = int(len(subj_df))
    if n_rpe_valid_rows == 0:
        raise ValueError(
            f"combined_df: no valid rpe/rpe_cat rows remain after filtering {trial_col} for {subj_id} cue_on"
        )

    subj_df = subj_df.copy()
    subj_df["trial_from_event"] = pd.to_numeric(subj_df[trial_col], errors="coerce")
    if subj_df["trial_from_event"].isna().any():
        raise ValueError(f"combined_df: non-numeric {trial_col} values for {subj_id} cue_on")
    if not np.allclose(subj_df["trial_from_event"], np.round(subj_df["trial_from_event"])):
        raise ValueError(f"combined_df: non-integer {trial_col} values for {subj_id} cue_on")

    subj_df["trial_from_event"] = subj_df["trial_from_event"].astype(int) - 1
    if subj_df["trial_from_event"].duplicated().any():
        dupes = subj_df.loc[subj_df["trial_from_event"].duplicated(), trial_col].tolist()
        raise ValueError(f"combined_df: duplicate {trial_col} values for {subj_id} cue_on: {dupes}")
    if not subj_df["trial_from_event"].between(0, n_epochs - 1).all():
        bad = subj_df.loc[~subj_df["trial_from_event"].between(0, n_epochs - 1), trial_col].tolist()
        raise ValueError(f"combined_df: out-of-range {trial_col} values for {subj_id} cue_on: {bad}")

    subj_df = subj_df.sort_values(by=trial_col).reset_index(drop=True)
    subj_df["trial_zero"] = np.arange(len(subj_df), dtype=int)
    subj_df["rpe_cat"] = normalize_rpe_cat_labels(subj_df["rpe_cat"])
    audit = {
        "n_day1_rows": n_day1_rows,
        "n_trial_rows": n_trial_rows,
        "n_rpe_valid_rows": n_rpe_valid_rows,
    }
    return subj_df, trial_col, audit


def load_event_epochs(ephys_root: str, subj_id: str, event: str) -> mne.Epochs:
    epoch_path = f"{ephys_root}/{subj_id}/neural/Day1/{event}-epo.fif"
    if not os.path.exists(epoch_path):
        raise FileNotFoundError(f"Missing epochs file: {epoch_path}")
    return mne.read_epochs(epoch_path, preload=True, verbose=False)


def get_aligned_behavior_for_event(
    event: str,
    subj_id: str,
    learn_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    n_epochs: int,
) -> Tuple[pd.DataFrame, str, str, Dict[str, int]]:
    if str(event).strip().lower() == "cue_on":
        aligned_behavior, trial_col, audit = get_aligned_behavior_from_combined_df(
            combined_df=combined_df,
            subj_id=subj_id,
            n_epochs=n_epochs,
        )
        return aligned_behavior, "combined_df", trial_col, audit

    aligned_behavior, trial_col, audit = get_aligned_behavior_from_learn_df(
        learn_df=learn_df,
        subj_id=subj_id,
        event=event,
        n_epochs=n_epochs,
    )
    return aligned_behavior, "learn_df", trial_col, audit


def apply_behavior_to_epochs(
    epochs: mne.Epochs,
    aligned_behavior: pd.DataFrame,
    behavior_source: str,
    behavior_trial_col: str,
) -> Tuple[mne.Epochs, pd.DataFrame]:
    metadata_cols = ["participant", "rpe", "rpe_cat", "trial_zero", "trial_from_event", behavior_trial_col]
    if "condition" in aligned_behavior.columns:
        metadata_cols.insert(1, "condition")
    for optional_col in ["trials", "trials_gamble", "trials_mem"]:
        if optional_col in aligned_behavior.columns and optional_col not in metadata_cols:
            metadata_cols.append(optional_col)

    if len(aligned_behavior) == 0:
        raise ValueError("Aligned behavior dataframe is empty")

    valid_indices = aligned_behavior["trial_from_event"].to_numpy(dtype=int)
    if len(valid_indices) != len(np.unique(valid_indices)):
        raise ValueError("Aligned behavior contains duplicate epoch mappings")

    aligned_epochs = epochs[valid_indices]
    aligned_behavior = aligned_behavior.copy().reset_index(drop=True)
    aligned_behavior["trial_zero"] = np.arange(len(aligned_behavior), dtype=int)
    aligned_behavior["behavior_source"] = behavior_source
    aligned_behavior["behavior_trial_col"] = behavior_trial_col
    metadata = aligned_behavior[metadata_cols + ["behavior_source", "behavior_trial_col"]].copy()
    metadata = metadata.rename(columns={"trial_zero": "trial"})
    aligned_epochs.metadata = metadata
    return aligned_epochs, aligned_behavior


def get_subject_electrodes(ephys_root: str, subj_id: str, rois_to_use: List[str]) -> pd.DataFrame:
    elec_path = f"{ephys_root}/{subj_id}/Day1_reref_elec_df"
    if not os.path.exists(elec_path):
        raise FileNotFoundError(f"Missing electrode table: {elec_path}")
    elec_df = pd.read_csv(elec_path)
    validate_required_columns(elec_df, ["label", "salman_region"], f"electrode table for {subj_id}")
    elec_df = elec_df.copy()
    elec_df["label"] = elec_df["label"].astype(str)
    valid = elec_df["salman_region"].isin(rois_to_use) & ~elec_df["label"].str.contains("lesion", case=False, na=False)
    return elec_df.loc[valid].copy().reset_index(drop=True)


def compute_exact_bosc_channel(
    chan_name: str,
    epochs: mne.Epochs,
    aligned_behavior: pd.DataFrame,
    elec_df: pd.DataFrame,
    subj_id: str,
    event: str,
    ephys_root: str,
    cfg_template: Dict,
    overwrite: bool = False,
) -> Dict[str, str]:
    from LFPAnalysis.oscillation_utils import eBOSC_wrapper  # lazy lab dependency
    output_dir = f"{ephys_root}/{subj_id}/scratch/eBOSC/{event}/dfs"
    os.makedirs(output_dir, exist_ok=True)
    time_resolved_file = f"{output_dir}/{chan_name}_time_resolved_exact_band_df.csv"
    period_file = f"{output_dir}/{chan_name}_period_averaged_exact_band_df.csv"
    freq_file = f"{output_dir}/{chan_name}_time_averaged_frequency_df.csv"
    if (not overwrite) and all(os.path.exists(path) for path in [time_resolved_file, period_file, freq_file]):
        data_df, diagnostics = prepare_bosc_channel_dataframe(
            epochs=epochs,
            chan_name=chan_name,
            subj_id=subj_id,
            event=event,
        )
        return {"channel": chan_name, "status": "cached", **diagnostics}

    data_df, diagnostics = prepare_bosc_channel_dataframe(
        epochs=epochs,
        chan_name=chan_name,
        subj_id=subj_id,
        event=event,
    )
    cfg = dict(cfg_template)
    cfg["fsample"] = float(epochs.info["sfreq"])
    cfg["threshold.duration"] = np.kron(np.ones((1, len(cfg["F"]))), 3)
    cfg["channel"] = [chan_name]
    cfg["trial"] = list(range(int(data_df["epoch"].nunique())))
    cfg["trial_background"] = list(range(int(data_df["epoch"].nunique())))

    e_bosc, _ = eBOSC_wrapper(cfg, data_df)
    detected = e_bosc["detected"].reset_index().rename(columns={0: "prop_detect"})
    detected["band"] = detected["frequency"].apply(assign_exact_band)

    behavior_merge_cols = [
        "trial_zero",
        "participant",
        "rpe",
        "rpe_cat",
        "behavior_source",
        "behavior_trial_col",
    ]
    for optional_col in ["condition", "trials", "trials_gamble", "trials_mem"]:
        if optional_col in aligned_behavior.columns:
            behavior_merge_cols.append(optional_col)
    behavior_merge = aligned_behavior[behavior_merge_cols].copy().rename(columns={"trial_zero": "trial"})

    roi_match = elec_df.loc[elec_df["label"] == chan_name, "salman_region"]
    if len(roi_match) != 1:
        raise ValueError(f"Expected exactly one ROI match for {subj_id} {chan_name}")
    roi = roi_match.iloc[0]
    unique_elec_id = f"{subj_id}_{chan_name}"

    exact_df = detected.dropna(subset=["band"]).copy()
    time_resolved_df = exact_df.groupby(["trial", "band", "time"], as_index=False)["prop_detect"].mean()
    period_averaged_df = exact_df.groupby(["trial", "band"], as_index=False)["prop_detect"].mean()
    freq_averaged_df = detected.groupby(["trial", "frequency"], as_index=False)["prop_detect"].mean()

    time_resolved_df = time_resolved_df.merge(behavior_merge, on="trial", how="left", validate="many_to_one")
    period_averaged_df = period_averaged_df.merge(behavior_merge, on="trial", how="left", validate="many_to_one")
    freq_averaged_df = freq_averaged_df.merge(behavior_merge, on="trial", how="left", validate="many_to_one")

    for df in [time_resolved_df, period_averaged_df, freq_averaged_df]:
        df.insert(0, "channel", chan_name)
        df.insert(0, "region", roi)
        df.insert(0, "event", event)
        df.insert(0, "unique_elec_id", unique_elec_id)

    time_resolved_df.to_csv(time_resolved_file, index=False)
    period_averaged_df.to_csv(period_file, index=False)
    freq_averaged_df.to_csv(freq_file, index=False)
    return {"channel": chan_name, "status": "computed", **diagnostics}


def run_subject_event_bosc(
    subj_id: str,
    event: str,
    learn_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    rois_to_use: List[str],
    ephys_root: str,
    cfg_template: Dict,
    overwrite: bool = False,
    n_jobs: int = -1,
) -> Dict[str, object]:
    epochs = load_event_epochs(ephys_root=ephys_root, subj_id=subj_id, event=event)
    aligned_behavior, behavior_source, behavior_trial_col, alignment_audit = get_aligned_behavior_for_event(
        event=event,
        subj_id=subj_id,
        learn_df=learn_df,
        combined_df=combined_df,
        n_epochs=len(epochs),
    )
    epochs, aligned_behavior = apply_behavior_to_epochs(
        epochs=epochs,
        aligned_behavior=aligned_behavior,
        behavior_source=behavior_source,
        behavior_trial_col=behavior_trial_col,
    )
    elec_df = get_subject_electrodes(ephys_root=ephys_root, subj_id=subj_id, rois_to_use=rois_to_use)
    channels = elec_df["label"].astype(str).tolist()
    results: List[Dict[str, str]] = []
    if len(channels) > 0:
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(compute_exact_bosc_channel)(
                chan_name=chan,
                epochs=epochs,
                aligned_behavior=aligned_behavior,
                elec_df=elec_df,
                subj_id=subj_id,
                event=event,
                ephys_root=ephys_root,
                cfg_template=cfg_template,
                overwrite=overwrite,
            )
            for chan in channels
        )

    return {
        "participant": subj_id,
        "event": event,
        "behavior_source": behavior_source,
        "behavior_trial_col": behavior_trial_col,
        "n_day1_rows": int(alignment_audit["n_day1_rows"]),
        "n_trial_rows": int(alignment_audit["n_trial_rows"]),
        "n_rpe_valid_rows": int(alignment_audit["n_rpe_valid_rows"]),
        "n_aligned_trials": int(len(aligned_behavior)),
        "n_electrodes": int(len(channels)),
        "n_cached": int(sum(row["status"] == "cached" for row in results)),
        "n_computed": int(sum(row["status"] == "computed" for row in results)),
        "n_bosc_trials": int(max((row.get("n_bosc_trials", 0) for row in results), default=0)),
        "n_bosc_timepoints": int(max((row.get("n_bosc_timepoints", 0) for row in results), default=0)),
        "epoch_labels_contiguous": bool(all(row.get("epoch_labels_contiguous", True) for row in results)),
    }


def load_group_bosc_outputs(
    subjects_to_use: List[str],
    rois_to_use: List[str],
    events_to_use: Dict[str, Tuple[float, float]],
    ephys_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_resolved_frames = []
    period_frames = []
    freq_frames = []

    for subj_id in subjects_to_use:
        elec_df = get_subject_electrodes(ephys_root=ephys_root, subj_id=subj_id, rois_to_use=rois_to_use)
        channels = elec_df["label"].astype(str).tolist()
        if len(channels) == 0:
            continue
        for event in events_to_use.keys():
            output_dir = f"{ephys_root}/{subj_id}/scratch/eBOSC/{event}/dfs"
            if not os.path.exists(output_dir):
                raise FileNotFoundError(f"Missing BOSC output directory: {output_dir}")
            for chan in channels:
                time_resolved_file = f"{output_dir}/{chan}_time_resolved_exact_band_df.csv"
                period_file = f"{output_dir}/{chan}_period_averaged_exact_band_df.csv"
                freq_file = f"{output_dir}/{chan}_time_averaged_frequency_df.csv"
                if not os.path.exists(time_resolved_file):
                    raise FileNotFoundError(f"Missing BOSC output file: {time_resolved_file}")
                if not os.path.exists(period_file):
                    raise FileNotFoundError(f"Missing BOSC output file: {period_file}")
                if not os.path.exists(freq_file):
                    raise FileNotFoundError(f"Missing BOSC output file: {freq_file}")
                time_resolved_frames.append(pd.read_csv(time_resolved_file))
                period_frames.append(pd.read_csv(period_file))
                freq_frames.append(pd.read_csv(freq_file))

    if len(time_resolved_frames) == 0 or len(period_frames) == 0 or len(freq_frames) == 0:
        raise ValueError("No BOSC outputs were loaded")

    time_resolved_df = pd.concat(time_resolved_frames, ignore_index=True)
    period_df = pd.concat(period_frames, ignore_index=True)
    freq_df = pd.concat(freq_frames, ignore_index=True)

    for name, df in [
        ("time_resolved_df", time_resolved_df),
        ("period_df", period_df),
        ("freq_df", freq_df),
    ]:
        validate_required_columns(df, ["region", "event", "participant", "rpe_cat"], name)
        df["rpe_cat"] = normalize_rpe_cat_labels(df["rpe_cat"])

    return time_resolved_df, period_df, freq_df


def build_time_plot_df(time_resolved_df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        time_resolved_df,
        ["participant", "region", "event", "band", "rpe_cat", "time", "prop_detect"],
        "time_resolved_df",
    )
    return (
        time_resolved_df
        .dropna(subset=["band", "rpe_cat"])
        .groupby(["participant", "region", "event", "band", "rpe_cat", "time"], as_index=False)["prop_detect"]
        .mean()
    )


def build_period_plot_df(period_df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        period_df,
        ["participant", "region", "event", "band", "rpe_cat", "prop_detect"],
        "period_df",
    )
    return (
        period_df
        .dropna(subset=["band", "rpe_cat"])
        .groupby(["participant", "region", "event", "band", "rpe_cat"], as_index=False)["prop_detect"]
        .mean()
    )


def compute_period_stats(
    period_df: pd.DataFrame,
    rois: List[str],
    events: Dict[str, Tuple[float, float]],
    bands: Dict[str, Tuple[float, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        period_df,
        ["region", "event", "band", "prop_detect", "rpe_cat", "participant", "unique_elec_id"],
        "period_df",
    )
    long_rows = []
    for roi in rois:
        for event in events.keys():
            for band in bands.keys():
                model_df = period_df[
                    (period_df["region"] == roi) &
                    (period_df["event"] == event) &
                    (period_df["band"] == band)
                ].dropna(subset=["prop_detect", "rpe_cat", "participant"]).copy()
                model_df["rpe_cat"] = pd.Categorical(model_df["rpe_cat"], categories=RPE_CATEGORY_ORDER, ordered=True)
                n_subjects = int(model_df["participant"].nunique())
                n_electrodes = int(model_df["unique_elec_id"].nunique())
                n_trials = int(len(model_df))
                if n_subjects < 2 or model_df["rpe_cat"].nunique() < 3:
                    for contrast in ["negative", "positive"]:
                        long_rows.append({
                            "region": roi,
                            "event": event,
                            "band": band,
                            "contrast": contrast,
                            "coef": np.nan,
                            "p_raw": np.nan,
                            "p_fdr": np.nan,
                            "significant_fdr": False,
                            "n_subjects": n_subjects,
                            "n_electrodes": n_electrodes,
                            "n_trials": n_trials,
                            "status": "insufficient_data",
                        })
                    continue

                model_res = smf.mixedlm(
                    "prop_detect ~ C(rpe_cat, Treatment('neutral'))",
                    data=model_df,
                    groups=model_df["participant"],
                ).fit(reml=False, method="lbfgs", disp=False)
                for contrast in ["negative", "positive"]:
                    term = f"C(rpe_cat, Treatment('neutral'))[T.{contrast}]"
                    long_rows.append({
                        "region": roi,
                        "event": event,
                        "band": band,
                        "contrast": contrast,
                        "coef": float(model_res.params.get(term, np.nan)),
                        "p_raw": float(model_res.pvalues.get(term, np.nan)),
                        "p_fdr": np.nan,
                        "significant_fdr": False,
                        "n_subjects": n_subjects,
                        "n_electrodes": n_electrodes,
                        "n_trials": n_trials,
                        "status": "ok",
                    })

    stats_long_df = pd.DataFrame(long_rows)
    valid = np.isfinite(stats_long_df["p_raw"].to_numpy(dtype=float))
    if int(valid.sum()) > 0:
        stats_long_df.loc[valid, "p_fdr"] = multipletests(stats_long_df.loc[valid, "p_raw"], method="fdr_bh")[1]
        stats_long_df.loc[valid, "significant_fdr"] = stats_long_df.loc[valid, "p_fdr"] < 0.05

    stats_summary_df = (
        stats_long_df
        .pivot_table(
            index=["region", "event", "band", "n_subjects", "n_electrodes", "n_trials"],
            columns="contrast",
            values=["coef", "p_raw", "p_fdr", "significant_fdr", "status"],
            aggfunc="first",
        )
        .reset_index()
    )
    stats_summary_df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in stats_summary_df.columns.to_flat_index()
    ]
    return stats_long_df, stats_summary_df


def plot_time_resolved_bosc(
    time_plot_df: pd.DataFrame,
    rois: List[str],
    events: Dict[str, Tuple[float, float]],
    bands: Dict[str, Tuple[float, float]],
    rpe_palette: Dict[str, str],
) -> Dict[str, plt.Figure]:
    figs: Dict[str, plt.Figure] = {}
    for band in bands.keys():
        fig, axes = plt.subplots(
            len(rois),
            len(events),
            figsize=(12, 3.2 * len(rois)),
            sharex=False,
            sharey=True,
            dpi=150,
            squeeze=False,
        )
        axes = np.asarray(axes)
        for i, roi in enumerate(rois):
            for j, event in enumerate(events.keys()):
                ax = axes[i, j]
                plot_df = time_plot_df[
                    (time_plot_df["region"] == roi) &
                    (time_plot_df["event"] == event) &
                    (time_plot_df["band"] == band)
                ].copy()
                n_subjects = int(plot_df["participant"].nunique())
                if plot_df.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=plot_df,
                    x="time",
                    y="prop_detect",
                    hue="rpe_cat",
                    hue_order=RPE_CATEGORY_ORDER,
                    palette=rpe_palette,
                    errorbar="se",
                    ax=ax,
                )
                ax.set_title(f"{roi} | {event} | n_subj={n_subjects}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Burst prevalence")
                ax.grid(False)
                sns.despine(ax=ax, top=True, right=True)
                if i != 0 or j != len(events) - 1:
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, frameon=False, loc="upper right")
        fig.suptitle(f"{band.capitalize()} BOSC prevalence vs RPE bin", y=1.02)
        fig.tight_layout()
        figs[band] = fig
    return figs


def plot_period_averaged_bosc(
    period_plot_df: pd.DataFrame,
    period_df_raw: pd.DataFrame,
    rois: List[str],
    events: Dict[str, Tuple[float, float]],
    bands: Dict[str, Tuple[float, float]],
    rpe_palette: Dict[str, str],
) -> Dict[str, plt.Figure]:
    figs: Dict[str, plt.Figure] = {}
    for band in bands.keys():
        fig, axes = plt.subplots(
            len(rois),
            len(events),
            figsize=(12, 3.2 * len(rois)),
            sharey=True,
            dpi=150,
            squeeze=False,
        )
        axes = np.asarray(axes)
        for i, roi in enumerate(rois):
            for j, event in enumerate(events.keys()):
                ax = axes[i, j]
                plot_df = period_plot_df[
                    (period_plot_df["region"] == roi) &
                    (period_plot_df["event"] == event) &
                    (period_plot_df["band"] == band)
                ].copy()
                raw_df = period_df_raw[
                    (period_df_raw["region"] == roi) &
                    (period_df_raw["event"] == event) &
                    (period_df_raw["band"] == band)
                ].copy()
                n_elecs = int(raw_df["unique_elec_id"].nunique())
                if plot_df.empty:
                    ax.set_visible(False)
                    continue
                sns.stripplot(
                    data=plot_df,
                    x="rpe_cat",
                    y="prop_detect",
                    order=RPE_CATEGORY_ORDER,
                    hue="rpe_cat",
                    hue_order=RPE_CATEGORY_ORDER,
                    palette=rpe_palette,
                    alpha=0.45,
                    size=4,
                    ax=ax,
                )
                sns.pointplot(
                    data=plot_df,
                    x="rpe_cat",
                    y="prop_detect",
                    order=RPE_CATEGORY_ORDER,
                    hue="rpe_cat",
                    hue_order=RPE_CATEGORY_ORDER,
                    palette=rpe_palette,
                    errorbar="se",
                    linestyle="none",
                    dodge=0.0,
                    markers="o",
                    ax=ax,
                )
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
                ax.set_title(f"{roi} | {event} | n_elecs={n_elecs}")
                ax.set_xlabel("RPE bin")
                ax.set_ylabel("Burst prevalence")
                ax.grid(False)
                sns.despine(ax=ax, top=True, right=True)
        fig.suptitle(f"{band.capitalize()} period-averaged BOSC prevalence", y=1.02)
        fig.tight_layout()
        figs[band] = fig
    return figs


def print_pairwise_rpe_ttests_period_plot(
    period_plot_df: pd.DataFrame,
    rois: List[str],
    events: Dict[str, Tuple[float, float]],
    bands: Dict[str, Tuple[float, float]],
) -> None:
    """
    Paired t-tests on subject-level mean prop_detect: positive vs negative, positive vs neutral.
    Uses participants with non-missing values for both categories in each pair.
    """
    validate_required_columns(
        period_plot_df,
        ["participant", "region", "event", "band", "rpe_cat", "prop_detect"],
        "period_plot_df",
    )
    band_order = list(bands.keys())
    event_names = list(events.keys())
    print("Paired t-tests on subject-level mean burst prevalence (period_plot_df)")
    for roi in rois:
        for event_name in event_names:
            for band in band_order:
                sub = period_plot_df[
                    (period_plot_df["region"] == roi)
                    & (period_plot_df["event"] == event_name)
                    & (period_plot_df["band"] == band)
                ].copy()
                if sub.empty:
                    print(f"  {roi} | {event_name} | {band}: no data")
                    continue
                wide = sub.pivot_table(
                    index="participant",
                    columns="rpe_cat",
                    values="prop_detect",
                    aggfunc="mean",
                )
                for cat in RPE_CATEGORY_ORDER:
                    if cat not in wide.columns:
                        wide[cat] = np.nan
                pos_neg = wide[["positive", "negative"]].dropna()
                pos_neu = wide[["positive", "neutral"]].dropna()
                print(f"{roi} | {event_name} | {band}")
                if len(pos_neg) >= 2:
                    t_pn, p_pn = ttest_rel(pos_neg["positive"], pos_neg["negative"])
                    print(
                        f"  positive vs negative: t={t_pn:.4f}, p={p_pn:.6g}, n_pairs={len(pos_neg)}"
                    )
                else:
                    print("  positive vs negative: insufficient paired data")
                if len(pos_neu) >= 2:
                    t_pu, p_pu = ttest_rel(pos_neu["positive"], pos_neu["neutral"])
                    print(
                        f"  positive vs neutral:  t={t_pu:.4f}, p={p_pu:.6g}, n_pairs={len(pos_neu)}"
                    )
                else:
                    print("  positive vs neutral: insufficient paired data")
