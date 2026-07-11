import os
import json
import hashlib
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from mne_connectivity import spectral_connectivity_epochs
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Minimum DPI for saving figures (never save below this).
FIG_SAVE_DPI = 300


def _apply_plot_kwargs(fig: plt.Figure, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Apply plot-related kwargs to fig axes (e.g. ylim, xlim). Returns copy of kwargs with those consumed."""
    kwargs = dict(kwargs)
    ylim = kwargs.pop("ylim", None)
    xlim = kwargs.pop("xlim", None)
    for ax in fig.axes:
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
    return kwargs


# -----------------------------------------------------------------------------
# Behavior loading / preprocessing
# -----------------------------------------------------------------------------

def _gelman_standardize(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return pd.Series((arr - mu) / (2.0 * sd), index=values.index)


def load_behavior_tables(load_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load behavior tables used by MV analyses.

    Returns
    -------
    (learn_df, combined_df)
    """
    learn_path = os.path.join(load_dir, "learn_df_RWH.csv")
    combined_path = os.path.join(load_dir, "full_df_RWH.csv")
    learn_df = pd.read_csv(learn_path)
    combined_df = pd.read_csv(combined_path)
    return learn_df, combined_df


def preprocess_learn_df(learn_df: pd.DataFrame) -> pd.DataFrame:
    """Replicates Subject-class learn_df preprocessing from mv_regression_utils.py."""
    df = learn_df.copy()

    if "gamble_rt" in df.columns and "gamble_rt_log" not in df.columns:
        with np.errstate(invalid="ignore"):
            df["gamble_rt_log"] = np.log(df["gamble_rt"])

    if {"rpe", "reward"}.issubset(df.columns):
        df["Q"] = df["rpe"] + df["reward"]

    if {"participant", "rpe"}.issubset(df.columns):
        df["prior_rpe"] = df.groupby("participant")["rpe"].shift(1)

    reversal_trials = [16, 33, 48, 65]
    if {"trials", "block"}.issubset(df.columns):
        def _tsr(row: pd.Series) -> float:
            blk = row.get("block", np.nan)
            tr = row.get("trials", np.nan)
            if not np.isfinite(blk) or not np.isfinite(tr):
                return np.nan
            if blk > 1:
                idx = int(blk - 2)
                if 0 <= idx < len(reversal_trials):
                    return float(np.mod(tr, reversal_trials[idx]))
            return float(tr)

        df["trials_since_reversal"] = df.apply(_tsr, axis=1)

    if "reward" in df.columns:
        df["preward"] = df["reward"].map({1: 0.5, 0: -0.5})

    if "gamble" in df.columns:
        df["pgamble"] = df["gamble"].map({1: 0.5, 0: -0.5})

    if "participant" in df.columns:
        for subj in df["participant"].dropna().unique():
            mask = df["participant"] == subj
            for src, dst in [
                ("Q", "zQ"),
                ("prior_rpe", "zprior_rpe"),
                ("rpe", "zrpe"),
                ("DPRIME", "zpm"),
                ("trials_since_reversal", "ztrials_since_reversal"),
            ]:
                if src in df.columns:
                    df.loc[mask, dst] = _gelman_standardize(df.loc[mask, src]).values

            if "gamble_rt_log" in df.columns and "reward" in df.columns:
                rt = pd.to_numeric(df.loc[mask, "gamble_rt_log"], errors="coerce").to_numpy(dtype=float)
                rt_min = np.nanmin(rt)
                rt_max = np.nanmax(rt)
                if np.isfinite(rt_min) and np.isfinite(rt_max) and (rt_max - rt_min) > 0:
                    rt_ev = 1.0 - ((rt - rt_min) / (rt_max - rt_min))
                    rt_rpe = pd.to_numeric(df.loc[mask, "reward"], errors="coerce").to_numpy(dtype=float) - rt_ev
                    df.loc[mask, "RT_rpe"] = rt_rpe
                    s = np.nanstd(rt_rpe)
                    if np.isfinite(s) and s > 0:
                        df.loc[mask, "zRT_rpe"] = (rt_rpe - np.nanmean(rt_rpe)) / (2.0 * s)

            if "rpe" in df.columns:
                vals = pd.to_numeric(df.loc[mask, "rpe"], errors="coerce")
                try:
                    df.loc[mask, "rpe_cat"] = pd.qcut(
                        vals,
                        q=3,
                        labels=["negative", "neutral", "positive"],
                        duplicates="drop",
                    )
                except Exception:
                    # Degenerate distributions can fail qcut.
                    df.loc[mask, "rpe_cat"] = np.nan

    return df


def preprocess_combined_df(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Replicates Subject-class combined_df preprocessing from mv_regression_utils.py."""
    df = combined_df.copy()

    outcome_cols = ["hits", "misses", "correct_rejections", "false_alarms"]
    present_outcomes = [c for c in outcome_cols if c in df.columns]
    if len(present_outcomes) > 0:
        df["memory_outcome"] = df[present_outcomes].idxmax(axis=1)

    if "gamble_rt_log" not in df.columns and "gamble_rt_gamble" in df.columns:
        with np.errstate(invalid="ignore"):
            df["gamble_rt_log"] = np.log(df["gamble_rt_gamble"])

    if {"rpe", "reward_gamble"}.issubset(df.columns):
        df["Q"] = df["rpe"] + df["reward_gamble"]

    if {"participant", "rpe"}.issubset(df.columns):
        df["prior_rpe"] = df.groupby("participant")["rpe"].shift(1)

    reversal_trials = [16, 33, 48, 65]
    if {"trials_gamble", "block_gamble"}.issubset(df.columns):
        def _tsr(row: pd.Series) -> float:
            blk = row.get("block_gamble", np.nan)
            tr = row.get("trials_gamble", np.nan)
            if not np.isfinite(blk) or not np.isfinite(tr):
                return np.nan
            if blk > 1:
                idx = int(blk - 2)
                if 0 <= idx < len(reversal_trials):
                    return float(np.mod(tr, reversal_trials[idx]))
            return float(tr)

        df["trials_since_reversal"] = df.apply(_tsr, axis=1)

    if "hits" in df.columns:
        hits_bool = df["hits"].astype(bool)
        df["phit"] = np.where(hits_bool, 0.5, -0.5)

    if "reward_gamble" in df.columns:
        df["preward"] = df["reward_gamble"].map({1: 0.5, 0: -0.5})

    if "gamble_gamble" in df.columns:
        df["pgamble"] = df["gamble_gamble"].map({1: 0.5, 0: -0.5})

    if "participant" in df.columns:
        for subj in df["participant"].dropna().unique():
            mask = df["participant"] == subj
            for src, dst in [
                ("Q", "zQ"),
                ("prior_rpe", "zprior_rpe"),
                ("rpe", "zrpe"),
                ("DPRIME", "zpm"),
                ("trials_since_reversal", "ztrials_since_reversal"),
            ]:
                if src in df.columns:
                    df.loc[mask, dst] = _gelman_standardize(df.loc[mask, src]).values

            if "gamble_rt_log" in df.columns and "reward_gamble" in df.columns:
                rt = pd.to_numeric(df.loc[mask, "gamble_rt_log"], errors="coerce").to_numpy(dtype=float)
                rt_min = np.nanmin(rt)
                rt_max = np.nanmax(rt)
                if np.isfinite(rt_min) and np.isfinite(rt_max) and (rt_max - rt_min) > 0:
                    rt_ev = 1.0 - ((rt - rt_min) / (rt_max - rt_min))
                    rt_rpe = pd.to_numeric(df.loc[mask, "reward_gamble"], errors="coerce").to_numpy(dtype=float) - rt_ev
                    df.loc[mask, "RT_rpe"] = rt_rpe
                    s = np.nanstd(rt_rpe)
                    if np.isfinite(s) and s > 0:
                        df.loc[mask, "zRT_rpe"] = (rt_rpe - np.nanmean(rt_rpe)) / (2.0 * s)

            if "rpe" in df.columns:
                vals = pd.to_numeric(df.loc[mask, "rpe"], errors="coerce")
                try:
                    df.loc[mask, "rpe_cat"] = pd.qcut(
                        vals,
                        q=3,
                        labels=["neg", "neutral", "pos"],
                        duplicates="drop",
                    )
                except Exception:
                    df.loc[mask, "rpe_cat"] = np.nan

    return df


# -----------------------------------------------------------------------------
# Epoch loading and ROI channel helpers
# -----------------------------------------------------------------------------

def get_roi_channels(elec_df: pd.DataFrame, epochs: mne.Epochs, roi: str, hemisphere: Optional[str] = None) -> Dict[str, Any]:
    roi_mask = elec_df["salman_region"] == roi
    if hemisphere is not None:
        hemi_mask = elec_df["hemisphere"].str.lower() == hemisphere.lower()
        roi_mask = roi_mask & hemi_mask
    roi_channels = elec_df.loc[roi_mask, "label"].tolist()
    valid_channels = [ch for ch in roi_channels if ch in epochs.ch_names]
    indices = np.array([epochs.ch_names.index(ch) for ch in valid_channels], dtype=int)
    return {"channels": valid_channels, "indices": indices, "n_channels": len(valid_channels)}


def make_multivariate_indices(source_indices: Sequence[int], target_indices: Sequence[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return ([np.array(source_indices)], [np.array(target_indices)])


def select_top_electrode_per_roi(
    epochs: mne.Epochs,
    roi_channels: Sequence[str],
    fmin: float,
    fmax: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    valid_channels = [ch for ch in roi_channels if ch in epochs.ch_names]
    if len(valid_channels) == 0:
        return {"electrode": None, "median_power": 0.0, "all_powers": {}}

    electrode_powers: Dict[str, float] = {}
    for ch_name in valid_channels:
        epochs_ch = epochs.copy().pick_channels([ch_name])
        epochs_ch = epochs_ch.copy().filter(fmin, fmax, verbose=False)
        power_data = epochs_ch.get_data() ** 2
        trial_medians = np.median(power_data[:, 0, :], axis=1)
        electrode_powers[ch_name] = float(np.median(trial_medians))

    top_electrode = max(electrode_powers, key=electrode_powers.get)
    top_power = electrode_powers[top_electrode]
    if verbose:
        print(f"Selected {top_electrode} (median power={top_power:.6f})")
    return {"electrode": top_electrode, "median_power": top_power, "all_powers": electrode_powers}


def load_subject_epochs(
    subj_id: str,
    base_dir: str,
    event: str,
    roi_source: str,
    roi_target: str,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    target_sfreq: float = 50,
    day: str = "Day1",
    filter_by_power: bool = False,
    power_percentile: float = 50,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Optional[Dict[str, Any]]:
    del filter_by_power, power_percentile, tfr_time_window  # currently unused in this thin port
    try:
        epochs_file = os.path.join(base_dir, "projects", "guLab", "Salman", "EphysAnalyses", subj_id, "neural", day, f"{event}-epo.fif")
        if not os.path.exists(epochs_file):
            print(f"Epochs file not found: {epochs_file}")
            return None

        epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
        original_sfreq = float(epochs.info["sfreq"])

        if tmin is not None or tmax is not None:
            epochs = epochs.crop(tmin=tmin, tmax=tmax)

        if target_sfreq < original_sfreq:
            epochs = epochs.resample(target_sfreq, verbose=False)

        actual_sfreq = float(epochs.info["sfreq"])

        elec_df_path = os.path.join(base_dir, "projects", "guLab", "Salman", "EphysAnalyses", subj_id, f"{day}_reref_elec_df")
        elec_df = pd.read_csv(elec_df_path)

        result: Dict[str, Any] = {
            "epochs": epochs,
            "times": epochs.times,
            "sfreq": actual_sfreq,
            "original_sfreq": original_sfreq,
            "n_epochs": len(epochs),
        }

        if use_top_electrode_only and (fmin is None or fmax is None):
            use_top_electrode_only = False

        for hemi in ("l", "r"):
            source_info = get_roi_channels(elec_df, epochs, roi_source, hemisphere=hemi)
            source_channels = source_info["channels"]
            if use_top_electrode_only and len(source_channels) > 0:
                top_source = select_top_electrode_per_roi(epochs, source_channels, fmin, fmax, verbose=False)
                source_channels = [top_source["electrode"]] if top_source["electrode"] is not None else []

            target_info = get_roi_channels(elec_df, epochs, roi_target, hemisphere=hemi)
            target_channels = target_info["channels"]
            if use_top_electrode_only and len(target_channels) > 0:
                top_target = select_top_electrode_per_roi(epochs, target_channels, fmin, fmax, verbose=False)
                target_channels = [top_target["electrode"]] if top_target["electrode"] is not None else []

            result[f"source_channels_{hemi}"] = source_channels
            result[f"target_channels_{hemi}"] = target_channels
            result[f"source_indices_{hemi}"] = np.array([epochs.ch_names.index(ch) for ch in source_channels if ch in epochs.ch_names])
            result[f"target_indices_{hemi}"] = np.array([epochs.ch_names.index(ch) for ch in target_channels if ch in epochs.ch_names])
            result[f"n_source_{hemi}"] = len(source_channels)
            result[f"n_target_{hemi}"] = len(target_channels)

        return result
    except Exception as exc:
        print(f"Error loading epochs for {subj_id}: {exc}")
        return None


# -----------------------------------------------------------------------------
# Trial alignment + hit/miss labels
# -----------------------------------------------------------------------------

def _resolve_trial_col_for_event(df: pd.DataFrame, event: str) -> str:
    """Resolve trial index column using event-aware preferences."""
    event_name = str(event).strip().lower()
    if event_name == "cue_on":
        candidates = ("trials_mem", "trials_gamble", "trials")
    else:
        candidates = ("trials", "trials_gamble", "trials_mem")
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not resolve trial column for event='{event}'. "
        f"Tried {list(candidates)}."
    )


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float) > 0.5
    lower = s.astype(str).str.strip().str.lower()
    return lower.isin(["1", "true", "t", "yes", "y", "hit", "hits"])


def build_hit_miss_trial_map(
    combined_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    day: str = "Day1",
    event: str = "feedback_start",
    trial_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Build epoch-aligned hit/miss map from an event-aligned behavior table."""
    if "hits" not in combined_df.columns:
        raise ValueError("behavior_df missing 'hits'")

    def _prep(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.dropna(subset=["hits"]).copy()
        if len(out) == 0:
            return out
        out["hits"] = _coerce_bool_series(out["hits"])
        out["hit_miss"] = np.where(out["hits"], "hit", "miss")
        return out

    trial_override = trial_col_override
    if trial_override is None and str(event).strip().lower() != "cue_on" and "trials_gamble" in combined_df.columns:
        # Preserve legacy hit/miss behavior outside cue-aligned analyses.
        trial_override = "trials_gamble"

    trial_map, _ = _build_condition_trial_map(
        behavior_df=combined_df,
        subj_id=subj_id,
        n_epochs=n_epochs,
        condition_col="hit_miss",
        day=day,
        event=event,
        trial_col_override=trial_override,
        preprocess_fn=_prep,
        allowed_conditions=("hit", "miss"),
        include_cols=("hits",),
    )
    return trial_map


REWARD_OUTCOME_CONDITIONS = ("rewarded", "unrewarded")


def build_reward_trial_map(
    behavior_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    day: str = "Day1",
    event: str = "feedback_start",
    trial_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Build epoch-aligned binary reward map from combined_df (`reward_gamble`) or learn_df (`reward`)."""
    if "reward_gamble" in behavior_df.columns:
        reward_col = "reward_gamble"
    elif "reward" in behavior_df.columns:
        reward_col = "reward"
    else:
        raise ValueError("behavior_df missing both 'reward_gamble' and 'reward'")

    def _prep(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.dropna(subset=[reward_col]).copy()
        if len(out) == 0:
            return out
        rv = pd.to_numeric(out[reward_col], errors="coerce")
        out = out.loc[rv.notna()].copy()
        rv = rv.loc[out.index].astype(int)
        mask = rv.isin([0, 1])
        out = out.loc[mask].copy()
        rv = rv.loc[mask]
        out["reward_outcome"] = np.where(rv.to_numpy() == 1, "rewarded", "unrewarded")
        return out

    trial_override = trial_col_override
    if trial_override is None and str(event).strip().lower() != "cue_on" and "trials_gamble" in behavior_df.columns:
        trial_override = "trials_gamble"

    trial_map, _ = _build_condition_trial_map(
        behavior_df=behavior_df,
        subj_id=subj_id,
        n_epochs=n_epochs,
        condition_col="reward_outcome",
        day=day,
        event=event,
        trial_col_override=trial_override,
        preprocess_fn=_prep,
        allowed_conditions=REWARD_OUTCOME_CONDITIONS,
        include_cols=(reward_col,),
    )
    return trial_map


MEMORY_OUTCOME_CONDITIONS = ("hits", "misses", "correct_rejections", "false_alarms")


def build_memory_outcome_trial_map(
    combined_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    day: str = "Day1",
    event: str = "feedback_start",
    trial_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Build epoch-aligned memory outcome map from combined_df (memory_outcome: hits, misses, correct_rejections, false_alarms)."""
    if "memory_outcome" not in combined_df.columns:
        raise ValueError("combined_df missing 'memory_outcome' (run preprocess_combined_df)")

    trial_map, _ = _build_condition_trial_map(
        behavior_df=combined_df,
        subj_id=subj_id,
        n_epochs=n_epochs,
        condition_col="memory_outcome",
        day=day,
        event=event,
        trial_col_override=trial_col_override,
        preprocess_fn=None,
        allowed_conditions=MEMORY_OUTCOME_CONDITIONS,
        include_cols=(),
    )
    return trial_map


def _normalize_rpe_cat_series(s: pd.Series) -> pd.Series:
    lower = s.astype(str).str.strip().str.lower()
    mapping = {
        "negative": "neg",
        "neg": "neg",
        "neutral": "neutral",
        "positive": "pos",
        "pos": "pos",
    }
    return lower.map(mapping)


def _build_condition_trial_map(
    behavior_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    condition_col: str,
    day: str = "Day1",
    event: str = "feedback_start",
    trial_col_override: Optional[str] = None,
    preprocess_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    allowed_conditions: Optional[Sequence[str]] = None,
    include_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build subject/day trial map aligned to epoch indices for an arbitrary condition column."""
    if "participant" not in behavior_df.columns:
        raise ValueError("behavior_df missing 'participant'")

    trial_col = trial_col_override or _resolve_trial_col_for_event(behavior_df, event)

    sub = behavior_df[behavior_df["participant"] == subj_id].copy()
    if "condition" in sub.columns:
        sub = sub[sub["condition"] == day].copy()
    if preprocess_fn is not None:
        sub = preprocess_fn(sub)

    include_cols = list(include_cols or [])
    out_cols = ["participant", "condition", trial_col, "epoch_idx", "alignment_valid", condition_col]
    for col in include_cols:
        if col not in out_cols:
            out_cols.append(col)

    if len(sub) == 0:
        return pd.DataFrame(columns=out_cols), trial_col
    if trial_col not in sub.columns:
        raise ValueError(f"behavior_df missing '{trial_col}'")
    if condition_col not in sub.columns:
        raise ValueError(f"behavior_df missing '{condition_col}'")

    sub = sub.dropna(subset=[trial_col, condition_col]).copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=out_cols), trial_col

    trial_num = pd.to_numeric(sub[trial_col], errors="coerce")
    sub = sub.loc[trial_num.notna()].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=out_cols), trial_col

    sub["epoch_idx"] = trial_num.loc[sub.index].astype(int) - 1
    sub["alignment_valid"] = (sub["epoch_idx"] >= 0) & (sub["epoch_idx"] < int(n_epochs))

    if allowed_conditions is not None:
        sub = sub[sub[condition_col].isin(list(allowed_conditions))].copy()

    keep_cols = [c for c in out_cols if c in sub.columns]
    return sub[keep_cols].reset_index(drop=True), trial_col


def build_rpe_trial_map(
    learn_df: pd.DataFrame,
    subj_id: str,
    n_epochs: int,
    day: str = "Day1",
    event: str = "feedback_start",
    trial_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """Build epoch-aligned RPE category map from an event-aligned behavior table."""
    if "rpe" not in learn_df.columns and "rpe_cat" not in learn_df.columns:
        raise ValueError("behavior_df missing both 'rpe' and 'rpe_cat'")

    def _prep(sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        if "rpe_cat" not in out.columns or out["rpe_cat"].isna().all():
            if "rpe" not in out.columns:
                raise ValueError("behavior_df missing both 'rpe' and 'rpe_cat'")
            vals = pd.to_numeric(out["rpe"], errors="coerce")
            try:
                out["rpe_cat"] = pd.qcut(vals, q=3, labels=["neg", "neutral", "pos"], duplicates="drop")
            except Exception:
                out["rpe_cat"] = np.nan
        out["rpe_cat"] = _normalize_rpe_cat_series(out["rpe_cat"])
        return out

    trial_map, _ = _build_condition_trial_map(
        behavior_df=learn_df,
        subj_id=subj_id,
        n_epochs=n_epochs,
        condition_col="rpe_cat",
        day=day,
        event=event,
        trial_col_override=trial_col_override,
        preprocess_fn=_prep,
        allowed_conditions=("neg", "neutral", "pos"),
    )
    return trial_map


# -----------------------------------------------------------------------------
# MVGC core
# -----------------------------------------------------------------------------

def compute_multivariate_gc(
    epochs: mne.Epochs,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    fmin: float = 4,
    fmax: float = 8,
    n_cycles: float = 5,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "gc_forward": None,
        "gc_backward": None,
        "gc_net": None,
        "times": None,
        "freqs": None,
        "success": False,
    }

    if len(source_indices) == 0 or len(target_indices) == 0:
        if verbose:
            print("No source/target channels")
        return result

    if len(epochs) < 5:
        if verbose:
            print(f"Insufficient epochs ({len(epochs)})")
        return result

    try:
        if n_freqs is None:
            bandwidth = fmax - fmin
            if bandwidth < 10:
                n_freqs = 20
            elif bandwidth < 20:
                n_freqs = 10
            else:
                n_freqs = 5

        cwt_freqs = np.linspace(fmin, fmax, int(n_freqs))

        con_fwd = spectral_connectivity_epochs(
            epochs,
            method="gc",
            indices=make_multivariate_indices(source_indices, target_indices),
            mode="cwt_morlet",
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=n_cycles,
            gc_n_lags=gc_n_lags,
            verbose=False,
        )
        con_bwd = spectral_connectivity_epochs(
            epochs,
            method="gc",
            indices=make_multivariate_indices(target_indices, source_indices),
            mode="cwt_morlet",
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=n_cycles,
            gc_n_lags=gc_n_lags,
            verbose=False,
        )

        gc_fwd = np.squeeze(con_fwd.get_data())
        gc_bwd = np.squeeze(con_bwd.get_data())

        if gc_fwd.ndim > 1:
            gc_fwd = np.mean(gc_fwd, axis=0)
            gc_bwd = np.mean(gc_bwd, axis=0)

        result["gc_forward"] = np.asarray(gc_fwd)
        result["gc_backward"] = np.asarray(gc_bwd)
        result["gc_net"] = np.asarray(gc_fwd) - np.asarray(gc_bwd)
        result["times"] = np.asarray(con_fwd.times)
        result["freqs"] = np.asarray(cwt_freqs)
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        if verbose:
            print(f"GC failure: {exc}")

    return result


def compute_multivariate_gc_from_arrays(
    source_data: np.ndarray,
    target_data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    fmin: float,
    fmax: float,
    gc_n_lags: int,
    n_freqs: Optional[int],
    n_cycles: float = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute multivariate GC from explicit source/target trial arrays."""
    source_data = np.asarray(source_data, dtype=np.float64)
    target_data = np.asarray(target_data, dtype=np.float64)

    if source_data.ndim != 3 or target_data.ndim != 3:
        return {"success": False, "error": "source_data and target_data must be 3D"}
    if source_data.shape[0] != target_data.shape[0] or source_data.shape[2] != target_data.shape[2]:
        return {"success": False, "error": "source/target array shape mismatch"}
    if source_data.shape[0] < 5:
        return {"success": False, "error": "insufficient_trials"}

    n_trials, n_source, n_times = source_data.shape
    _, n_target, _ = target_data.shape
    data = np.concatenate([source_data, target_data], axis=1)
    ch_names = [f"src_{i}" for i in range(n_source)] + [f"tgt_{i}" for i in range(n_target)]
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=["seeg"] * len(ch_names))
    tmin = float(np.asarray(times)[0]) if np.asarray(times).size > 0 else 0.0
    epochs = mne.EpochsArray(data, info, tmin=tmin, baseline=None, verbose=False)

    source_idx = np.arange(n_source, dtype=int)
    target_idx = np.arange(n_source, n_source + n_target, dtype=int)
    out = compute_multivariate_gc(
        epochs=epochs,
        source_indices=source_idx,
        target_indices=target_idx,
        fmin=fmin,
        fmax=fmax,
        n_cycles=n_cycles,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        verbose=verbose,
    )
    if out.get("success", False):
        out["times"] = np.asarray(times, dtype=np.float64)
    out["n_trials"] = int(n_trials)
    out["n_source"] = int(n_source)
    out["n_target"] = int(n_target)
    out["n_times"] = int(n_times)
    return out


def compute_null_stats_one_sided(real: np.ndarray, surr: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """Compute one-sided null stats: p = P(null >= observed)."""
    real = np.asarray(real, dtype=np.float64)
    surr = np.asarray(surr, dtype=np.float64)
    if surr.ndim != 2:
        raise ValueError("surr must be 2D [n_surrogates, n_times]")
    n = surr.shape[0]
    if n == 0:
        nan = np.full_like(real, np.nan, dtype=np.float64)
        return {"mean": nan, "std": nan, "z": nan, "p": nan}

    mean = np.nanmean(surr, axis=0)
    std = np.nanstd(surr, axis=0)
    std = np.maximum(std, eps)
    z = (real - mean) / std
    p = (np.sum(surr >= real[None, :], axis=0) + 1.0) / (n + 1.0)
    return {"mean": mean, "std": std, "z": z, "p": p}


def compute_gc_surrogates_source_shuffle(
    epochs: mne.Epochs,
    source_idx: Sequence[int],
    target_idx: Sequence[int],
    trial_ix: np.ndarray,
    n_surrogates: int = 500,
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    surrogate_tail: str = "greater",
    store_surrogates: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Surrogate null via source-only trial shuffling (within condition)."""
    if surrogate_tail != "greater":
        raise ValueError("Only one-sided 'greater' surrogate tail is currently supported.")
    if len(source_idx) == 0 or len(target_idx) == 0:
        return {"success": False, "error": "empty_source_or_target_indices"}
    trial_ix = np.asarray(trial_ix, dtype=int)
    if len(trial_ix) < 5:
        return {"success": False, "error": "insufficient_trials"}

    data = epochs.get_data()
    source_trials = np.asarray(data[trial_ix][:, np.asarray(source_idx, dtype=int), :], dtype=np.float64)
    target_trials = np.asarray(data[trial_ix][:, np.asarray(target_idx, dtype=int), :], dtype=np.float64)

    real = compute_multivariate_gc_from_arrays(
        source_data=source_trials,
        target_data=target_trials,
        sfreq=float(epochs.info["sfreq"]),
        times=np.asarray(epochs.times),
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        verbose=verbose,
    )
    if not real.get("success", False):
        return {"success": False, "error": f"real_gc_failed:{real.get('error', 'unknown')}"}

    n_trials = source_trials.shape[0]
    if n_trials < 2:
        return {"success": False, "error": "insufficient_trials_for_shuffle"}

    # Reproducible seeds for each permutation
    seed_seq = np.random.SeedSequence(int(random_state))
    child_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(int(n_surrogates))]

    def _single_surrogate(seed: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_trials)
        src_shuf = source_trials[perm]
        out = compute_multivariate_gc_from_arrays(
            source_data=src_shuf,
            target_data=target_trials,
            sfreq=float(epochs.info["sfreq"]),
            times=np.asarray(epochs.times),
            fmin=fmin,
            fmax=fmax,
            gc_n_lags=gc_n_lags,
            n_freqs=n_freqs,
            verbose=False,
        )
        if not out.get("success", False):
            return None
        return (
            np.asarray(out["gc_forward"], dtype=np.float64),
            np.asarray(out["gc_backward"], dtype=np.float64),
            np.asarray(out["gc_net"], dtype=np.float64),
        )

    # Guardrail: avoid parallel overhead for tiny surrogate counts
    use_parallel = int(n_jobs_surrogates) > 1 and int(n_surrogates) >= 20
    if use_parallel:
        n_jobs_eff = max(1, int(n_jobs_surrogates))
        chunks = np.array_split(np.asarray(child_seeds, dtype=np.int64), n_jobs_eff)
        chunks = [c.tolist() for c in chunks if len(c) > 0]

        def _run_chunk(seed_chunk: List[int]) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            return [_single_surrogate(seed) for seed in seed_chunk]

        chunk_results = Parallel(
            n_jobs=n_jobs_eff,
            backend=surrogate_parallel_backend,
            batch_size=1,
            prefer=None,
            verbose=0,
        )(delayed(_run_chunk)(chunk) for chunk in chunks)
        surrogate_out = [item for chunk in chunk_results for item in chunk]
    else:
        surrogate_out = [_single_surrogate(seed) for seed in child_seeds]

    valid = [x for x in surrogate_out if x is not None]
    if len(valid) == 0:
        return {"success": False, "error": "all_surrogates_failed"}

    surr_forward = np.stack([x[0] for x in valid], axis=0)
    surr_backward = np.stack([x[1] for x in valid], axis=0)
    surr_net = np.stack([x[2] for x in valid], axis=0)

    stats_fwd = compute_null_stats_one_sided(np.asarray(real["gc_forward"]), surr_forward)
    stats_bwd = compute_null_stats_one_sided(np.asarray(real["gc_backward"]), surr_backward)
    stats_net = compute_null_stats_one_sided(np.asarray(real["gc_net"]), surr_net)

    out: Dict[str, Any] = {
        "success": True,
        "real_forward": np.asarray(real["gc_forward"]),
        "real_backward": np.asarray(real["gc_backward"]),
        "real_net": np.asarray(real["gc_net"]),
        "times": np.asarray(real["times"]),
        "freqs": np.asarray(real["freqs"]),
        "z_forward": stats_fwd["z"],
        "z_backward": stats_bwd["z"],
        "z_net": stats_net["z"],
        "p_forward": stats_fwd["p"],
        "p_backward": stats_bwd["p"],
        "p_net": stats_net["p"],
        "surr_mean_forward": stats_fwd["mean"],
        "surr_mean_backward": stats_bwd["mean"],
        "surr_mean_net": stats_net["mean"],
        "surr_std_forward": stats_fwd["std"],
        "surr_std_backward": stats_bwd["std"],
        "surr_std_net": stats_net["std"],
        "n_surrogates": int(len(valid)),
        "n_surrogates_requested": int(n_surrogates),
        "random_state": int(random_state),
        "surrogate_tail": surrogate_tail,
        "surrogate_parallel_backend": surrogate_parallel_backend,
        "source_shuffle": True,
    }
    if store_surrogates:
        out["surr_forward"] = surr_forward
        out["surr_backward"] = surr_backward
        out["surr_net"] = surr_net
    return out


def _compute_gc_by_condition(
    epoch_data: Dict[str, Any],
    trial_map: pd.DataFrame,
    condition_col: str,
    condition_order: Sequence[str],
    trial_value_col: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    seed_offset_by_condition: Optional[Dict[str, int]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    valid_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map
    n_dropped_alignment = int((~trial_map["alignment_valid"]).sum()) if len(trial_map) > 0 else 0
    if seed_offset_by_condition is None:
        seed_offset_by_condition = {cond: idx + 1 for idx, cond in enumerate(condition_order)}

    results: Dict[str, Any] = {
        "trial_map": trial_map,
        "condition_col": condition_col,
        "trial_value_col": trial_value_col,
        "condition_order": list(condition_order),
        "n_alignment_total": int(len(trial_map)),
        "n_alignment_valid": int(len(valid_map)),
        "n_dropped_alignment": n_dropped_alignment,
        "surrogate_config": {
            "use_surrogates": bool(use_surrogates),
            "n_surrogates": int(n_surrogates),
            "surrogate_tail": surrogate_tail,
            "random_state": int(random_state),
            "n_jobs_surrogates": int(n_jobs_surrogates),
            "surrogate_parallel_backend": surrogate_parallel_backend,
            "source_shuffle_only": True,
        },
    }

    for hemi in ("l", "r"):
        source_idx = epoch_data.get(f"source_indices_{hemi}", np.array([]))
        target_idx = epoch_data.get(f"target_indices_{hemi}", np.array([]))
        n_source = int(epoch_data.get(f"n_source_{hemi}", 0))
        n_target = int(epoch_data.get(f"n_target_{hemi}", 0))
        if n_source == 0 or n_target == 0:
            continue

        hemi_results: Dict[str, Any] = {
            "n_source": n_source,
            "n_target": n_target,
            "source_channels": epoch_data.get(f"source_channels_{hemi}", []),
            "target_channels": epoch_data.get(f"target_channels_{hemi}", []),
        }

        for cond in condition_order:
            cond_map = valid_map[valid_map[condition_col] == cond].copy()
            trial_ix = cond_map["epoch_idx"].to_numpy(dtype=int)
            trial_vals = cond_map[trial_value_col].tolist() if trial_value_col in cond_map.columns else []

            if len(trial_ix) < min_trials:
                hemi_results[cond] = {
                    "success": False,
                    "n_trials": int(len(trial_ix)),
                    "trial_indices": trial_ix.tolist(),
                    trial_value_col: trial_vals,
                }
                continue

            if use_surrogates:
                gc_sur = compute_gc_surrogates_source_shuffle(
                    epochs=epochs,
                    source_idx=source_idx,
                    target_idx=target_idx,
                    trial_ix=trial_ix,
                    n_surrogates=n_surrogates,
                    random_state=random_state + (0 if hemi == "l" else 10000) + int(seed_offset_by_condition.get(cond, 1)),
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    surrogate_tail=surrogate_tail,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                )
                if gc_sur.get("success", False):
                    cond_result: Dict[str, Any] = {
                        "success": True,
                        "gc_forward": gc_sur["real_forward"],
                        "gc_backward": gc_sur["real_backward"],
                        "gc_net": gc_sur["real_net"],
                        "gc_forward_z": gc_sur["z_forward"],
                        "gc_backward_z": gc_sur["z_backward"],
                        "gc_net_z": gc_sur["z_net"],
                        "p_forward": gc_sur["p_forward"],
                        "p_backward": gc_sur["p_backward"],
                        "p_net": gc_sur["p_net"],
                        "surr_mean_forward": gc_sur["surr_mean_forward"],
                        "surr_mean_backward": gc_sur["surr_mean_backward"],
                        "surr_mean_net": gc_sur["surr_mean_net"],
                        "surr_std_forward": gc_sur["surr_std_forward"],
                        "surr_std_backward": gc_sur["surr_std_backward"],
                        "surr_std_net": gc_sur["surr_std_net"],
                        "times": gc_sur["times"],
                        "freqs": gc_sur["freqs"],
                        "n_surrogates": gc_sur["n_surrogates"],
                        "surrogate_tail": gc_sur["surrogate_tail"],
                        "source_shuffle": gc_sur["source_shuffle"],
                    }
                    if "surr_forward" in gc_sur:
                        cond_result["surr_forward"] = gc_sur["surr_forward"]
                        cond_result["surr_backward"] = gc_sur["surr_backward"]
                        cond_result["surr_net"] = gc_sur["surr_net"]
                else:
                    cond_result = {"success": False, "error": gc_sur.get("error", "surrogate_failed")}
            else:
                cond_result = compute_multivariate_gc(
                    epochs=epochs[trial_ix],
                    source_indices=source_idx,
                    target_indices=target_idx,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    verbose=verbose,
                )

            cond_result["n_trials"] = int(len(trial_ix))
            cond_result["trial_indices"] = trial_ix.tolist()
            cond_result[trial_value_col] = trial_vals
            hemi_results[cond] = cond_result

        results[hemi] = hemi_results

    return results


def compute_gc_by_hit_miss(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    event: str = "feedback_start",
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    trial_map = build_hit_miss_trial_map(
        combined_df,
        subj_id,
        n_epochs=len(epochs),
        day=day,
        event=event,
    )
    if str(event).strip().lower() == "cue_on":
        pref = ("trials_mem", "trials_gamble", "trials")
    else:
        pref = ("trials_gamble", "trials", "trials_mem")
    trial_value_col = next((c for c in pref if c in trial_map.columns), _resolve_trial_col_for_event(combined_df, event))
    return _compute_gc_by_condition(
        epoch_data=epoch_data,
        trial_map=trial_map,
        condition_col="hit_miss",
        condition_order=("hit", "miss"),
        trial_value_col=trial_value_col,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        seed_offset_by_condition={"hit": 1, "miss": 2},
        verbose=verbose,
    )


def compute_gc_by_reward(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    event: str = "feedback_start",
) -> Dict[str, Any]:
    """Compute MVGC split by binary reward (rewarded vs unrewarded) from combined_df or learn-like table."""
    epochs = epoch_data["epochs"]
    trial_map = build_reward_trial_map(
        combined_df,
        subj_id,
        n_epochs=len(epochs),
        day=day,
        event=event,
    )
    if str(event).strip().lower() == "cue_on":
        pref = ("trials_mem", "trials_gamble", "trials")
    else:
        pref = ("trials_gamble", "trials", "trials_mem")
    trial_value_col = next((c for c in pref if c in trial_map.columns), _resolve_trial_col_for_event(combined_df, event))
    return _compute_gc_by_condition(
        epoch_data=epoch_data,
        trial_map=trial_map,
        condition_col="reward_outcome",
        condition_order=REWARD_OUTCOME_CONDITIONS,
        trial_value_col=trial_value_col,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        seed_offset_by_condition={"rewarded": 1, "unrewarded": 2},
        verbose=verbose,
    )


def compute_gc_by_memory_outcome(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    event: str = "feedback_start",
) -> Dict[str, Any]:
    """Compute MVGC split by memory_outcome (hits, misses, correct_rejections, false_alarms) from combined_df."""
    epochs = epoch_data["epochs"]
    trial_map = build_memory_outcome_trial_map(
        combined_df,
        subj_id,
        n_epochs=len(epochs),
        day=day,
        event=event,
    )
    if str(event).strip().lower() == "cue_on":
        pref = ("trials_mem", "trials_gamble", "trials")
    else:
        pref = ("trials_gamble", "trials", "trials_mem")
    trial_value_col = next((c for c in pref if c in trial_map.columns), _resolve_trial_col_for_event(combined_df, event))
    seed_offset = {cond: idx + 1 for idx, cond in enumerate(MEMORY_OUTCOME_CONDITIONS)}
    return _compute_gc_by_condition(
        epoch_data=epoch_data,
        trial_map=trial_map,
        condition_col="memory_outcome",
        condition_order=MEMORY_OUTCOME_CONDITIONS,
        trial_value_col=trial_value_col,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        seed_offset_by_condition=seed_offset,
        verbose=verbose,
    )


def compute_gc_by_rpe_category(
    epoch_data: Dict[str, Any],
    learn_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    event: str = "feedback_start",
) -> Dict[str, Any]:
    """Compute RPE-category MVGC from an event-aligned behavior table.

    The `learn_df` argument may be any dataframe containing `participant`, an
    event-appropriate trial column, and either `rpe_cat` or `rpe`.
    """
    epochs = epoch_data["epochs"]
    trial_map = build_rpe_trial_map(
        learn_df,
        subj_id,
        n_epochs=len(epochs),
        day=day,
        event=event,
    )
    trial_value_col = _resolve_trial_col_for_event(learn_df, event)
    return _compute_gc_by_condition(
        epoch_data=epoch_data,
        trial_map=trial_map,
        condition_col="rpe_cat",
        condition_order=("neg", "neutral", "pos"),
        trial_value_col=trial_value_col,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        seed_offset_by_condition={"neg": 1, "neutral": 2, "pos": 3},
        verbose=verbose,
    )


def combine_hemispheres(
    gc_by_hemi: Dict[str, Any],
    condition_keys: Tuple[str, ...] = ("hit", "miss"),
) -> Dict[str, Any]:
    def _mean_if_present(rows: List[Dict[str, Any]], key: str) -> Optional[np.ndarray]:
        vals = [np.asarray(r[key]) for r in rows if key in r and r[key] is not None]
        if len(vals) == 0:
            return None
        return np.mean(vals, axis=0)

    trial_value_col = str(gc_by_hemi.get("trial_value_col", "trials_gamble"))
    combined: Dict[str, Any] = {}
    for cond in condition_keys:
        valid_results = []
        total_trials = 0
        times = None
        trial_indices_all: List[int] = []
        trial_values_all: List[int] = []
        trials_gamble_all: List[int] = []
        trials_mem_all: List[int] = []
        trials_all: List[int] = []

        for hemi in ("l", "r"):
            if hemi not in gc_by_hemi:
                continue
            hemi_data = gc_by_hemi[hemi]
            if cond not in hemi_data:
                continue
            cond_data = hemi_data[cond]
            if not cond_data.get("success", False):
                continue
            valid_results.append(cond_data)
            total_trials += int(cond_data.get("n_trials", 0))
            if times is None:
                times = cond_data.get("times", None)
            trial_indices_all.extend(cond_data.get("trial_indices", []))
            trial_values_all.extend(cond_data.get(trial_value_col, []))
            trials_gamble_all.extend(cond_data.get("trials_gamble", []))
            trials_mem_all.extend(cond_data.get("trials_mem", []))
            trials_all.extend(cond_data.get("trials", []))

        if len(valid_results) == 0:
            combined[cond] = {"success": False, "n_trials": 0}
            continue

        gc_forward = np.mean([np.asarray(r["gc_forward"]) for r in valid_results], axis=0)
        gc_backward = np.mean([np.asarray(r["gc_backward"]) for r in valid_results], axis=0)
        gc_net = np.mean([np.asarray(r["gc_net"]) for r in valid_results], axis=0)
        gc_forward_z = _mean_if_present(valid_results, "gc_forward_z")
        gc_backward_z = _mean_if_present(valid_results, "gc_backward_z")
        gc_net_z = _mean_if_present(valid_results, "gc_net_z")
        p_forward = _mean_if_present(valid_results, "p_forward")
        p_backward = _mean_if_present(valid_results, "p_backward")
        p_net = _mean_if_present(valid_results, "p_net")
        surr_mean_forward = _mean_if_present(valid_results, "surr_mean_forward")
        surr_mean_backward = _mean_if_present(valid_results, "surr_mean_backward")
        surr_mean_net = _mean_if_present(valid_results, "surr_mean_net")
        surr_std_forward = _mean_if_present(valid_results, "surr_std_forward")
        surr_std_backward = _mean_if_present(valid_results, "surr_std_backward")
        surr_std_net = _mean_if_present(valid_results, "surr_std_net")
        n_surrogates = [int(r.get("n_surrogates", 0)) for r in valid_results if "n_surrogates" in r]

        unique_trial_indices = sorted(set([int(x) for x in trial_indices_all]))
        out = {
            "success": True,
            "gc_forward": gc_forward,
            "gc_backward": gc_backward,
            "gc_net": gc_net,
            "times": np.asarray(times) if times is not None else None,
            "n_trials": len(unique_trial_indices),
            "n_hemispheres": int(len(valid_results)),
            "trial_indices": unique_trial_indices,
            trial_value_col: sorted(set([int(x) for x in trial_values_all if pd.notna(x)])),
            "trials_gamble": sorted(set([int(x) for x in trials_gamble_all if pd.notna(x)])),
            "trials_mem": sorted(set([int(x) for x in trials_mem_all if pd.notna(x)])),
            "trials": sorted(set([int(x) for x in trials_all if pd.notna(x)])),
        }
        if gc_forward_z is not None:
            out["gc_forward_z"] = gc_forward_z
        if gc_backward_z is not None:
            out["gc_backward_z"] = gc_backward_z
        if gc_net_z is not None:
            out["gc_net_z"] = gc_net_z
        if p_forward is not None:
            out["p_forward"] = p_forward
        if p_backward is not None:
            out["p_backward"] = p_backward
        if p_net is not None:
            out["p_net"] = p_net
        if surr_mean_forward is not None:
            out["surr_mean_forward"] = surr_mean_forward
        if surr_mean_backward is not None:
            out["surr_mean_backward"] = surr_mean_backward
        if surr_mean_net is not None:
            out["surr_mean_net"] = surr_mean_net
        if surr_std_forward is not None:
            out["surr_std_forward"] = surr_std_forward
        if surr_std_backward is not None:
            out["surr_std_backward"] = surr_std_backward
        if surr_std_net is not None:
            out["surr_std_net"] = surr_std_net
        if len(n_surrogates) > 0:
            out["n_surrogates"] = int(np.round(np.mean(n_surrogates)))

        combined[cond] = out

    return combined


def process_subject_gc_hit_miss(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Optional[Dict[str, Any]]:
    epoch_data = load_subject_epochs(
        subj_id=subj_id,
        base_dir=base_dir,
        event=event,
        roi_source=roi_source,
        roi_target=roi_target,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        day=day,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        fmin=fmin,
        fmax=fmax,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )

    if epoch_data is None:
        return None

    gc_results = compute_gc_by_hit_miss(
        epoch_data=epoch_data,
        combined_df=combined_df,
        subj_id=subj_id,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        event=event,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )

    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "times": epoch_data["times"],
        "sfreq": epoch_data["sfreq"],
        "original_sfreq": epoch_data.get("original_sfreq", epoch_data["sfreq"]),
        "n_epochs_total": epoch_data["n_epochs"],
        "gc_by_hemi": gc_results,
        "epoch_data": epoch_data,
        "surrogate_config": gc_results.get("surrogate_config", {}),
    }


def process_subject_gc_reward(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Full pipeline: load epochs, compute GC by binary reward (rewarded vs unrewarded)."""
    epoch_data = load_subject_epochs(
        subj_id=subj_id,
        base_dir=base_dir,
        event=event,
        roi_source=roi_source,
        roi_target=roi_target,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        day=day,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        fmin=fmin,
        fmax=fmax,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )

    if epoch_data is None:
        return None

    gc_results = compute_gc_by_reward(
        epoch_data=epoch_data,
        combined_df=combined_df,
        subj_id=subj_id,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        event=event,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )

    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "times": epoch_data["times"],
        "sfreq": epoch_data["sfreq"],
        "original_sfreq": epoch_data.get("original_sfreq", epoch_data["sfreq"]),
        "n_epochs_total": epoch_data["n_epochs"],
        "gc_by_hemi": gc_results,
        "epoch_data": epoch_data,
        "surrogate_config": gc_results.get("surrogate_config", {}),
    }


def process_subject_gc_memory_outcome(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Full pipeline: load epochs, compute GC by memory_outcome (hits, misses, correct_rejections, false_alarms)."""
    epoch_data = load_subject_epochs(
        subj_id=subj_id,
        base_dir=base_dir,
        event=event,
        roi_source=roi_source,
        roi_target=roi_target,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        day=day,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        fmin=fmin,
        fmax=fmax,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )

    if epoch_data is None:
        return None

    gc_results = compute_gc_by_memory_outcome(
        epoch_data=epoch_data,
        combined_df=combined_df,
        subj_id=subj_id,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        event=event,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )

    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "times": epoch_data["times"],
        "sfreq": epoch_data["sfreq"],
        "original_sfreq": epoch_data.get("original_sfreq", epoch_data["sfreq"]),
        "n_epochs_total": epoch_data["n_epochs"],
        "gc_by_hemi": gc_results,
        "epoch_data": epoch_data,
        "surrogate_config": gc_results.get("surrogate_config", {}),
    }


def process_subject_gc_rpe(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    learn_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Optional[Dict[str, Any]]:
    epoch_data = load_subject_epochs(
        subj_id=subj_id,
        base_dir=base_dir,
        event=event,
        roi_source=roi_source,
        roi_target=roi_target,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        day=day,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        fmin=fmin,
        fmax=fmax,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )
    if epoch_data is None:
        return None

    gc_results = compute_gc_by_rpe_category(
        epoch_data=epoch_data,
        learn_df=learn_df,
        subj_id=subj_id,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        event=event,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )

    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "times": epoch_data["times"],
        "sfreq": epoch_data["sfreq"],
        "original_sfreq": epoch_data.get("original_sfreq", epoch_data["sfreq"]),
        "n_epochs_total": epoch_data["n_epochs"],
        "gc_by_hemi": gc_results,
        "epoch_data": epoch_data,
        "surrogate_config": gc_results.get("surrogate_config", {}),
    }


def export_gc_to_dataframe(
    result_dict: Dict[str, Any],
    subj_id: str,
    roi_source: str,
    roi_target: str,
    condition_keys: Tuple[str, ...] = ("hit", "miss"),
    condition_col: str = "hit_miss",
    trial_col: str = "trials_gamble",
) -> pd.DataFrame:
    combined_gc = combine_hemispheres(result_dict["gc_by_hemi"], condition_keys=condition_keys)
    n_dropped_alignment = int(result_dict["gc_by_hemi"].get("n_dropped_alignment", 0))

    rows: List[Dict[str, Any]] = []
    for cond in condition_keys:
        cond_data = combined_gc.get(cond, {})
        if not cond_data.get("success", False):
            continue

        times = np.asarray(cond_data["times"])
        for t_idx, t in enumerate(times):
            rows.append(
                {
                    "participant": subj_id,
                    "roi_source": roi_source,
                    "roi_target": roi_target,
                    condition_col: cond,
                    "time": float(t),
                    "gc_forward": float(cond_data["gc_forward"][t_idx]),
                    "gc_backward": float(cond_data["gc_backward"][t_idx]),
                    "gc_net": float(cond_data["gc_net"][t_idx]),
                    "gc_forward_z": float(cond_data["gc_forward_z"][t_idx]) if "gc_forward_z" in cond_data else np.nan,
                    "gc_backward_z": float(cond_data["gc_backward_z"][t_idx]) if "gc_backward_z" in cond_data else np.nan,
                    "gc_net_z": float(cond_data["gc_net_z"][t_idx]) if "gc_net_z" in cond_data else np.nan,
                    "p_forward": float(cond_data["p_forward"][t_idx]) if "p_forward" in cond_data else np.nan,
                    "p_backward": float(cond_data["p_backward"][t_idx]) if "p_backward" in cond_data else np.nan,
                    "p_net": float(cond_data["p_net"][t_idx]) if "p_net" in cond_data else np.nan,
                    "surr_mean_forward": float(cond_data["surr_mean_forward"][t_idx]) if "surr_mean_forward" in cond_data else np.nan,
                    "surr_mean_backward": float(cond_data["surr_mean_backward"][t_idx]) if "surr_mean_backward" in cond_data else np.nan,
                    "surr_mean_net": float(cond_data["surr_mean_net"][t_idx]) if "surr_mean_net" in cond_data else np.nan,
                    "surr_std_forward": float(cond_data["surr_std_forward"][t_idx]) if "surr_std_forward" in cond_data else np.nan,
                    "surr_std_backward": float(cond_data["surr_std_backward"][t_idx]) if "surr_std_backward" in cond_data else np.nan,
                    "surr_std_net": float(cond_data["surr_std_net"][t_idx]) if "surr_std_net" in cond_data else np.nan,
                    "n_trials": int(cond_data.get("n_trials", 0)),
                    "n_hemispheres": int(cond_data.get("n_hemispheres", 0)),
                    "n_surrogates": int(cond_data.get("n_surrogates", 0)),
                    "n_dropped_alignment": n_dropped_alignment,
                    "trial_indices": ",".join(map(str, cond_data.get("trial_indices", []))),
                    trial_col: ",".join(map(str, cond_data.get(trial_col, []))),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# DataFrame cache helpers
# -----------------------------------------------------------------------------

def _stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _subject_day_behavior_fingerprint(combined_df: pd.DataFrame, subj_id: str, day: str = "Day1") -> str:
    sub = combined_df[combined_df["participant"] == subj_id].copy() if "participant" in combined_df.columns else combined_df.copy()
    if "condition" in sub.columns:
        sub = sub[sub["condition"] == day].copy()
    keep_cols = [c for c in ["participant", "condition", "trials_gamble", "hits", "reward_gamble", "gamble_gamble", "rpe"] if c in sub.columns]
    if len(keep_cols) > 0:
        sub = sub[keep_cols].copy()
    if "trials_gamble" in sub.columns:
        sub["trials_gamble"] = pd.to_numeric(sub["trials_gamble"], errors="coerce")
        sub = sub.sort_values(["trials_gamble"] + [c for c in ["participant", "condition"] if c in sub.columns]).reset_index(drop=True)
    csv_text = sub.to_csv(index=False)
    return hashlib.sha1(csv_text.encode("utf-8")).hexdigest()


def _build_gc_df_cache_key(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    tmin: float,
    tmax: float,
    target_sfreq: float,
    fmin: float,
    fmax: float,
    gc_n_lags: int,
    n_freqs: Optional[int],
    min_trials: int,
    use_surrogates: bool,
    n_surrogates: int,
    surrogate_tail: str,
    random_state: int,
    surrogate_parallel_backend: str,
    filter_by_power: bool,
    power_percentile: float,
    tfr_time_window: Optional[Tuple[float, float]],
    use_top_electrode_only: bool,
    behavior_fingerprint: str,
    behavior_trial_col: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "day": day,
        "tmin": float(tmin),
        "tmax": float(tmax),
        "target_sfreq": float(target_sfreq),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "gc_n_lags": int(gc_n_lags),
        "n_freqs": None if n_freqs is None else int(n_freqs),
        "min_trials": int(min_trials),
        "use_surrogates": bool(use_surrogates),
        "n_surrogates": int(n_surrogates),
        "surrogate_tail": surrogate_tail,
        "random_state": int(random_state),
        "surrogate_parallel_backend": surrogate_parallel_backend,
        "filter_by_power": bool(filter_by_power),
        "power_percentile": float(power_percentile),
        "tfr_time_window": None if tfr_time_window is None else [float(tfr_time_window[0]), float(tfr_time_window[1])],
        "use_top_electrode_only": bool(use_top_electrode_only),
        "behavior_fingerprint": behavior_fingerprint,
        "behavior_trial_col": behavior_trial_col,
    }


def build_gc_df_cache_path(
    cache_dir: str,
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    cache_key_hash: str,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{subj_id}_{roi_source}_to_{roi_target}_{event}_{day}_{cache_key_hash[:12]}_gc_timeseries.csv.gz"
    return os.path.join(cache_dir, filename)


def build_gc_memory_outcome_cache_path(
    cache_dir: str,
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    cache_key_hash: str,
) -> str:
    """Cache path for memory_outcome GC; distinct from hit_miss so caches do not collide."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{subj_id}_{roi_source}_to_{roi_target}_{event}_{day}_{cache_key_hash[:12]}_memory_outcome_gc_timeseries.csv.gz"
    return os.path.join(cache_dir, filename)


def build_gc_reward_cache_path(
    cache_dir: str,
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    cache_key_hash: str,
) -> str:
    """Cache path for binary-reward GC; distinct from hit_miss / memory_outcome."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{subj_id}_{roi_source}_to_{roi_target}_{event}_{day}_{cache_key_hash[:12]}_reward_gc_timeseries.csv.gz"
    return os.path.join(cache_dir, filename)


def load_gc_df_cache(cache_file: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return None
    required = {"participant", "roi_source", "roi_target", "hit_miss", "time", "gc_forward", "gc_backward", "gc_net"}
    if len(required.difference(df.columns)) > 0:
        return None
    return df


def load_gc_memory_outcome_df_cache(cache_file: str) -> Optional[pd.DataFrame]:
    """Load cached GC DataFrame with memory_outcome condition column."""
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return None
    required = {"participant", "roi_source", "roi_target", "memory_outcome", "time", "gc_forward", "gc_backward", "gc_net"}
    if len(required.difference(df.columns)) > 0:
        return None
    return df


def load_gc_reward_df_cache(cache_file: str) -> Optional[pd.DataFrame]:
    """Load cached GC DataFrame with reward_outcome condition column."""
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return None
    required = {"participant", "roi_source", "roi_target", "reward_outcome", "time", "gc_forward", "gc_backward", "gc_net"}
    if len(required.difference(df.columns)) > 0:
        return None
    return df


def save_gc_df_cache(df: pd.DataFrame, cache_file: str) -> str:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, index=False, compression="gzip")
    return cache_file


def run_or_load_subject_gc_df(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    cache_dir: str,
    overwrite_cache: bool = False,
    return_result_dict: bool = False,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Dict[str, Any]]:
    behavior_fp = _subject_day_behavior_fingerprint(combined_df, subj_id=subj_id, day=day)
    behavior_trial_col = _resolve_trial_col_for_event(combined_df, event)
    key_dict = _build_gc_df_cache_key(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        surrogate_parallel_backend=surrogate_parallel_backend,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
        behavior_fingerprint=behavior_fp,
        behavior_trial_col=behavior_trial_col,
    )
    key_hash = hashlib.sha1(_stable_json_dumps(key_dict).encode("utf-8")).hexdigest()
    cache_file = build_gc_df_cache_path(
        cache_dir=cache_dir,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        cache_key_hash=key_hash,
    )

    if (not overwrite_cache) and os.path.exists(cache_file):
        cached_df = load_gc_df_cache(cache_file)
        if cached_df is not None:
            meta = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cache_key_hash": key_hash,
                "cache_key": key_dict,
            }
            result = None
            if return_result_dict:
                result = process_subject_gc_hit_miss(
                    subj_id=subj_id,
                    roi_source=roi_source,
                    roi_target=roi_target,
                    base_dir=base_dir,
                    combined_df=combined_df,
                    event=event,
                    tmin=tmin,
                    tmax=tmax,
                    target_sfreq=target_sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    min_trials=min_trials,
                    day=day,
                    use_surrogates=use_surrogates,
                    n_surrogates=n_surrogates,
                    surrogate_tail=surrogate_tail,
                    random_state=random_state,
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                    filter_by_power=filter_by_power,
                    power_percentile=power_percentile,
                    tfr_time_window=tfr_time_window,
                    use_top_electrode_only=use_top_electrode_only,
                )
            return cached_df, result, meta

    result = process_subject_gc_hit_miss(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        combined_df=combined_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )
    if result is None:
        return pd.DataFrame(), None, {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key_hash": key_hash,
            "cache_key": key_dict,
            "error": "process_subject_gc_hit_miss returned None",
        }

    trial_col = str(result.get("gc_by_hemi", {}).get("trial_value_col", behavior_trial_col))
    df_subj = export_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        trial_col=trial_col,
    )
    save_gc_df_cache(df_subj, cache_file)
    meta = {
        "cache_hit": False,
        "cache_file": cache_file,
        "cache_key_hash": key_hash,
        "cache_key": key_dict,
    }
    if return_result_dict:
        return df_subj, result, meta
    return df_subj, None, meta


def run_or_load_subject_gc_memory_outcome_df(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    cache_dir: str,
    overwrite_cache: bool = False,
    return_result_dict: bool = False,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run or load cached GC split by memory_outcome (hits, misses, correct_rejections, false_alarms)."""
    behavior_fp = _subject_day_behavior_fingerprint(combined_df, subj_id=subj_id, day=day)
    behavior_trial_col = _resolve_trial_col_for_event(combined_df, event)
    key_dict = _build_gc_df_cache_key(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        surrogate_parallel_backend=surrogate_parallel_backend,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
        behavior_fingerprint=behavior_fp,
        behavior_trial_col=behavior_trial_col,
    )
    key_hash = hashlib.sha1(_stable_json_dumps(key_dict).encode("utf-8")).hexdigest()
    cache_file = build_gc_memory_outcome_cache_path(
        cache_dir=cache_dir,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        cache_key_hash=key_hash,
    )

    if (not overwrite_cache) and os.path.exists(cache_file):
        cached_df = load_gc_memory_outcome_df_cache(cache_file)
        if cached_df is not None:
            meta = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cache_key_hash": key_hash,
                "cache_key": key_dict,
            }
            result = None
            if return_result_dict:
                result = process_subject_gc_memory_outcome(
                    subj_id=subj_id,
                    roi_source=roi_source,
                    roi_target=roi_target,
                    base_dir=base_dir,
                    combined_df=combined_df,
                    event=event,
                    tmin=tmin,
                    tmax=tmax,
                    target_sfreq=target_sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    min_trials=min_trials,
                    day=day,
                    use_surrogates=use_surrogates,
                    n_surrogates=n_surrogates,
                    surrogate_tail=surrogate_tail,
                    random_state=random_state,
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                    filter_by_power=filter_by_power,
                    power_percentile=power_percentile,
                    tfr_time_window=tfr_time_window,
                    use_top_electrode_only=use_top_electrode_only,
                )
            return cached_df, result, meta

    result = process_subject_gc_memory_outcome(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        combined_df=combined_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )
    if result is None:
        return pd.DataFrame(), None, {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key_hash": key_hash,
            "cache_key": key_dict,
            "error": "process_subject_gc_memory_outcome returned None",
        }

    trial_col = str(result.get("gc_by_hemi", {}).get("trial_value_col", behavior_trial_col))
    df_subj = export_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        condition_keys=MEMORY_OUTCOME_CONDITIONS,
        condition_col="memory_outcome",
        trial_col=trial_col,
    )
    save_gc_df_cache(df_subj, cache_file)
    meta = {
        "cache_hit": False,
        "cache_file": cache_file,
        "cache_key_hash": key_hash,
        "cache_key": key_dict,
    }
    if return_result_dict:
        return df_subj, result, meta
    return df_subj, None, meta


def run_or_load_subject_gc_reward_df(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    cache_dir: str,
    overwrite_cache: bool = False,
    return_result_dict: bool = False,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run or load cached GC split by binary reward (rewarded vs unrewarded)."""
    behavior_fp = _subject_day_behavior_fingerprint(combined_df, subj_id=subj_id, day=day)
    behavior_trial_col = _resolve_trial_col_for_event(combined_df, event)
    key_dict = _build_gc_df_cache_key(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        surrogate_parallel_backend=surrogate_parallel_backend,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
        behavior_fingerprint=behavior_fp,
        behavior_trial_col=behavior_trial_col,
    )
    key_hash = hashlib.sha1(_stable_json_dumps(key_dict).encode("utf-8")).hexdigest()
    cache_file = build_gc_reward_cache_path(
        cache_dir=cache_dir,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        cache_key_hash=key_hash,
    )

    if (not overwrite_cache) and os.path.exists(cache_file):
        cached_df = load_gc_reward_df_cache(cache_file)
        if cached_df is not None:
            meta = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cache_key_hash": key_hash,
                "cache_key": key_dict,
            }
            result = None
            if return_result_dict:
                result = process_subject_gc_reward(
                    subj_id=subj_id,
                    roi_source=roi_source,
                    roi_target=roi_target,
                    base_dir=base_dir,
                    combined_df=combined_df,
                    event=event,
                    tmin=tmin,
                    tmax=tmax,
                    target_sfreq=target_sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    min_trials=min_trials,
                    day=day,
                    use_surrogates=use_surrogates,
                    n_surrogates=n_surrogates,
                    surrogate_tail=surrogate_tail,
                    random_state=random_state,
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                    filter_by_power=filter_by_power,
                    power_percentile=power_percentile,
                    tfr_time_window=tfr_time_window,
                    use_top_electrode_only=use_top_electrode_only,
                )
            return cached_df, result, meta

    result = process_subject_gc_reward(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        combined_df=combined_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )
    if result is None:
        return pd.DataFrame(), None, {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key_hash": key_hash,
            "cache_key": key_dict,
            "error": "process_subject_gc_reward returned None",
        }

    trial_col = str(result.get("gc_by_hemi", {}).get("trial_value_col", behavior_trial_col))
    df_subj = export_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        condition_keys=REWARD_OUTCOME_CONDITIONS,
        condition_col="reward_outcome",
        trial_col=trial_col,
    )
    save_gc_df_cache(df_subj, cache_file)
    meta = {
        "cache_hit": False,
        "cache_file": cache_file,
        "cache_key_hash": key_hash,
        "cache_key": key_dict,
    }
    if return_result_dict:
        return df_subj, result, meta
    return df_subj, None, meta


def _subject_day_learn_fingerprint(
    learn_df: pd.DataFrame,
    subj_id: str,
    day: str = "Day1",
    event: str = "feedback_start",
) -> str:
    sub = learn_df[learn_df["participant"] == subj_id].copy() if "participant" in learn_df.columns else learn_df.copy()
    if "condition" in sub.columns:
        sub = sub[sub["condition"] == day].copy()
    trial_col = None
    try:
        trial_col = _resolve_trial_col_for_event(sub, event)
    except Exception:
        trial_col = None
    if "rpe_cat" not in sub.columns and "rpe" in sub.columns:
        vals = pd.to_numeric(sub["rpe"], errors="coerce")
        try:
            sub["rpe_cat"] = pd.qcut(vals, q=3, labels=["neg", "neutral", "pos"], duplicates="drop")
        except Exception:
            sub["rpe_cat"] = np.nan
    if "rpe_cat" in sub.columns:
        sub["rpe_cat"] = _normalize_rpe_cat_series(sub["rpe_cat"])
    keep_pref = ["participant", "condition", "rpe", "rpe_cat", "reward", "reward_gamble", "gamble", "gamble_gamble"]
    if trial_col is not None:
        keep_pref.insert(2, trial_col)
    keep_cols = [c for c in keep_pref if c in sub.columns]
    if len(keep_cols) > 0:
        sub = sub[keep_cols].copy()
    if trial_col is not None and trial_col in sub.columns:
        sub[trial_col] = pd.to_numeric(sub[trial_col], errors="coerce")
        sort_cols = [trial_col] + [c for c in ["participant", "condition"] if c in sub.columns]
        sub = sub.sort_values(sort_cols).reset_index(drop=True)
    csv_text = sub.to_csv(index=False) + f"\n__event__={event}\n__trial_col__={trial_col}"
    return hashlib.sha1(csv_text.encode("utf-8")).hexdigest()


def build_gc_rpe_df_cache_path(
    cache_dir: str,
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    cache_key_hash: str,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{subj_id}_{roi_source}_to_{roi_target}_{event}_{day}_{cache_key_hash[:12]}_gc_rpe_timeseries.csv.gz"
    return os.path.join(cache_dir, filename)


def load_gc_rpe_df_cache(cache_file: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return None
    required = {"participant", "roi_source", "roi_target", "rpe_cat", "time", "gc_forward", "gc_backward", "gc_net"}
    if len(required.difference(df.columns)) > 0:
        return None
    return df


def save_gc_rpe_df_cache(df: pd.DataFrame, cache_file: str) -> str:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, index=False, compression="gzip")
    return cache_file


def run_or_load_subject_gc_rpe_df(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    learn_df: pd.DataFrame,
    cache_dir: str,
    overwrite_cache: bool = False,
    return_result_dict: bool = False,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    use_top_electrode_only: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Dict[str, Any]]:
    behavior_fp = _subject_day_learn_fingerprint(learn_df, subj_id=subj_id, day=day, event=event)
    behavior_trial_col = _resolve_trial_col_for_event(learn_df, event)
    key_dict = _build_gc_df_cache_key(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        surrogate_parallel_backend=surrogate_parallel_backend,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
        behavior_fingerprint=behavior_fp,
        behavior_trial_col=behavior_trial_col,
    )
    key_hash = hashlib.sha1(_stable_json_dumps(key_dict).encode("utf-8")).hexdigest()
    cache_file = build_gc_rpe_df_cache_path(
        cache_dir=cache_dir,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        cache_key_hash=key_hash,
    )

    if (not overwrite_cache) and os.path.exists(cache_file):
        cached_df = load_gc_rpe_df_cache(cache_file)
        if cached_df is not None:
            meta = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cache_key_hash": key_hash,
                "cache_key": key_dict,
            }
            result = None
            if return_result_dict:
                result = process_subject_gc_rpe(
                    subj_id=subj_id,
                    roi_source=roi_source,
                    roi_target=roi_target,
                    base_dir=base_dir,
                    learn_df=learn_df,
                    event=event,
                    tmin=tmin,
                    tmax=tmax,
                    target_sfreq=target_sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    min_trials=min_trials,
                    day=day,
                    use_surrogates=use_surrogates,
                    n_surrogates=n_surrogates,
                    surrogate_tail=surrogate_tail,
                    random_state=random_state,
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                    filter_by_power=filter_by_power,
                    power_percentile=power_percentile,
                    tfr_time_window=tfr_time_window,
                    use_top_electrode_only=use_top_electrode_only,
                )
            return cached_df, result, meta

    result = process_subject_gc_rpe(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        learn_df=learn_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=use_top_electrode_only,
    )
    if result is None:
        return pd.DataFrame(), None, {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key_hash": key_hash,
            "cache_key": key_dict,
            "error": "process_subject_gc_rpe returned None",
        }

    trial_col = str(result.get("gc_by_hemi", {}).get("trial_value_col", behavior_trial_col))
    df_subj = export_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        condition_keys=("neg", "neutral", "pos"),
        condition_col="rpe_cat",
        trial_col=trial_col,
    )
    save_gc_rpe_df_cache(df_subj, cache_file)
    meta = {
        "cache_hit": False,
        "cache_file": cache_file,
        "cache_key_hash": key_hash,
        "cache_key": key_dict,
    }
    if return_result_dict:
        return df_subj, result, meta
    return df_subj, None, meta


def enumerate_electrode_pairs(
    epoch_data: Dict[str, Any],
    same_hemisphere_only: bool = True,
) -> List[Tuple[str, str, str, int, int]]:
    """Enumerate source-target electrode pairs from loaded epoch metadata."""
    pairs: List[Tuple[str, str, str, int, int]] = []
    if same_hemisphere_only:
        hemi_specs = [("l", "l"), ("r", "r")]
    else:
        hemi_specs = [("l", "l"), ("l", "r"), ("r", "l"), ("r", "r")]

    for source_hemi, target_hemi in hemi_specs:
        source_channels = list(epoch_data.get(f"source_channels_{source_hemi}", []))
        target_channels = list(epoch_data.get(f"target_channels_{target_hemi}", []))
        source_indices = np.asarray(epoch_data.get(f"source_indices_{source_hemi}", np.array([])), dtype=int)
        target_indices = np.asarray(epoch_data.get(f"target_indices_{target_hemi}", np.array([])), dtype=int)
        if len(source_channels) == 0 or len(target_channels) == 0:
            continue
        if len(source_channels) != len(source_indices) or len(target_channels) != len(target_indices):
            continue

        hemi_label = source_hemi if same_hemisphere_only else f"{source_hemi}_to_{target_hemi}"
        for source_ch, source_idx in zip(source_channels, source_indices):
            for target_ch, target_idx in zip(target_channels, target_indices):
                pairs.append((hemi_label, str(source_ch), str(target_ch), int(source_idx), int(target_idx)))
    return pairs


def compute_pairwise_gc_by_rpe(
    epoch_data: Dict[str, Any],
    learn_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 7,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    event: str = "feedback_start",
    same_hemisphere_only: bool = True,
) -> Dict[str, Any]:
    """Compute single-electrode GC for every source-target pair split by RPE category."""
    epochs = epoch_data["epochs"]
    trial_map = build_rpe_trial_map(
        learn_df,
        subj_id,
        n_epochs=len(epochs),
        day=day,
        event=event,
    )
    valid_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map
    n_dropped_alignment = int((~trial_map["alignment_valid"]).sum()) if len(trial_map) > 0 else 0

    trial_value_col = next(
        (c for c in ("trials_mem", "trials_gamble", "trials") if c in trial_map.columns),
        _resolve_trial_col_for_event(learn_df, event),
    )
    condition_order = ("neg", "neutral", "pos")
    seed_offset_by_condition = {"neg": 1, "neutral": 2, "pos": 3}
    pairs = enumerate_electrode_pairs(epoch_data, same_hemisphere_only=same_hemisphere_only)

    results: Dict[str, Any] = {
        "trial_map": trial_map,
        "condition_col": "rpe_cat",
        "trial_value_col": trial_value_col,
        "condition_order": list(condition_order),
        "same_hemisphere_only": bool(same_hemisphere_only),
        "pairs": [],
        "n_pairs": int(len(pairs)),
        "n_alignment_total": int(len(trial_map)),
        "n_alignment_valid": int(len(valid_map)),
        "n_dropped_alignment": n_dropped_alignment,
        "surrogate_config": {
            "use_surrogates": bool(use_surrogates),
            "n_surrogates": int(n_surrogates),
            "surrogate_tail": surrogate_tail,
            "random_state": int(random_state),
            "n_jobs_surrogates": int(n_jobs_surrogates),
            "surrogate_parallel_backend": surrogate_parallel_backend,
            "source_shuffle_only": True,
        },
    }

    for pair_idx, (hemisphere, source_electrode, target_electrode, source_idx, target_idx) in enumerate(pairs):
        pair_id = f"{hemisphere}:{source_electrode}->{target_electrode}"
        pair_result: Dict[str, Any] = {
            "pair_index": int(pair_idx),
            "pair_id": pair_id,
            "hemisphere": hemisphere,
            "source_electrode": source_electrode,
            "target_electrode": target_electrode,
            "source_index": int(source_idx),
            "target_index": int(target_idx),
        }

        for cond in condition_order:
            cond_map = valid_map[valid_map["rpe_cat"] == cond].copy()
            trial_ix = cond_map["epoch_idx"].to_numpy(dtype=int)
            trial_vals = cond_map[trial_value_col].tolist() if trial_value_col in cond_map.columns else []

            if len(trial_ix) < min_trials:
                pair_result[cond] = {
                    "success": False,
                    "n_trials": int(len(trial_ix)),
                    "trial_indices": trial_ix.tolist(),
                    trial_value_col: trial_vals,
                }
                continue

            if use_surrogates:
                gc_sur = compute_gc_surrogates_source_shuffle(
                    epochs=epochs,
                    source_idx=[source_idx],
                    target_idx=[target_idx],
                    trial_ix=trial_ix,
                    n_surrogates=n_surrogates,
                    random_state=random_state + (pair_idx * 100) + int(seed_offset_by_condition.get(cond, 1)),
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    surrogate_tail=surrogate_tail,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                )
                if gc_sur.get("success", False):
                    cond_result = {
                        "success": True,
                        "gc_forward": gc_sur["real_forward"],
                        "gc_backward": gc_sur["real_backward"],
                        "gc_net": gc_sur["real_net"],
                        "gc_forward_z": gc_sur["z_forward"],
                        "gc_backward_z": gc_sur["z_backward"],
                        "gc_net_z": gc_sur["z_net"],
                        "p_forward": gc_sur["p_forward"],
                        "p_backward": gc_sur["p_backward"],
                        "p_net": gc_sur["p_net"],
                        "surr_mean_forward": gc_sur["surr_mean_forward"],
                        "surr_mean_backward": gc_sur["surr_mean_backward"],
                        "surr_mean_net": gc_sur["surr_mean_net"],
                        "surr_std_forward": gc_sur["surr_std_forward"],
                        "surr_std_backward": gc_sur["surr_std_backward"],
                        "surr_std_net": gc_sur["surr_std_net"],
                        "times": gc_sur["times"],
                        "freqs": gc_sur["freqs"],
                        "n_surrogates": gc_sur["n_surrogates"],
                        "surrogate_tail": gc_sur["surrogate_tail"],
                        "source_shuffle": gc_sur["source_shuffle"],
                    }
                    if "surr_forward" in gc_sur:
                        cond_result["surr_forward"] = gc_sur["surr_forward"]
                        cond_result["surr_backward"] = gc_sur["surr_backward"]
                        cond_result["surr_net"] = gc_sur["surr_net"]
                else:
                    cond_result = {"success": False, "error": gc_sur.get("error", "surrogate_failed")}
            else:
                cond_result = compute_multivariate_gc(
                    epochs=epochs[trial_ix],
                    source_indices=[source_idx],
                    target_indices=[target_idx],
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    verbose=verbose,
                )

            cond_result["n_trials"] = int(len(trial_ix))
            cond_result["trial_indices"] = trial_ix.tolist()
            cond_result[trial_value_col] = trial_vals
            pair_result[cond] = cond_result

        results["pairs"].append(pair_result)

    return results


def process_subject_pairwise_gc_rpe(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    learn_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    same_hemisphere_only: bool = True,
) -> Optional[Dict[str, Any]]:
    epoch_data = load_subject_epochs(
        subj_id=subj_id,
        base_dir=base_dir,
        event=event,
        roi_source=roi_source,
        roi_target=roi_target,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        day=day,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        fmin=fmin,
        fmax=fmax,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=False,
    )
    if epoch_data is None:
        return None

    gc_results = compute_pairwise_gc_by_rpe(
        epoch_data=epoch_data,
        learn_df=learn_df,
        subj_id=subj_id,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        event=event,
        same_hemisphere_only=same_hemisphere_only,
    )

    return {
        "participant": subj_id,
        "roi_source": roi_source,
        "roi_target": roi_target,
        "event": event,
        "times": epoch_data["times"],
        "sfreq": epoch_data["sfreq"],
        "original_sfreq": epoch_data.get("original_sfreq", epoch_data["sfreq"]),
        "n_epochs_total": epoch_data["n_epochs"],
        "pairwise_gc": gc_results,
        "epoch_data": epoch_data,
        "surrogate_config": gc_results.get("surrogate_config", {}),
    }


def export_pairwise_gc_to_dataframe(
    result_dict: Dict[str, Any],
    subj_id: str,
    roi_source: str,
    roi_target: str,
    condition_keys: Tuple[str, ...] = ("neg", "neutral", "pos"),
    condition_col: str = "rpe_cat",
    trial_col: str = "trials_gamble",
) -> pd.DataFrame:
    pairwise_gc = result_dict["pairwise_gc"]
    n_dropped_alignment = int(pairwise_gc.get("n_dropped_alignment", 0))
    rows: List[Dict[str, Any]] = []

    for pair_result in pairwise_gc.get("pairs", []):
        base_row = {
            "participant": subj_id,
            "roi_source": roi_source,
            "roi_target": roi_target,
            "hemisphere": pair_result.get("hemisphere"),
            "source_electrode": pair_result.get("source_electrode"),
            "target_electrode": pair_result.get("target_electrode"),
            "pair_id": pair_result.get("pair_id"),
        }
        for cond in condition_keys:
            cond_data = pair_result.get(cond, {})
            if not cond_data.get("success", False):
                continue
            times = np.asarray(cond_data["times"])
            for t_idx, t in enumerate(times):
                row = dict(base_row)
                row.update(
                    {
                        condition_col: cond,
                        "time": float(t),
                        "gc_forward": float(cond_data["gc_forward"][t_idx]),
                        "gc_backward": float(cond_data["gc_backward"][t_idx]),
                        "gc_net": float(cond_data["gc_net"][t_idx]),
                        "gc_forward_z": float(cond_data["gc_forward_z"][t_idx]) if "gc_forward_z" in cond_data else np.nan,
                        "gc_backward_z": float(cond_data["gc_backward_z"][t_idx]) if "gc_backward_z" in cond_data else np.nan,
                        "gc_net_z": float(cond_data["gc_net_z"][t_idx]) if "gc_net_z" in cond_data else np.nan,
                        "p_forward": float(cond_data["p_forward"][t_idx]) if "p_forward" in cond_data else np.nan,
                        "p_backward": float(cond_data["p_backward"][t_idx]) if "p_backward" in cond_data else np.nan,
                        "p_net": float(cond_data["p_net"][t_idx]) if "p_net" in cond_data else np.nan,
                        "surr_mean_forward": float(cond_data["surr_mean_forward"][t_idx]) if "surr_mean_forward" in cond_data else np.nan,
                        "surr_mean_backward": float(cond_data["surr_mean_backward"][t_idx]) if "surr_mean_backward" in cond_data else np.nan,
                        "surr_mean_net": float(cond_data["surr_mean_net"][t_idx]) if "surr_mean_net" in cond_data else np.nan,
                        "surr_std_forward": float(cond_data["surr_std_forward"][t_idx]) if "surr_std_forward" in cond_data else np.nan,
                        "surr_std_backward": float(cond_data["surr_std_backward"][t_idx]) if "surr_std_backward" in cond_data else np.nan,
                        "surr_std_net": float(cond_data["surr_std_net"][t_idx]) if "surr_std_net" in cond_data else np.nan,
                        "n_trials": int(cond_data.get("n_trials", 0)),
                        "n_surrogates": int(cond_data.get("n_surrogates", 0)),
                        "n_dropped_alignment": n_dropped_alignment,
                        "trial_indices": ",".join(map(str, cond_data.get("trial_indices", []))),
                        trial_col: ",".join(map(str, cond_data.get(trial_col, []))),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def build_pairwise_gc_rpe_df_cache_path(
    cache_dir: str,
    subj_id: str,
    roi_source: str,
    roi_target: str,
    event: str,
    day: str,
    cache_key_hash: str,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{subj_id}_{roi_source}_to_{roi_target}_{event}_{day}_{cache_key_hash[:12]}_pairwise_gc_rpe_timeseries.csv.gz"
    return os.path.join(cache_dir, filename)


def load_pairwise_gc_rpe_df_cache(cache_file: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_csv(cache_file)
    except Exception:
        return None
    required = {
        "participant",
        "roi_source",
        "roi_target",
        "hemisphere",
        "source_electrode",
        "target_electrode",
        "pair_id",
        "rpe_cat",
        "time",
        "gc_forward",
        "gc_backward",
        "gc_net",
    }
    if len(required.difference(df.columns)) > 0:
        return None
    return df


def save_pairwise_gc_rpe_df_cache(df: pd.DataFrame, cache_file: str) -> str:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, index=False, compression="gzip")
    return cache_file


def run_or_load_subject_pairwise_gc_rpe_df(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    learn_df: pd.DataFrame,
    cache_dir: str,
    overwrite_cache: bool = False,
    return_result_dict: bool = False,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 500,
    surrogate_tail: str = "greater",
    random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = False,
    verbose: bool = False,
    filter_by_power: bool = False,
    power_percentile: float = 50,
    tfr_time_window: Optional[Tuple[float, float]] = None,
    same_hemisphere_only: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Dict[str, Any]]:
    behavior_fp = _subject_day_learn_fingerprint(learn_df, subj_id=subj_id, day=day, event=event)
    behavior_trial_col = _resolve_trial_col_for_event(learn_df, event)
    key_dict = _build_gc_df_cache_key(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        surrogate_parallel_backend=surrogate_parallel_backend,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        use_top_electrode_only=False,
        behavior_fingerprint=behavior_fp,
        behavior_trial_col=behavior_trial_col,
    )
    key_dict["analysis_mode"] = "pairwise_rpe"
    key_dict["same_hemisphere_only"] = bool(same_hemisphere_only)
    key_hash = hashlib.sha1(_stable_json_dumps(key_dict).encode("utf-8")).hexdigest()
    cache_file = build_pairwise_gc_rpe_df_cache_path(
        cache_dir=cache_dir,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        event=event,
        day=day,
        cache_key_hash=key_hash,
    )

    if (not overwrite_cache) and os.path.exists(cache_file):
        cached_df = load_pairwise_gc_rpe_df_cache(cache_file)
        if cached_df is not None:
            meta = {
                "cache_hit": True,
                "cache_file": cache_file,
                "cache_key_hash": key_hash,
                "cache_key": key_dict,
            }
            result = None
            if return_result_dict:
                result = process_subject_pairwise_gc_rpe(
                    subj_id=subj_id,
                    roi_source=roi_source,
                    roi_target=roi_target,
                    base_dir=base_dir,
                    learn_df=learn_df,
                    event=event,
                    tmin=tmin,
                    tmax=tmax,
                    target_sfreq=target_sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    gc_n_lags=gc_n_lags,
                    n_freqs=n_freqs,
                    min_trials=min_trials,
                    day=day,
                    use_surrogates=use_surrogates,
                    n_surrogates=n_surrogates,
                    surrogate_tail=surrogate_tail,
                    random_state=random_state,
                    n_jobs_surrogates=n_jobs_surrogates,
                    surrogate_parallel_backend=surrogate_parallel_backend,
                    store_surrogates=store_surrogates,
                    verbose=verbose,
                    filter_by_power=filter_by_power,
                    power_percentile=power_percentile,
                    tfr_time_window=tfr_time_window,
                    same_hemisphere_only=same_hemisphere_only,
                )
            return cached_df, result, meta

    result = process_subject_pairwise_gc_rpe(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        learn_df=learn_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
        filter_by_power=filter_by_power,
        power_percentile=power_percentile,
        tfr_time_window=tfr_time_window,
        same_hemisphere_only=same_hemisphere_only,
    )
    if result is None:
        return pd.DataFrame(), None, {
            "cache_hit": False,
            "cache_file": cache_file,
            "cache_key_hash": key_hash,
            "cache_key": key_dict,
            "error": "process_subject_pairwise_gc_rpe returned None",
        }

    trial_col = str(result.get("pairwise_gc", {}).get("trial_value_col", behavior_trial_col))
    df_subj = export_pairwise_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        condition_keys=("neg", "neutral", "pos"),
        condition_col="rpe_cat",
        trial_col=trial_col,
    )
    save_pairwise_gc_rpe_df_cache(df_subj, cache_file)
    meta = {
        "cache_hit": False,
        "cache_file": cache_file,
        "cache_key_hash": key_hash,
        "cache_key": key_dict,
    }
    if return_result_dict:
        return df_subj, result, meta
    return df_subj, None, meta


# -----------------------------------------------------------------------------
# Band power + plotting
# -----------------------------------------------------------------------------

def compute_band_power_by_hit_miss(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    day: str = "Day1",
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    times = np.asarray(epoch_data["times"])
    trial_map = build_hit_miss_trial_map(combined_df, subj_id, n_epochs=len(epochs), day=day)
    trial_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map

    results: Dict[str, Any] = {
        "source": {"hit": None, "miss": None},
        "target": {"hit": None, "miss": None},
        "times": times,
    }

    for roi_type in ("source", "target"):
        all_channels: List[str] = []
        for hemi in ("l", "r"):
            all_channels.extend(epoch_data.get(f"{roi_type}_channels_{hemi}", []))
        all_channels = [ch for ch in all_channels if ch in epochs.ch_names]
        if len(all_channels) == 0:
            continue

        epochs_roi = epochs.copy().pick_channels(all_channels)
        epochs_roi = epochs_roi.copy().filter(fmin, fmax, verbose=False)
        power_data = epochs_roi.get_data() ** 2
        power_roi = np.mean(power_data, axis=1)

        for cond in ("hit", "miss"):
            cond_map = trial_map[trial_map["hit_miss"] == cond]
            idx = cond_map["epoch_idx"].to_numpy(dtype=int)
            if len(idx) == 0:
                continue
            cond_power = power_roi[idx, :]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(len(idx))
            results[roi_type][cond] = {
                "mean": mean_power,
                "sem": sem_power,
                "n_trials": int(len(idx)),
            }

    return results


def compute_band_power_by_reward(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    day: str = "Day1",
    event: str = "feedback_start",
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    times = np.asarray(epoch_data["times"])
    trial_map = build_reward_trial_map(combined_df, subj_id, n_epochs=len(epochs), day=day, event=event)
    trial_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map

    results: Dict[str, Any] = {
        "source": {"rewarded": None, "unrewarded": None},
        "target": {"rewarded": None, "unrewarded": None},
        "times": times,
    }

    for roi_type in ("source", "target"):
        all_channels: List[str] = []
        for hemi in ("l", "r"):
            all_channels.extend(epoch_data.get(f"{roi_type}_channels_{hemi}", []))
        all_channels = [ch for ch in all_channels if ch in epochs.ch_names]
        if len(all_channels) == 0:
            continue

        epochs_roi = epochs.copy().pick_channels(all_channels)
        epochs_roi = epochs_roi.copy().filter(fmin, fmax, verbose=False)
        power_data = epochs_roi.get_data() ** 2
        power_roi = np.mean(power_data, axis=1)

        for cond in REWARD_OUTCOME_CONDITIONS:
            cond_map = trial_map[trial_map["reward_outcome"] == cond]
            idx = cond_map["epoch_idx"].to_numpy(dtype=int)
            if len(idx) == 0:
                continue
            cond_power = power_roi[idx, :]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(len(idx))
            results[roi_type][cond] = {
                "mean": mean_power,
                "sem": sem_power,
                "n_trials": int(len(idx)),
            }

    return results


def get_frequency_band_params(fmin: float, fmax: float) -> Dict[str, Any]:
    center = 0.5 * (fmin + fmax)
    if fmin >= 4 and fmax <= 8:
        name = "theta"
    elif fmin >= 8 and fmax <= 12:
        name = "alpha"
    elif fmin >= 12 and fmax <= 30:
        name = "beta"
    elif fmin >= 30:
        name = "gamma"
    else:
        name = f"{fmin:.1f}-{fmax:.1f}Hz"
    return {"band_name": name, "center_freq": center, "fmin": fmin, "fmax": fmax}


def _format_band_suffix(fmin: Optional[float] = None, fmax: Optional[float] = None) -> str:
    if fmin is None:
        return ""
    if fmax is None:
        return f"_{fmin}"
    return f"_{fmin}-{fmax}Hz"


def generate_subject_plot_hit_miss(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    result_dict: Dict[str, Any],
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    fmin: float = 4,
    fmax: float = 8,
    band_name: Optional[str] = None,
    savepath: Optional[str] = None,
    event: str = "feedback_start",
) -> Optional[plt.Figure]:
    combined_gc = combine_hemispheres(result_dict["gc_by_hemi"], condition_keys=("hit", "miss"))
    band_power = compute_band_power_by_hit_miss(epoch_data, combined_df, subj_id, fmin=fmin, fmax=fmax)
    valid = [k for k in ("hit", "miss") if combined_gc.get(k, {}).get("success", False)]
    if len(valid) == 0:
        return None

    colors = {"hit": "#2ecc71", "miss": "#e74c3c"}
    labels = {"hit": "Hit", "miss": "Miss"}

    times = np.asarray(combined_gc[valid[0]]["times"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: band power
    for col, roi_type in enumerate(("source", "target")):
        ax = axes[0, col]
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={entry['n_trials']})")
            ax.fill_between(times, entry["mean"] - entry["sem"], entry["mean"] + entry["sem"], color=colors[cond], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band Power (a.u.)")
        ax.set_title(f"{'Source' if roi_type=='source' else 'Target'}: {roi_source if roi_type=='source' else roi_target}")
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 2]
    for roi_type, clr in (("source", "#1abc9c"), ("target", "#9b59b6")):
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=clr, linestyle="-" if cond == "hit" else "--", linewidth=2,
                    label=f"{roi_source if roi_type=='source' else roi_target} ({labels[cond]})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band Power (a.u.)")
    ax.set_title("Source vs Target Overlay")
    ax.legend(fontsize=8, loc="upper right")

    # Row 2: GC
    net_metric = "gc_net_z" if all("gc_net_z" in combined_gc[k] for k in valid) else "gc_net"
    net_label = "Net GC z-score" if net_metric == "gc_net_z" else "Net GC"
    metrics = [("gc_forward", f"Forward GC: {roi_source}→{roi_target}"),
               ("gc_backward", f"Backward GC: {roi_target}→{roi_source}"),
               (net_metric, f"{net_label}: {roi_source}↔{roi_target}")]
    for col, (metric, title) in enumerate(metrics):
        ax = axes[1, col]
        for cond in valid:
            gc_data = combined_gc[cond]
            vals = np.asarray(gc_data[metric])
            ax.plot(times, vals, color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={gc_data['n_trials']})")
            if metric in ("gc_net", "gc_net_z"):
                ax.fill_between(times, 0, vals, color=colors[cond], alpha=0.15)
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GC" if metric != "gc_net_z" else "GC z-score")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    if band_name is None:
        band_name = get_frequency_band_params(fmin, fmax)["band_name"]
    fig.suptitle(f"{subj_id}: {roi_source}→{roi_target} | {band_name} band ({fmin}-{fmax} Hz) | Hit vs Miss", fontsize=14, y=1.02)
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        band_suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"{subj_id}_{roi_source}_{roi_target}_{event}{band_suffix}_subject_hitmiss_summary.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


def generate_subject_plot_reward(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    result_dict: Dict[str, Any],
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    fmin: float = 4,
    fmax: float = 8,
    band_name: Optional[str] = None,
    savepath: Optional[str] = None,
    event: str = "feedback_start",
) -> Optional[plt.Figure]:
    combined_gc = combine_hemispheres(result_dict["gc_by_hemi"], condition_keys=REWARD_OUTCOME_CONDITIONS)
    band_power = compute_band_power_by_reward(epoch_data, combined_df, subj_id, fmin=fmin, fmax=fmax, event=event)
    valid = [k for k in REWARD_OUTCOME_CONDITIONS if combined_gc.get(k, {}).get("success", False)]
    if len(valid) == 0:
        return None

    colors = {"rewarded": "#2ecc71", "unrewarded": "#e74c3c"}
    labels = {"rewarded": "Rewarded", "unrewarded": "Unrewarded"}

    times = np.asarray(combined_gc[valid[0]]["times"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, roi_type in enumerate(("source", "target")):
        ax = axes[0, col]
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={entry['n_trials']})")
            ax.fill_between(times, entry["mean"] - entry["sem"], entry["mean"] + entry["sem"], color=colors[cond], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band Power (a.u.)")
        ax.set_title(f"{'Source' if roi_type=='source' else 'Target'}: {roi_source if roi_type=='source' else roi_target}")
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 2]
    for roi_type, clr in (("source", "#1abc9c"), ("target", "#9b59b6")):
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(
                times,
                entry["mean"],
                color=clr,
                linestyle="-" if cond == "rewarded" else "--",
                linewidth=2,
                label=f"{roi_source if roi_type=='source' else roi_target} ({labels[cond]})",
            )
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band Power (a.u.)")
    ax.set_title("Source vs Target Overlay")
    ax.legend(fontsize=8, loc="upper right")

    net_metric = "gc_net_z" if all("gc_net_z" in combined_gc[k] for k in valid) else "gc_net"
    net_label = "Net GC z-score" if net_metric == "gc_net_z" else "Net GC"
    metrics = [
        ("gc_forward", f"Forward GC: {roi_source}→{roi_target}"),
        ("gc_backward", f"Backward GC: {roi_target}→{roi_source}"),
        (net_metric, f"{net_label}: {roi_source}↔{roi_target}"),
    ]
    for col, (metric, title) in enumerate(metrics):
        ax = axes[1, col]
        for cond in valid:
            gc_data = combined_gc[cond]
            vals = np.asarray(gc_data[metric])
            ax.plot(times, vals, color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={gc_data['n_trials']})")
            if metric in ("gc_net", "gc_net_z"):
                ax.fill_between(times, 0, vals, color=colors[cond], alpha=0.15)
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GC" if metric != "gc_net_z" else "GC z-score")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    if band_name is None:
        band_name = get_frequency_band_params(fmin, fmax)["band_name"]
    fig.suptitle(f"{subj_id}: {roi_source}→{roi_target} | {band_name} band ({fmin}-{fmax} Hz) | Rewarded vs Unrewarded", fontsize=14, y=1.02)
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        band_suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"{subj_id}_{roi_source}_{roi_target}_{event}{band_suffix}_subject_reward_summary.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


# -----------------------------------------------------------------------------
# Merge + stats + aggregate plotting
# -----------------------------------------------------------------------------

def merge_with_behavior_hit_miss(mvgc_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    """Attach behavior summaries for hit/miss and optional trial-level mapping.

    If mvgc_df includes numeric `trials_gamble`, merge on participant+trials_gamble.
    Otherwise merge participant+hit_miss aggregated behavior.
    """
    df = mvgc_df.copy()
    cdf = combined_df.copy()

    if "condition" in cdf.columns:
        cdf = cdf[cdf["condition"] == "Day1"].copy()

    if "hits" in cdf.columns:
        cdf["hits"] = _coerce_bool_series(cdf["hits"])
        cdf["hit_miss"] = np.where(cdf["hits"], "hit", "miss")

    if "trials_gamble" in df.columns and "trials_gamble" in cdf.columns:
        merge_cols = ["participant", "trials_gamble"]
        keep_cols = [c for c in ["participant", "trials_gamble", "hit_miss", "reward_gamble", "gamble_gamble", "rpe", "zrpe", "memory_outcome"] if c in cdf.columns]
        return df.merge(cdf[keep_cols], on=merge_cols, how="left")

    agg_dict = {}
    for col in ["reward_gamble", "gamble_gamble", "rpe", "zrpe", "trials_gamble"]:
        if col in cdf.columns:
            agg_dict[col] = "mean"
    agg_dict["hits"] = "mean" if "hits" in cdf.columns else None
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None}

    if len(agg_dict) > 0 and {"participant", "hit_miss"}.issubset(cdf.columns):
        beh_agg = cdf.groupby(["participant", "hit_miss"], as_index=False).agg(agg_dict)
        return df.merge(beh_agg, on=["participant", "hit_miss"], how="left")
    return df


def merge_with_behavior_reward(mvgc_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    """Attach behavior summaries for binary reward GC; merge on participant+trials_gamble or aggregate by reward_outcome."""
    df = mvgc_df.copy()
    cdf = combined_df.copy()

    if "condition" in cdf.columns:
        cdf = cdf[cdf["condition"] == "Day1"].copy()

    reward_src = "reward_gamble" if "reward_gamble" in cdf.columns else "reward"
    if reward_src in cdf.columns:
        rv = pd.to_numeric(cdf[reward_src], errors="coerce")
        cdf = cdf.loc[rv.notna()].copy()
        rv = rv.loc[cdf.index].astype(int)
        cdf = cdf.loc[rv.isin([0, 1])].copy()
        rv = rv.loc[cdf.index]
        cdf["reward_outcome"] = np.where(rv.to_numpy() == 1, "rewarded", "unrewarded")

    if "trials_gamble" in df.columns and "trials_gamble" in cdf.columns:
        merge_cols = ["participant", "trials_gamble"]
        keep_cols = [
            c
            for c in [
                "participant",
                "trials_gamble",
                "reward_outcome",
                "reward_gamble",
                "reward",
                "hits",
                "gamble_gamble",
                "rpe",
                "zrpe",
                "memory_outcome",
            ]
            if c in cdf.columns
        ]
        return df.merge(cdf[keep_cols], on=merge_cols, how="left")

    agg_dict = {}
    for col in ["reward_gamble", "reward", "gamble_gamble", "rpe", "zrpe", "trials_gamble"]:
        if col in cdf.columns:
            agg_dict[col] = "mean"
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None}

    if len(agg_dict) > 0 and {"participant", "reward_outcome"}.issubset(cdf.columns):
        beh_agg = cdf.groupby(["participant", "reward_outcome"], as_index=False).agg(agg_dict)
        return df.merge(beh_agg, on=["participant", "reward_outcome"], how="left")
    return df


def fit_mixed_effects_hit_miss(model_df: pd.DataFrame, reference: str = "miss") -> Dict[str, Any]:
    """Fit mixed-effects models per target ROI: gc_net_z ~ hit_miss + (1|participant)."""
    required = {"participant", "target_roi", "gc_net_z", "hit_miss"}
    missing = sorted(required.difference(model_df.columns))
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for mixed model: {missing}")

    df = model_df.copy()
    df = df[df["gc_net_z"].notna() & df["hit_miss"].notna()].copy()
    df["hit_miss"] = pd.Categorical(df["hit_miss"], categories=[reference, "hit" if reference == "miss" else "miss"], ordered=False)

    model_results: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []

    for target in sorted(df["target_roi"].dropna().unique()):
        target_data = df[df["target_roi"] == target].copy()
        if len(target_data) < 20 or target_data["participant"].nunique() < 2:
            continue
        try:
            formula = f"gc_net_z ~ C(hit_miss, Treatment('{reference}'))"
            model = smf.mixedlm(formula, data=target_data, groups=target_data["participant"])
            fit = model.fit(method="lbfgs", maxiter=500)
            model_results[target] = fit

            conf_int = fit.conf_int()
            for param in fit.params.index:
                pval = float(fit.pvalues[param])
                coef = float(fit.params[param])
                ci_low, ci_high = conf_int.loc[param]
                records.append(
                    {
                        "target_roi": target,
                        "param": param,
                        "coef": coef,
                        "pval": pval,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }
                )
        except Exception as exc:
            model_results[target] = {"error": str(exc)}

    stats_df = pd.DataFrame(records)
    if len(stats_df) > 0:
        mask = ~stats_df["param"].str.contains("Intercept|Group Var", regex=True)
        p = stats_df.loc[mask, "pval"].to_numpy(dtype=float)
        if len(p) > 0:
            _, qvals, _, _ = multipletests(p, method="fdr_bh")
            stats_df.loc[mask, "pval_fdr"] = qvals

    return {"model_results": model_results, "stats_df": stats_df, "model_df": df}


def plot_mixed_effects_hit_miss(
    model_out: Dict[str, Any],
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    model_df = model_out.get("model_df", pd.DataFrame())
    stats_df = model_out.get("stats_df", pd.DataFrame())
    if len(model_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: model coefficient hit-vs-miss per target
    ax = axes[0]
    coef_df = stats_df[stats_df["param"].str.contains("hit_miss", na=False)].copy() if len(stats_df) > 0 else pd.DataFrame()
    if len(coef_df) > 0:
        coef_df = coef_df.sort_values("target_roi")
        x = np.arange(len(coef_df))
        y = coef_df["coef"].to_numpy(dtype=float)
        err_low = y - coef_df["ci_low"].to_numpy(dtype=float)
        err_high = coef_df["ci_high"].to_numpy(dtype=float) - y
        ax.bar(x, y, color="#2ecc71", alpha=0.8)
        ax.errorbar(x, y, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(coef_df["target_roi"].tolist(), rotation=0)
        ax.set_ylabel("Coef on gc_net_z (hit vs miss)")
        ax.set_title("Mixed-effects coefficient ±95% CI")
    else:
        ax.text(0.5, 0.5, "No fitted coefficients", ha="center", va="center", transform=ax.transAxes)

    # Panel 2: raw means by hit/miss
    ax = axes[1]
    means = (
        model_df.groupby(["target_roi", "hit_miss"])["gc_net_z"].agg(["mean", "sem"]).reset_index()
    )
    if len(means) > 0:
        target_order = sorted(model_df["target_roi"].unique())
        x = np.arange(len(target_order))
        width = 0.35
        miss_vals = means[means["hit_miss"] == "miss"].set_index("target_roi").reindex(target_order)
        hit_vals = means[means["hit_miss"] == "hit"].set_index("target_roi").reindex(target_order)

        ax.bar(x - width / 2, miss_vals["mean"], width, yerr=1.96 * miss_vals["sem"], color="#e74c3c", alpha=0.8, capsize=2, label="Miss")
        ax.bar(x + width / 2, hit_vals["mean"], width, yerr=1.96 * hit_vals["sem"], color="#2ecc71", alpha=0.8, capsize=2, label="Hit")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order)
        ax.set_ylabel("Mean gc_net_z")
        ax.set_title("Net GC by Hit/Miss (mean ±95% CI)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No model_df rows", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle(f"Mixed Effects: {source_roi}→targets | hit/miss")
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"mixed_model_gc_hitmiss_{source_roi}_to_targets_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


def plot_aggregate_hit_miss(
    agg_df: pd.DataFrame,
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    req = {"target_roi", "gc_forward", "gc_backward", "gc_net", "hit_miss"}
    if len(agg_df) == 0 or len(req.difference(agg_df.columns)) > 0:
        return None

    df = agg_df.copy()
    net_col = "gc_net_z" if ("gc_net_z" in df.columns and df["gc_net_z"].notna().any()) else "gc_net"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: net GC distribution by target
    ax = axes[0, 0]
    sns.violinplot(data=df, x="target_roi", y=net_col, hue="hit_miss", split=True, inner="quart", palette={"hit": "#2ecc71", "miss": "#e74c3c"}, ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"{'Net GC z-score' if net_col=='gc_net_z' else 'Net GC'} distribution by target ROI")

    # Panel 2: forward/backward means
    ax = axes[0, 1]
    means = df.groupby("target_roi")[["gc_forward", "gc_backward"]].mean().reset_index()
    x = np.arange(len(means))
    w = 0.35
    ax.bar(x - w / 2, means["gc_forward"], w, color="#3498db", label="Forward")
    ax.bar(x + w / 2, means["gc_backward"], w, color="#f39c12", label="Backward")
    ax.set_xticks(x)
    ax.set_xticklabels(means["target_roi"].tolist())
    ax.set_title("Forward vs Backward GC")
    ax.legend()

    # Panel 3: hit-miss delta by target
    ax = axes[1, 0]
    hm = df.groupby(["target_roi", "hit_miss"])[net_col].mean().unstack("hit_miss")
    hm["delta_hit_minus_miss"] = hm.get("hit", np.nan) - hm.get("miss", np.nan)
    hm["delta_hit_minus_miss"].plot(kind="bar", color="#8e44ad", ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"Mean {'z-scored Net GC' if net_col=='gc_net_z' else 'Net GC'}: Hit - Miss")
    ax.set_ylabel("Delta")

    # Panel 4: sample counts
    ax = axes[1, 1]
    counts = df.groupby(["target_roi", "hit_miss"]).size().reset_index(name="n")
    sns.barplot(data=counts, x="target_roi", y="n", hue="hit_miss", palette={"hit": "#2ecc71", "miss": "#e74c3c"}, ax=ax)
    ax.set_title("Sample counts")

    plt.suptitle(f"Aggregate MVGC summary (hit/miss) | source={source_roi}")
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"aggregate_{source_roi}_to_targets_hitmiss_summary_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


def fit_mixed_effects_reward(model_df: pd.DataFrame, reference: str = "unrewarded") -> Dict[str, Any]:
    """Fit mixed-effects models per target ROI: gc_net_z ~ reward_outcome + (1|participant)."""
    required = {"participant", "target_roi", "gc_net_z", "reward_outcome"}
    missing = sorted(required.difference(model_df.columns))
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for mixed model: {missing}")

    df = model_df.copy()
    df = df[df["gc_net_z"].notna() & df["reward_outcome"].notna()].copy()
    other = "rewarded" if reference == "unrewarded" else "unrewarded"
    df["reward_outcome"] = pd.Categorical(df["reward_outcome"], categories=[reference, other], ordered=False)

    model_results: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []

    for target in sorted(df["target_roi"].dropna().unique()):
        target_data = df[df["target_roi"] == target].copy()
        if len(target_data) < 20 or target_data["participant"].nunique() < 2:
            continue
        try:
            formula = f"gc_net_z ~ C(reward_outcome, Treatment('{reference}'))"
            model = smf.mixedlm(formula, data=target_data, groups=target_data["participant"])
            fit = model.fit(method="lbfgs", maxiter=500)
            model_results[target] = fit

            conf_int = fit.conf_int()
            for param in fit.params.index:
                pval = float(fit.pvalues[param])
                coef = float(fit.params[param])
                ci_low, ci_high = conf_int.loc[param]
                records.append(
                    {
                        "target_roi": target,
                        "param": param,
                        "coef": coef,
                        "pval": pval,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }
                )
        except Exception as exc:
            model_results[target] = {"error": str(exc)}

    stats_df = pd.DataFrame(records)
    if len(stats_df) > 0:
        mask = ~stats_df["param"].str.contains("Intercept|Group Var", regex=True)
        p = stats_df.loc[mask, "pval"].to_numpy(dtype=float)
        if len(p) > 0:
            _, qvals, _, _ = multipletests(p, method="fdr_bh")
            stats_df.loc[mask, "pval_fdr"] = qvals

    return {"model_results": model_results, "stats_df": stats_df, "model_df": df}


def plot_mixed_effects_reward(
    model_out: Dict[str, Any],
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    model_df = model_out.get("model_df", pd.DataFrame())
    stats_df = model_out.get("stats_df", pd.DataFrame())
    if len(model_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    coef_df = stats_df[stats_df["param"].str.contains("reward_outcome", na=False)].copy() if len(stats_df) > 0 else pd.DataFrame()
    if len(coef_df) > 0:
        coef_df = coef_df.sort_values("target_roi")
        x = np.arange(len(coef_df))
        y = coef_df["coef"].to_numpy(dtype=float)
        err_low = y - coef_df["ci_low"].to_numpy(dtype=float)
        err_high = coef_df["ci_high"].to_numpy(dtype=float) - y
        ax.bar(x, y, color="#2ecc71", alpha=0.8)
        ax.errorbar(x, y, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(coef_df["target_roi"].tolist(), rotation=0)
        ax.set_ylabel("Coef on gc_net_z (rewarded vs unrewarded)")
        ax.set_title("Mixed-effects coefficient ±95% CI")
    else:
        ax.text(0.5, 0.5, "No fitted coefficients", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1]
    means = model_df.groupby(["target_roi", "reward_outcome"])["gc_net_z"].agg(["mean", "sem"]).reset_index()
    if len(means) > 0:
        target_order = sorted(model_df["target_roi"].unique())
        x = np.arange(len(target_order))
        width = 0.35
        unrev = means[means["reward_outcome"] == "unrewarded"].set_index("target_roi").reindex(target_order)
        rew = means[means["reward_outcome"] == "rewarded"].set_index("target_roi").reindex(target_order)

        ax.bar(x - width / 2, unrev["mean"], width, yerr=1.96 * unrev["sem"], color="#e74c3c", alpha=0.8, capsize=2, label="Unrewarded")
        ax.bar(x + width / 2, rew["mean"], width, yerr=1.96 * rew["sem"], color="#2ecc71", alpha=0.8, capsize=2, label="Rewarded")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order)
        ax.set_ylabel("Mean gc_net_z")
        ax.set_title("Net GC by reward (mean ±95% CI)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No model_df rows", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle(f"Mixed Effects: {source_roi}→targets | binary reward")
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"mixed_model_gc_reward_{source_roi}_to_targets_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


def plot_aggregate_reward(
    agg_df: pd.DataFrame,
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    req = {"target_roi", "gc_forward", "gc_backward", "gc_net", "reward_outcome"}
    if len(agg_df) == 0 or len(req.difference(agg_df.columns)) > 0:
        return None

    df = agg_df.copy()
    net_col = "gc_net_z" if ("gc_net_z" in df.columns and df["gc_net_z"].notna().any()) else "gc_net"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    sns.violinplot(
        data=df,
        x="target_roi",
        y=net_col,
        hue="reward_outcome",
        split=True,
        inner="quart",
        palette={"rewarded": "#2ecc71", "unrewarded": "#e74c3c"},
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"{'Net GC z-score' if net_col=='gc_net_z' else 'Net GC'} distribution by target ROI")

    ax = axes[0, 1]
    means = df.groupby("target_roi")[["gc_forward", "gc_backward"]].mean().reset_index()
    x = np.arange(len(means))
    w = 0.35
    ax.bar(x - w / 2, means["gc_forward"], w, color="#3498db", label="Forward")
    ax.bar(x + w / 2, means["gc_backward"], w, color="#f39c12", label="Backward")
    ax.set_xticks(x)
    ax.set_xticklabels(means["target_roi"].tolist())
    ax.set_title("Forward vs Backward GC")
    ax.legend()

    ax = axes[1, 0]
    hm = df.groupby(["target_roi", "reward_outcome"])[net_col].mean().unstack("reward_outcome")
    hm["delta_rewarded_minus_unrewarded"] = hm.get("rewarded", np.nan) - hm.get("unrewarded", np.nan)
    hm["delta_rewarded_minus_unrewarded"].plot(kind="bar", color="#8e44ad", ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"Mean {'z-scored Net GC' if net_col=='gc_net_z' else 'Net GC'}: Rewarded - Unrewarded")
    ax.set_ylabel("Delta")

    ax = axes[1, 1]
    counts = df.groupby(["target_roi", "reward_outcome"]).size().reset_index(name="n")
    sns.barplot(
        data=counts,
        x="target_roi",
        y="n",
        hue="reward_outcome",
        palette={"rewarded": "#2ecc71", "unrewarded": "#e74c3c"},
        ax=ax,
    )
    ax.set_title("Sample counts")

    plt.suptitle(f"Aggregate MVGC summary (binary reward) | source={source_roi}")
    plt.tight_layout()

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"aggregate_{source_roi}_to_targets_reward_summary_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")

    return fig


# -----------------------------------------------------------------------------
# Memory outcome (4 categories) merge + stats + plotting
# -----------------------------------------------------------------------------

def merge_with_behavior_memory_outcome(mvgc_df: pd.DataFrame, combined_df: pd.DataFrame) -> pd.DataFrame:
    """Attach behavior for memory_outcome GC: merge on participant + memory_outcome or trial column."""
    df = mvgc_df.copy()
    cdf = combined_df.copy()
    if "condition" in cdf.columns:
        cdf = cdf[cdf["condition"] == "Day1"].copy()
    if "memory_outcome" not in cdf.columns:
        return df
    trial_col = None
    for col in ("trials_mem", "trials_gamble", "trials"):
        if col in df.columns and col in cdf.columns:
            trial_series = df[col].astype(str)
            if not trial_series.str.contains(",", regex=False).any():
                trial_col = col
                break
    if trial_col is not None:
        keep_cols = [c for c in ["participant", trial_col, "memory_outcome", "reward_gamble", "gamble_gamble", "rpe", "zrpe"] if c in cdf.columns]
        df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce")
        cdf[trial_col] = pd.to_numeric(cdf[trial_col], errors="coerce")
        return df.merge(cdf[keep_cols], on=["participant", trial_col], how="left")
    agg_dict = {}
    for col in ["reward_gamble", "gamble_gamble", "rpe", "zrpe"]:
        if col in cdf.columns:
            agg_dict[col] = "mean"
    if len(agg_dict) > 0 and "memory_outcome" in cdf.columns:
        beh_agg = cdf.groupby(["participant", "memory_outcome"], as_index=False).agg(agg_dict)
        return df.merge(beh_agg, on=["participant", "memory_outcome"], how="left")
    return df


def fit_mixed_effects_memory_outcome(model_df: pd.DataFrame, reference: str = "misses") -> Dict[str, Any]:
    """Fit mixed-effects: gc_net_z ~ C(memory_outcome, Treatment(reference)) + (1|participant)."""
    required = {"participant", "target_roi", "gc_net_z", "memory_outcome"}
    missing = sorted(required.difference(model_df.columns))
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for mixed model: {missing}")

    df = model_df.copy()
    df = df[df["gc_net_z"].notna() & df["memory_outcome"].notna()].copy()
    others = [c for c in MEMORY_OUTCOME_CONDITIONS if c != reference]
    df["memory_outcome"] = pd.Categorical(df["memory_outcome"], categories=[reference] + others, ordered=False)

    model_results: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []

    for target in sorted(df["target_roi"].dropna().unique()):
        target_data = df[df["target_roi"] == target].copy()
        if len(target_data) < 20 or target_data["participant"].nunique() < 2:
            continue
        try:
            formula = f"gc_net_z ~ C(memory_outcome, Treatment('{reference}'))"
            model = smf.mixedlm(formula, data=target_data, groups=target_data["participant"])
            fit = model.fit(method="lbfgs", maxiter=500)
            model_results[target] = fit
            conf_int = fit.conf_int()
            for param in fit.params.index:
                pval = float(fit.pvalues[param])
                coef = float(fit.params[param])
                ci_low, ci_high = conf_int.loc[param]
                records.append({
                    "target_roi": target,
                    "param": param,
                    "coef": coef,
                    "pval": pval,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                })
        except Exception as exc:
            model_results[target] = {"error": str(exc)}

    stats_df = pd.DataFrame(records)
    if len(stats_df) > 0:
        mask = ~stats_df["param"].str.contains("Intercept|Group Var", regex=True)
        p = stats_df.loc[mask, "pval"].to_numpy(dtype=float)
        if len(p) > 0:
            _, qvals, _, _ = multipletests(p, method="fdr_bh")
            stats_df.loc[mask, "pval_fdr"] = qvals
    return {"model_results": model_results, "stats_df": stats_df, "model_df": df}


def plot_mixed_effects_memory_outcome(
    model_out: Dict[str, Any],
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    model_df = model_out.get("model_df", pd.DataFrame())
    stats_df = model_out.get("stats_df", pd.DataFrame())
    if len(model_df) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    coef_df = stats_df[stats_df["param"].str.contains("memory_outcome", na=False)].copy() if len(stats_df) > 0 else pd.DataFrame()
    if len(coef_df) > 0:
        coef_df = coef_df.sort_values("target_roi")
        x = np.arange(len(coef_df))
        y = coef_df["coef"].to_numpy(dtype=float)
        err_low = y - coef_df["ci_low"].to_numpy(dtype=float)
        err_high = coef_df["ci_high"].to_numpy(dtype=float) - y
        ax.bar(x, y, color="#9b59b6", alpha=0.8)
        ax.errorbar(x, y, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=3)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(coef_df["target_roi"].tolist(), rotation=0)
        ax.set_ylabel("Coef on gc_net_z (memory outcome vs reference)")
        ax.set_title("Mixed-effects coefficient ±95% CI")
    else:
        ax.text(0.5, 0.5, "No fitted coefficients", ha="center", va="center", transform=ax.transAxes)
    ax = axes[1]
    means = model_df.groupby(["target_roi", "memory_outcome"])["gc_net_z"].agg(["mean", "sem"]).reset_index()
    if len(means) > 0:
        target_order = sorted(model_df["target_roi"].unique())
        x = np.arange(len(target_order))
        width = 0.2
        palette = {"hits": "#2ecc71", "misses": "#e74c3c", "correct_rejections": "#3498db", "false_alarms": "#f39c12"}
        for i, cond in enumerate(MEMORY_OUTCOME_CONDITIONS):
            cond_vals = means[means["memory_outcome"] == cond].set_index("target_roi").reindex(target_order)
            offset = (i - 1.5) * width
            ax.bar(x + offset, cond_vals["mean"], width, yerr=cond_vals["sem"], color=palette.get(cond, "gray"), alpha=0.8, capsize=2, label=cond)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order)
        ax.set_ylabel("Mean gc_net_z")
        ax.set_title("Net GC by memory outcome (mean ±SEM)")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No model_df rows", ha="center", va="center", transform=ax.transAxes)
    plt.suptitle(f"Mixed Effects: {source_roi}→targets | memory outcome")
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"mixed_model_gc_memory_outcome_{source_roi}_to_targets_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    return fig


def plot_aggregate_memory_outcome(
    agg_df: pd.DataFrame,
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    req = {"target_roi", "gc_forward", "gc_backward", "gc_net", "memory_outcome"}
    if len(agg_df) == 0 or len(req.difference(agg_df.columns)) > 0:
        return None
    df = agg_df.copy()
    if "target_roi" not in df.columns and "roi_target" in df.columns:
        df["target_roi"] = df["roi_target"]
    net_col = "gc_net_z" if ("gc_net_z" in df.columns and df["gc_net_z"].notna().any()) else "gc_net"
    palette = {"hits": "#2ecc71", "misses": "#e74c3c", "correct_rejections": "#3498db", "false_alarms": "#f39c12"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    sns.violinplot(data=df, x="target_roi", y=net_col, hue="memory_outcome", inner="quart", palette=palette, ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"{'Net GC z-score' if net_col=='gc_net_z' else 'Net GC'} by memory outcome")
    ax = axes[0, 1]
    means = df.groupby("target_roi")[["gc_forward", "gc_backward"]].mean().reset_index()
    x = np.arange(len(means))
    w = 0.35
    ax.bar(x - w / 2, means["gc_forward"], w, color="#3498db", label="Forward")
    ax.bar(x + w / 2, means["gc_backward"], w, color="#f39c12", label="Backward")
    ax.set_xticks(x)
    ax.set_xticklabels(means["target_roi"].tolist())
    ax.set_title("Forward vs Backward GC")
    ax.legend()
    ax = axes[1, 0]
    mo_means = df.groupby(["target_roi", "memory_outcome"])[net_col].mean().unstack("memory_outcome")
    mo_means["mean"] = mo_means.mean(axis=1)
    mo_means["mean"].plot(kind="bar", color="#8e44ad", ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"Mean {'z-scored Net GC' if net_col=='gc_net_z' else 'Net GC'} by target")
    ax.set_ylabel("Mean")
    ax = axes[1, 1]
    counts = df.groupby(["target_roi", "memory_outcome"]).size().reset_index(name="n")
    sns.barplot(data=counts, x="target_roi", y="n", hue="memory_outcome", palette=palette, ax=ax)
    ax.set_title("Sample counts by memory outcome")
    plt.suptitle(f"Aggregate MVGC summary (memory outcome) | source={source_roi}")
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"aggregate_{source_roi}_to_targets_memory_outcome_summary_{event}{suffix}.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    return fig


def compute_band_power_by_memory_outcome(
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    day: str = "Day1",
    event: str = "feedback_start",
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    times = np.asarray(epoch_data["times"])
    trial_map = build_memory_outcome_trial_map(combined_df, subj_id, n_epochs=len(epochs), day=day, event=event)
    trial_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map
    results: Dict[str, Any] = {
        "source": {c: None for c in MEMORY_OUTCOME_CONDITIONS},
        "target": {c: None for c in MEMORY_OUTCOME_CONDITIONS},
        "times": times,
    }
    for roi_type in ("source", "target"):
        all_channels = []
        for hemi in ("l", "r"):
            all_channels.extend(epoch_data.get(f"{roi_type}_channels_{hemi}", []))
        all_channels = [ch for ch in all_channels if ch in epochs.ch_names]
        if len(all_channels) == 0:
            continue
        epochs_roi = epochs.copy().pick_channels(all_channels)
        epochs_roi = epochs_roi.copy().filter(fmin, fmax, verbose=False)
        power_data = epochs_roi.get_data() ** 2
        power_roi = np.mean(power_data, axis=1)
        for cond in MEMORY_OUTCOME_CONDITIONS:
            cond_map = trial_map[trial_map["memory_outcome"] == cond]
            idx = cond_map["epoch_idx"].to_numpy(dtype=int)
            if len(idx) == 0:
                continue
            cond_power = power_roi[idx, :]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(len(idx))
            results[roi_type][cond] = {"mean": mean_power, "sem": sem_power, "n_trials": int(len(idx))}
    return results


def generate_subject_plot_memory_outcome(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    result_dict: Dict[str, Any],
    epoch_data: Dict[str, Any],
    combined_df: pd.DataFrame,
    fmin: float = 4,
    fmax: float = 8,
    band_name: Optional[str] = None,
    savepath: Optional[str] = None,
    event: str = "feedback_start",
) -> Optional[plt.Figure]:
    combined_gc = combine_hemispheres(result_dict["gc_by_hemi"], condition_keys=MEMORY_OUTCOME_CONDITIONS)
    band_power = compute_band_power_by_memory_outcome(epoch_data, combined_df, subj_id, fmin=fmin, fmax=fmax, day="Day1", event=event)
    valid = [k for k in MEMORY_OUTCOME_CONDITIONS if combined_gc.get(k, {}).get("success", False)]
    if len(valid) == 0:
        return None
    colors = {"hits": "#2ecc71", "misses": "#e74c3c", "correct_rejections": "#3498db", "false_alarms": "#f39c12"}
    times = np.asarray(combined_gc[valid[0]]["times"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for col, roi_type in enumerate(("source", "target")):
        ax = axes[0, col]
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=colors.get(cond, "gray"), linewidth=2, label=f"{cond} (n={entry['n_trials']})")
            ax.fill_between(times, entry["mean"] - entry["sem"], entry["mean"] + entry["sem"], color=colors.get(cond, "gray"), alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band Power (a.u.)")
        ax.set_title(f"{'Source' if roi_type=='source' else 'Target'}: {roi_source if roi_type=='source' else roi_target}")
        ax.legend(fontsize=8, loc="upper right")
    ax = axes[0, 2]
    for roi_type, clr in (("source", "#1abc9c"), ("target", "#9b59b6")):
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=clr, linewidth=2, label=f"{roi_source if roi_type=='source' else roi_target} ({cond})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band Power (a.u.)")
    ax.set_title("Source vs Target Overlay")
    ax.legend(fontsize=8, loc="upper right")
    net_metric = "gc_net_z" if all("gc_net_z" in combined_gc[k] for k in valid) else "gc_net"
    metrics = [
        ("gc_forward", f"Forward GC: {roi_source}→{roi_target}"),
        ("gc_backward", f"Backward GC: {roi_target}→{roi_source}"),
        (net_metric, f"Net GC: {roi_source}↔{roi_target}"),
    ]
    for col, (metric, title) in enumerate(metrics):
        ax = axes[1, col]
        for cond in valid:
            gc_data = combined_gc[cond]
            vals = np.asarray(gc_data[metric])
            ax.plot(times, vals, color=colors.get(cond, "gray"), linewidth=2, label=f"{cond} (n={gc_data['n_trials']})")
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GC" if metric != "gc_net_z" else "GC z-score")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
    if band_name is None:
        band_name = get_frequency_band_params(fmin, fmax)["band_name"]
    fig.suptitle(f"{subj_id}: {roi_source}→{roi_target} | {band_name} band | memory outcome", fontsize=14, y=1.02)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        band_suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"{subj_id}_{roi_source}_{roi_target}_{event}{band_suffix}_subject_memory_outcome_summary.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    return fig


def validate_smoke_outputs_memory_outcome(
    smoke_pack: Dict[str, Any],
    combined_df: pd.DataFrame,
) -> None:
    required_beh_cols = {"participant", "condition", "memory_outcome"}
    missing_beh = sorted(required_beh_cols.difference(combined_df.columns))
    if "memory_outcome" not in combined_df.columns:
        missing_beh.append("memory_outcome")
    if len(missing_beh) > 0:
        raise AssertionError(f"Missing required behavior columns: {missing_beh}")

    result = smoke_pack["result"]
    smoke_df = smoke_pack["smoke_df"]
    surrogate_cfg = result.get("surrogate_config", result.get("gc_by_hemi", {}).get("surrogate_config", {}))

    trial_map = result["gc_by_hemi"].get("trial_map", pd.DataFrame())
    if len(trial_map) == 0:
        raise AssertionError("Trial map empty in smoke output")
    if "alignment_valid" not in trial_map.columns:
        raise AssertionError("alignment_valid missing from trial map")

    labels = set(trial_map.loc[trial_map["alignment_valid"], "memory_outcome"].dropna().unique().tolist())
    if not labels.issubset(set(MEMORY_OUTCOME_CONDITIONS)):
        raise AssertionError(f"Unexpected memory_outcome labels: {labels}")
    if len(labels) == 0:
        raise AssertionError("No valid memory_outcome labels after alignment")

    gc_by_hemi = result["gc_by_hemi"]
    for hemi in ("l", "r"):
        if hemi not in gc_by_hemi:
            continue
        hemi_data = gc_by_hemi[hemi]
        for cond in MEMORY_OUTCOME_CONDITIONS:
            entry = hemi_data.get(cond)
            if not isinstance(entry, dict) or not entry.get("success", False):
                continue
            for k in ("gc_forward", "gc_backward", "gc_net", "times"):
                if k not in entry or entry[k] is None:
                    raise AssertionError(f"Missing {k} in {hemi}/{cond} output")
            n_t = len(entry["times"])
            if not (len(entry["gc_forward"]) == len(entry["gc_backward"]) == len(entry["gc_net"]) == n_t):
                raise AssertionError(f"GC vector length mismatch in {hemi}/{cond}")

    required_export_cols = {
        "participant", "roi_source", "roi_target", "memory_outcome", "time",
        "gc_forward", "gc_backward", "gc_net", "n_trials", "n_dropped_alignment",
    }
    missing_export = sorted(required_export_cols.difference(smoke_df.columns))
    if len(missing_export) > 0:
        raise AssertionError(f"Missing export columns: {missing_export}")

    if surrogate_cfg.get("use_surrogates", False):
        if "gc_net_z" not in smoke_df.columns:
            raise AssertionError("gc_net_z missing in surrogate-enabled smoke export")
        if "p_net" not in smoke_df.columns:
            raise AssertionError("p_net missing in surrogate-enabled smoke export")
        if smoke_df["gc_net_z"].isna().all():
            raise AssertionError("gc_net_z all NaN in surrogate-enabled smoke export")

    if smoke_pack.get("fig", None) is None:
        raise AssertionError("Smoke plot figure is None")


def validate_smoke_outputs_reward(
    smoke_pack: Dict[str, Any],
    combined_df: pd.DataFrame,
) -> None:
    reward_src = "reward_gamble" if "reward_gamble" in combined_df.columns else "reward"
    required_beh_cols = {"participant", "condition", reward_src}
    missing_beh = sorted(required_beh_cols.difference(combined_df.columns))
    if len(missing_beh) > 0:
        raise AssertionError(f"Missing required behavior columns: {missing_beh}")

    result = smoke_pack["result"]
    smoke_df = smoke_pack["smoke_df"]
    surrogate_cfg = result.get("surrogate_config", result.get("gc_by_hemi", {}).get("surrogate_config", {}))

    trial_map = result["gc_by_hemi"].get("trial_map", pd.DataFrame())
    if len(trial_map) == 0:
        raise AssertionError("Trial map empty in smoke output")
    if "alignment_valid" not in trial_map.columns:
        raise AssertionError("alignment_valid missing from trial map")

    labels = set(trial_map.loc[trial_map["alignment_valid"], "reward_outcome"].dropna().unique().tolist())
    if not labels.issubset(set(REWARD_OUTCOME_CONDITIONS)):
        raise AssertionError(f"Unexpected reward_outcome labels: {labels}")
    if len(labels) == 0:
        raise AssertionError("No valid reward_outcome labels after alignment")

    gc_by_hemi = result["gc_by_hemi"]
    for hemi in ("l", "r"):
        if hemi not in gc_by_hemi:
            continue
        hemi_data = gc_by_hemi[hemi]
        for cond in REWARD_OUTCOME_CONDITIONS:
            entry = hemi_data.get(cond)
            if not isinstance(entry, dict) or not entry.get("success", False):
                continue
            for k in ("gc_forward", "gc_backward", "gc_net", "times"):
                if k not in entry or entry[k] is None:
                    raise AssertionError(f"Missing {k} in {hemi}/{cond} output")
            n_t = len(entry["times"])
            if not (len(entry["gc_forward"]) == len(entry["gc_backward"]) == len(entry["gc_net"]) == n_t):
                raise AssertionError(f"GC vector length mismatch in {hemi}/{cond}")

    required_export_cols = {
        "participant",
        "roi_source",
        "roi_target",
        "reward_outcome",
        "time",
        "gc_forward",
        "gc_backward",
        "gc_net",
        "n_trials",
        "n_dropped_alignment",
    }
    missing_export = sorted(required_export_cols.difference(smoke_df.columns))
    if len(missing_export) > 0:
        raise AssertionError(f"Missing export columns: {missing_export}")

    if surrogate_cfg.get("use_surrogates", False):
        if "gc_net_z" not in smoke_df.columns:
            raise AssertionError("gc_net_z missing in surrogate-enabled smoke export")
        if "p_net" not in smoke_df.columns:
            raise AssertionError("p_net missing in surrogate-enabled smoke export")
        if smoke_df["gc_net_z"].isna().all():
            raise AssertionError("gc_net_z all NaN in surrogate-enabled smoke export")

    if smoke_pack.get("fig", None) is None:
        raise AssertionError("Smoke plot figure is None")


def plot_smoke_reward(smoke_pack: Dict[str, Any]) -> Optional[plt.Figure]:
    fig = smoke_pack.get("fig", None)
    if fig is not None:
        plt.show()
    return fig


def plot_smoke_memory_outcome(smoke_pack: Dict[str, Any]) -> Optional[plt.Figure]:
    fig = smoke_pack.get("fig", None)
    if fig is not None:
        plt.show()
    return fig


def compute_band_power_by_rpe(
    epoch_data: Dict[str, Any],
    learn_df: pd.DataFrame,
    subj_id: str,
    fmin: float = 4,
    fmax: float = 8,
    day: str = "Day1",
    event: str = "feedback_start",
) -> Dict[str, Any]:
    epochs = epoch_data["epochs"]
    times = np.asarray(epoch_data["times"])
    trial_map = build_rpe_trial_map(learn_df, subj_id, n_epochs=len(epochs), day=day, event=event)
    trial_map = trial_map[trial_map["alignment_valid"]].copy() if len(trial_map) > 0 else trial_map

    results: Dict[str, Any] = {
        "source": {"neg": None, "neutral": None, "pos": None},
        "target": {"neg": None, "neutral": None, "pos": None},
        "times": times,
    }

    for roi_type in ("source", "target"):
        all_channels: List[str] = []
        for hemi in ("l", "r"):
            all_channels.extend(epoch_data.get(f"{roi_type}_channels_{hemi}", []))
        all_channels = [ch for ch in all_channels if ch in epochs.ch_names]
        if len(all_channels) == 0:
            continue

        epochs_roi = epochs.copy().pick_channels(all_channels)
        epochs_roi = epochs_roi.copy().filter(fmin, fmax, verbose=False)
        power_data = epochs_roi.get_data() ** 2
        power_roi = np.mean(power_data, axis=1)

        for cond in ("neg", "neutral", "pos"):
            cond_map = trial_map[trial_map["rpe_cat"] == cond]
            idx = cond_map["epoch_idx"].to_numpy(dtype=int)
            if len(idx) == 0:
                continue
            cond_power = power_roi[idx, :]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(len(idx))
            results[roi_type][cond] = {
                "mean": mean_power,
                "sem": sem_power,
                "n_trials": int(len(idx)),
            }
    return results


def generate_subject_plot_rpe(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    result_dict: Dict[str, Any],
    epoch_data: Dict[str, Any],
    learn_df: pd.DataFrame,
    fmin: float = 4,
    fmax: float = 8,
    band_name: Optional[str] = None,
    savepath: Optional[str] = None,
    event: str = "feedback_start",
) -> Optional[plt.Figure]:
    condition_keys = ("neg", "neutral", "pos")
    combined_gc = combine_hemispheres(result_dict["gc_by_hemi"], condition_keys=condition_keys)
    band_power = compute_band_power_by_rpe(epoch_data, learn_df, subj_id, fmin=fmin, fmax=fmax, event=event)
    valid = [k for k in condition_keys if combined_gc.get(k, {}).get("success", False)]
    if len(valid) == 0:
        return None

    colors = {"neg": "#3498db", "neutral": "#95a5a6", "pos": "#e74c3c"}
    labels = {"neg": "Negative", "neutral": "Neutral", "pos": "Positive"}

    times = np.asarray(combined_gc[valid[0]]["times"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, roi_type in enumerate(("source", "target")):
        ax = axes[0, col]
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            ax.plot(times, entry["mean"], color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={entry['n_trials']})")
            ax.fill_between(times, entry["mean"] - entry["sem"], entry["mean"] + entry["sem"], color=colors[cond], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Band Power (a.u.)")
        ax.set_title(f"{'Source' if roi_type=='source' else 'Target'}: {roi_source if roi_type=='source' else roi_target}")
        ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 2]
    for roi_type, clr in (("source", "#1abc9c"), ("target", "#9b59b6")):
        for cond in valid:
            entry = band_power[roi_type].get(cond)
            if entry is None:
                continue
            linestyle = "-" if cond == "neutral" else "--" if cond == "neg" else ":"
            ax.plot(times, entry["mean"], color=clr, linestyle=linestyle, linewidth=2,
                    label=f"{roi_source if roi_type=='source' else roi_target} ({labels[cond]})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band Power (a.u.)")
    ax.set_title("Source vs Target Overlay")
    ax.legend(fontsize=8, loc="upper right")

    net_metric = "gc_net_z" if all("gc_net_z" in combined_gc[k] for k in valid) else "gc_net"
    net_label = "Net GC z-score" if net_metric == "gc_net_z" else "Net GC"
    metrics = [
        ("gc_forward", f"Forward GC: {roi_source}→{roi_target}"),
        ("gc_backward", f"Backward GC: {roi_target}→{roi_source}"),
        (net_metric, f"{net_label}: {roi_source}↔{roi_target}")
    ]
    for col, (metric, title) in enumerate(metrics):
        ax = axes[1, col]
        for cond in valid:
            gc_data = combined_gc[cond]
            vals = np.asarray(gc_data[metric])
            ax.plot(times, vals, color=colors[cond], linewidth=2, label=f"{labels[cond]} (n={gc_data['n_trials']})")
            if metric in ("gc_net", "gc_net_z"):
                ax.fill_between(times, 0, vals, color=colors[cond], alpha=0.12)
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GC" if metric != "gc_net_z" else "GC z-score")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    if band_name is None:
        band_name = get_frequency_band_params(fmin, fmax)["band_name"]
    fig.suptitle(f"{subj_id}: {roi_source}→{roi_target} | {band_name} band ({fmin}-{fmax} Hz) | RPE categories", fontsize=14, y=1.02)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        band_suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"{subj_id}_{roi_source}_{roi_target}_{event}{band_suffix}_subject_rpe_summary.pdf")
        fig.savefig(out_file, dpi=FIG_SAVE_DPI, bbox_inches="tight")
    return fig


def merge_with_behavior_rpe(
    mvgc_df: pd.DataFrame,
    learn_df: pd.DataFrame,
    event: str = "feedback_start",
) -> pd.DataFrame:
    df = mvgc_df.copy()
    ldf = learn_df.copy()
    if "condition" in ldf.columns:
        ldf = ldf[ldf["condition"] == "Day1"].copy()
    if "rpe_cat" not in ldf.columns and "rpe" in ldf.columns and "participant" in ldf.columns:
        for subj in ldf["participant"].dropna().unique():
            mask = ldf["participant"] == subj
            vals = pd.to_numeric(ldf.loc[mask, "rpe"], errors="coerce")
            try:
                ldf.loc[mask, "rpe_cat"] = pd.qcut(vals, q=3, labels=["neg", "neutral", "pos"], duplicates="drop")
            except Exception:
                ldf.loc[mask, "rpe_cat"] = np.nan
    if "rpe_cat" in ldf.columns:
        ldf["rpe_cat"] = _normalize_rpe_cat_series(ldf["rpe_cat"])

    event_name = str(event).strip().lower()
    trial_candidates = ("trials_mem", "trials_gamble", "trials") if event_name == "cue_on" else ("trials", "trials_gamble", "trials_mem")
    for trial_col in trial_candidates:
        if trial_col not in df.columns or trial_col not in ldf.columns:
            continue
        # Use trial-level merge only when the trial column is scalar-like (not comma-joined lists).
        trial_series = df[trial_col].astype(str)
        has_list_like = trial_series.str.contains(",", regex=False).any()
        if has_list_like:
            continue
        df = df.copy()
        ldf = ldf.copy()
        df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce")
        ldf[trial_col] = pd.to_numeric(ldf[trial_col], errors="coerce")
        keep_cols = [
            c for c in [
                "participant", trial_col, "rpe", "zrpe", "rpe_cat", "reward", "reward_gamble",
                "gamble", "gamble_gamble", "gamble_rt", "gamble_rt_gamble", "hits", "memory_outcome"
            ] if c in ldf.columns
        ]
        return df.merge(ldf[keep_cols], on=["participant", trial_col], how="left")

    agg_dict = {}
    for col in ["rpe", "zrpe", "reward", "reward_gamble", "gamble", "gamble_gamble", "trials", "trials_gamble", "trials_mem", "hits"]:
        if col in ldf.columns:
            agg_dict[col] = "mean"
    if "memory_outcome" in ldf.columns:
        agg_dict["memory_outcome"] = lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size > 0 else np.nan

    if len(agg_dict) > 0 and {"participant", "rpe_cat"}.issubset(ldf.columns):
        beh_agg = ldf.groupby(["participant", "rpe_cat"], as_index=False).agg(agg_dict)
        return df.merge(beh_agg, on=["participant", "rpe_cat"], how="left")
    return df


def fit_mixed_effects_rpe(model_df: pd.DataFrame, reference: str = "neutral") -> Dict[str, Any]:
    required = {"participant", "target_roi", "gc_net_z", "rpe_cat"}
    missing = sorted(required.difference(model_df.columns))
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for mixed model: {missing}")

    df = model_df.copy()
    df = df[df["gc_net_z"].notna() & df["rpe_cat"].notna()].copy()
    df["rpe_cat"] = _normalize_rpe_cat_series(df["rpe_cat"])
    df["rpe_cat"] = pd.Categorical(df["rpe_cat"], categories=[reference, "neg", "pos"], ordered=False)

    model_results: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    for target in sorted(df["target_roi"].dropna().unique()):
        target_data = df[df["target_roi"] == target].copy()
        if len(target_data) < 20 or target_data["participant"].nunique() < 2:
            continue
        try:
            formula = f"gc_net_z ~ C(rpe_cat, Treatment('{reference}'))"
            model = smf.mixedlm(formula, data=target_data, groups=target_data["participant"])
            fit = model.fit(method="lbfgs", maxiter=500)
            model_results[target] = fit
            conf_int = fit.conf_int()
            for param in fit.params.index:
                pval = float(fit.pvalues[param])
                coef = float(fit.params[param])
                ci_low, ci_high = conf_int.loc[param]
                records.append(
                    {
                        "target_roi": target,
                        "param": param,
                        "coef": coef,
                        "pval": pval,
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }
                )
        except Exception as exc:
            model_results[target] = {"error": str(exc)}

    stats_df = pd.DataFrame(records)
    if len(stats_df) > 0:
        mask = ~stats_df["param"].str.contains("Intercept|Group Var", regex=True)
        p = stats_df.loc[mask, "pval"].to_numpy(dtype=float)
        if len(p) > 0:
            _, qvals, _, _ = multipletests(p, method="fdr_bh")
            stats_df.loc[mask, "pval_fdr"] = qvals
    return {"model_results": model_results, "stats_df": stats_df, "model_df": df}


def plot_mixed_effects_rpe(
    model_out: Dict[str, Any],
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    add_strip: bool = True,
    **kwargs: Any,
) -> Optional[plt.Figure]:
    """
    Two-panel event-split plot: left = feedback_start, right = cue_on.
    Expects model_df to include an 'event' column when comparing events;
    shared axes and one legend. Participant-mean dots over bars when add_strip=True.

    Plot kwargs applied before saving: ylim, xlim (e.g. ylim=(-1, 1)).
    savefig kwargs: dpi (min 300), bbox_inches, etc.
    """
    model_df = model_out.get("model_df", pd.DataFrame())
    if len(model_df) == 0:
        return None

    has_event = "event" in model_df.columns
    event_left = "feedback_start"
    event_right = "cue_on"
    colors = {"neg": "#d62728", "neutral": "#7f7f7f", "pos": "#2ca02c"}
    cats = ["neg", "neutral", "pos"]

    if has_event:
        df_left = model_df[model_df["event"] == event_left]
        df_right = model_df[model_df["event"] == event_right]
        target_order = sorted(model_df["target_roi"].unique())
    else:
        df_left = model_df.copy()
        df_right = pd.DataFrame()
        target_order = sorted(model_df["target_roi"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_lims = [np.nan, np.nan]
    legend_handles, legend_labels = None, None

    for idx, (ax, ev_name, df_ev) in enumerate([(axes[0], event_left, df_left), (axes[1], event_right, df_right)]):
        if len(df_ev) == 0:
            ax.text(0.5, 0.5, f"{ev_name}\n(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks(np.arange(len(target_order)))
            ax.set_xticklabels(target_order)
            ax.set_ylabel("Mean gc_net_z")
            ax.set_title(ev_name)
            continue
        means = df_ev.groupby(["target_roi", "rpe_cat"])["gc_net_z"].agg(["mean", "sem"]).reset_index()
        x = np.arange(len(target_order))
        width = 0.25
        for i, cat in enumerate(cats):
            vals = means[means["rpe_cat"] == cat].set_index("target_roi").reindex(target_order)
            ax.bar(
                x + (i - 1) * width,
                vals["mean"],
                width,
                yerr=vals["sem"],
                color=colors[cat],
                alpha=0.85,
                capsize=2,
                label=cat,
            )
        if add_strip and "gc_net_z" in df_ev.columns:
            subj_means = (
                df_ev.groupby(["participant", "target_roi", "rpe_cat"])["gc_net_z"]
                .mean()
                .reset_index()
            )
            sns.stripplot(
                data=subj_means,
                x="target_roi",
                y="gc_net_z",
                hue="rpe_cat",
                order=target_order,
                hue_order=cats,
                dodge=True,
                palette=colors,
                alpha=0.6,
                edgecolor="black",
                linewidth=0.3,
                size=3,
                ax=ax,
                zorder=10,
            )
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(target_order)
        ax.set_ylabel("Mean gc_net_z")
        ax.set_title(ev_name)
        y_min, y_max = ax.get_ylim()
        y_lims[0] = np.nanmin([y_lims[0], y_min])
        y_lims[1] = np.nanmax([y_lims[1], y_max])
        handles, labels_leg = ax.get_legend_handles_labels()
        if handles and labels_leg and legend_handles is None:
            by_label = dict(zip(labels_leg, handles))
            legend_handles, legend_labels = list(by_label.values()), list(by_label.keys())
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if legend_handles is not None and legend_labels is not None:
        axes[1].legend(legend_handles, legend_labels, loc="upper right")

    if np.isfinite(y_lims[0]) and np.isfinite(y_lims[1]):
        for ax in axes:
            ax.set_ylim(y_lims[0], y_lims[1])

    plt.suptitle(f"Mixed Effects: {source_roi}→targets | RPE category")
    plt.tight_layout()
    save_kw = _apply_plot_kwargs(fig, kwargs)
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_name = (
            f"mixed_model_gc_rpe_{source_roi}_to_targets_feedback_cue{suffix}.pdf"
            if has_event
            else f"mixed_model_gc_rpe_{source_roi}_to_targets_{event}{suffix}.pdf"
        )
        out_file = os.path.join(savepath, out_name)
        sns.despine()
        dpi = max(FIG_SAVE_DPI, save_kw.pop("dpi", FIG_SAVE_DPI))
        fig.savefig(out_file, dpi=dpi, bbox_inches=save_kw.pop("bbox_inches", "tight"), **save_kw)
    return fig


def plot_aggregate_rpe(
    agg_df: pd.DataFrame,
    source_roi: str = "ACC",
    savepath: Optional[str] = None,
    event: str = "feedback_start",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    **kwargs: Any,
) -> Optional[plt.Figure]:
    """
    Plot kwargs applied before saving: ylim, xlim.
    savefig kwargs: dpi (min 300), bbox_inches, etc.
    """
    req = {"target_roi", "gc_forward", "gc_backward", "gc_net", "rpe_cat"}
    if len(agg_df) == 0 or len(req.difference(agg_df.columns)) > 0:
        return None
    df = agg_df.copy()
    df["rpe_cat"] = _normalize_rpe_cat_series(df["rpe_cat"])
    net_col = "gc_net_z" if ("gc_net_z" in df.columns and df["gc_net_z"].notna().any()) else "gc_net"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = {"neg": "#d62728", "neutral": "#7f7f7f", "pos": "#2ca02c"}

    ax = axes[0, 0]
    sns.violinplot(data=df, x="target_roi", y=net_col, hue="rpe_cat", inner="quart", palette=palette, ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"{'Net GC z-score' if net_col=='gc_net_z' else 'Net GC'} distribution by target ROI")

    ax = axes[0, 1]
    means = df.groupby("target_roi")[["gc_forward", "gc_backward"]].mean().reset_index()
    x = np.arange(len(means))
    w = 0.35
    ax.bar(x - w / 2, means["gc_forward"], w, color="#2ecc71", label="Forward")
    ax.bar(x + w / 2, means["gc_backward"], w, color="#f39c12", label="Backward")
    ax.set_xticks(x)
    ax.set_xticklabels(means["target_roi"].tolist())
    ax.set_title("Forward vs Backward GC")
    ax.legend()

    ax = axes[1, 0]
    hm = df.groupby(["target_roi", "rpe_cat"])[net_col].mean().unstack("rpe_cat")
    hm["delta_pos_minus_neg"] = hm.get("pos", np.nan) - hm.get("neg", np.nan)
    hm["delta_pos_minus_neg"].plot(kind="bar", color="#8e44ad", ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title(f"Mean {'z-scored Net GC' if net_col=='gc_net_z' else 'Net GC'}: Pos - Neg")
    ax.set_ylabel("Delta")

    ax = axes[1, 1]
    counts = df.groupby(["target_roi", "rpe_cat"]).size().reset_index(name="n")
    sns.barplot(data=counts, x="target_roi", y="n", hue="rpe_cat", palette=palette, ax=ax)
    ax.set_title("Sample counts")

    plt.suptitle(f"Aggregate MVGC summary (RPE) | source={source_roi}")
    plt.tight_layout()
    save_kw = _apply_plot_kwargs(fig, kwargs)
    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        suffix = _format_band_suffix(fmin=fmin, fmax=fmax)
        out_file = os.path.join(savepath, f"aggregate_{source_roi}_to_targets_rpe_summary_{event}{suffix}.pdf")
        dpi = max(FIG_SAVE_DPI, save_kw.pop("dpi", FIG_SAVE_DPI))
        fig.savefig(out_file, dpi=dpi, bbox_inches=save_kw.pop("bbox_inches", "tight"), **save_kw)
    return fig


# -----------------------------------------------------------------------------
# Smoke helpers
# -----------------------------------------------------------------------------

def run_smoke_subject_gc(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    combined_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 200,
    surrogate_tail: str = "greater",
    surrogate_random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = True,
    verbose: bool = True,
    savepath: Optional[str] = None,
) -> Dict[str, Any]:
    result = process_subject_gc_hit_miss(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        combined_df=combined_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=surrogate_random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )
    if result is None:
        raise RuntimeError("Smoke run failed to produce result")

    smoke_df = export_gc_to_dataframe(result, subj_id=subj_id, roi_source=roi_source, roi_target=roi_target)
    fig = generate_subject_plot_hit_miss(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        result_dict=result,
        epoch_data=result["epoch_data"],
        combined_df=combined_df,
        fmin=fmin,
        fmax=fmax,
        savepath=savepath,
    )
    return {"result": result, "smoke_df": smoke_df, "fig": fig}


def validate_smoke_outputs(
    smoke_pack: Dict[str, Any],
    combined_df: pd.DataFrame,
) -> None:
    # 1) behavior columns
    required_beh_cols = {"participant", "condition", "trials_gamble", "hits"}
    missing_beh = sorted(required_beh_cols.difference(combined_df.columns))
    if len(missing_beh) > 0:
        raise AssertionError(f"Missing required behavior columns: {missing_beh}")

    result = smoke_pack["result"]
    smoke_df = smoke_pack["smoke_df"]
    surrogate_cfg = result.get("surrogate_config", result.get("gc_by_hemi", {}).get("surrogate_config", {}))

    # 2) alignment diagnostics
    trial_map = result["gc_by_hemi"].get("trial_map", pd.DataFrame())
    if len(trial_map) == 0:
        raise AssertionError("Trial map empty in smoke output")
    if "alignment_valid" not in trial_map.columns:
        raise AssertionError("alignment_valid missing from trial map")

    # 3) hit/miss labels present
    labels = set(trial_map.loc[trial_map["alignment_valid"], "hit_miss"].dropna().unique().tolist())
    if not labels.issubset({"hit", "miss"}):
        raise AssertionError(f"Unexpected hit_miss labels: {labels}")
    if len(labels) == 0:
        raise AssertionError("No valid hit/miss labels after alignment")

    # 4) vector consistency
    gc_by_hemi = result["gc_by_hemi"]
    for hemi in ("l", "r"):
        if hemi not in gc_by_hemi:
            continue
        hemi_data = gc_by_hemi[hemi]
        for cond in ("hit", "miss"):
            entry = hemi_data.get(cond)
            if not isinstance(entry, dict) or not entry.get("success", False):
                continue
            for k in ("gc_forward", "gc_backward", "gc_net", "times"):
                if k not in entry or entry[k] is None:
                    raise AssertionError(f"Missing {k} in {hemi}/{cond} output")
            n_t = len(entry["times"])
            if not (len(entry["gc_forward"]) == len(entry["gc_backward"]) == len(entry["gc_net"]) == n_t):
                raise AssertionError(f"GC vector length mismatch in {hemi}/{cond}")

    # 5) export schema
    required_export_cols = {
        "participant", "roi_source", "roi_target", "hit_miss", "time",
        "gc_forward", "gc_backward", "gc_net", "n_trials",
        "n_dropped_alignment",
    }
    missing_export = sorted(required_export_cols.difference(smoke_df.columns))
    if len(missing_export) > 0:
        raise AssertionError(f"Missing export columns: {missing_export}")

    # Surrogate-enabled checks
    if surrogate_cfg.get("use_surrogates", False):
        if "gc_net_z" not in smoke_df.columns:
            raise AssertionError("gc_net_z missing in surrogate-enabled smoke export")
        if "p_net" not in smoke_df.columns:
            raise AssertionError("p_net missing in surrogate-enabled smoke export")
        if "n_surrogates" not in smoke_df.columns:
            raise AssertionError("n_surrogates missing in surrogate-enabled smoke export")
        if smoke_df["gc_net_z"].isna().all():
            raise AssertionError("gc_net_z all NaN in surrogate-enabled smoke export")
        if smoke_df["p_net"].isna().all():
            raise AssertionError("p_net all NaN in surrogate-enabled smoke export")
        pvals = smoke_df["p_net"].dropna().to_numpy(dtype=float)
        if pvals.size > 0 and (np.nanmin(pvals) < 0 or np.nanmax(pvals) > 1):
            raise AssertionError("p_net values out of [0,1] range")
        if (smoke_df["n_surrogates"].fillna(0) <= 0).all():
            raise AssertionError("n_surrogates not positive in surrogate-enabled smoke export")

    # 6) plot object
    if smoke_pack.get("fig", None) is None:
        raise AssertionError("Smoke plot figure is None")


def plot_smoke_hit_miss(smoke_pack: Dict[str, Any]) -> Optional[plt.Figure]:
    fig = smoke_pack.get("fig", None)
    if fig is not None:
        plt.show()
    return fig


def run_smoke_subject_gc_rpe(
    subj_id: str,
    roi_source: str,
    roi_target: str,
    base_dir: str,
    learn_df: pd.DataFrame,
    event: str = "feedback_start",
    tmin: float = 0.0,
    tmax: float = 1.5,
    target_sfreq: float = 50,
    fmin: float = 4,
    fmax: float = 8,
    gc_n_lags: int = 10,
    n_freqs: Optional[int] = None,
    min_trials: int = 5,
    day: str = "Day1",
    use_surrogates: bool = True,
    n_surrogates: int = 200,
    surrogate_tail: str = "greater",
    surrogate_random_state: int = 42,
    n_jobs_surrogates: int = 1,
    surrogate_parallel_backend: str = "loky",
    store_surrogates: bool = True,
    verbose: bool = True,
    savepath: Optional[str] = None,
) -> Dict[str, Any]:
    result = process_subject_gc_rpe(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        base_dir=base_dir,
        learn_df=learn_df,
        event=event,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
        fmin=fmin,
        fmax=fmax,
        gc_n_lags=gc_n_lags,
        n_freqs=n_freqs,
        min_trials=min_trials,
        day=day,
        use_surrogates=use_surrogates,
        n_surrogates=n_surrogates,
        surrogate_tail=surrogate_tail,
        random_state=surrogate_random_state,
        n_jobs_surrogates=n_jobs_surrogates,
        surrogate_parallel_backend=surrogate_parallel_backend,
        store_surrogates=store_surrogates,
        verbose=verbose,
    )
    if result is None:
        raise RuntimeError("Smoke run failed to produce result")

    trial_col = str(result.get("gc_by_hemi", {}).get("trial_value_col", _resolve_trial_col_for_event(learn_df, event)))
    smoke_df = export_gc_to_dataframe(
        result,
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        condition_keys=("neg", "neutral", "pos"),
        condition_col="rpe_cat",
        trial_col=trial_col,
    )
    fig = generate_subject_plot_rpe(
        subj_id=subj_id,
        roi_source=roi_source,
        roi_target=roi_target,
        result_dict=result,
        epoch_data=result["epoch_data"],
        learn_df=learn_df,
        fmin=fmin,
        fmax=fmax,
        event=event,
        savepath=savepath,
    )
    return {"result": result, "smoke_df": smoke_df, "fig": fig}


def validate_smoke_outputs_rpe(
    smoke_pack: Dict[str, Any],
    learn_df: pd.DataFrame,
    event: str = "feedback_start",
) -> None:
    required_beh_cols = {"participant", _resolve_trial_col_for_event(learn_df, event)}
    missing_beh = sorted(required_beh_cols.difference(learn_df.columns))
    if len(missing_beh) > 0:
        raise AssertionError(f"Missing required behavior columns: {missing_beh}")

    result = smoke_pack["result"]
    smoke_df = smoke_pack["smoke_df"]
    surrogate_cfg = result.get("surrogate_config", result.get("gc_by_hemi", {}).get("surrogate_config", {}))

    trial_map = result["gc_by_hemi"].get("trial_map", pd.DataFrame())
    if len(trial_map) == 0:
        raise AssertionError("Trial map empty in smoke output")
    if "alignment_valid" not in trial_map.columns:
        raise AssertionError("alignment_valid missing from trial map")

    labels = set(trial_map.loc[trial_map["alignment_valid"], "rpe_cat"].dropna().unique().tolist())
    if not labels.issubset({"neg", "neutral", "pos"}):
        raise AssertionError(f"Unexpected rpe_cat labels: {labels}")
    if len(labels) == 0:
        raise AssertionError("No valid rpe_cat labels after alignment")

    gc_by_hemi = result["gc_by_hemi"]
    for hemi in ("l", "r"):
        if hemi not in gc_by_hemi:
            continue
        hemi_data = gc_by_hemi[hemi]
        for cond in ("neg", "neutral", "pos"):
            entry = hemi_data.get(cond)
            if not isinstance(entry, dict) or not entry.get("success", False):
                continue
            for k in ("gc_forward", "gc_backward", "gc_net", "times"):
                if k not in entry or entry[k] is None:
                    raise AssertionError(f"Missing {k} in {hemi}/{cond} output")
            n_t = len(entry["times"])
            if not (len(entry["gc_forward"]) == len(entry["gc_backward"]) == len(entry["gc_net"]) == n_t):
                raise AssertionError(f"GC vector length mismatch in {hemi}/{cond}")

    required_export_cols = {
        "participant", "roi_source", "roi_target", "rpe_cat", "time",
        "gc_forward", "gc_backward", "gc_net", "n_trials",
        "n_dropped_alignment",
    }
    missing_export = sorted(required_export_cols.difference(smoke_df.columns))
    if len(missing_export) > 0:
        raise AssertionError(f"Missing export columns: {missing_export}")

    if surrogate_cfg.get("use_surrogates", False):
        for col in ("gc_net_z", "p_net", "n_surrogates"):
            if col not in smoke_df.columns:
                raise AssertionError(f"{col} missing in surrogate-enabled smoke export")
        if smoke_df["gc_net_z"].isna().all():
            raise AssertionError("gc_net_z all NaN in surrogate-enabled smoke export")
        if smoke_df["p_net"].isna().all():
            raise AssertionError("p_net all NaN in surrogate-enabled smoke export")
        pvals = smoke_df["p_net"].dropna().to_numpy(dtype=float)
        if pvals.size > 0 and (np.nanmin(pvals) < 0 or np.nanmax(pvals) > 1):
            raise AssertionError("p_net values out of [0,1] range")
        if (smoke_df["n_surrogates"].fillna(0) <= 0).all():
            raise AssertionError("n_surrogates not positive in surrogate-enabled smoke export")

    if smoke_pack.get("fig", None) is None:
        raise AssertionError("Smoke plot figure is None")


def plot_smoke_rpe(smoke_pack: Dict[str, Any]) -> Optional[plt.Figure]:
    fig = smoke_pack.get("fig", None)
    if fig is not None:
        plt.show()
    return fig
