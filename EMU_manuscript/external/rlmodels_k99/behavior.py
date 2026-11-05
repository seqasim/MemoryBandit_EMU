"""
Canonical behavior table loading and preprocessing for MemoryBandit EMU analyses.

Unified conventions:
- ``rpe_cat`` uses labels ``negative``, ``neutral``, ``positive`` for both learn and combined tables.
- ``resolve_trial_for_event`` replaces both ``_resolve_trial_var_for_event`` (multivariate) and
  ``_resolve_trial_col_for_event`` (granger).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# RPE category normalization (single canonical mapping)
# -----------------------------------------------------------------------------

RPE_CATEGORY_ORDER = ["negative", "neutral", "positive"]

RPE_LABEL_MAP = {
    "neg": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}

# Lean columns for R / lme4 mixed models on studied recognition trials.
LME4_COMBINED_COLUMNS = (
    "participant",
    "trials_gamble",
    "block_gamble",
    "hit",
    "hits",
    "misses",
    "rpe_new",
    "zrpe_new",
    "rpe",
    "zrpe",
    "rpe_cat",
    "reward_gamble",
    "preward",
    "Q",
    "zQ",
    # Subject-level learn_df RW params + across-participant Gelman z-scores.
    "alpha",
    "beta",
    "zalpha",
    "zbeta",
    # Hierarchical rw subject params + across-participant Gelman z-scores.
    "alpha_new",
    "beta_new",
    "zalpha_new",
    "zbeta_new",
    # Hierarchical rwus_upe RPE / subject params (model-specific names).
    "rpe_rwus_upe",
    "zrpe_rwus_upe",
    "alpha_rwus_upe",
    "beta_rwus_upe",
    "zalpha_rwus_upe",
    "zbeta_rwus_upe",
    "dprime",
    "DPRIME",
    "image_dprime",
    "zpm",
    "condition",
    "repeated_presentation",
    "recog_time",
    "good_dm_performance_gamble",
    "trials_since_reversal",
    "ztrials_since_reversal",
    "phit",
    "memory_outcome",
)


def normalize_rpe_cat_labels(series: pd.Series) -> pd.Series:
    """Map mixed RPE category strings to canonical negative/neutral/positive."""
    normalized = series.astype(str).str.strip().str.lower().map(RPE_LABEL_MAP)
    return normalized


def resolve_trial_for_event(df: pd.DataFrame, event: str) -> str:
    """
    Resolve trial index column for an event-specific dataframe.

    For ``cue_on``, prefer memory-trial columns when present.
    """
    event_name = str(event).strip().lower()
    if event_name == "cue_on":
        for candidate in ("trials_mem", "trials_gamble", "trials"):
            if candidate in df.columns:
                return candidate
    for candidate in ("trials", "trials_gamble", "trials_mem"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not resolve trial column for event='{event}'. "
        f"Available columns include: {list(df.columns)[:30]}..."
    )


# -----------------------------------------------------------------------------
# Gelman-style standardization (mean-centered, scaled by 2*SD)
# -----------------------------------------------------------------------------


def _gelman_standardize(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return pd.Series((arr - mu) / (2.0 * sd), index=values.index)


def load_behavior_tables(load_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load behavior tables used by MV analyses.

    Returns
    -------
    (learn_df, combined_df) from ``learn_df_RWH.csv`` and ``full_df_RWH.csv``.
    """
    load_dir = Path(load_dir)
    learn_path = load_dir / "learn_df_RWH.csv"
    combined_path = load_dir / "full_df_RWH.csv"
    if not learn_path.exists():
        raise FileNotFoundError(learn_path)
    if not combined_path.exists():
        raise FileNotFoundError(combined_path)
    learn_df = pd.read_csv(learn_path)
    combined_df = pd.read_csv(combined_path)
    return learn_df, combined_df


def preprocess_learn_df(learn_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess learn_df: logs, Q, z-scores, tercile ``rpe_cat`` (negative/neutral/positive)."""
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
                    df.loc[mask, "rpe_cat"] = np.nan

    return df


def preprocess_combined_df(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess combined_df; ``rpe_cat`` uses negative/neutral/positive (same as learn)."""
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
                ("trials_since_reversal", "ztrials_since_reversal"),
            ]:
                if src in df.columns:
                    df.loc[mask, dst] = _gelman_standardize(df.loc[mask, src]).values

            # Image memorability → zpm (prefer image_dprime; EMU uses image-level DPRIME).
            # Do not use subject-level recognition d′ here.
            pm_src = (
                "image_dprime"
                if "image_dprime" in df.columns
                else ("DPRIME" if "DPRIME" in df.columns else None)
            )
            if pm_src is not None:
                df.loc[mask, "zpm"] = _gelman_standardize(df.loc[mask, pm_src]).values

            if "gamble_rt_log" in df.columns and "reward_gamble" in df.columns:
                rt = pd.to_numeric(df.loc[mask, "gamble_rt_log"], errors="coerce").to_numpy(dtype=float)
                rt_min = np.nanmin(rt)
                rt_max = np.nanmax(rt)
                if np.isfinite(rt_min) and np.isfinite(rt_max) and (rt_max - rt_min) > 0:
                    rt_ev = 1.0 - ((rt - rt_min) / (rt_max - rt_min))
                    rt_rpe = (
                        pd.to_numeric(df.loc[mask, "reward_gamble"], errors="coerce").to_numpy(dtype=float)
                        - rt_ev
                    )
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
                    df.loc[mask, "rpe_cat"] = np.nan

    return df


def gelman_standardize_across_subjects(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    participant_col: str = "participant",
    prefix: str = "z",
) -> pd.DataFrame:
    """Gelman-standardize subject-level columns across participants: (x-mean)/(2*SD).

    Takes one value per participant (first non-null), standardizes that subject vector,
    then broadcasts back to all rows. Matches the Gelman scaling used for ``zrpe`` / ``zpm``.
    """
    out = df.copy()
    if participant_col not in out.columns:
        raise ValueError(f"missing participant column {participant_col!r}")
    for col in cols:
        if col not in out.columns:
            continue
        zcol = f"{prefix}{col}" if not col.startswith(prefix) else col
        # Prefer rename mapping alpha -> zalpha rather than zzalpha.
        if col.startswith("z"):
            zcol = col
        elif prefix == "z" and not col.startswith("z"):
            zcol = f"z{col}"
        sub = out.groupby(participant_col, sort=False)[col].first()
        z_sub = _gelman_standardize(sub)
        out[zcol] = out[participant_col].map(z_sub)
    return out


def _default_rpe_colnames(model: str) -> tuple[str, str]:
    """Return (rpe_col, zrpe_col) for a hierarchical model id."""
    if model == "rw":
        return "rpe_new", "zrpe_new"
    return f"rpe_{model}", f"zrpe_{model}"


def _default_param_colnames(model: str) -> tuple[str, str, str, str]:
    """Return (alpha_col, beta_col, zalpha_col, zbeta_col) for a hierarchical model id."""
    if model == "rw":
        return "alpha_new", "beta_new", "zalpha_new", "zbeta_new"
    return (
        f"alpha_{model}",
        f"beta_{model}",
        f"zalpha_{model}",
        f"zbeta_{model}",
    )


def attach_model_rpe(
    combined_df: pd.DataFrame,
    trial_cmp: pd.DataFrame,
    *,
    model: str = "rw",
    rpe_col: str | None = None,
    zrpe_col: str | None = None,
) -> pd.DataFrame:
    """Merge hierarchical trial-wise RPE onto combined_df with model-specific names.

    Default names:
    - ``rw`` → ``rpe_new`` / ``zrpe_new``
    - other models → ``rpe_<model>`` / ``zrpe_<model>``

    Trial RPE is Gelman-standardized within participant.
    """
    if rpe_col is None or zrpe_col is None:
        d_rpe, d_zrpe = _default_rpe_colnames(model)
        rpe_col = rpe_col or d_rpe
        zrpe_col = zrpe_col or d_zrpe

    df = combined_df.copy()
    drop_cols = [c for c in (rpe_col, zrpe_col) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    rpe_src = (
        trial_cmp.loc[trial_cmp["model"] == model, ["participant", "trials", "rpe_fit"]]
        .rename(columns={"trials": "trials_gamble", "rpe_fit": rpe_col})
        .copy()
    )
    if len(rpe_src) == 0:
        raise ValueError(f"no trial_cmp rows for model={model!r}")
    rpe_src["trials_gamble"] = rpe_src["trials_gamble"].astype(float)
    df = df.merge(rpe_src, on=["participant", "trials_gamble"], how="left", validate="m:1")

    df[zrpe_col] = np.nan
    for subj in df["participant"].dropna().unique():
        mask = df["participant"] == subj
        df.loc[mask, zrpe_col] = _gelman_standardize(df.loc[mask, rpe_col]).values
    return df


def attach_rw_rpe_new(
    combined_df: pd.DataFrame,
    trial_cmp: pd.DataFrame,
    *,
    model: str = "rw",
) -> pd.DataFrame:
    """Backward-compatible wrapper: attach ``rw`` RPE as ``rpe_new`` / ``zrpe_new``."""
    return attach_model_rpe(combined_df, trial_cmp, model=model)


def attach_model_subject_params(
    combined_df: pd.DataFrame,
    param_cmp: pd.DataFrame,
    *,
    model: str = "rw",
    alpha_col: str | None = None,
    beta_col: str | None = None,
    standardize_learn_df_params: bool = True,
    recompute_zpm: bool = True,
) -> pd.DataFrame:
    """Attach hierarchical subject α/β with model-specific names and Gelman z-scores.

    Default names:
    - ``rw`` → ``alpha_new`` / ``beta_new`` (+ ``zalpha_new`` / ``zbeta_new``)
    - other models → ``alpha_<model>`` / ``beta_<model>`` (+ z-prefixed versions)

    Subject-level params use across-participant Gelman scaling: (x - mean) / (2 * SD).
    When ``standardize_learn_df_params`` is True, also creates/updates ``zalpha`` / ``zbeta``
    from the original learn_df ``alpha`` / ``beta`` columns.
    """
    if alpha_col is None or beta_col is None:
        d_a, d_b, _, _ = _default_param_colnames(model)
        alpha_col = alpha_col or d_a
        beta_col = beta_col or d_b
    zalpha_col = f"z{alpha_col}"
    zbeta_col = f"z{beta_col}"

    df = combined_df.copy()
    drop_cols = [c for c in (alpha_col, beta_col, zalpha_col, zbeta_col) if c in df.columns]
    # Only drop learn_df z-scores when we intend to recompute them.
    if standardize_learn_df_params:
        drop_cols.extend([c for c in ("zalpha", "zbeta") if c in df.columns and c not in drop_cols])
    if drop_cols:
        df = df.drop(columns=drop_cols)

    sub = (
        param_cmp.loc[param_cmp["model"] == model, ["participant", "alpha_fit", "beta_fit"]]
        .drop_duplicates("participant")
        .rename(columns={"alpha_fit": alpha_col, "beta_fit": beta_col})
    )
    if len(sub) == 0:
        raise ValueError(f"no param_cmp rows for model={model!r}")
    df = df.merge(sub, on="participant", how="left", validate="m:1")

    subject_cols = [c for c in (alpha_col, beta_col) if c in df.columns]
    if standardize_learn_df_params:
        subject_cols = [c for c in ("alpha", "beta") if c in df.columns] + subject_cols
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered = []
    for c in subject_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if ordered:
        df = gelman_standardize_across_subjects(df, ordered)

    if recompute_zpm:
        # Prefer trial-level image memorability (within-participant Gelman).
        # Fall back to across-subject subject-d′ only when no trial-varying source exists.
        trial_pm = None
        for cand in ("image_dprime", "DPRIME"):
            if cand not in df.columns:
                continue
            nuniq_med = float(df.groupby("participant", sort=False)[cand].nunique().median())
            if np.isfinite(nuniq_med) and nuniq_med > 1:
                trial_pm = cand
                break
        if trial_pm is not None:
            df["zpm"] = np.nan
            for pid, g in df.groupby("participant", sort=False):
                vals = pd.to_numeric(g[trial_pm], errors="coerce")
                z = _gelman_standardize(vals)
                df.loc[g.index, "zpm"] = z.to_numpy(dtype=float)
        else:
            dprime_col = (
                "DPRIME" if "DPRIME" in df.columns else ("dprime" if "dprime" in df.columns else None)
            )
            if dprime_col is not None:
                subj = df.groupby("participant", sort=False)[dprime_col].first()
                z_sub = _gelman_standardize(subj)
                df["zpm"] = df["participant"].map(z_sub)
    return df


def attach_rw_subject_params(
    combined_df: pd.DataFrame,
    param_cmp: pd.DataFrame,
    *,
    model: str = "rw",
) -> pd.DataFrame:
    """Backward-compatible wrapper for hierarchical ``rw`` subject params."""
    return attach_model_subject_params(combined_df, param_cmp, model=model)


def trim_combined_df_for_lme4(
    combined_df: pd.DataFrame,
    *,
    studied_only: bool = True,
    require_rpe_new: bool = True,
    require_rpe_col: str | None = None,
    columns: Sequence[str] = LME4_COMBINED_COLUMNS,
) -> pd.DataFrame:
    """Return a lean combined_df suitable for R ``lme4`` mixed models.

    Default: studied items (hit or miss) with non-null RPE column, plus a binary
    ``hit`` column (0/1). Only columns present in ``combined_df`` are kept.

    ``require_rpe_col`` overrides which RPE column must be non-null when filtering
    (default ``rpe_new`` when ``require_rpe_new`` is True).
    """
    df = combined_df.copy()
    if "hits" in df.columns and "hit" not in df.columns:
        df["hit"] = df["hits"].astype(bool).astype(int)

    # Ensure Gelman z-scores for any present subject-level α/β columns.
    subject_like = [
        c
        for c in df.columns
        if c in {"alpha", "beta", "alpha_new", "beta_new"}
        or c.startswith("alpha_")
        or c.startswith("beta_")
    ]
    missing_z = [c for c in subject_like if f"z{c}" not in df.columns]
    if missing_z:
        df = gelman_standardize_across_subjects(df, missing_z)

    if studied_only and {"hits", "misses"}.issubset(df.columns):
        studied = df["hits"].astype(bool) | df["misses"].astype(bool)
        df = df.loc[studied].copy()

    rpe_required = require_rpe_col
    if rpe_required is None and require_rpe_new:
        rpe_required = "rpe_new"
    if rpe_required is not None:
        if rpe_required not in df.columns:
            raise ValueError(f"require_rpe_col={rpe_required!r} missing; attach it first")
        df = df.loc[df[rpe_required].notna()].copy()

    keep = [c for c in columns if c in df.columns]
    out = df.loc[:, keep].reset_index(drop=True)
    if "rpe_cat" in out.columns:
        out["rpe_cat"] = normalize_rpe_cat_labels(out["rpe_cat"])
    return out
