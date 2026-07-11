"""
Canonical behavior table loading and preprocessing for MemoryBandit EMU analyses.

Unified conventions:
- ``rpe_cat`` uses labels ``negative``, ``neutral``, ``positive`` for both learn and combined tables.
- ``resolve_trial_for_event`` replaces both ``_resolve_trial_var_for_event`` (multivariate) and
  ``_resolve_trial_col_for_event`` (granger).
"""

from __future__ import annotations

from typing import Tuple

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


def normalize_rpe_cat_labels(series: pd.Series) -> pd.Series:
    """Map mixed RPE category strings to canonical negative/neutral/positive."""
    normalized = (
        series.astype(str).str.strip().str.lower().map(RPE_LABEL_MAP)
    )
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


def load_behavior_tables(load_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load behavior tables used by MV analyses.

    Returns
    -------
    (learn_df, combined_df) from ``learn_df_RWH.csv`` and ``full_df_RWH.csv``.
    """
    import os

    learn_path = os.path.join(load_dir, "learn_df_RWH.csv")
    combined_path = os.path.join(load_dir, "full_df_RWH.csv")
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
                        labels=["negative", "neutral", "positive"],
                        duplicates="drop",
                    )
                except Exception:
                    df.loc[mask, "rpe_cat"] = np.nan

    return df
