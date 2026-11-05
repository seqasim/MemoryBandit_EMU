"""Validated learn_df → pyEM subject-major data adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from rlmodels_k99.task import block_lengths, validate_binary_series

IncludeSubjects = Literal[True, False, "good_dm", "all"]


@dataclass
class LearnDFBundle:
    subject_ids: list[Any]
    all_data: list[list[np.ndarray]]
    choices: list[np.ndarray]
    rewards: list[np.ndarray]
    meta: pd.DataFrame
    n_trials: int
    trial_col: str


def _resolve_trial_col(df: pd.DataFrame, trial_col: str | None) -> str:
    if trial_col is not None:
        if trial_col not in df.columns:
            raise ValueError(f"trial_col={trial_col!r} not in columns")
        return trial_col
    for cand in ("trials", "learn_trials"):
        if cand in df.columns:
            return cand
    raise ValueError("Could not find trial column 'trials' or 'learn_trials'")


def learn_df_to_pyem(
    learn_df: pd.DataFrame,
    *,
    trial_col: str | None = None,
    include_subjects: IncludeSubjects = "good_dm",
    n_trials: int = 80,
    require_complete: bool = True,
) -> LearnDFBundle:
    """Convert a MemoryBandit learn_df into pyEM subject-major arrays.

    Fitting uses raw ``gamble`` / ``reward`` columns (0/1 integers).
    Never uses reshape(n_subj, n_trials).
    """
    if "participant" not in learn_df.columns:
        raise ValueError("learn_df must contain 'participant'")
    for col in ("gamble", "reward"):
        if col not in learn_df.columns:
            raise ValueError(f"learn_df must contain '{col}'")

    df = learn_df.copy()
    tcol = _resolve_trial_col(df, trial_col)

    if include_subjects == "good_dm":
        if "good_dm_performance" not in df.columns:
            raise ValueError(
                "include_subjects='good_dm' requires column 'good_dm_performance'"
            )
        df = df.loc[df["good_dm_performance"].astype(bool)].copy()
    elif include_subjects is False:
        raise ValueError("include_subjects=False is invalid; use 'all' or True")
    elif include_subjects is True or include_subjects == "all":
        pass
    else:
        raise ValueError(f"Unknown include_subjects={include_subjects!r}")

    df = df.sort_values(["participant", tcol], kind="mergesort").reset_index(drop=True)

    subject_ids: list[Any] = []
    choices: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    rows = []

    for pid, sub in df.groupby("participant", sort=True):
        trials = pd.to_numeric(sub[tcol], errors="coerce").to_numpy()
        if np.any(~np.isfinite(trials)):
            raise ValueError(f"participant {pid}: non-numeric trials")
        trials_i = trials.astype(int)
        if require_complete:
            expected = np.arange(1, n_trials + 1)
            if len(trials_i) != n_trials:
                raise ValueError(
                    f"participant {pid}: expected {n_trials} trials, got {len(trials_i)}"
                )
            if not np.array_equal(trials_i, expected):
                raise ValueError(
                    f"participant {pid}: trials must be contiguous unique 1..{n_trials}"
                )
        else:
            if len(np.unique(trials_i)) != len(trials_i):
                raise ValueError(f"participant {pid}: duplicate trial indices")

        a = validate_binary_series(sub["gamble"].to_numpy(), f"gamble[{pid}]")
        r = validate_binary_series(sub["reward"].to_numpy(), f"reward[{pid}]")
        subject_ids.append(pid)
        choices.append(a)
        rewards.append(r)
        meta_row = {"participant": pid, "n_trials": len(a)}
        if "good_dm_performance" in sub.columns:
            meta_row["good_dm_performance"] = bool(sub["good_dm_performance"].iloc[0])
        if "block" in sub.columns:
            meta_row["n_blocks"] = int(pd.to_numeric(sub["block"]).nunique())
        rows.append(meta_row)

    all_data = [[c, r] for c, r in zip(choices, rewards)]
    meta = pd.DataFrame(rows)
    # Attach expected block structure for documentation/plots.
    meta.attrs["block_lengths"] = block_lengths(n_trials) if n_trials in (60, 80) else None
    return LearnDFBundle(
        subject_ids=subject_ids,
        all_data=all_data,
        choices=choices,
        rewards=rewards,
        meta=meta,
        n_trials=n_trials,
        trial_col=tcol,
    )
