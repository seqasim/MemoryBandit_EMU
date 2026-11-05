"""Canonical continuous-session reversal bandit task.

Schedule convention (verified against MemoryBandit learn_df / HierarchicalRW):
the public markers ``(16, 33, 48, 65)`` are the **0-based** first trials of each
new contingency. Flipping ``mu`` before those indices yields block lengths
``(16, 17, 15, 17, 15)``.

An early-flip compatibility schedule (markers interpreted as 1-based boundaries
converted via ``boundary - 1``) is retained for regression only.
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np

ScheduleName = Literal["canonical", "early_flip"]

# Public markers used in plots / notebooks (also = 0-based flip indices).
REVERSAL_BOUNDARIES_80 = (16, 33, 48, 65)
REVERSAL_BOUNDARIES_60 = (13, 24, 35, 48)
BLOCK_LENGTHS_80 = (16, 17, 15, 17, 15)
BLOCK_LENGTHS_60 = (13, 11, 11, 13, 12)
DEFAULT_MU = np.array([0.8, 0.2], dtype=float)

# Off-by-one schedule used briefly after the first pyEM port (boundary - 1).
EARLY_FLIP_INDICES_80 = (15, 32, 47, 64)
EARLY_FLIP_INDICES_60 = (12, 23, 34, 47)
EARLY_FLIP_BLOCK_LENGTHS_80 = (15, 17, 15, 17, 16)


def reversal_boundaries(n_trials: int) -> tuple[int, ...]:
    if n_trials == 80:
        return REVERSAL_BOUNDARIES_80
    if n_trials == 60:
        return REVERSAL_BOUNDARIES_60
    raise ValueError(f"Unsupported n_trials={n_trials}; expected 60 or 80")


def reversal_indices_0based(
    n_trials: int,
    schedule: ScheduleName = "canonical",
) -> tuple[int, ...]:
    """Return 0-based trial indices at which ``mu`` flips before sampling.

    For ``n_trials`` other than 60/80, returns an empty tuple (stable bandit) so
    diagnostic factorials can vary trial count without inventing a schedule.
    """
    if schedule == "canonical":
        if n_trials in (60, 80):
            return reversal_boundaries(n_trials)
        return ()
    if schedule == "early_flip":
        if n_trials == 80:
            return EARLY_FLIP_INDICES_80
        if n_trials == 60:
            return EARLY_FLIP_INDICES_60
        return ()
    raise ValueError(f"Unknown schedule={schedule!r}")


def block_lengths(n_trials: int, schedule: ScheduleName = "canonical") -> tuple[int, ...]:
    if schedule == "early_flip":
        if n_trials == 80:
            return EARLY_FLIP_BLOCK_LENGTHS_80
        if n_trials == 60:
            flips = EARLY_FLIP_INDICES_60
            ends = list(flips) + [n_trials]
            starts = [0] + list(flips)
            return tuple(e - s for s, e in zip(starts, ends))
        return (n_trials,)
    if n_trials == 80:
        return BLOCK_LENGTHS_80
    if n_trials == 60:
        return BLOCK_LENGTHS_60
    return (n_trials,)


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def reward_schedule_over_trials(
    n_trials: int,
    mu: Sequence[float] = (0.8, 0.2),
    schedule: ScheduleName = "canonical",
) -> np.ndarray:
    """Return (n_trials, 2) scheduled P(reward|arm) after applying reversals."""
    mu_t = np.asarray(mu, dtype=float).copy()
    if mu_t.shape != (2,):
        raise ValueError("mu must have shape (2,)")
    out = np.zeros((n_trials, 2), dtype=float)
    flips = set(reversal_indices_0based(n_trials, schedule=schedule))
    for t in range(n_trials):
        if t in flips:
            mu_t = mu_t[::-1].copy()
        out[t] = mu_t
    return out


def sample_reward(mu_arm: float, rng: np.random.Generator) -> int:
    return int(rng.random() < mu_arm)


def validate_binary_series(x: Iterable, name: str) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.isin(arr, [0.0, 1.0])):
        raise ValueError(f"{name} must be binary {{0,1}}")
    return arr.astype(int)
