"""Shared helpers for continuous-session bandit ModelSpecs."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rlmodels_k99.task import (
    DEFAULT_MU,
    make_rng,
    reversal_indices_0based,
    sample_reward,
)


def as_1d_int(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D trial series, got shape {arr.shape}")
    return arr.astype(int)


def simulate_bandit_loop(
    n_subjects: int,
    n_trials: int,
    params: np.ndarray,
    choose_and_update,
    seed: int | None = None,
    mu: Sequence[float] = (0.8, 0.2),
    schedule: str = "canonical",
) -> dict:
    """Generic simulate skeleton: flip schedule, sample choice/reward, update."""
    rng = make_rng(seed)
    flips = set(reversal_indices_0based(n_trials, schedule=schedule))
    choices = np.zeros((n_subjects, n_trials), dtype=int)
    rewards = np.zeros((n_subjects, n_trials), dtype=int)
    choice_prob = np.zeros((n_subjects, n_trials, 2), dtype=float)
    nll = np.zeros((n_subjects, n_trials), dtype=float)

    for s in range(n_subjects):
        mu_t = np.asarray(mu, dtype=float).copy()
        state = None
        for t in range(n_trials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p, state = choose_and_update(state, params[s], rng, mode="choose")
            a = int(rng.choice([0, 1], p=p))
            r = sample_reward(float(mu_t[a]), rng)
            state = choose_and_update(state, params[s], rng, mode="update", action=a, reward=r, probs=p)
            choices[s, t] = a
            rewards[s, t] = r
            choice_prob[s, t] = p
            nll[s, t] = -np.log(p[a] + 1e-12)

    return {
        "params": np.asarray(params, dtype=float),
        "choices": choices,
        "rewards": rewards,
        "choice_prob": choice_prob,
        "nll": nll,
        "n_trials": n_trials,
    }
