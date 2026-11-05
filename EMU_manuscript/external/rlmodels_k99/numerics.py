"""Stable numerical helpers shared across models."""

from __future__ import annotations

import numpy as np


def softplus(x: float | np.ndarray) -> float | np.ndarray:
    """Maps unbounded values to (0, inf) with softplus."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softmax(values: np.ndarray, beta: float) -> np.ndarray:
    """Temperature-scaled softmax with max-subtraction for stability."""
    z = beta * np.asarray(values, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def choice_nll(prob_chosen: float, eps: float = 1e-12) -> float:
    return float(-np.log(np.clip(prob_chosen, eps, 1.0)))


def check_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
