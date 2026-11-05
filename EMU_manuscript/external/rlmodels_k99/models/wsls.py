"""Win-Stay Lose-Shift model (corrected pyEM port)."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward


def _wsls_probs(prev_a: int | None, prev_r: int | None, epsilon: float) -> np.ndarray:
    if prev_a is None:
        return np.array([0.5, 0.5], dtype=float)
    p = np.empty(2, dtype=float)
    if prev_r == 1:
        p[prev_a] = 1.0 - epsilon / 2.0
        p[1 - prev_a] = epsilon / 2.0
    else:
        p[prev_a] = epsilon / 2.0
        p[1 - prev_a] = 1.0 - epsilon / 2.0
    return p


def wsls_sim(params: np.ndarray, ntrials: int = 80, seed: int | None = None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    eps = params[:, 0]
    if not ((eps >= 0.0) & (eps <= 1.0)).all():
        raise ValueError("epsilon out of bounds")

    rng = make_rng(seed)
    flips = set(reversal_indices_0based(ntrials, schedule=schedule))
    choices = np.zeros((nsubj, ntrials), dtype=int)
    rewards = np.zeros((nsubj, ntrials), dtype=int)
    choice_prob = np.zeros((nsubj, ntrials, 2), dtype=float)
    nll = np.zeros((nsubj, ntrials), dtype=float)

    for s in range(nsubj):
        mu_t = np.asarray(mu, dtype=float).copy()
        prev_a = prev_r = None
        for t in range(ntrials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p = _wsls_probs(prev_a, prev_r, float(eps[s]))
            a = int(rng.choice([0, 1], p=p))
            r = sample_reward(float(mu_t[a]), rng)
            choices[s, t] = a
            rewards[s, t] = r
            choice_prob[s, t] = p
            nll[s, t] = -np.log(p[a] + 1e-12)
            prev_a, prev_r = a, r

    return {
        "params": params.copy(),
        "choices": choices,
        "rewards": rewards,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def wsls_fit(params, choices, rewards, *, prior=None, output="npl"):
    epsilon = float(norm2alpha(params[0]))
    if not (0.0 <= epsilon <= 1.0):
        return 1e7

    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    if len(a) != len(r):
        return 1e7

    nll = 0.0
    prev_a = prev_r = None
    probs = np.zeros((len(a), 2), dtype=float)
    for t in range(len(a)):
        p = _wsls_probs(prev_a, prev_r, epsilon)
        probs[t] = p
        nll += -np.log(p[a[t]] + 1e-12)
        prev_a, prev_r = int(a[t]), int(r[t])

    if output == "all":
        return {
            "params": [epsilon],
            "choices": a,
            "rewards": r,
            "choice_prob": probs,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


wsls_desc = (
    "Win-Stay Lose-Shift with noise epsilon. "
    "After wins, stay with probability 1-epsilon/2; after losses, stay with epsilon/2. "
    "Free parameter: epsilon."
)
wsls_id = "wsls"
wsls_spec = {"rl": {"wsls": ["epsilon"]}}
wsls_model = ModelSpec(
    id=wsls_id,
    spec=wsls_spec,
    desc=wsls_desc,
    params=None,
    sim=wsls_sim,
    fit=wsls_fit,
)
