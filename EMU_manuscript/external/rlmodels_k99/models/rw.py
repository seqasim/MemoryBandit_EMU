"""Standard Rescorla-Wagner model (corrected pyEM port)."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.numerics import softmax
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward


def rw_sim(params: np.ndarray, ntrials: int = 80, seed: int | None = None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, alphas = params[:, 0], params[:, 1]
    if not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")
    if not ((alphas >= 0.0) & (alphas <= 1.0)).all():
        raise ValueError("alpha out of bounds")

    rng = make_rng(seed)
    flips = set(reversal_indices_0based(ntrials, schedule=schedule))
    choices = np.zeros((nsubj, ntrials), dtype=int)
    rewards = np.zeros((nsubj, ntrials), dtype=int)
    EV = np.zeros((nsubj, ntrials + 1, 2), dtype=float)
    PE = np.zeros((nsubj, ntrials), dtype=float)
    choice_prob = np.zeros((nsubj, ntrials, 2), dtype=float)
    nll = np.zeros((nsubj, ntrials), dtype=float)

    for s in range(nsubj):
        Q = np.array([0.5, 0.5], dtype=float)
        EV[s, 0] = Q
        mu_t = np.asarray(mu, dtype=float).copy()
        for t in range(ntrials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p = softmax(Q, float(betas[s]))
            a = int(rng.choice([0, 1], p=p))
            r = sample_reward(float(mu_t[a]), rng)
            pe = r - Q[a]
            Q = Q.copy()
            Q[a] = Q[a] + float(alphas[s]) * pe
            choices[s, t] = a
            rewards[s, t] = r
            PE[s, t] = pe
            EV[s, t + 1] = Q
            choice_prob[s, t] = p
            nll[s, t] = -np.log(p[a] + 1e-12)

    return {
        "params": params.copy(),
        "choices": choices,
        "rewards": rewards,
        "EV": EV,
        "PE": PE,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def rw_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha <= 1.0):
        return 1e7

    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    T = len(a)
    Q = np.array([0.5, 0.5], dtype=float)
    EV = np.zeros((T + 1, 2), dtype=float)
    PE = np.zeros(T, dtype=float)
    EV[0] = Q
    nll = 0.0
    for t in range(T):
        p = softmax(Q, beta)
        nll += -np.log(p[a[t]] + 1e-12)
        pe = r[t] - Q[a[t]]
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q

    if output == "all":
        return {"params": [beta, alpha], "choices": a, "rewards": r, "EV": EV, "PE": PE, "nll": nll}
    return calc_fval(nll, params, prior=prior, output=output)


rw_desc = (
    "Rescorla-Wagner with one learning rate and softmax choice. "
    "Free parameters: beta (inverse temperature), alpha (learning rate)."
)
rw_id = "rw"
rw_spec = {"rl": {"softmax": ["beta"], "rw": ["alpha"]}}
rw_model = ModelSpec(id=rw_id, spec=rw_spec, desc=rw_desc, params=None, sim=rw_sim, fit=rw_fit)
