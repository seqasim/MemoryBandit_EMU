"""Volatile Kalman Filter (corrected pyEM port)."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.numerics import softplus, softmax
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _vkf_step(m, w, v, o, lam, omega):
    mpre = m.copy()
    wpre = w.copy()
    delta_m = o - _sigmoid(m)
    k = (w + v) / (w + v + omega)
    alpha = np.sqrt(w + v)
    m = m + alpha * delta_m
    w = (1.0 - k) * (w + v)
    wcov = (1.0 - k) * wpre
    delta_v = (m - mpre) ** 2 + w + wpre - 2.0 * wcov - v
    v = v + lam * delta_v
    return m, w, v, delta_m, alpha


def vkf_sim(params: np.ndarray, ntrials: int = 80, seed: int | None = None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, lams, v0s, omegas = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    if not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")
    if not ((lams >= 0.0) & (lams <= 1.0)).all():
        raise ValueError("lambda out of bounds")
    if not ((v0s > 0.0) & (omegas > 0.0)).all():
        raise ValueError("v0/omega must be positive")

    rng = make_rng(seed)
    flips = set(reversal_indices_0based(ntrials, schedule=schedule))
    choices = np.zeros((nsubj, ntrials), dtype=int)
    rewards = np.zeros((nsubj, ntrials), dtype=int)
    M_hist = np.zeros((nsubj, ntrials + 1, 2), dtype=float)
    choice_prob = np.zeros((nsubj, ntrials, 2), dtype=float)
    nll = np.zeros((nsubj, ntrials), dtype=float)

    for s in range(nsubj):
        m = np.zeros(2, dtype=float)
        w = float(omegas[s]) * np.ones(2, dtype=float)
        v = float(v0s[s]) * np.ones(2, dtype=float)
        M_hist[s, 0] = m
        mu_t = np.asarray(mu, dtype=float).copy()
        for t in range(ntrials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p = softmax(m, float(betas[s]))
            a = int(rng.choice([0, 1], p=p))
            r = sample_reward(float(mu_t[a]), rng)
            o = np.zeros(2, dtype=float)
            o[a] = r
            o[1 - a] = 1 - r
            m, w, v, _, _ = _vkf_step(m, w, v, o, float(lams[s]), float(omegas[s]))
            choices[s, t] = a
            rewards[s, t] = r
            M_hist[s, t + 1] = m
            choice_prob[s, t] = p
            nll[s, t] = -np.log(p[a] + 1e-12)

    return {
        "params": params.copy(),
        "choices": choices,
        "rewards": rewards,
        "M": M_hist,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def vkf_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    lam = float(norm2alpha(params[1]))
    v0 = float(softplus(params[2]))
    omega = float(softplus(params[3]))
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= lam <= 1.0) or not (v0 > 0.0 and omega > 0.0):
        return 1e7

    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    T = len(a)
    m = np.zeros(2, dtype=float)
    w = omega * np.ones(2, dtype=float)
    v = v0 * np.ones(2, dtype=float)
    M_hist = np.zeros((T + 1, 2), dtype=float)
    M_hist[0] = m
    nll = 0.0
    for t in range(T):
        p = softmax(m, beta)
        nll += -np.log(p[a[t]] + 1e-12)
        o = np.zeros(2, dtype=float)
        o[a[t]] = r[t]
        o[1 - a[t]] = 1 - r[t]
        m, w, v, _, _ = _vkf_step(m, w, v, o, lam, omega)
        M_hist[t + 1] = m

    if output == "all":
        return {
            "params": [beta, lam, v0, omega],
            "choices": a,
            "rewards": r,
            "M": M_hist,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


vkf_desc = (
    "Volatile Kalman Filter over both arms with counterfactual outcomes. "
    "Free parameters: beta, lambda (volatility learning rate), v0, omega (noise)."
)
vkf_id = "vkf"
vkf_spec = {"rl": {"softmax": ["beta"], "vkf": ["lambda", "v0", "omega"]}}
vkf_model = ModelSpec(id=vkf_id, spec=vkf_spec, desc=vkf_desc, params=None, sim=vkf_sim, fit=vkf_fit)
