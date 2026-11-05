"""Reduced VKF candidates with fixed initial volatility and/or lambda."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.models.vkf import _vkf_step
from rlmodels_k99.numerics import softplus, softmax
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward

# Preregistered fixed constants for reduced VKF candidates.
FIXED_LAMBDA = 0.2
FIXED_V0 = 1.0
# Decoupled initial uncertainty (legacy VKF sets W0=omega).
FIXED_W0 = 1.0


def _vkf_loop(params_cols, ntrials, seed, mu, schedule, *, free_lambda: bool, w0s=None):
    """Shared sim loop. params_cols provides natural-scale arrays.

    When ``w0s`` is None, initial uncertainty equals observation noise (legacy).
    """
    betas = params_cols["beta"]
    lams = params_cols["lambda"]
    v0s = params_cols["v0"]
    omegas = params_cols["omega"]
    nsubj = len(betas)
    if w0s is None:
        w0s = omegas
    if not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")
    if free_lambda and not ((lams >= 0.0) & (lams <= 1.0)).all():
        raise ValueError("lambda out of bounds")
    if not ((v0s > 0.0) & (omegas > 0.0)).all():
        raise ValueError("v0/omega must be positive")
    if not (np.asarray(w0s) > 0.0).all():
        raise ValueError("w0 must be positive")

    rng = make_rng(seed)
    flips = set(reversal_indices_0based(ntrials, schedule=schedule))
    choices = np.zeros((nsubj, ntrials), dtype=int)
    rewards = np.zeros((nsubj, ntrials), dtype=int)
    M_hist = np.zeros((nsubj, ntrials + 1, 2), dtype=float)
    choice_prob = np.zeros((nsubj, ntrials, 2), dtype=float)
    nll = np.zeros((nsubj, ntrials), dtype=float)
    packed = np.column_stack([betas, lams, v0s, omegas])

    for s in range(nsubj):
        m = np.zeros(2, dtype=float)
        w = float(w0s[s]) * np.ones(2, dtype=float)
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
        "params": packed,
        "choices": choices,
        "rewards": rewards,
        "M": M_hist,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def vkf_omega_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    return _vkf_loop(
        {
            "beta": params[:, 0],
            "lambda": np.full(nsubj, FIXED_LAMBDA),
            "v0": np.full(nsubj, FIXED_V0),
            "omega": params[:, 1],
        },
        ntrials,
        seed,
        mu,
        schedule,
        free_lambda=False,
    )


def vkf_omega_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    omega = float(softplus(params[1]))
    lam = FIXED_LAMBDA
    v0 = FIXED_V0
    if not (1e-5 <= beta <= 20.0) or not (omega > 0.0):
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
            "params": [beta, omega],
            "choices": a,
            "rewards": r,
            "M": M_hist,
            "nll": nll,
            "fixed_lambda": lam,
            "fixed_v0": v0,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def vkf_fixed_v0_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    return _vkf_loop(
        {
            "beta": params[:, 0],
            "lambda": params[:, 1],
            "v0": np.full(nsubj, FIXED_V0),
            "omega": params[:, 2],
        },
        ntrials,
        seed,
        mu,
        schedule,
        free_lambda=True,
    )


def vkf_fixed_v0_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    lam = float(norm2alpha(params[1]))
    omega = float(softplus(params[2]))
    v0 = FIXED_V0
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= lam <= 1.0) or not (omega > 0.0):
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
            "params": [beta, lam, omega],
            "choices": a,
            "rewards": r,
            "M": M_hist,
            "nll": nll,
            "fixed_v0": v0,
        }
    return calc_fval(nll, params, prior=prior, output=output)


vkf_omega_model = ModelSpec(
    id="vkf_omega",
    spec={"rl": {"softmax": ["beta"], "vkf_omega": ["omega"]}},
    desc=(
        f"Reduced VKF with free beta/omega; lambda={FIXED_LAMBDA} and v0={FIXED_V0} fixed "
        f"(legacy W0=omega)."
    ),
    params=None,
    sim=vkf_omega_sim,
    fit=vkf_omega_fit,
)

vkf_fixed_v0_model = ModelSpec(
    id="vkf_fixed_v0",
    spec={"rl": {"softmax": ["beta"], "vkf_fixed_v0": ["lambda", "omega"]}},
    desc=f"VKF with free beta/lambda/omega; v0={FIXED_V0} fixed.",
    params=None,
    sim=vkf_fixed_v0_sim,
    fit=vkf_fixed_v0_fit,
)


def vkf_w0_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    """Reduced VKF with free beta/omega; W0 fixed independently of omega."""
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    return _vkf_loop(
        {
            "beta": params[:, 0],
            "lambda": np.full(nsubj, FIXED_LAMBDA),
            "v0": np.full(nsubj, FIXED_V0),
            "omega": params[:, 1],
        },
        ntrials,
        seed,
        mu,
        schedule,
        free_lambda=False,
        w0s=np.full(nsubj, FIXED_W0),
    )


def vkf_w0_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    omega = float(softplus(params[1]))
    lam = FIXED_LAMBDA
    v0 = FIXED_V0
    w0 = FIXED_W0
    if not (1e-5 <= beta <= 20.0) or not (omega > 0.0):
        return 1e7

    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    T = len(a)
    m = np.zeros(2, dtype=float)
    w = w0 * np.ones(2, dtype=float)
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
            "params": [beta, omega],
            "choices": a,
            "rewards": r,
            "M": M_hist,
            "nll": nll,
            "fixed_lambda": lam,
            "fixed_v0": v0,
            "fixed_w0": w0,
        }
    return calc_fval(nll, params, prior=prior, output=output)


vkf_w0_model = ModelSpec(
    id="vkf_w0",
    spec={"rl": {"softmax": ["beta"], "vkf_w0": ["omega"]}},
    desc=(
        f"Reduced VKF with free beta/omega; lambda={FIXED_LAMBDA}, v0={FIXED_V0}, "
        f"w0={FIXED_W0} fixed (W0 decoupled from omega)."
    ),
    params=None,
    sim=vkf_w0_sim,
    fit=vkf_w0_fit,
)
