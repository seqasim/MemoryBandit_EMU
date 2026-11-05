"""RW family variants: signed learning rates, side bias, and stickiness."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import alpha2norm, calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.numerics import softmax
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward


def _choice_probs_from_values(Q: np.ndarray, beta: float, *, bias: float = 0.0, stick: float = 0.0, prev: int | None = None):
    """Softmax on Q with optional right-arm logit bias and previous-choice stickiness."""
    logits = beta * np.asarray(Q, dtype=float)
    logits = logits.copy()
    logits[1] = logits[1] + bias
    if prev is not None:
        # Prefer previous choice: +stick on prev arm, equivalently +/- stick on right arm.
        logits[prev] = logits[prev] + stick
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def rw_signed_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, apos, aneg = params[:, 0], params[:, 1], params[:, 2]
    if not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")
    if not ((apos >= 0.0) & (apos <= 1.0)).all() or not ((aneg >= 0.0) & (aneg <= 1.0)).all():
        raise ValueError("alpha_pos/alpha_neg out of bounds")

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
            alpha = float(apos[s]) if pe >= 0 else float(aneg[s])
            Q = Q.copy()
            Q[a] = Q[a] + alpha * pe
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


def rw_signed_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    alpha_pos = float(norm2alpha(params[1]))
    alpha_neg = float(norm2alpha(params[2]))
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha_pos <= 1.0) or not (0.0 <= alpha_neg <= 1.0):
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
        alpha = alpha_pos if pe >= 0 else alpha_neg
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q
    if output == "all":
        return {
            "params": [beta, alpha_pos, alpha_neg],
            "choices": a,
            "rewards": r,
            "EV": EV,
            "PE": PE,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def rw_side_bias_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, alphas, biases = params[:, 0], params[:, 1], params[:, 2]
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
            p = _choice_probs_from_values(Q, float(betas[s]), bias=float(biases[s]))
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


def rw_side_bias_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))
    bias = float(params[2])  # identity transform
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha <= 1.0) or not np.isfinite(bias):
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
        p = _choice_probs_from_values(Q, beta, bias=bias)
        nll += -np.log(p[a[t]] + 1e-12)
        pe = r[t] - Q[a[t]]
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q
    if output == "all":
        return {"params": [beta, alpha, bias], "choices": a, "rewards": r, "EV": EV, "PE": PE, "nll": nll}
    return calc_fval(nll, params, prior=prior, output=output)


def rw_sticky_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, alphas, sticks = params[:, 0], params[:, 1], params[:, 2]
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
        prev = None
        for t in range(ntrials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p = _choice_probs_from_values(Q, float(betas[s]), stick=float(sticks[s]), prev=prev)
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
            prev = a
    return {
        "params": params.copy(),
        "choices": choices,
        "rewards": rewards,
        "EV": EV,
        "PE": PE,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def rw_sticky_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))
    stickiness = float(params[2])
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha <= 1.0) or not np.isfinite(stickiness):
        return 1e7
    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    T = len(a)
    Q = np.array([0.5, 0.5], dtype=float)
    EV = np.zeros((T + 1, 2), dtype=float)
    PE = np.zeros(T, dtype=float)
    EV[0] = Q
    nll = 0.0
    prev = None
    for t in range(T):
        p = _choice_probs_from_values(Q, beta, stick=stickiness, prev=prev)
        nll += -np.log(p[a[t]] + 1e-12)
        pe = r[t] - Q[a[t]]
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q
        prev = int(a[t])
    if output == "all":
        return {
            "params": [beta, alpha, stickiness],
            "choices": a,
            "rewards": r,
            "EV": EV,
            "PE": PE,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def _signed_rates_from_mean_asym(alpha_mean_latent: float, alpha_asym_latent: float) -> tuple[float, float]:
    """Map mean/asymmetry latent coords to (alpha_pos, alpha_neg).

    Uses the same logistic transform as ``rw_signed`` so the two-rate likelihood
    is unchanged for equivalent natural rates:
    α₊ = σ(m + d), α₋ = σ(m - d) with m=alpha_mean_latent, d=alpha_asym_latent.
    """
    alpha_pos = float(norm2alpha(alpha_mean_latent + alpha_asym_latent))
    alpha_neg = float(norm2alpha(alpha_mean_latent - alpha_asym_latent))
    return alpha_pos, alpha_neg


def rw_signed_natural_from_asym(alpha_mean_latent: float, alpha_asym_latent: float) -> tuple[float, float]:
    """Public helper for parity tests / documentation."""
    return _signed_rates_from_mean_asym(alpha_mean_latent, alpha_asym_latent)


def rw_signed_asym_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    """Signed RW with mean + asymmetry parameterization (identity natural m,d).

    Free natural parameters are ``[beta, alpha_mean, alpha_asym]`` where
    ``alpha_mean``/``alpha_asym`` are the latent mid/diff coordinates m,d.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    betas = params[:, 0]
    apos = np.empty(len(betas))
    aneg = np.empty(len(betas))
    for s in range(len(betas)):
        apos[s], aneg[s] = _signed_rates_from_mean_asym(float(params[s, 1]), float(params[s, 2]))
    natural = np.column_stack([betas, apos, aneg])
    out = rw_signed_sim(natural, ntrials=ntrials, seed=seed, mu=mu, schedule=schedule, **kwargs)
    # Keep free-parameter matrix for recovery bookkeeping.
    out = dict(out)
    out["params"] = params.copy()
    out["alpha_pos"] = apos
    out["alpha_neg"] = aneg
    return out


def rw_signed_asym_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    # alpha_mean / alpha_asym use identity transforms (latent mid/diff).
    m = float(params[1])
    d = float(params[2])
    alpha_pos, alpha_neg = _signed_rates_from_mean_asym(m, d)
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha_pos <= 1.0) or not (0.0 <= alpha_neg <= 1.0):
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
        alpha = alpha_pos if pe >= 0 else alpha_neg
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q
    if output == "all":
        return {
            "params": [beta, m, d],
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "choices": a,
            "rewards": r,
            "EV": EV,
            "PE": PE,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


rw_signed_model = ModelSpec(
    id="rw_signed",
    spec={"rl": {"softmax": ["beta"], "rw_signed": ["alpha_pos", "alpha_neg"]}},
    desc="RW with separate learning rates for positive and negative prediction errors.",
    params=None,
    sim=rw_signed_sim,
    fit=rw_signed_fit,
)

rw_signed_asym_model = ModelSpec(
    id="rw_signed_asym",
    spec={"rl": {"softmax": ["beta"], "rw_signed_asym": ["alpha_mean", "alpha_asym"]}},
    desc=(
        "Signed RW reparameterized as latent mean learning rate + asymmetry "
        "(α₊=σ(m+d), α₋=σ(m-d)); same two-rate likelihood as rw_signed."
    ),
    params=None,
    sim=rw_signed_asym_sim,
    fit=rw_signed_asym_fit,
)


# Predeclared fixed mean learning rate for the two-parameter asymmetry model.
# Matches the hierarchical RW prior mean (alpha ≈ 0.35).
FIXED_ALPHA_MEAN = 0.35
FIXED_ALPHA_MEAN_LATENT = float(alpha2norm(FIXED_ALPHA_MEAN))


def rw_asym_rates(d: float, *, c: float = FIXED_ALPHA_MEAN_LATENT) -> tuple[float, float]:
    """α₊=σ(c+d), α₋=σ(c-d) with fixed mean latent c."""
    return _signed_rates_from_mean_asym(c, float(d))


def rw_asym_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    """Two-parameter signed RW: free beta and asymmetry d; mean rate fixed."""
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    betas = params[:, 0]
    ds = params[:, 1]
    apos = np.empty(len(betas))
    aneg = np.empty(len(betas))
    for s in range(len(betas)):
        apos[s], aneg[s] = rw_asym_rates(float(ds[s]))
    natural = np.column_stack([betas, apos, aneg])
    out = rw_signed_sim(natural, ntrials=ntrials, seed=seed, mu=mu, schedule=schedule, **kwargs)
    out = dict(out)
    out["params"] = params.copy()
    out["alpha_pos"] = apos
    out["alpha_neg"] = aneg
    out["fixed_alpha_mean"] = FIXED_ALPHA_MEAN
    return out


def rw_asym_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    d = float(params[1])  # identity transform
    alpha_pos, alpha_neg = rw_asym_rates(d)
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha_pos <= 1.0) or not (0.0 <= alpha_neg <= 1.0):
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
        alpha = alpha_pos if pe >= 0 else alpha_neg
        Q = Q.copy()
        Q[a[t]] = Q[a[t]] + alpha * pe
        PE[t] = pe
        EV[t + 1] = Q
    if output == "all":
        return {
            "params": [beta, d],
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "fixed_alpha_mean": FIXED_ALPHA_MEAN,
            "choices": a,
            "rewards": r,
            "EV": EV,
            "PE": PE,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


rw_asym_model = ModelSpec(
    id="rw_asym",
    spec={"rl": {"softmax": ["beta"], "rw_asym": ["d"]}},
    desc=(
        f"Two-parameter valence-asymmetric RW: free beta and asymmetry d; "
        f"mean learning rate fixed at alpha={FIXED_ALPHA_MEAN} "
        f"(α₊=σ(c+d), α₋=σ(c-d); d=0 recovers symmetric RW)."
    ),
    params=None,
    sim=rw_asym_sim,
    fit=rw_asym_fit,
)

rw_side_bias_model = ModelSpec(
    id="rw_side_bias",
    spec={"rl": {"softmax": ["beta"], "rw": ["alpha"], "side_bias": ["bias"]}},
    desc="Standard RW with an additive right-arm logit bias.",
    params=None,
    sim=rw_side_bias_sim,
    fit=rw_side_bias_fit,
)

rw_sticky_model = ModelSpec(
    id="rw_sticky",
    spec={"rl": {"softmax": ["beta"], "rw": ["alpha"], "stickiness": ["stickiness"]}},
    desc="Standard RW with previous-choice stickiness/perseveration.",
    params=None,
    sim=rw_sticky_sim,
    fit=rw_sticky_fit,
)
