"""Bayesian change-point / reversal filter (corrected pyEM port)."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward

P_NOISY = 1e-5


def _likelihoods(action: int, reward: int, p_reward: float, p_noisy: float = P_NOISY):
    # LC: likelihood of outcome given "right (arm 1) is correct"
    lc = (
        reward * (action * p_reward + (1 - action) * p_noisy)
        + (1 - reward) * (action * (1 - p_reward) + (1 - action) * (1 - p_noisy))
    )
    li = (
        reward * (action * p_noisy + (1 - action) * p_reward)
        + (1 - reward) * (action * (1 - p_noisy) + (1 - action) * (1 - p_reward))
    )
    return float(lc), float(li)


def _update_belief(belief: float, action: int, reward: int, p_switch: float, p_reward: float) -> float:
    lc, li = _likelihoods(action, reward, p_reward)
    denom = lc * belief + li * (1.0 - belief)
    if denom <= 0:
        post = 0.5
    else:
        post = (lc * belief) / denom
    return (1.0 - p_switch) * post + p_switch * (1.0 - post)


def _choice_p_right(belief: float, beta: float) -> float:
    return 1.0 / (1.0 + np.exp(beta * (0.5 - belief)))


def bayes_sim(params: np.ndarray, ntrials: int = 80, seed: int | None = None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    betas, p_switches, p_rewards = params[:, 0], params[:, 1], params[:, 2]
    if not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")
    if not ((p_switches >= 0.0) & (p_switches <= 1.0)).all():
        raise ValueError("p_switch out of bounds")
    if not ((p_rewards >= 0.0) & (p_rewards <= 1.0)).all():
        raise ValueError("p_reward out of bounds")

    rng = make_rng(seed)
    flips = set(reversal_indices_0based(ntrials, schedule=schedule))
    choices = np.zeros((nsubj, ntrials), dtype=int)
    rewards = np.zeros((nsubj, ntrials), dtype=int)
    beliefs = np.zeros((nsubj, ntrials + 1), dtype=float)
    choice_prob = np.zeros((nsubj, ntrials, 2), dtype=float)
    nll = np.zeros((nsubj, ntrials), dtype=float)

    for s in range(nsubj):
        belief = 0.5
        beliefs[s, 0] = belief
        mu_t = np.asarray(mu, dtype=float).copy()
        for t in range(ntrials):
            if t in flips:
                mu_t = mu_t[::-1].copy()
            p_right = _choice_p_right(belief, float(betas[s]))
            p = np.array([1.0 - p_right, p_right], dtype=float)
            a = int(rng.choice([0, 1], p=p))
            r = sample_reward(float(mu_t[a]), rng)
            belief = _update_belief(belief, a, r, float(p_switches[s]), float(p_rewards[s]))
            choices[s, t] = a
            rewards[s, t] = r
            beliefs[s, t + 1] = belief
            choice_prob[s, t] = p
            nll[s, t] = -np.log(p[a] + 1e-12)

    return {
        "params": params.copy(),
        "choices": choices,
        "rewards": rewards,
        "belief": beliefs,
        "choice_prob": choice_prob,
        "nll": nll,
    }


def bayes_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    p_switch = float(norm2alpha(params[1]))
    p_reward = float(norm2alpha(params[2]))
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    if not (0.0 <= p_switch <= 1.0) or not (0.0 <= p_reward <= 1.0):
        return 1e7

    a = as_1d_int(choices)
    r = as_1d_int(rewards)
    T = len(a)
    belief = 0.5
    beliefs = np.zeros(T + 1, dtype=float)
    beliefs[0] = belief
    nll = 0.0
    for t in range(T):
        p_right = _choice_p_right(belief, beta)
        p = np.array([1.0 - p_right, p_right], dtype=float)
        nll += -np.log(p[a[t]] + 1e-12)
        belief = _update_belief(belief, int(a[t]), int(r[t]), p_switch, p_reward)
        beliefs[t + 1] = belief

    if output == "all":
        return {
            "params": [beta, p_switch, p_reward],
            "choices": a,
            "rewards": r,
            "belief": beliefs,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


bayes_desc = (
    "Bayesian belief-updating model of a latent correct arm with switch probability. "
    "Free parameters: beta, p_switch, p_reward (p_noisy fixed at 1e-5)."
)
bayes_id = "bayes"
bayes_spec = {"rl": {"bayes_filter": ["beta", "p_switch", "p_reward"]}}
bayes_model = ModelSpec(
    id=bayes_id, spec=bayes_spec, desc=bayes_desc, params=None, sim=bayes_sim, fit=bayes_fit
)
