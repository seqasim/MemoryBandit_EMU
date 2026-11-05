"""Bayes model candidates for recovery / identifiability benchmarks.

Legacy ``bayes`` (three-parameter, fixed p_noisy=1e-5) remains the canonical
entry in the registry. These variants are opt-in until promotion criteria pass.
"""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2alpha, norm2beta

from rlmodels_k99.models._common import as_1d_int
from rlmodels_k99.models.bayes import P_NOISY, _choice_p_right, _likelihoods, _update_belief
from rlmodels_k99.task import make_rng, reversal_indices_0based, sample_reward


def _likelihoods_symmetric(action: int, reward: int, p_reward: float):
    """Symmetric emissions: incorrect-arm reward probability = 1 - p_reward."""
    p_noisy = 1.0 - p_reward
    return _likelihoods(action, reward, p_reward, p_noisy=p_noisy)


def _update_belief_generic(
    belief: float,
    action: int,
    reward: int,
    p_switch: float,
    p_reward: float,
    lik_fn,
) -> float:
    lc, li = lik_fn(action, reward, p_reward)
    denom = lc * belief + li * (1.0 - belief)
    post = 0.5 if denom <= 0 else (lc * belief) / denom
    return (1.0 - p_switch) * post + p_switch * (1.0 - post)


def _bayes_loop(
    params: np.ndarray,
    ntrials: int,
    seed,
    mu,
    schedule,
    *,
    constrain: bool,
    lik_fn,
    n_params: int,
):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubj = params.shape[0]
    if n_params == 3:
        betas, p_switches, p_rewards = params[:, 0], params[:, 1], params[:, 2]
    else:
        betas = np.ones(nsubj)
        p_switches, p_rewards = params[:, 0], params[:, 1]

    if constrain:
        if not ((p_switches >= 0.0) & (p_switches < 0.5)).all():
            raise ValueError("constrained p_switch must be in [0, 0.5)")
        if not ((p_rewards > 0.5) & (p_rewards <= 1.0)).all():
            raise ValueError("constrained p_reward must be in (0.5, 1]")
    else:
        if not ((p_switches >= 0.0) & (p_switches <= 1.0)).all():
            raise ValueError("p_switch out of bounds")
        if not ((p_rewards >= 0.0) & (p_rewards <= 1.0)).all():
            raise ValueError("p_reward out of bounds")
    if n_params == 3 and not ((betas >= 1e-5) & (betas <= 20.0)).all():
        raise ValueError("beta out of bounds")

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
            belief = _update_belief_generic(
                belief, a, r, float(p_switches[s]), float(p_rewards[s]), lik_fn
            )
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


def _bayes_fit_generic(
    params,
    choices,
    rewards,
    *,
    prior=None,
    output="npl",
    constrain: bool,
    lik_fn,
    n_params: int,
):
    if n_params == 3:
        beta = float(norm2beta(params[0]))
        p_switch = float(norm2alpha(params[1]))
        p_reward = float(norm2alpha(params[2]))
        natural = [beta, p_switch, p_reward]
        if not (1e-5 <= beta <= 20.0):
            return 1e7
    else:
        beta = 1.0
        p_switch = float(norm2alpha(params[0]))
        p_reward = float(norm2alpha(params[1]))
        natural = [p_switch, p_reward]

    if constrain:
        if not (0.0 <= p_switch < 0.5) or not (0.5 < p_reward <= 1.0):
            return 1e7
    else:
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
        belief = _update_belief_generic(belief, int(a[t]), int(r[t]), p_switch, p_reward, lik_fn)
        beliefs[t + 1] = belief

    if output == "all":
        return {
            "params": natural,
            "choices": a,
            "rewards": r,
            "belief": beliefs,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def bayes_constrained_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    return _bayes_loop(
        params,
        ntrials,
        seed,
        mu,
        schedule,
        constrain=True,
        lik_fn=lambda a, r, pr: _likelihoods(a, r, pr, p_noisy=P_NOISY),
        n_params=3,
    )


def bayes_constrained_fit(params, choices, rewards, *, prior=None, output="npl"):
    return _bayes_fit_generic(
        params,
        choices,
        rewards,
        prior=prior,
        output=output,
        constrain=True,
        lik_fn=lambda a, r, pr: _likelihoods(a, r, pr, p_noisy=P_NOISY),
        n_params=3,
    )


def bayes_symmetric_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    return _bayes_loop(
        params,
        ntrials,
        seed,
        mu,
        schedule,
        constrain=False,
        lik_fn=_likelihoods_symmetric,
        n_params=3,
    )


def bayes_symmetric_fit(params, choices, rewards, *, prior=None, output="npl"):
    return _bayes_fit_generic(
        params,
        choices,
        rewards,
        prior=prior,
        output=output,
        constrain=False,
        lik_fn=_likelihoods_symmetric,
        n_params=3,
    )


def bayes_two_param_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    return _bayes_loop(
        params,
        ntrials,
        seed,
        mu,
        schedule,
        constrain=False,
        lik_fn=lambda a, r, pr: _likelihoods(a, r, pr, p_noisy=P_NOISY),
        n_params=2,
    )


def bayes_two_param_fit(params, choices, rewards, *, prior=None, output="npl"):
    return _bayes_fit_generic(
        params,
        choices,
        rewards,
        prior=prior,
        output=output,
        constrain=False,
        lik_fn=lambda a, r, pr: _likelihoods(a, r, pr, p_noisy=P_NOISY),
        n_params=2,
    )


bayes_constrained_model = ModelSpec(
    id="bayes_constrained",
    spec={"rl": {"bayes_filter_constrained": ["beta", "p_switch", "p_reward"]}},
    desc=(
        "Bayes filter with semantic constraints p_switch<0.5 and p_reward>0.5 "
        "(fixed p_noisy=1e-5)."
    ),
    params=None,
    sim=bayes_constrained_sim,
    fit=bayes_constrained_fit,
)

bayes_symmetric_model = ModelSpec(
    id="bayes_symmetric",
    spec={"rl": {"bayes_filter_symmetric": ["beta", "p_switch", "p_reward"]}},
    desc="Bayes filter with symmetric emissions p_noisy = 1 - p_reward.",
    params=None,
    sim=bayes_symmetric_sim,
    fit=bayes_symmetric_fit,
)

bayes_two_param_model = ModelSpec(
    id="bayes_two_param",
    spec={"rl": {"bayes_filter_two_param": ["p_switch", "p_reward"]}},
    desc="Table-consistent two-parameter Bayes filter with beta fixed at 1.",
    params=None,
    sim=bayes_two_param_sim,
    fit=bayes_two_param_fit,
)


# Task-matched fixed emissions for the 80/20 reversal bandit.
FIXED_P_REWARD = 0.8
FIXED_P_NOISY = 0.2


def _lik_fixed_reward(action: int, reward: int, _p_reward_unused: float = FIXED_P_REWARD):
    return _likelihoods(action, reward, FIXED_P_REWARD, p_noisy=FIXED_P_NOISY)


def bayes_fixed_reward_sim(params, ntrials=80, seed=None, mu=(0.8, 0.2), schedule="canonical", **kwargs):
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    # Pack as [beta, p_switch, unused_p_reward_placeholder] for the shared 3-param loop
    # by expanding to include the fixed emission so _bayes_loop can reuse lik_fn.
    nsubj = params.shape[0]
    packed = np.column_stack(
        [params[:, 0], params[:, 1], np.full(nsubj, FIXED_P_REWARD)]
    )
    return _bayes_loop(
        packed,
        ntrials,
        seed,
        mu,
        schedule,
        constrain=False,
        lik_fn=_lik_fixed_reward,
        n_params=3,
    )


def bayes_fixed_reward_fit(params, choices, rewards, *, prior=None, output="npl"):
    beta = float(norm2beta(params[0]))
    p_switch = float(norm2alpha(params[1]))
    if not (1e-5 <= beta <= 20.0) or not (0.0 <= p_switch <= 1.0):
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
        belief = _update_belief_generic(
            belief, int(a[t]), int(r[t]), p_switch, FIXED_P_REWARD, _lik_fixed_reward
        )
        beliefs[t + 1] = belief

    if output == "all":
        return {
            "params": [beta, p_switch],
            "choices": a,
            "rewards": r,
            "belief": beliefs,
            "nll": nll,
            "fixed_p_reward": FIXED_P_REWARD,
            "fixed_p_noisy": FIXED_P_NOISY,
        }
    return calc_fval(nll, params, prior=prior, output=output)


bayes_fixed_reward_model = ModelSpec(
    id="bayes_fixed_reward",
    spec={"rl": {"bayes_filter_fixed_reward": ["beta", "p_switch"]}},
    desc=(
        "Bayes filter with free beta and p_switch; emissions fixed to task "
        f"p_high={FIXED_P_REWARD}, p_low={FIXED_P_NOISY}."
    ),
    params=None,
    sim=bayes_fixed_reward_sim,
    fit=bayes_fixed_reward_fit,
)
