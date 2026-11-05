"""Hierarchical equal-variance SDT for recognition with zRPE → d′ and condition-specific c."""

from __future__ import annotations

import numpy as np
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import calc_fval, norm2beta
from scipy.stats import norm


def sdt_zrpe_day_param_names(n_conditions: int) -> list[str]:
    """Per-condition: dprime, criterion, b_zrpe (3 params × C conditions)."""
    if n_conditions < 1:
        raise ValueError("n_conditions must be >= 1")
    names: list[str] = []
    for c in range(1, n_conditions + 1):
        names.extend([f"dprime_{c}", f"criterion_{c}", f"b_zrpe_{c}"])
    return names


def _unpack_params(params: np.ndarray, n_conditions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Natural-scale dprime, criterion, b_zrpe arrays of length n_conditions."""
    params = np.asarray(params, dtype=float).ravel()
    expected = 3 * n_conditions
    if params.size != expected:
        raise ValueError(f"Expected {expected} params, got {params.size}")
    dprime = np.array([float(norm2beta(params[3 * i])) for i in range(n_conditions)])
    criterion = np.array([float(params[3 * i + 1]) for i in range(n_conditions)])
    b_zrpe = np.array([float(params[3 * i + 2]) for i in range(n_conditions)])
    return dprime, criterion, b_zrpe


def sdt_zrpe_day_fit(
    params,
    is_old,
    resp_old,
    zrpe,
    cond_idx,
    n_conditions: int,
    prior=None,
    output="npl",
):
    """
    Trial-level equal-variance SDT with zRPE modulating d′ on studied trials only.

    Evidence mean on trial n in condition c:
      μ = is_old * (d'[c]/2 + b_zrpe[c] * zrpe) + (1-is_old) * (-d'[c]/2)
    P(resp_old=1) = Φ(μ - c[c])
    """
    try:
        dprime, criterion, b_zrpe = _unpack_params(params, n_conditions)
    except ValueError:
        return 1e7

    if not ((dprime >= 1e-5) & (dprime <= 20.0)).all():
        return 1e7

    is_old = np.asarray(is_old, dtype=float)
    resp_old = np.asarray(resp_old, dtype=float)
    zrpe = np.asarray(zrpe, dtype=float)
    cond_idx = np.asarray(cond_idx, dtype=int)

    if not (
        is_old.shape == resp_old.shape == zrpe.shape == cond_idx.shape
    ):
        return 1e7

    nll = 0.0
    for i in range(is_old.size):
        c = int(cond_idx[i]) - 1
        if c < 0 or c >= n_conditions:
            return 1e7
        dp = float(dprime[c])
        crit = float(criterion[c])
        bz = float(b_zrpe[c])
        z = float(zrpe[i])
        if is_old[i] == 1.0:
            mu = dp / 2.0 + bz * z
        else:
            mu = -dp / 2.0
        signal = mu - crit
        p_old = float(norm.cdf(signal))
        p_old = np.clip(p_old, 1e-12, 1.0 - 1e-12)
        ro = resp_old[i]
        nll -= ro * np.log(p_old) + (1.0 - ro) * np.log(1.0 - p_old)

    if output == "all":
        return {
            "params": np.column_stack([dprime, criterion, b_zrpe]),
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def sdt_zrpe_day_sim(
    params: np.ndarray,
    ntrials: int = 200,
    n_conditions: int = 2,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Simulate old/new recognition with condition-balanced trials per subject."""
    params = np.asarray(params, dtype=float)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    nsubjects = params.shape[0]
    rng = np.random.default_rng(seed)

    all_is_old = np.zeros((nsubjects, ntrials), dtype=float)
    all_resp = np.zeros((nsubjects, ntrials), dtype=float)
    all_zrpe = np.zeros((nsubjects, ntrials), dtype=float)
    all_cond = np.zeros((nsubjects, ntrials), dtype=int)

    for s in range(nsubjects):
        dprime, criterion, b_zrpe = _unpack_params(params[s], n_conditions)
        trial_is_old = np.zeros(ntrials, dtype=float)
        trial_is_old[: ntrials // 2] = 1.0
        rng.shuffle(trial_is_old)
        if ntrials % 2 == 1:
            trial_is_old[-1] = float(rng.integers(0, 2))

        trial_cond = rng.integers(1, n_conditions + 1, size=ntrials)
        trial_zrpe = np.zeros(ntrials, dtype=float)
        old_mask = trial_is_old == 1.0
        trial_zrpe[old_mask] = rng.normal(0.0, 0.5, size=int(old_mask.sum()))

        resp = np.zeros(ntrials, dtype=float)
        for t in range(ntrials):
            c = int(trial_cond[t]) - 1
            dp = float(dprime[c])
            crit = float(criterion[c])
            bz = float(b_zrpe[c])
            z = float(trial_zrpe[t])
            if trial_is_old[t] == 1.0:
                mu = dp / 2.0 + bz * z
            else:
                mu = -dp / 2.0
            p_old = float(norm.cdf(mu - crit))
            resp[t] = float(rng.random() < p_old)

        all_is_old[s] = trial_is_old
        all_resp[s] = resp
        all_zrpe[s] = trial_zrpe
        all_cond[s] = trial_cond

    return {
        "params": params.copy(),
        "is_old": all_is_old,
        "resp_old": all_resp,
        "zrpe": all_zrpe,
        "cond_idx": all_cond,
    }


def make_sdt_zrpe_day_model(n_conditions: int) -> ModelSpec:
    """Build a pyEM ModelSpec for C presentation conditions."""

    def _fit(params, is_old, resp_old, zrpe, cond_idx, prior=None, output="npl"):
        return sdt_zrpe_day_fit(
            params, is_old, resp_old, zrpe, cond_idx, n_conditions, prior, output
        )

    def _sim(params, ntrials=200, seed=None, **kwargs):
        return sdt_zrpe_day_sim(params, ntrials, n_conditions, seed=seed, **kwargs)

    desc = (
        f"Equal-variance SDT with zRPE→d′ and condition-specific criterion "
        f"({n_conditions} presentation conditions)."
    )
    pid = f"sdt_zrpe_day_{n_conditions}"
    spec = {pid: {f"gaussian_ev_zrpe_{n_conditions}c": sdt_zrpe_day_param_names(n_conditions)}}
    return ModelSpec(id=pid, spec=spec, desc=desc, params=None, sim=_sim, fit=_fit)


sdt_zrpe_day_2_model = make_sdt_zrpe_day_model(2)
sdt_zrpe_day_3_model = make_sdt_zrpe_day_model(3)
