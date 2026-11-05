"""Diagnostic-gated hierarchical fitting wrappers around pyEM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from pyem import EMModel
from pyem.core.priors import GaussianPrior
from scipy.optimize import minimize

from rlmodels_k99.parameters import MODEL_PARAM_NAMES, space_for
from rlmodels_k99.registry import make_em_model


@dataclass
class FitDiagnostics:
    convergence: bool
    n_iterations_requested: int
    mean_npl: float
    good_hess_frac: float
    boundary_frac: float
    posterior_mu: np.ndarray
    posterior_sigma: np.ndarray
    subject_params: np.ndarray
    npl: np.ndarray
    passed_gates: bool
    gate_failures: list[str] = field(default_factory=list)
    backend: str = "pyem"
    raw: Any = None
    lme: float = float("nan")


def _boundary_frac(params: np.ndarray, names: Sequence[str], tol: float = 1e-3) -> float:
    space = space_for(names)
    hits = 0
    total = params.size
    for j, d in enumerate(space.defs):
        col = params[:, j]
        if np.isfinite(d.low):
            hits += int(np.sum(np.abs(col - d.low) < tol))
        if np.isfinite(d.high):
            hits += int(np.sum(np.abs(col - d.high) < tol))
    return float(hits) / float(max(total, 1))


def evaluate_gates(
    *,
    convergence: bool,
    good_hess_frac: float,
    boundary_frac: float,
    require_convergence: bool = True,
    min_good_hess: float = 0.8,
    max_boundary_frac: float = 0.35,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if require_convergence and not convergence:
        failures.append("em_not_converged")
    if good_hess_frac < min_good_hess:
        failures.append(f"good_hess_frac={good_hess_frac:.3f}<{min_good_hess}")
    if boundary_frac > max_boundary_frac:
        failures.append(f"boundary_frac={boundary_frac:.3f}>{max_boundary_frac}")
    return len(failures) == 0, failures


# Model-specific recovery budgets (overrides for hard models).
RECOVERY_BUDGETS: dict[str, dict[str, Any]] = {
    "vkf": dict(mstep_maxit=120, max_restarts=4),
    "vkf_omega": dict(mstep_maxit=100, max_restarts=4),
    "vkf_fixed_v0": dict(mstep_maxit=100, max_restarts=4),
    "vkf_w0": dict(mstep_maxit=100, max_restarts=4),
    "rwus": dict(mstep_maxit=100, max_restarts=4),
    "rwus_upe": dict(mstep_maxit=100, max_restarts=4),
    "rw_signed": dict(mstep_maxit=100, max_restarts=4),
    "rw_signed_asym": dict(mstep_maxit=100, max_restarts=4),
    "rw_asym": dict(mstep_maxit=100, max_restarts=4),
}


def default_fit_kwargs(
    *,
    seed: int | None = 0,
    mstep_maxit: int = 100,
    max_restarts: int = 4,
    verbose: int = 0,
    njobs: int = 1,
    use_informed_prior: bool = True,
    model_name: str | None = None,
    convergence_crit: float = 1e-5,
) -> dict:
    kw: dict[str, Any] = dict(
        mstep_maxit=mstep_maxit,
        max_restarts=max_restarts,
        verbose=verbose,
        njobs=njobs,
        seed=seed,
        convergence_custom="relative_npl",
        convergence_crit=convergence_crit,
    )
    if use_informed_prior and model_name is not None:
        names = MODEL_PARAM_NAMES[model_name]
        mu, var = space_for(names).prior_moments()
        kw["prior_mu"] = mu
        kw["prior_sigma"] = var
    return kw


def production_recovery_fit_kwargs(
    model_name: str,
    *,
    seed: int | None = 0,
    njobs: int = 1,
    verbose: int = 0,
    screening: bool = False,
) -> dict:
    """Recovery-oriented EM settings with model-specific budgets.

    Screening uses a slightly looser convergence tolerance; final claims use 1e-5.
    """
    budget = RECOVERY_BUDGETS.get(
        model_name,
        dict(mstep_maxit=80 if not screening else 60, max_restarts=4),
    )
    return default_fit_kwargs(
        model_name=model_name,
        seed=seed,
        njobs=njobs,
        verbose=verbose,
        mstep_maxit=int(budget.get("mstep_maxit", 80)),
        max_restarts=int(budget.get("max_restarts", 4)),
        convergence_crit=1e-4 if screening else 1e-5,
        use_informed_prior=True,
    )


def fit_hierarchical(
    model_name: str,
    all_data: list,
    *,
    fit_kwargs: dict | None = None,
    n_starts: int = 1,
    require_convergence: bool = True,
    min_good_hess: float = 0.8,
    max_boundary_frac: float = 0.35,
) -> tuple[EMModel, FitDiagnostics]:
    """Fit with optional multi-seed starts; keep the lowest mean-NPL success."""
    base = default_fit_kwargs(model_name=model_name) | (fit_kwargs or {})
    seed0 = base.get("seed", 0) or 0
    best: tuple[EMModel, FitDiagnostics, float] | None = None

    for s in range(n_starts):
        kw = dict(base)
        kw["seed"] = int(seed0) + 17 * s
        em = make_em_model(model_name, all_data=all_data)
        res = em.fit(**kw)
        subj = em.subject_params()
        lme_val = float("nan")
        try:
            _lap, _lme, good = em.compute_lme()
            good_frac = float(np.mean(good)) if len(good) else 0.0
            lme_val = float(_lme) if np.isfinite(_lme) else float("nan")
        except Exception:
            good_frac = 0.0
        names = MODEL_PARAM_NAMES[model_name]
        bfrac = _boundary_frac(subj, names)
        mean_npl = float(np.mean(res.NPL))
        passed, fails = evaluate_gates(
            convergence=bool(res.convergence),
            good_hess_frac=good_frac,
            boundary_frac=bfrac,
            require_convergence=require_convergence,
            min_good_hess=min_good_hess,
            max_boundary_frac=max_boundary_frac,
        )
        diag = FitDiagnostics(
            convergence=bool(res.convergence),
            n_iterations_requested=int(kw.get("mstep_maxit", 100)),
            mean_npl=mean_npl,
            good_hess_frac=good_frac,
            boundary_frac=bfrac,
            posterior_mu=np.asarray(res.posterior_mu, dtype=float),
            posterior_sigma=np.asarray(res.posterior_sigma, dtype=float),
            subject_params=subj,
            npl=np.asarray(res.NPL, dtype=float),
            passed_gates=passed,
            gate_failures=fails,
            backend="pyem",
            raw=res,
            lme=lme_val,
        )
        score = mean_npl + (0.0 if passed else 1e6)
        if best is None or score < best[2]:
            best = (em, diag, score)

    assert best is not None
    return best[0], best[1]


def make_sdt_zrpe_day_em_model(n_conditions: int, all_data: list | None = None) -> EMModel:
    """Build pyEM EMModel for recognition SDT with C presentation conditions."""
    if n_conditions not in (2, 3):
        raise ValueError("n_conditions must be 2 (Prolific) or 3 (EMU)")
    model_name = f"sdt_zrpe_day_{n_conditions}"
    return make_em_model(model_name, all_data=all_data)


def fit_sdt_zrpe_day(
    all_data: list,
    n_conditions: int,
    *,
    fit_kwargs: dict | None = None,
    n_starts: int = 2,
    require_convergence: bool = False,
) -> tuple[EMModel, FitDiagnostics]:
    """Hierarchical EM fit for zRPE→d′ SDT with condition-specific parameters."""
    model_name = f"sdt_zrpe_day_{n_conditions}"
    kw = production_recovery_fit_kwargs(model_name, seed=0, njobs=1, verbose=0)
    kw["mstep_maxit"] = 80
    kw.update(fit_kwargs or {})
    return fit_hierarchical(
        model_name,
        all_data,
        fit_kwargs=kw,
        n_starts=n_starts,
        require_convergence=require_convergence,
        min_good_hess=0.5,
        max_boundary_frac=0.5,
    )


def independent_mle(
    model_name: str,
    all_data: list,
    *,
    seed: int = 0,
    max_restarts: int = 5,
    prior_centered: bool = True,
) -> np.ndarray:
    """Multi-start subject-level MLE (no hierarchical prior) in latent space."""
    em = make_em_model(model_name, all_data=all_data)
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    mu, _ = space.prior_moments()
    nparams = len(names)
    nsubj = len(all_data)
    out = np.zeros((nsubj, nparams), dtype=float)
    rng = np.random.default_rng(seed)

    for i, args in enumerate(all_data):
        if not isinstance(args, (list, tuple)):
            args = (args,)
        best_fun = np.inf
        best_x = None
        for r in range(max_restarts):
            if prior_centered and r == 0:
                x0 = mu + 0.15 * rng.standard_normal(nparams)
            else:
                x0 = (mu if prior_centered else 0.0) + 0.35 * rng.standard_normal(nparams)
            res = minimize(
                lambda x, a=args: float(em.fit_func(x, *a, prior=None, output="nll")),
                x0=x0,
                method="BFGS",
                options={"gtol": 1e-5},
            )
            if res.fun < best_fun:
                best_fun = float(res.fun)
                best_x = res.x
        assert best_x is not None
        out[i] = space.to_natural(best_x)
    return out


def fixed_prior_map(
    model_name: str,
    all_data: list,
    *,
    seed: int = 0,
    max_restarts: int = 4,
) -> np.ndarray:
    """Subject MAP under the informed Gaussian prior (no EM M-step updates)."""
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    mu, var = space.prior_moments()
    prior = GaussianPrior(mu=mu, sigma=var)
    em = make_em_model(model_name, all_data=all_data)
    nparams = len(names)
    nsubj = len(all_data)
    out = np.zeros((nsubj, nparams), dtype=float)
    rng = np.random.default_rng(seed)

    for i, args in enumerate(all_data):
        if not isinstance(args, (list, tuple)):
            args = (args,)
        best_fun = np.inf
        best_x = None
        for r in range(max_restarts):
            # Prior-centered starts; first restart is closest to mu.
            scale = 0.1 if r == 0 else 0.25
            x0 = mu + scale * rng.standard_normal(nparams)
            res = minimize(
                lambda x, a=args: float(em.fit_func(x, *a, prior=prior, output="npl")),
                x0=x0,
                method="BFGS",
                options={"gtol": 1e-5},
            )
            if res.fun < best_fun:
                best_fun = float(res.fun)
                best_x = res.x
        assert best_x is not None
        out[i] = space.to_natural(best_x)
    return out


def oracle_profile(
    fit_func: Callable,
    choices: np.ndarray,
    rewards: np.ndarray,
    true_latent: np.ndarray,
    free_index: int,
    *,
    bounds: tuple[float, float] = (-8.0, 8.0),
) -> float:
    """Optimize one latent parameter with others fixed at truth; return natural estimate."""
    from scipy.optimize import minimize_scalar

    true_latent = np.asarray(true_latent, dtype=float)

    def obj(x):
        q = true_latent.copy()
        q[free_index] = x
        return float(fit_func(q, choices, rewards, prior=None, output="nll"))

    res = minimize_scalar(obj, bounds=bounds, method="bounded", options={"xatol": 1e-5})
    return float(res.x)


def stabilized_laplace_em(
    model_name: str,
    all_data: list,
    *,
    mstep_maxit: int = 40,
    seed: int = 0,
    variance_floor: float = 0.05,
    max_restarts: int = 3,
) -> tuple[np.ndarray, FitDiagnostics]:
    """Opt-in warm-started Laplace-EM with variance floors.

    Retains stock pyEM as the reference; this path exists for shrinkage/collapse
    comparisons and is not the default recovery backend.
    """
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    mu, var = space.prior_moments()
    nparams = len(names)
    nsubj = len(all_data)
    rng = np.random.default_rng(seed)
    em_proto = make_em_model(model_name, all_data=all_data)
    fit_func = em_proto.fit_func

    m = np.zeros((nparams, nsubj))
    inv_h = np.zeros((nparams, nparams, nsubj))
    npl = np.zeros(nsubj)
    warm = [None] * nsubj
    hist = []

    for it in range(mstep_maxit):
        prior = GaussianPrior(mu=mu, sigma=np.maximum(var, variance_floor))
        for i, args in enumerate(all_data):
            if not isinstance(args, (list, tuple)):
                args = (args,)
            best_fun = np.inf
            best_x = None
            best_hess = None
            starts = []
            if warm[i] is not None:
                starts.append(np.asarray(warm[i], dtype=float))
            starts.append(mu + 0.1 * rng.standard_normal(nparams))
            for _ in range(max_restarts):
                starts.append(0.2 * rng.standard_normal(nparams))
            for x0 in starts:
                res = minimize(
                    lambda x, a=args: float(fit_func(x, *a, prior=prior, output="npl")),
                    x0=x0,
                    method="BFGS",
                    options={"gtol": 1e-5},
                )
                if res.fun < best_fun:
                    best_fun = float(res.fun)
                    best_x = res.x
                    if hasattr(res, "hess_inv"):
                        try:
                            best_hess = np.asarray(res.hess_inv, dtype=float)
                        except Exception:
                            best_hess = None
            assert best_x is not None
            warm[i] = best_x
            m[:, i] = best_x
            npl[i] = best_fun
            if best_hess is None or best_hess.shape != (nparams, nparams):
                best_hess = np.eye(nparams)
            # Force PD diagonal floor.
            diag = np.maximum(np.diag(best_hess), variance_floor)
            best_hess = best_hess.copy()
            np.fill_diagonal(best_hess, diag)
            # Symmetrize
            best_hess = 0.5 * (best_hess + best_hess.T)
            inv_h[:, :, i] = best_hess

        mu_new = np.mean(m, axis=1)
        var_new = np.zeros(nparams)
        for s in range(nsubj):
            var_new += m[:, s] ** 2 + np.diag(inv_h[:, :, s])
        var_new = var_new / nsubj - mu_new**2
        var_new = np.maximum(var_new, variance_floor)
        delta = float(np.max(np.abs(mu_new - mu)) + np.max(np.abs(var_new - var)))
        hist.append(delta)
        mu, var = mu_new, var_new
        if it > 1 and delta < 1e-4:
            converged = True
            break
    else:
        converged = False

    natural = space.to_natural(m.T)
    bfrac = _boundary_frac(natural, names)
    passed, fails = evaluate_gates(
        convergence=converged,
        good_hess_frac=1.0,
        boundary_frac=bfrac,
        require_convergence=True,
    )
    diag = FitDiagnostics(
        convergence=converged,
        n_iterations_requested=mstep_maxit,
        mean_npl=float(np.mean(npl)),
        good_hess_frac=1.0,
        boundary_frac=bfrac,
        posterior_mu=mu,
        posterior_sigma=var,
        subject_params=natural,
        npl=npl,
        passed_gates=passed,
        gate_failures=fails,
        backend="stabilized_laplace_em",
        raw={"history": hist, "m": m, "inv_h": inv_h},
    )
    return natural, diag
