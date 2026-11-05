"""Parameter- and model-recovery helpers plus identifiability diagnostics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rlmodels_k99.estimation import (
    default_fit_kwargs,
    fit_hierarchical,
    fixed_prior_map,
    independent_mle,
    oracle_profile,
    production_recovery_fit_kwargs,
    stabilized_laplace_em,
)
from rlmodels_k99.parameters import (
    MODEL_PARAM_NAMES,
    GeneratorMode,
    build_params,
    space_for,
)
from rlmodels_k99.registry import MODEL_ORDER, get_model, make_em_model
from rlmodels_k99.runtime import append_checkpoint_row, has_checkpoint_keys


def _stable_round_seed(base_seed: int, rnd: int, gen_name: str) -> int:
    """Deterministic seed so model-recovery checkpoints resume reproducibly."""
    digest = hashlib.md5(f"{base_seed}:{rnd}:{gen_name}".encode()).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def sim_to_all_data(sim: dict) -> list[list[np.ndarray]]:
    return [[sim["choices"][s], sim["rewards"][s]] for s in range(sim["choices"].shape[0])]


def parameter_recovery_metrics(
    true_params: np.ndarray,
    est_params: np.ndarray,
    *,
    scale: str = "natural",
) -> pd.DataFrame:
    """Pearson/Spearman recovery metrics plus calibration and shrinkage.

    ``scale`` is a label only (``natural`` or ``latent``); arrays must already
    be expressed on that scale.
    """
    rows = []
    for j in range(true_params.shape[1]):
        t = true_params[:, j]
        e = est_params[:, j]
        if np.std(t) > 0 and np.std(e) > 0:
            corr = float(np.corrcoef(t, e)[0, 1])
            sp = stats.spearmanr(t, e)
            spear = float(getattr(sp, "statistic", getattr(sp, "correlation")))
            slope, intercept = np.polyfit(t, e, 1)
        else:
            corr = spear = slope = intercept = np.nan
        rows.append(
            {
                "param_index": j,
                "scale": scale,
                "correlation": corr,
                "spearman": spear,
                "rmse": float(np.sqrt(np.mean((t - e) ** 2))),
                "mae": float(np.mean(np.abs(t - e))),
                "bias": float(np.mean(e - t)),
                "calibration_slope": float(slope) if slope == slope else np.nan,
                "calibration_intercept": float(intercept) if intercept == intercept else np.nan,
                "true_sd": float(np.std(t)),
                "est_sd": float(np.std(e)),
                "shrinkage_ratio": float(np.std(e) / np.std(t)) if np.std(t) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parameter_error_correlations(true_params: np.ndarray, est_params: np.ndarray) -> dict:
    """Pairwise correlations of (est - true) across parameters."""
    err = np.asarray(est_params, dtype=float) - np.asarray(true_params, dtype=float)
    n = err.shape[1]
    out: dict[str, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(err[:, i]) > 0 and np.std(err[:, j]) > 0:
                out[f"err_corr_{i}_{j}"] = float(np.corrcoef(err[:, i], err[:, j])[0, 1])
            else:
                out[f"err_corr_{i}_{j}"] = float("nan")
    return out


def group_hyperparameter_recovery(
    true_natural: np.ndarray,
    posterior_mu_latent: np.ndarray,
    param_names: Sequence[str],
) -> dict:
    """Compare estimated group latent means to the true sample latent means."""
    space = space_for(param_names)
    true_latent_mean = space.to_latent(np.asarray(true_natural, dtype=float)).mean(axis=0)
    est_mu = np.asarray(posterior_mu_latent, dtype=float).reshape(-1)
    rows = {}
    for j, name in enumerate(param_names):
        rows[f"group_mu_true_{name}"] = float(true_latent_mean[j])
        rows[f"group_mu_est_{name}"] = float(est_mu[j]) if j < len(est_mu) else float("nan")
        rows[f"group_mu_err_{name}"] = (
            float(est_mu[j] - true_latent_mean[j]) if j < len(est_mu) else float("nan")
        )
    return rows


def run_parameter_recovery(
    model_name: str,
    n_subjects: int = 40,
    ntrials: int = 80,
    seed: int = 0,
    fit_kwargs: dict | None = None,
    generator_mode: GeneratorMode = "hierarchical_latent",
    n_starts: int = 1,
    require_convergence: bool = False,
    backend: str = "pyem",
    use_production_budget: bool = False,
) -> dict:
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    # Keep parameter generation and behavioral simulation seeds independent.
    param_seed = int(seed)
    sim_seed = int(seed) + 10_000
    _, _, true_params = build_params(
        names, n_subjects, seed=param_seed, mode=generator_mode, model_name=model_name
    )
    sim = get_model(model_name).sim(true_params, ntrials=ntrials, seed=sim_seed)
    all_data = sim_to_all_data(sim)

    if use_production_budget:
        base_kw = production_recovery_fit_kwargs(model_name, seed=seed)
    else:
        base_kw = default_fit_kwargs(model_name=model_name, seed=seed)
    merged = base_kw | (fit_kwargs or {})

    if backend == "stabilized":
        est, diag = stabilized_laplace_em(
            model_name,
            all_data,
            mstep_maxit=int(merged.get("mstep_maxit", 40)),
            seed=seed,
        )
        em = None
        convergence = diag.convergence
    else:
        em, diag = fit_hierarchical(
            model_name,
            all_data,
            fit_kwargs=merged,
            n_starts=n_starts,
            require_convergence=require_convergence,
        )
        est = diag.subject_params
        convergence = diag.convergence

    metrics = parameter_recovery_metrics(true_params, est, scale="natural")
    true_lat = space.to_latent(true_params)
    est_lat = space.to_latent(est)
    metrics_lat = parameter_recovery_metrics(true_lat, est_lat, scale="latent")
    # Merge latent columns alongside natural (primary rows remain natural).
    for col in ("correlation", "spearman", "rmse", "calibration_slope", "shrinkage_ratio"):
        metrics[f"latent_{col}"] = metrics_lat[col].values
    metrics.insert(0, "param", names)
    metrics.insert(0, "model", model_name)
    metrics["generator_mode"] = generator_mode
    metrics["backend"] = diag.backend
    metrics["convergence"] = convergence
    metrics["passed_gates"] = diag.passed_gates
    metrics["good_hess_frac"] = diag.good_hess_frac
    metrics["boundary_frac"] = diag.boundary_frac
    group = group_hyperparameter_recovery(true_params, diag.posterior_mu, names)
    for k, v in group.items():
        metrics[k] = v
    err_corr = parameter_error_correlations(true_params, est)
    for k, v in err_corr.items():
        metrics[k] = v
    return {
        "model": model_name,
        "true_params": true_params,
        "estimated_params": est,
        "metrics": metrics,
        "metrics_latent": metrics_lat,
        "diagnostics": diag,
        "convergence": convergence,
        "em": em,
        "sim": sim,
        "group_hyperparameters": group,
    }


def summarize_model_recovery_rounds(
    rounds_df: pd.DataFrame,
    model_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate round-level model-recovery rows into a confusion summary."""
    if rounds_df is None or len(rounds_df) == 0:
        return pd.DataFrame()
    model_names = list(model_names or sorted(set(rounds_df["Simulated"]) | set(rounds_df["Estimated"])))
    summary_rows = []
    for gen_name in model_names:
        for fit_name in model_names:
            sub = rounds_df[(rounds_df.Simulated == gen_name) & (rounds_df.Estimated == fit_name)]
            gen_rounds = rounds_df[rounds_df.Simulated == gen_name]
            best_lme = 0
            best_bic = 0
            nrounds = int(gen_rounds["round"].nunique()) if len(gen_rounds) else 0
            for rnd in gen_rounds["round"].unique() if len(gen_rounds) else []:
                block = gen_rounds[gen_rounds["round"] == rnd]
                if block["LME"].notna().any():
                    winner = block.loc[block["LME"].idxmax(), "Estimated"]
                    best_lme += int(winner == fit_name)
                if "BICint" in block.columns and block["BICint"].notna().any():
                    winner = block.loc[block["BICint"].idxmin(), "Estimated"]
                    best_bic += int(winner == fit_name)
            summary_rows.append(
                {
                    "Simulated": gen_name,
                    "Estimated": fit_name,
                    "LME": float(sub["LME"].mean()) if len(sub) else np.nan,
                    "BICint": float(sub["BICint"].mean()) if len(sub) and "BICint" in sub else np.nan,
                    "bestlme": best_lme,
                    "bestbic": best_bic,
                    "nrounds": nrounds,
                    "mean_convergence": float(sub["convergence"].mean()) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(summary_rows)


def true_est_pairs_frame(
    model_name: str,
    seed: int,
    true_params: np.ndarray,
    est_params: np.ndarray,
    param_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Long-form true vs estimated subject parameters for scatter plots."""
    names = list(param_names or MODEL_PARAM_NAMES[model_name])
    true_params = np.asarray(true_params, dtype=float)
    est_params = np.asarray(est_params, dtype=float)
    rows = []
    for s in range(true_params.shape[0]):
        for j, pname in enumerate(names):
            rows.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "subject": int(s),
                    "param": pname,
                    "true": float(true_params[s, j]),
                    "estimated": float(est_params[s, j]),
                }
            )
    return pd.DataFrame(rows)


def run_parameter_recovery_batch(
    model_names: Sequence[str],
    *,
    seeds: Sequence[int],
    n_subjects: int = 40,
    ntrials: int = 80,
    fit_kwargs: dict | None = None,
    generator_mode: GeneratorMode = "hierarchical_latent",
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    require_convergence: bool = False,
    n_starts: int = 1,
    use_production_budget: bool = False,
) -> pd.DataFrame:
    """Run parameter recovery for each (model, seed), checkpointing metrics + pairs.

    Replications are expressed as an explicit ``seeds`` list (no seed×round product).
    Also writes long-form true/estimated pairs for scatter plots
    (``pr_pairs_{model}_seed{seed}.csv`` and ``parameter_recovery_pairs.csv``).
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = checkpoint_dir / "parameter_recovery_metrics.csv"
        pairs_agg_path = checkpoint_dir / "parameter_recovery_pairs.csv"
    else:
        aggregate_path = None
        pairs_agg_path = None

    frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    if resume and aggregate_path is not None and aggregate_path.exists():
        existing = pd.read_csv(aggregate_path)
        frames.append(existing)
    else:
        existing = pd.DataFrame()
    if resume and pairs_agg_path is not None and pairs_agg_path.exists():
        pair_frames.append(pd.read_csv(pairs_agg_path))

    for model_name in model_names:
        for seed in seeds:
            seed = int(seed)
            pairs_unit = (
                checkpoint_dir / f"pr_pairs_{model_name}_seed{seed}.csv"
                if checkpoint_dir is not None
                else None
            )
            metrics_done = False
            if resume and len(existing):
                hit = existing[(existing["model"] == model_name) & (existing["seed"] == seed)]
                metrics_done = len(hit) > 0
            pairs_done = pairs_unit is not None and pairs_unit.exists()
            # Require both metrics and scatter pairs before skipping (one-time refill if pairs missing).
            if resume and metrics_done and pairs_done:
                print(f"resume skip PR {model_name} seed={seed}")
                continue
            print(f"parameter recovery: {model_name} seed={seed}")
            out = run_parameter_recovery(
                model_name,
                n_subjects=n_subjects,
                ntrials=ntrials,
                seed=seed,
                fit_kwargs=fit_kwargs,
                generator_mode=generator_mode,
                require_convergence=require_convergence,
                n_starts=n_starts,
                use_production_budget=use_production_budget,
            )
            m = out["metrics"].copy()
            m["seed"] = seed
            m["n_subjects"] = n_subjects
            pairs = true_est_pairs_frame(
                model_name, seed, out["true_params"], out["estimated_params"]
            )
            del out
            frames.append(m)
            pair_frames.append(pairs)
            if aggregate_path is not None:
                pd.concat(frames, ignore_index=True).drop_duplicates(
                    subset=["model", "param", "seed"], keep="last"
                ).to_csv(aggregate_path, index=False)
                unit_path = checkpoint_dir / f"pr_{model_name}_seed{seed}.csv"
                m.to_csv(unit_path, index=False)
                pairs.to_csv(pairs_unit, index=False)
                pd.concat(pair_frames, ignore_index=True).drop_duplicates(
                    subset=["model", "seed", "subject", "param"], keep="last"
                ).to_csv(pairs_agg_path, index=False)

    if not frames:
        return pd.DataFrame()
    pr_df = pd.concat(frames, ignore_index=True)
    if {"model", "param", "seed"}.issubset(pr_df.columns):
        pr_df = pr_df.drop_duplicates(subset=["model", "param", "seed"], keep="last")
    if aggregate_path is not None:
        pr_df.to_csv(aggregate_path, index=False)
    if pair_frames and pairs_agg_path is not None:
        pairs_df = pd.concat(pair_frames, ignore_index=True).drop_duplicates(
            subset=["model", "seed", "subject", "param"], keep="last"
        )
        pairs_df.to_csv(pairs_agg_path, index=False)
    return pr_df


def run_model_recovery(
    model_names: Sequence[str] | None = None,
    n_subjects: int = 30,
    ntrials: int = 80,
    nrounds: int = 2,
    seed: int = 0,
    fit_kwargs: dict | None = None,
    bic_nsamples: int = 200,
    generator_mode: GeneratorMode = "hierarchical_latent",
    *,
    compute_bic: bool = False,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Custom outer loop retaining round-level results.

    By default ``compute_bic=False`` because eligibility gates use LME.
    Integrated BIC is expensive (nsamples × n_subjects likelihoods per cell).
    """
    model_names = list(model_names or MODEL_ORDER)
    # Recovery-specific EM budget; caller fit_kwargs may override.
    base_kw = default_fit_kwargs(seed=seed, mstep_maxit=30, use_informed_prior=False, njobs=1)
    fit_kwargs = base_kw | (fit_kwargs or {})

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    rounds_path = checkpoint_dir / "model_recovery_rounds.csv" if checkpoint_dir else None
    if rounds_path is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if rounds_path is not None and not resume and rounds_path.exists():
        rounds_path.unlink()
    if resume and rounds_path is not None and rounds_path.exists():
        rounds_df = pd.read_csv(rounds_path)
        round_rows = rounds_df.to_dict("records")
    else:
        rounds_df = pd.DataFrame()
        round_rows = []

    def _done(rnd: int, gen_name: str, fit_name: str) -> bool:
        if not resume or rounds_path is None or not rounds_path.exists():
            return False
        return has_checkpoint_keys(
            rounds_path,
            {"round": rnd, "Simulated": gen_name, "Estimated": fit_name},
        )

    for rnd in range(nrounds):
        for gen_name in model_names:
            round_seed = _stable_round_seed(int(seed), int(rnd), gen_name)

            # Skip simulating if every fit for this generator/round is done.
            if all(_done(rnd, gen_name, fit_name) for fit_name in model_names):
                print(f"resume skip MR round={rnd} gen={gen_name}")
                continue

            names = MODEL_PARAM_NAMES[gen_name]
            _, _, true_params = build_params(
                names,
                n_subjects,
                seed=round_seed,
                mode=generator_mode,
                model_name=gen_name,
            )
            gen_model = make_em_model(gen_name)
            sim = gen_model.simulate(true_params, ntrials=ntrials, seed=round_seed)
            all_data = sim_to_all_data(sim)

            for fit_name in model_names:
                if _done(rnd, gen_name, fit_name):
                    print(f"resume skip MR round={rnd} {gen_name}->{fit_name}")
                    continue
                print(f"model recovery: round={rnd} {gen_name}->{fit_name}")
                per_fit_kw = default_fit_kwargs(model_name=fit_name, seed=round_seed) | {
                    k: v for k, v in fit_kwargs.items() if k not in ("prior_mu", "prior_sigma")
                }
                per_fit_kw["seed"] = round_seed
                em, diag = fit_hierarchical(
                    fit_name,
                    all_data,
                    fit_kwargs=per_fit_kw,
                    require_convergence=False,
                )
                # Reuse LME from fit diagnostics (already computed in fit_hierarchical).
                lme = float(diag.lme) if diag.lme == diag.lme else np.nan
                good_hess = float(diag.good_hess_frac)
                bic = np.nan
                if compute_bic:
                    try:
                        bic = em.compute_integrated_bic(
                            nsamples=bic_nsamples,
                            ntrials_total=ntrials,
                            func_output="all",
                            nll_key="nll",
                        )
                        bic = float(bic) if np.isfinite(bic) else np.nan
                    except Exception:
                        bic = np.nan
                row = {
                    "round": rnd,
                    "seed": round_seed,
                    "Simulated": gen_name,
                    "Estimated": fit_name,
                    "LME": lme,
                    "BICint": bic,
                    "convergence": bool(diag.convergence),
                    "good_hess_frac": good_hess,
                    "passed_gates": diag.passed_gates,
                    "boundary_frac": diag.boundary_frac,
                }
                round_rows.append(row)
                if rounds_path is not None:
                    append_checkpoint_row(
                        rounds_path,
                        row,
                        key_cols=("round", "Simulated", "Estimated"),
                    )
                # Release fitted model promptly.
                del em

    rounds_df = pd.DataFrame(round_rows)
    if rounds_path is not None and len(rounds_df):
        # Reload canonical checkpoint to avoid duplicates after resume.
        rounds_df = pd.read_csv(rounds_path)
        rounds_df = rounds_df.drop_duplicates(
            subset=["round", "Simulated", "Estimated"], keep="last"
        )
        rounds_df.to_csv(rounds_path, index=False)
    summary_df = summarize_model_recovery_rounds(rounds_df, model_names)
    if checkpoint_dir is not None and len(summary_df):
        summary_df.to_csv(checkpoint_dir / "model_recovery_summary.csv", index=False)
    return rounds_df, summary_df


def compute_finalist_bic(
    model_name: str,
    all_data: list,
    *,
    ntrials: int = 80,
    fit_kwargs: dict | None = None,
    bic_nsamples: int = 500,
    require_convergence: bool = False,
) -> dict:
    """Fit one finalist model and compute integrated BIC once."""
    em, diag = fit_hierarchical(
        model_name,
        all_data,
        fit_kwargs=fit_kwargs,
        require_convergence=require_convergence,
    )
    try:
        bic = em.compute_integrated_bic(
            nsamples=bic_nsamples,
            ntrials_total=ntrials,
            func_output="all",
            nll_key="nll",
        )
        bic = float(bic) if np.isfinite(bic) else np.nan
    except Exception:
        bic = np.nan
    return {
        "model": model_name,
        "LME": float(diag.lme) if diag.lme == diag.lme else np.nan,
        "BICint": bic,
        "convergence": bool(diag.convergence),
        "passed_gates": diag.passed_gates,
        "good_hess_frac": float(diag.good_hess_frac),
        "boundary_frac": float(diag.boundary_frac),
        "em": em,
        "diagnostics": diag,
    }


def likelihood_profile_2d(
    model_name: str,
    choices: np.ndarray,
    rewards: np.ndarray,
    true_natural: np.ndarray,
    idx_x: int,
    idx_y: int,
    *,
    n_grid: int = 21,
    span: float = 2.0,
) -> dict:
    """Evaluate NLL on a 2-D latent grid around the true latent parameters."""
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    fit = get_model(model_name).fit
    true_latent = space.to_latent(np.asarray(true_natural, dtype=float))
    xs = np.linspace(true_latent[idx_x] - span, true_latent[idx_x] + span, n_grid)
    ys = np.linspace(true_latent[idx_y] - span, true_latent[idx_y] + span, n_grid)
    Z = np.zeros((n_grid, n_grid), dtype=float)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            q = true_latent.copy()
            q[idx_x] = xv
            q[idx_y] = yv
            Z[j, i] = float(fit(q, choices, rewards, prior=None, output="nll"))
    return {
        "x": xs,
        "y": ys,
        "nll": Z,
        "param_x": names[idx_x],
        "param_y": names[idx_y],
        "true_latent": true_latent,
    }


def run_identifiability_study(
    model_name: str,
    *,
    n_subjects: int = 40,
    ntrials: int = 80,
    seeds: Sequence[int] = (0, 1, 2),
    fit_kwargs: dict | None = None,
    generator_mode: GeneratorMode = "hierarchical_latent",
    include_oracle: bool = True,
    include_mle: bool = True,
    include_map: bool = True,
    include_em: bool = True,
    include_stabilized: bool = False,
    n_em_starts: int = 3,
    use_production_budget: bool = True,
) -> pd.DataFrame:
    """Multi-seed oracle / MLE / MAP / EM recovery on natural and latent scales."""
    names = MODEL_PARAM_NAMES[model_name]
    space = space_for(names)
    rows = []
    fit_kwargs = fit_kwargs or {}

    for seed in seeds:
        _, _, true = build_params(
            names, n_subjects, seed=seed, mode=generator_mode, model_name=model_name
        )
        sim = get_model(model_name).sim(true, ntrials=ntrials, seed=seed + 10_000)
        data = sim_to_all_data(sim)
        true_latent = space.to_latent(true)

        estimators: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}

        if include_oracle:
            est = np.zeros_like(true)
            for s in range(n_subjects):
                for j in range(len(names)):
                    lat_j = oracle_profile(
                        get_model(model_name).fit,
                        sim["choices"][s],
                        sim["rewards"][s],
                        true_latent[s],
                        j,
                    )
                    tmp = true_latent[s].copy()
                    tmp[j] = lat_j
                    est[s, j] = space.to_natural(tmp)[j]
            estimators["oracle"] = est
            meta["oracle"] = {"convergence": True, "passed_gates": True}

        if include_mle:
            estimators["mle"] = independent_mle(model_name, data, seed=seed)
            meta["mle"] = {"convergence": True, "passed_gates": True}

        if include_map:
            estimators["map"] = fixed_prior_map(model_name, data, seed=seed)
            meta["map"] = {"convergence": True, "passed_gates": True}

        if include_em:
            if use_production_budget:
                em_kw = production_recovery_fit_kwargs(model_name, seed=seed) | fit_kwargs
            else:
                em_kw = default_fit_kwargs(model_name=model_name, seed=seed) | fit_kwargs
            _em, diag = fit_hierarchical(
                model_name,
                data,
                fit_kwargs=em_kw,
                n_starts=n_em_starts,
                require_convergence=False,
            )
            estimators["em"] = diag.subject_params
            group = group_hyperparameter_recovery(true, diag.posterior_mu, names)
            meta["em"] = {
                "convergence": diag.convergence,
                "passed_gates": diag.passed_gates,
                "good_hess_frac": diag.good_hess_frac,
                "boundary_frac": diag.boundary_frac,
                **group,
            }

        if include_stabilized:
            est, diag = stabilized_laplace_em(
                model_name,
                data,
                mstep_maxit=int(fit_kwargs.get("mstep_maxit", 40)),
                seed=seed,
            )
            estimators["stabilized"] = est
            meta["stabilized"] = {
                "convergence": diag.convergence,
                "passed_gates": diag.passed_gates,
                "good_hess_frac": diag.good_hess_frac,
                "boundary_frac": diag.boundary_frac,
            }

        for est_name, est in estimators.items():
            est_lat = space.to_latent(est)
            for scale, t_arr, e_arr in (
                ("natural", true, est),
                ("latent", true_latent, est_lat),
            ):
                m = parameter_recovery_metrics(t_arr, e_arr, scale=scale)
                err_corr = parameter_error_correlations(t_arr, e_arr)
                for _, row in m.iterrows():
                    rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "estimator": est_name,
                            "param": names[int(row["param_index"])],
                            **{k: row[k] for k in m.columns if k != "param_index"},
                            **meta.get(est_name, {}),
                            **err_corr,
                            "generator_mode": generator_mode,
                            "n_subjects": n_subjects,
                            "ntrials": ntrials,
                        }
                    )

    return pd.DataFrame(rows)


def classify_recovery_ladder(id_df: pd.DataFrame, *, min_r: float = 0.6) -> pd.DataFrame:
    """Classify each parameter using oracle/MLE/MAP/EM mean correlations.

    Labels:
    - structural_or_task: oracle or MLE weak
    - prior_shrinkage: MLE good but MAP/EM weak
    - optimizer_or_convergence: all good except EM
    - recoverable: all estimators pass
    - mixed: otherwise
    """
    if id_df is None or len(id_df) == 0:
        return pd.DataFrame()
    df = id_df[id_df.get("scale", "natural") == "natural"] if "scale" in id_df.columns else id_df
    rows = []
    for (model, param), sub in df.groupby(["model", "param"]):
        means = sub.groupby("estimator")["correlation"].mean()
        oracle_r = float(means.get("oracle", np.nan))
        mle_r = float(means.get("mle", np.nan))
        map_r = float(means.get("map", np.nan))
        em_r = float(means.get("em", np.nan))
        em_sub = sub[sub["estimator"] == "em"]
        conv = float(em_sub["convergence"].mean()) if len(em_sub) and "convergence" in em_sub else np.nan

        def _ok(r: float) -> bool:
            return bool(np.isfinite(r) and r > min_r)

        if not _ok(oracle_r) or not _ok(mle_r):
            label = "structural_or_task"
        elif _ok(oracle_r) and _ok(mle_r) and _ok(map_r) and _ok(em_r):
            label = "recoverable"
        elif _ok(oracle_r) and _ok(mle_r) and _ok(map_r) and not _ok(em_r):
            label = "optimizer_or_convergence"
        elif _ok(mle_r) and (not _ok(map_r) or not _ok(em_r)):
            label = "prior_shrinkage"
        else:
            label = "mixed"
        rows.append(
            {
                "model": model,
                "param": param,
                "oracle_r": oracle_r,
                "mle_r": mle_r,
                "map_r": map_r,
                "em_r": em_r,
                "em_convergence_rate": conv,
                "classification": label,
                "min_r": min_r,
            }
        )
    return pd.DataFrame(rows)


def recovery_evidence_status(
    metrics_df: pd.DataFrame,
    *,
    min_subjects: int = 40,
    min_seeds: int = 3,
    min_convergence_rate: float = 0.8,
) -> dict:
    """Return an explicit sufficiency judgment for recovery claims."""
    n_subjects = int(metrics_df.get("n_subjects", pd.Series([0])).max() or 0)
    n_seeds = int(metrics_df["seed"].nunique()) if "seed" in metrics_df.columns else 1
    if "convergence" in metrics_df.columns:
        conv = float(metrics_df["convergence"].mean())
    else:
        conv = np.nan
    reasons = []
    if n_subjects and n_subjects < min_subjects:
        reasons.append(f"n_subjects={n_subjects}<{min_subjects}")
    if n_seeds < min_seeds:
        reasons.append(f"n_seeds={n_seeds}<{min_seeds}")
    if conv == conv and conv < min_convergence_rate:
        reasons.append(f"convergence_rate={conv:.2f}<{min_convergence_rate}")
    ok = len(reasons) == 0
    return {
        "sufficient": ok,
        "status": "ok" if ok else "insufficient_recovery_evidence",
        "reasons": reasons,
        "n_subjects": n_subjects,
        "n_seeds": n_seeds,
        "convergence_rate": conv,
    }


def summarize_parameter_recovery(
    pr_metrics: pd.DataFrame,
    *,
    converged_only: bool = False,
    scale: str | None = "natural",
) -> pd.DataFrame:
    """Mean recovery metrics per model/parameter across seeds/rounds.

    When ``converged_only=True``, non-converged EM fits are excluded from the
    primary summary (all-fit diagnostics remain available on the raw table).
    """
    if pr_metrics is None or len(pr_metrics) == 0:
        return pd.DataFrame()
    df = pr_metrics
    if scale is not None and "scale" in df.columns:
        df = df[df["scale"] == scale]
    if converged_only and "convergence" in df.columns:
        # Keep rows where convergence is True; for estimators without a flag, keep all.
        mask = df["convergence"].fillna(True).astype(bool)
        df = df[mask]
    if len(df) == 0:
        return pd.DataFrame()
    group_cols = ["model", "param"]
    agg_kwargs: dict = dict(
        mean_r=("correlation", "mean"),
        min_r=("correlation", "min"),
        mean_spearman=("spearman", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_slope=("calibration_slope", "mean"),
        mean_shrinkage=("shrinkage_ratio", "mean"),
        n_rows=("correlation", "size"),
    )
    if "convergence" in df.columns:
        agg_kwargs["convergence_rate"] = ("convergence", "mean")
    agg = df.groupby(group_cols, as_index=False).agg(**agg_kwargs)
    return agg


def summarize_parameter_recovery_all_and_converged(pr_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return both all-fit and converged-only primary summaries."""
    return {
        "all_fits": summarize_parameter_recovery(pr_metrics, converged_only=False),
        "converged": summarize_parameter_recovery(pr_metrics, converged_only=True),
    }


def model_recovery_self_hit_rates(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Diagonal (self) win rates from a model-recovery summary table."""
    if summary_df is None or len(summary_df) == 0:
        return pd.DataFrame()
    rows = []
    for model in summary_df["Simulated"].unique():
        diag = summary_df[
            (summary_df["Simulated"] == model) & (summary_df["Estimated"] == model)
        ]
        if len(diag) == 0:
            continue
        nrounds = int(diag["nrounds"].iloc[0])
        bestlme = float(diag["bestlme"].iloc[0])
        bestbic = float(diag["bestbic"].iloc[0])
        rows.append(
            {
                "model": model,
                "nrounds": nrounds,
                "bestlme": bestlme,
                "bestbic": bestbic,
                "lme_hit_rate": bestlme / nrounds if nrounds else np.nan,
                "bic_hit_rate": bestbic / nrounds if nrounds else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_real_data_gates(
    pr_metrics: pd.DataFrame,
    model_recovery_summary: pd.DataFrame,
    *,
    min_param_r: float = 0.6,
    min_model_hit_rate: float = 0.5,
    min_convergence_rate: float = 0.8,
    min_calibration_slope: float = 0.2,
    min_shrinkage_ratio: float = 0.15,
    criterion: str = "lme",
    converged_only: bool = True,
) -> pd.DataFrame:
    """Decide which models may be fit to real data / promoted.

    Gates:
    - every free parameter has mean Pearson r > ``min_param_r`` on converged fits
    - EM convergence rate ≥ ``min_convergence_rate``
    - no severe variance collapse (mean calibration slope and shrinkage above floors)
    - the model recovers itself at least ``min_model_hit_rate`` of rounds (LME/BIC)
    """
    pr_sum = summarize_parameter_recovery(pr_metrics, converged_only=converged_only)
    # Convergence rate is assessed on the full (unfiltered) table.
    pr_all = summarize_parameter_recovery(pr_metrics, converged_only=False)
    hits = model_recovery_self_hit_rates(model_recovery_summary)
    if len(pr_sum) == 0:
        return pd.DataFrame()

    hit_col = "lme_hit_rate" if criterion == "lme" else "bic_hit_rate"
    rows = []
    models = sorted(pr_sum["model"].unique())
    hit_map = hits.set_index("model") if len(hits) else None

    for model in models:
        sub = pr_sum[pr_sum["model"] == model]
        worst_r = float(sub["mean_r"].min()) if len(sub) else np.nan
        all_params_ok = bool((sub["mean_r"] > min_param_r).all()) if len(sub) else False
        mean_slope = float(sub["mean_slope"].mean()) if "mean_slope" in sub else np.nan
        mean_shrink = float(sub["mean_shrinkage"].mean()) if "mean_shrinkage" in sub else np.nan
        # Prefer unfiltered convergence rate (includes failed EM starts).
        conv_rate = np.nan
        if "convergence_rate" in pr_all.columns:
            conv_block = pr_all.loc[pr_all["model"] == model, "convergence_rate"]
            if len(conv_block):
                conv_rate = float(conv_block.mean())
        elif "convergence_rate" in sub.columns and sub["convergence_rate"].notna().any():
            conv_rate = float(sub["convergence_rate"].mean())
        conv_ok = bool(np.isfinite(conv_rate) and conv_rate >= min_convergence_rate)
        collapse_ok = True
        collapse_reasons = []
        if np.isfinite(mean_slope) and mean_slope < min_calibration_slope:
            collapse_ok = False
            collapse_reasons.append(f"mean_slope={mean_slope:.3f}<{min_calibration_slope}")
        if np.isfinite(mean_shrink) and mean_shrink < min_shrinkage_ratio:
            collapse_ok = False
            collapse_reasons.append(f"mean_shrinkage={mean_shrink:.3f}<{min_shrinkage_ratio}")
        if hit_map is not None and model in hit_map.index:
            hit_rate = float(hit_map.loc[model, hit_col])
        else:
            hit_rate = np.nan
        model_ok = bool(np.isfinite(hit_rate) and hit_rate >= min_model_hit_rate)
        reasons = []
        if not all_params_ok:
            bad = sub.loc[sub["mean_r"] <= min_param_r, ["param", "mean_r"]]
            reasons.append(
                "param_r_fail:"
                + ",".join(f"{p}={r:.3f}" for p, r in bad.itertuples(index=False))
            )
        if not conv_ok:
            reasons.append(f"convergence_rate={conv_rate}")
        if not collapse_ok:
            reasons.append("variance_collapse:" + ",".join(collapse_reasons))
        if not model_ok:
            reasons.append(f"model_recovery_{criterion}={hit_rate}")
        eligible = all_params_ok and conv_ok and collapse_ok and model_ok
        rows.append(
            {
                "model": model,
                "worst_param_r": worst_r,
                "all_params_r_gt_threshold": all_params_ok,
                "convergence_rate": conv_rate,
                "convergence_ok": conv_ok,
                "mean_calibration_slope": mean_slope,
                "mean_shrinkage_ratio": mean_shrink,
                "no_severe_collapse": collapse_ok,
                "model_hit_rate": hit_rate,
                "model_recovers": model_ok,
                "eligible_for_real_data": eligible,
                "promote": eligible,
                "min_param_r": min_param_r,
                "min_model_hit_rate": min_model_hit_rate,
                "min_convergence_rate": min_convergence_rate,
                "criterion": criterion,
                "reasons": "; ".join(reasons) if reasons else "",
            }
        )
    return pd.DataFrame(rows)


def evaluate_promotion_decisions(
    pr_metrics: pd.DataFrame,
    model_recovery_summary: pd.DataFrame,
    *,
    legacy_predictive_nll: dict[str, float] | None = None,
    candidate_predictive_nll: dict[str, float] | None = None,
    max_nll_degradation: float = 0.05,
    **gate_kwargs,
) -> pd.DataFrame:
    """Promotion CSV: recovery gates plus optional predictive-NLL vs legacy family.

    ``legacy_predictive_nll`` / ``candidate_predictive_nll`` map model_id -> mean NLL.
    A candidate fails if its NLL is worse than its legacy counterpart by more than
    ``max_nll_degradation`` (absolute, natural nats/trial units as provided).
    """
    gates = evaluate_real_data_gates(pr_metrics, model_recovery_summary, **gate_kwargs)
    if len(gates) == 0:
        return gates
    legacy_predictive_nll = legacy_predictive_nll or {}
    candidate_predictive_nll = candidate_predictive_nll or {}
    # Family pairing for predictive checks.
    family_legacy = {
        "vkf_omega": "vkf",
        "vkf_fixed_v0": "vkf",
        "vkf_w0": "vkf",
        "rwus_upe": "rwus",
        "rw_signed_asym": "rw_signed",
        "rw_asym": "rw",
    }
    rows = []
    for _, row in gates.iterrows():
        model = row["model"]
        pred_ok = True
        pred_note = ""
        legacy = family_legacy.get(model)
        if legacy and model in candidate_predictive_nll and legacy in legacy_predictive_nll:
            cand_nll = float(candidate_predictive_nll[model])
            leg_nll = float(legacy_predictive_nll[legacy])
            deg = cand_nll - leg_nll
            pred_ok = deg <= max_nll_degradation
            pred_note = f"nll_delta_vs_{legacy}={deg:.4f}"
            if not pred_ok:
                pred_note += f">max={max_nll_degradation}"
        promote = bool(row["eligible_for_real_data"] and pred_ok)
        reasons = row["reasons"]
        if not pred_ok:
            reasons = (reasons + "; " if reasons else "") + f"predictive_fail:{pred_note}"
        rec = dict(row)
        rec["predictive_ok"] = pred_ok
        rec["predictive_note"] = pred_note
        rec["promote"] = promote
        rec["eligible_for_real_data"] = promote
        rec["reasons"] = reasons
        rows.append(rec)
    return pd.DataFrame(rows)


def eligible_models_for_real_data(gate_df: pd.DataFrame) -> list[str]:
    if gate_df is None or len(gate_df) == 0:
        return []
    return list(gate_df.loc[gate_df["eligible_for_real_data"], "model"])


def behavioral_sensitivity_summary(sim: dict) -> dict:
    """Cheap behavioral summaries for sensitivity / identifiability plots."""
    choices = sim["choices"]
    rewards = sim["rewards"]
    nsubj, T = choices.shape
    stay = []
    win_stay = []
    lose_shift = []
    for s in range(nsubj):
        a = choices[s]
        r = rewards[s]
        stay.append(float(np.mean(a[1:] == a[:-1])))
        wins = r[:-1] == 1
        losses = r[:-1] == 0
        if wins.any():
            win_stay.append(float(np.mean(a[1:][wins] == a[:-1][wins])))
        if losses.any():
            lose_shift.append(float(np.mean(a[1:][losses] != a[:-1][losses])))
    return {
        "mean_stay": float(np.mean(stay)),
        "mean_win_stay": float(np.mean(win_stay)) if win_stay else np.nan,
        "mean_lose_shift": float(np.mean(lose_shift)) if lose_shift else np.nan,
        "mean_reward": float(np.mean(rewards)),
        "choice_entropy": float(
            np.mean([-np.sum(p * np.log(p + 1e-12)) for p in [np.bincount(choices[s], minlength=2) / T for s in range(nsubj)]])
        ),
    }


def mean_predictive_nll(
    model_name: str,
    all_data: list,
    *,
    fit_kwargs: dict | None = None,
    n_starts: int = 1,
) -> float:
    """Fit model and return mean subject NLL (nats) at estimated subject params."""
    em, diag = fit_hierarchical(
        model_name,
        all_data,
        fit_kwargs=fit_kwargs,
        n_starts=n_starts,
        require_convergence=False,
    )
    space = space_for(MODEL_PARAM_NAMES[model_name])
    nlls = []
    for i, args in enumerate(all_data):
        if not isinstance(args, (list, tuple)):
            args = (args,)
        x = space.to_latent(diag.subject_params[i])
        nlls.append(float(em.fit_func(x, *args, prior=None, output="nll")))
    del em
    return float(np.mean(nlls)) if nlls else float("nan")


def run_redesign_benchmark(
    model_names: Sequence[str],
    *,
    seeds: Sequence[int],
    n_subjects: int = 30,
    ntrials: int = 80,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    n_starts: int = 3,
    run_identifiability: bool = True,
    id_seeds: Sequence[int] | None = None,
    mi_rounds: int = 2,
    run_model_recovery_flag: bool = True,
) -> dict:
    """Equal-design recovery benchmark for legacy vs candidate redesign models.

    Writes into ``checkpoint_dir`` without touching other result trees. Returns
    dict with PR metrics, optional identifiability / model-recovery tables, and
    a promotion-decision frame.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pr_df = run_parameter_recovery_batch(
        model_names,
        seeds=seeds,
        n_subjects=n_subjects,
        ntrials=ntrials,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        require_convergence=False,
        n_starts=n_starts,
        use_production_budget=True,
    )

    id_frames = []
    if run_identifiability:
        id_path = checkpoint_dir / "identifiability_ladder.csv" if checkpoint_dir else None
        if resume and id_path is not None and id_path.exists():
            id_frames.append(pd.read_csv(id_path))
            done = set(zip(id_frames[-1]["model"], id_frames[-1]["seed"]))
        else:
            done = set()
        for model_name in model_names:
            for seed in id_seeds or seeds[:1]:
                if (model_name, int(seed)) in done:
                    continue
                print(f"identifiability ladder: {model_name} seed={seed}")
                id_df = run_identifiability_study(
                    model_name,
                    n_subjects=n_subjects,
                    ntrials=ntrials,
                    seeds=(int(seed),),
                    n_em_starts=n_starts,
                    use_production_budget=True,
                )
                id_frames.append(id_df)
                if id_path is not None:
                    pd.concat(id_frames, ignore_index=True).to_csv(id_path, index=False)

    id_df = pd.concat(id_frames, ignore_index=True) if id_frames else pd.DataFrame()
    ladder = classify_recovery_ladder(id_df) if len(id_df) else pd.DataFrame()
    if checkpoint_dir is not None and len(ladder):
        ladder.to_csv(checkpoint_dir / "recovery_ladder_classification.csv", index=False)

    rounds_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    if run_model_recovery_flag:
        rounds_df, summary_df = run_model_recovery(
            model_names,
            n_subjects=n_subjects,
            ntrials=ntrials,
            nrounds=mi_rounds,
            seed=0,
            compute_bic=False,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            fit_kwargs=dict(njobs=1, verbose=0),
        )

    promotion = evaluate_promotion_decisions(pr_df, summary_df)
    if checkpoint_dir is not None and len(promotion):
        promotion.to_csv(checkpoint_dir / "promotion_decisions.csv", index=False)
        summaries = summarize_parameter_recovery_all_and_converged(pr_df)
        if len(summaries["converged"]):
            summaries["converged"].to_csv(
                checkpoint_dir / "parameter_recovery_summary_converged.csv", index=False
            )
        if len(summaries["all_fits"]):
            summaries["all_fits"].to_csv(
                checkpoint_dir / "parameter_recovery_summary_all.csv", index=False
            )

    return {
        "parameter_recovery": pr_df,
        "identifiability": id_df,
        "ladder": ladder,
        "model_recovery_rounds": rounds_df,
        "model_recovery_summary": summary_df,
        "promotion": promotion,
    }
