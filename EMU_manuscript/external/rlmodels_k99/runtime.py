"""Laptop-safe run configuration, thread controls, workload preview, and checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rlmodels_k99.registry import MODEL_ORDER


BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_blas_threads(n_threads: int = 1) -> dict[str, str]:
    """Cap BLAS/OpenMP threads to avoid oversubscription with joblib.

    Call early in a process (before heavy numpy/scipy imports if possible).
    Returns the env vars that were set.
    """
    n = max(1, int(n_threads))
    applied = {}
    for var in BLAS_THREAD_VARS:
        os.environ[var] = str(n)
        applied[var] = str(n)
    return applied


@dataclass(frozen=True)
class RunConfig:
    """Staged recovery/run settings for one notebook MODE."""

    mode: str
    n_jobs: int = 1
    blas_threads: int = 1
    ntrials: int = 80
    seed: int = 0
    # Recovery EM budget (screening). Real-data fits use real_fit_kwargs.
    recovery_mstep_maxit: int = 40
    recovery_max_restarts: int = 2
    real_mstep_maxit: int = 100
    real_max_restarts: int = 4
    verbose: int = 0
    # Parameter recovery: one explicit seed list (no seed×round product).
    pr_subjects: int = 20
    pr_seeds: tuple[int, ...] = (0, 1)
    # Model recovery
    mi_subjects: int = 20
    mi_rounds: int = 2
    mi_models: tuple[str, ...] = ()
    compute_bic_in_model_recovery: bool = False
    bic_nsamples: int = 200
    # Gates
    min_param_r: float = 0.6
    min_model_hit: float = 0.5
    min_convergence_rate: float = 0.8
    # Optional expensive stages
    run_identifiability: bool = False
    id_models: tuple[str, ...] = ()
    run_candidates: bool = False
    candidate_seeds: tuple[int, ...] = (0, 1)
    # Redesign benchmark (writes a separate results tree; does not overwrite laptop/)
    run_redesign: bool = False
    redesign_models: tuple[str, ...] = ()
    redesign_n_starts: int = 3
    result_subdir: str = ""
    # I/O
    show_plots: bool = False
    resume: bool = True
    require_full_confirmation: bool = False

    def recovery_fit_kwargs(self) -> dict[str, Any]:
        return dict(
            mstep_maxit=self.recovery_mstep_maxit,
            max_restarts=self.recovery_max_restarts,
            verbose=self.verbose,
            njobs=self.n_jobs,
            seed=self.seed,
        )

    def real_fit_kwargs(self) -> dict[str, Any]:
        return dict(
            mstep_maxit=self.real_mstep_maxit,
            max_restarts=self.real_max_restarts,
            verbose=self.verbose,
            njobs=self.n_jobs,
            seed=self.seed,
        )


def resolve_mode(mode: str | None = None) -> str:
    """Default to laptop-safe local mode unless RLMODELS_MODE/full is set."""
    raw = mode if mode is not None else os.environ.get("RLMODELS_MODE", "laptop")
    return str(raw).strip().lower()


def build_run_config(
    mode: str | None = None,
    *,
    model_order: Sequence[str] | None = None,
    n_jobs: int | None = None,
    show_plots: bool | None = None,
) -> RunConfig:
    """Build a RunConfig for smoke | laptop | diagnostic | full."""
    mode = resolve_mode(mode)
    models = tuple(model_order or MODEL_ORDER)
    env_jobs = os.environ.get("RLMODELS_NJOBS")
    if n_jobs is None and env_jobs is not None:
        n_jobs = int(env_jobs)

    if mode == "smoke":
        cfg = RunConfig(
            mode=mode,
            n_jobs=1 if n_jobs is None else n_jobs,
            recovery_mstep_maxit=15,
            recovery_max_restarts=2,
            real_mstep_maxit=15,
            real_max_restarts=2,
            pr_subjects=10,
            pr_seeds=(0,),
            mi_subjects=8,
            mi_rounds=1,
            mi_models=("wsls", "rw", "rwf", "bayes_fixed_reward"),
            compute_bic_in_model_recovery=False,
            bic_nsamples=100,
            run_identifiability=False,
            run_candidates=False,
            show_plots=False if show_plots is None else show_plots,
            resume=True,
        )
    elif mode == "laptop":
        # Staged local default: LME gates, no MR BIC, modest EM budget, few threads.
        cfg = RunConfig(
            mode=mode,
            n_jobs=1 if n_jobs is None else n_jobs,
            recovery_mstep_maxit=40,
            recovery_max_restarts=2,
            real_mstep_maxit=80,
            real_max_restarts=3,
            pr_subjects=30,
            pr_seeds=(0, 1, 2),
            mi_subjects=25,
            mi_rounds=2,
            mi_models=models,
            compute_bic_in_model_recovery=False,
            bic_nsamples=500,
            run_identifiability=False,
            id_models=(),
            run_candidates=False,
            show_plots=False if show_plots is None else show_plots,
            resume=True,
        )
    elif mode == "diagnostic":
        cfg = RunConfig(
            mode=mode,
            n_jobs=2 if n_jobs is None else n_jobs,
            recovery_mstep_maxit=60,
            recovery_max_restarts=3,
            real_mstep_maxit=100,
            real_max_restarts=4,
            pr_subjects=40,
            pr_seeds=(0, 1, 2),
            mi_subjects=30,
            mi_rounds=2,
            mi_models=models,
            compute_bic_in_model_recovery=False,
            bic_nsamples=500,
            run_identifiability=True,
            id_models=("rw", "rw_signed", "rw_side_bias", "rw_sticky", "bayes_fixed_reward"),
            run_candidates=True,
            candidate_seeds=(0, 1, 2),
            show_plots=False if show_plots is None else show_plots,
            resume=True,
        )
    elif mode == "full":
        # Publication scale, but still staged: no BIC inside the MR grid.
        cfg = RunConfig(
            mode=mode,
            n_jobs=2 if n_jobs is None else n_jobs,
            recovery_mstep_maxit=80,
            recovery_max_restarts=3,
            real_mstep_maxit=150,
            real_max_restarts=5,
            pr_subjects=80,
            # Explicit replications (replaces old seeds×rounds product).
            pr_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
            mi_subjects=50,
            mi_rounds=5,
            mi_models=models,
            compute_bic_in_model_recovery=False,
            bic_nsamples=2000,
            run_identifiability=True,
            id_models=models,
            run_candidates=True,
            candidate_seeds=(0, 1, 2),
            show_plots=False if show_plots is None else show_plots,
            resume=True,
            require_full_confirmation=True,
        )
    elif mode == "redesign":
        # Equal-design recoverable-model redesign; new result directory only.
        from rlmodels_k99.registry import REDESIGN_BENCHMARK_ORDER

        redesign = tuple(REDESIGN_BENCHMARK_ORDER)
        cfg = RunConfig(
            mode=mode,
            n_jobs=1 if n_jobs is None else n_jobs,
            recovery_mstep_maxit=100,
            recovery_max_restarts=4,
            real_mstep_maxit=100,
            real_max_restarts=4,
            pr_subjects=30,
            pr_seeds=(0, 1, 2),
            mi_subjects=25,
            mi_rounds=2,
            mi_models=redesign,
            compute_bic_in_model_recovery=False,
            bic_nsamples=500,
            run_identifiability=True,
            id_models=("wsls", "rw", "rwus_upe", "vkf_omega", "bayes_fixed_reward", "rw_asym"),
            run_candidates=False,
            run_redesign=True,
            redesign_models=redesign,
            redesign_n_starts=3,
            result_subdir="redesign_v2",
            min_convergence_rate=0.8,
            show_plots=False if show_plots is None else show_plots,
            resume=False,
        )
    else:
        raise ValueError(
            f"Unknown MODE={mode!r}; choose smoke|laptop|diagnostic|full|redesign"
        )
    return cfg


def estimate_workload(
    cfg: RunConfig,
    *,
    n_main_models: int | None = None,
    n_candidates: int = 0,
) -> dict[str, Any]:
    """Estimate hierarchical fit counts and optional BIC evaluation counts."""
    n_main = int(n_main_models if n_main_models is not None else len(MODEL_ORDER))
    n_mi = len(cfg.mi_models) if cfg.mi_models else n_main
    pr_fits = n_main * len(cfg.pr_seeds)
    mr_fits = cfg.mi_rounds * n_mi * n_mi
    mr_bic_evals = 0
    if cfg.compute_bic_in_model_recovery:
        mr_bic_evals = mr_fits * cfg.mi_subjects * cfg.bic_nsamples
    id_fits = len(cfg.id_models) * min(3, len(cfg.pr_seeds)) if cfg.run_identifiability else 0
    cand_fits = n_candidates * len(cfg.candidate_seeds) if cfg.run_candidates else 0
    return {
        "mode": cfg.mode,
        "n_jobs": cfg.n_jobs,
        "blas_threads": cfg.blas_threads,
        "parameter_recovery_fits": pr_fits,
        "model_recovery_fits": mr_fits,
        "model_recovery_bic_evals": mr_bic_evals,
        "identifiability_em_fits": id_fits,
        "candidate_pr_fits": cand_fits,
        "total_em_fits_estimate": pr_fits + mr_fits + id_fits + cand_fits,
        "compute_bic_in_model_recovery": cfg.compute_bic_in_model_recovery,
        "resume": cfg.resume,
        "notes": (
            "BIC is deferred to eligible finalists when compute_bic_in_model_recovery=False."
        ),
    }


def print_workload_preview(workload: dict[str, Any]) -> None:
    print("=== Workload preview ===")
    for k in (
        "mode",
        "n_jobs",
        "blas_threads",
        "parameter_recovery_fits",
        "model_recovery_fits",
        "model_recovery_bic_evals",
        "identifiability_em_fits",
        "candidate_pr_fits",
        "total_em_fits_estimate",
        "compute_bic_in_model_recovery",
        "resume",
        "notes",
    ):
        if k in workload:
            print(f"  {k}: {workload[k]}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def checkpoint_path(checkpoint_dir: Path, stem: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{stem}.csv"


def load_checkpoint_csv(path: Path):
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def save_checkpoint_csv(df, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_checkpoint_row(path: Path, row: dict[str, Any], *, key_cols: Sequence[str]) -> None:
    """Append a row unless a matching key already exists."""
    import pandas as pd

    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        mask = pd.Series(True, index=old.index)
        for c in key_cols:
            mask &= old[c] == row[c]
        if mask.any():
            return
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def has_checkpoint_keys(path: Path, key_vals: dict[str, Any]) -> bool:
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return False
    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    for k, v in key_vals.items():
        if k not in df.columns:
            return False
        mask &= df[k] == v
    return bool(mask.any())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def config_to_manifest(cfg: RunConfig, **extra) -> dict[str, Any]:
    d = asdict(cfg)
    d.update(extra)
    return d
