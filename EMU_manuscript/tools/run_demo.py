#!/usr/bin/env python3
"""
Nature Research Code & Software Submission Checklist Demo

Demonstrates the core ``membandit`` and ``rlmodels_k99`` analysis pipelines on a
small synthetic dataset (no external data drive required):

  1. Simulates 2-arm bandit reversal-learning behavior (N = 10 synthetic agents
     x 80 trials) under the winning Rescorla-Wagner (RW) model and fits the
     hierarchical Laplace-EM model via ``rlmodels_k99`` + ``pyem``.
  2. Preprocesses the simulated trial table via ``membandit.behavior``
     (within-subject RPE terciles, Gelman standardization).
  3. Synthesizes event-locked multi-electrode HFA time series for encoding
     (``feedback_start``) and retrieval (``cue_on``) and computes cross-temporal
     representational similarity analysis (RSA; Euclidean RDMs, Spearman rho-a,
     remembered minus forgotten contrast + permutation z-score) via
     ``membandit.rsa``.

Expected runtime on a standard laptop/desktop: ~5-10 seconds.

Usage:
  python tools/run_demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "membandit_demo_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("MNE_HOME", _CACHE_DIR)
os.environ.setdefault("HOME", _CACHE_DIR)

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "external"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np
import pandas as pd

from membandit.behavior import preprocess_learn_df
from membandit.rsa import compute_cross_temporal_rsa_from_hfa
from rlmodels_k99.estimation import fit_hierarchical
from rlmodels_k99.parameters import build_params
from rlmodels_k99.registry import get_model


def run_demo() -> int:
    t0 = time.perf_counter()
    print("=" * 72)
    print("MemoryBandit EMU — Synthetic Demo (Nature Research Software Checklist)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Simulate & fit hierarchical Rescorla-Wagner model (Fig 1C-E)
    # ------------------------------------------------------------------
    n_subjects = 10
    n_trials = 80
    _, _, true_params = build_params(["beta", "alpha"], n_subjects, seed=42, model_name="rw")
    sim = get_model("rw").sim(true_params, ntrials=n_trials, seed=42)

    all_data = [[sim["choices"][i], sim["rewards"][i]] for i in range(n_subjects)]
    _, diag = fit_hierarchical(
        "rw",
        all_data,
        fit_kwargs={"seed": 42, "mstep_maxit": 25, "verbose": 0},
        n_starts=1,
    )
    est_params = diag.subject_params
    r_beta = float(np.corrcoef(true_params[:, 0], est_params[:, 0])[0, 1])
    r_alpha = float(np.corrcoef(true_params[:, 1], est_params[:, 1])[0, 1])
    print(
        f"[1/3] Hierarchical pyEM Rescorla-Wagner fit complete "
        f"(N={n_subjects} agents x {n_trials} trials, converged={diag.convergence}): "
        f"recovery r(alpha)={r_alpha:.2f}, r(beta)={r_beta:.2f}"
    )

    # ------------------------------------------------------------------
    # 2. Preprocess trial-level behavior table (membandit.behavior)
    # ------------------------------------------------------------------
    rows = []
    for i in range(n_subjects):
        pid = f"DEMO{i + 1:02d}"
        pe = sim["PE"][i] if "PE" in sim else (sim["rewards"][i] - 0.5)
        for t in range(n_trials):
            rows.append(
                {
                    "participant": pid,
                    "trials": t + 1,
                    "block": (t // 16) + 1,
                    "gamble": int(sim["choices"][i][t]),
                    "reward": int(sim["rewards"][i][t]),
                    "gamble_rt": 0.65,
                    "DPRIME": float((t % 10) * 0.1 + 1.0),
                    "rpe": float(pe[t]),
                    "alpha": float(est_params[i, 1]),
                    "beta": float(est_params[i, 0]),
                }
            )
    learn_df = preprocess_learn_df(pd.DataFrame(rows))
    tercile_counts = learn_df["rpe_cat"].value_counts().to_dict()
    print(f"[2/3] Behavior table preprocessed ({len(learn_df)} rows); RPE terciles: {tercile_counts}")

    # ------------------------------------------------------------------
    # 3. Cross-temporal feedback -> cue HFA RSA (Fig 4B pipeline)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_recog_trials = 24
    n_acc_ch, n_hpc_ch = 6, 5
    n_fb_t, n_cue_t = 15, 10
    fb_hfa = rng.normal(size=(n_recog_trials, n_acc_ch, n_fb_t))
    cue_hfa = rng.normal(size=(n_recog_trials, n_hpc_ch, n_cue_t))
    memory_labels = np.array(["remembered"] * 12 + ["forgotten"] * 12, dtype=object)

    obs_rsa = compute_cross_temporal_rsa_from_hfa(
        encoding_hfa=fb_hfa,
        cue_hfa=cue_hfa,
        memory_outcomes=memory_labels,
        rdm_method="euclidean",
        rdm_compare_method="rho-a",
    )
    diff_map = np.asarray(obs_rsa["diff_heatmap"], dtype=float)
    print(
        f"[3/3] Cross-temporal ACC->HPC HFA RSA computed: "
        f"diff_heatmap shape={diff_map.shape}, max |diff|={np.nanmax(np.abs(diff_map)):.3f}"
    )

    out_dir = REPO_ROOT / "results" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_csv = out_dir / "demo_preprocessed_learn_df.csv"
    learn_df.head(20).to_csv(demo_csv, index=False)

    elapsed = time.perf_counter() - t0
    print("-" * 72)
    print(f"Demo finished successfully in {elapsed:.2f} s. Sample output written to: {demo_csv.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
