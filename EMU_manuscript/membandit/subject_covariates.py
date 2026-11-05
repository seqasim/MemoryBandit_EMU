"""
Subject-level covariate comparison helpers for connectivity notebooks.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, eye, kron
from scipy.stats import pearsonr, spearmanr


def load_acc_reinstatement_table(base_dir: str) -> pd.DataFrame:
    """Load ACC RPE reinstatement table used in cross-decoding analyses."""
    acc_reinstatement_csv = os.path.join(
        base_dir,
        "scratch/MemoryBandit/MVPA/crossdecoding/rpe_category/"
        "ACC_rpe_category_HFA_subject_reinstatement.csv",
    )
    if not os.path.isfile(acc_reinstatement_csv):
        raise FileNotFoundError(f"ACC reinstatement CSV not found: {acc_reinstatement_csv}")
    acc_tbl = pd.read_csv(acc_reinstatement_csv)
    required = ["Subject", "MeanAccuracy", "NumElectrodes"]
    missing = [c for c in required if c not in acc_tbl.columns]
    if missing:
        raise ValueError(f"ACC reinstatement table missing {missing}")
    out = acc_tbl.rename(
        columns={
            "MeanAccuracy": "ACC_MeanAccuracy",
            "NumElectrodes": "ACC_NumElectrodes",
        }
    )[["Subject", "ACC_MeanAccuracy", "ACC_NumElectrodes"]]
    return out


def load_hpc_memory_classification_table(
    base_dir: str,
    freq: str = "theta",
) -> pd.DataFrame:
    """Load HPC memory-outcome decoding table (hit/miss) for subject-level covariates."""
    freq = str(freq).strip().lower()
    if freq not in {"theta", "hfa"}:
        raise ValueError("freq must be 'theta' or 'hfa'")
    hpc_memory_csv = os.path.join(
        base_dir,
        "scratch/MemoryBandit/MVPA/decoding_cue/hit_miss/"
        f"HPC_memory_outcome_hit_miss_{freq}_subject_reinstatement.csv",
    )
    if not os.path.isfile(hpc_memory_csv):
        raise FileNotFoundError(f"HPC memory classification CSV not found: {hpc_memory_csv}")
    hpc_tbl = pd.read_csv(hpc_memory_csv)
    required = ["Subject", "MeanAccuracy", "NumElectrodes"]
    missing = [c for c in required if c not in hpc_tbl.columns]
    if missing:
        raise ValueError(f"HPC memory classification table missing {missing}")
    out = hpc_tbl.rename(
        columns={
            "MeanAccuracy": "HPC_MemoryClassAccuracy",
            "NumElectrodes": "HPC_MemoryClass_NumElectrodes",
        }
    )[["Subject", "HPC_MemoryClassAccuracy", "HPC_MemoryClass_NumElectrodes"]]
    return out


def load_acc_hpc_hit_rsa_table(
    base_dir: str,
    n_jobs: int = 4,
    alpha: float = 0.05,
    n_permutations: int = 5000,
    seed: int = 42,
    tail: int = 1,
    hit_memory_label: str = "remembered",
) -> pd.DataFrame:
    """Load ACC->HPC hit-RSA cluster means from group_results pickle."""
    candidates = [
        os.path.join(
            base_dir,
            "scratch/MemoryBandit/RSA_MVTemplate/group_level/group_results_ACC_to_HPC.pkl",
        ),
        os.path.join(
            base_dir,
            "scratch/MemoryBandit/RSA/crossregion/group_results_ACC_to_HPC.pkl",
        ),
    ]
    rsa_group_results_pkl = next((p for p in candidates if os.path.isfile(p)), candidates[0])
    if not os.path.isfile(rsa_group_results_pkl):
        raise FileNotFoundError(f"RSA group results PKL not found (tried {candidates})")

    with open(rsa_group_results_pkl, "rb") as f:
        rsa_group_results = pickle.load(f)
    required = [
        "observed_diffs_stack",
        "feedback_times",
        "cue_times",
        "subject_results",
        "successful_subjects",
        "n_subjects",
    ]
    missing = [k for k in required if k not in rsa_group_results]
    if missing:
        raise ValueError(f"RSA group results missing {missing}")

    X = np.asarray(rsa_group_results["observed_diffs_stack"], dtype=float)
    feedback_times = np.asarray(rsa_group_results["feedback_times"], dtype=float)
    cue_times = np.asarray(rsa_group_results["cue_times"], dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected observed_diffs_stack 3D; got {X.shape}")
    if X.shape[1] != len(feedback_times) or X.shape[2] != len(cue_times):
        raise ValueError(
            f"RSA stack {X.shape} vs feedback {len(feedback_times)} / cue {len(cue_times)}"
        )

    n_fb, n_cue = X.shape[1], X.shape[2]
    fb_adjacency = mne.stats.combine_adjacency(n_fb)
    cue_adjacency = mne.stats.combine_adjacency(n_cue)
    adjacency_2d = csr_matrix(kron(eye(n_fb), cue_adjacency) + kron(fb_adjacency, eye(n_cue)))
    X_flat = X.reshape(X.shape[0], -1)
    _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(
        X_flat,
        n_permutations=n_permutations,
        threshold=None,
        adjacency=adjacency_2d,
        tail=tail,
        n_jobs=max(1, int(n_jobs)),
        seed=seed,
        verbose=False,
    )
    sig_cluster_indices = [i for i, pval in enumerate(cluster_pv) if pval < alpha]
    if len(sig_cluster_indices) != 1:
        raise ValueError(
            f"Expected exactly 1 significant ACC->HPC cluster at alpha={alpha}; "
            f"found {len(sig_cluster_indices)}."
        )
    sig_cluster = clusters[sig_cluster_indices[0]]
    sig_mask_flat = np.zeros(n_fb * n_cue, dtype=bool)
    sig_mask_flat[sig_cluster[0]] = True
    sig_cluster_mask = sig_mask_flat.reshape(n_fb, n_cue)

    rows = []
    for subject_result in rsa_group_results["subject_results"]:
        observed = subject_result["observed"]
        subject = observed["subject"]
        category_heatmaps = observed.get("category_heatmaps", {})
        if hit_memory_label not in category_heatmaps:
            raise ValueError(f"Subject {subject} missing category_heatmaps[{hit_memory_label!r}].")
        hit_heatmap = np.asarray(category_heatmaps[hit_memory_label], dtype=float)
        if hit_heatmap.shape != sig_cluster_mask.shape:
            raise ValueError(
                f"Subject {subject} heatmap {hit_heatmap.shape} vs mask {sig_cluster_mask.shape}"
            )
        rows.append(
            {
                "Subject": subject,
                "ACC_HPC_HitRSA_cluster_mean": float(np.nanmean(hit_heatmap[sig_cluster_mask])),
            }
        )
    return pd.DataFrame(rows).sort_values("Subject").reset_index(drop=True)


def _extract_corr(res) -> Tuple[float, float]:
    if hasattr(res, "statistic") and hasattr(res, "pvalue"):
        return float(res.statistic), float(res.pvalue)
    return float(res[0]), float(res[1])


def run_correlation_analysis(
    x: np.ndarray,
    y: np.ndarray,
    corr_type: str = "spearman",
    run_secondary_pearson: bool = True,
) -> Dict[str, float]:
    """Compute per-panel correlation summary."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[mask], y[mask]
    n = int(len(x_valid))
    if n < 3:
        raise ValueError(f"Need at least 3 valid observations; got {n}.")
    rho_s, p_s = _extract_corr(spearmanr(x_valid, y_valid, nan_policy="omit"))
    if run_secondary_pearson or corr_type.lower() == "pearson":
        r_p, p_p = _extract_corr(pearsonr(x_valid, y_valid))
    else:
        r_p, p_p = np.nan, np.nan
    corr_type_norm = corr_type.lower()
    if corr_type_norm == "spearman":
        primary_r, primary_p = rho_s, p_s
    elif corr_type_norm == "pearson":
        primary_r, primary_p = r_p, p_p
    else:
        raise ValueError(f"Unsupported corr_type: {corr_type}")
    return {
        "n": n,
        "spearman_rho": rho_s,
        "spearman_p": p_s,
        "pearson_r": r_p,
        "pearson_p": p_p,
        "primary_corr_type": corr_type_norm,
        "primary_r": primary_r,
        "primary_p": primary_p,
    }


def _fit_and_plot_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 2 or len(np.unique(x)) < 2:
        return
    coeff = np.polyfit(x, y, 1)
    xfit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(xfit, np.polyval(coeff, xfit), color="#222222", linewidth=1.2)


def correlate_subject_metric(
    x_tbl: pd.DataFrame,
    metric_tbl: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    rpe_cats: Iterable[str],
    plot_savepath: str,
    fig_stem: str,
    title: str,
    corr_type: str = "spearman",
    label_points: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join, correlate per RPE category, and plot 3-panel subject scatter."""
    rpe_cats = tuple(rpe_cats)
    merged = x_tbl.merge(metric_tbl, on="Subject", how="inner")
    print(f"\n{y_label}: inner-join N subjects = {merged['Subject'].nunique()}  rows = {len(merged)}")
    rows = []
    for cat in rpe_cats:
        panel = merged[merged["rpe_cat"] == cat]
        x = panel[x_col].to_numpy(dtype=float)
        y = panel[y_col].to_numpy(dtype=float)
        rr = run_correlation_analysis(x, y, corr_type=corr_type)
        rr.update({"rpe_cat": cat, "target": y_col})
        rows.append(rr)
        print(
            f"  {cat}: spearman r={rr['spearman_rho']:.3f} p={rr['spearman_p']:.3g}  "
            f"pearson r={rr['pearson_r']:.3f} p={rr['pearson_p']:.3g}  n={rr['n']}"
        )
    results_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(rpe_cats), figsize=(6 * len(rpe_cats), 5), constrained_layout=True)
    if len(rpe_cats) == 1:
        axes = [axes]
    for ax, cat in zip(axes, rpe_cats):
        panel = merged[merged["rpe_cat"] == cat].copy()
        panel = panel[np.isfinite(panel[x_col]) & np.isfinite(panel[y_col])]
        ax.set_title(cat)
        if len(panel) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        x = panel[x_col].to_numpy(dtype=float)
        y = panel[y_col].to_numpy(dtype=float)
        sns.scatterplot(x=x, y=y, ax=ax, s=52, color="#4c78a8", edgecolor="black", linewidth=0.7)
        _fit_and_plot_line(ax, x, y)
        if label_points:
            for row in panel.itertuples(index=False):
                ax.text(getattr(row, x_col), getattr(row, y_col), f" {row.Subject}", fontsize=8)
        res_match = results_df[results_df["rpe_cat"] == cat]
        if len(res_match):
            rr = res_match.iloc[0]
            ann = (
                f"{rr['primary_corr_type']} r={rr['primary_r']:.3f}, "
                f"p={rr['primary_p']:.3g}, n={int(rr['n'])}"
            )
            ax.text(0.02, 0.98, ann, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)
    fig.suptitle(title, fontsize=14)
    fig_path = os.path.join(plot_savepath, f"{fig_stem}.pdf")
    fig.savefig(fig_path, dpi=300)
    print("saved", fig_path)
    plt.show()
    return merged, results_df


__all__ = [
    "load_acc_reinstatement_table",
    "load_hpc_memory_classification_table",
    "load_acc_hpc_hit_rsa_table",
    "run_correlation_analysis",
    "correlate_subject_metric",
]
