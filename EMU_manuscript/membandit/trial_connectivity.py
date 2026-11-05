"""
Shared trial-level connectivity statistics and plotting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2, wilcoxon
from statsmodels.stats.multitest import multipletests


@dataclass
class TrialModelBundle:
    full_model: object
    null_model: object
    diagnostics: pd.DataFrame
    fixed_effects: pd.DataFrame
    pairwise_subject: pd.DataFrame


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def with_pair_id(df: pd.DataFrame, subject_col: str = "subject", pair_col: str = "pair_label") -> pd.DataFrame:
    require_columns(df, [subject_col, pair_col])
    out = df.copy()
    out["pair_id"] = out[subject_col].astype(str) + "::" + out[pair_col].astype(str)
    return out


def filter_min_per_category(
    df: pd.DataFrame,
    group_col: str,
    category_col: str,
    categories: Sequence[str],
    min_n: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    require_columns(df, [group_col, category_col])
    counts = df.groupby([group_col, category_col], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(categories), fill_value=0)
    keep = counts.index[(counts[list(categories)] >= int(min_n)).all(axis=1)].to_numpy()
    return df[df[group_col].isin(keep)].copy(), keep


def _fit_mixedlm(
    formula: str,
    data: pd.DataFrame,
    group_col: str,
    pair_vc: bool,
):
    if pair_vc:
        md = smf.mixedlm(
            formula,
            data=data,
            groups=data[group_col],
            re_formula="1",
            vc_formula={"pair": "0 + C(pair_id)"},
        )
    else:
        md = smf.mixedlm(
            formula,
            data=data,
            groups=data[group_col],
            re_formula="1",
        )
    return md.fit(method="lbfgs", maxiter=500, reml=False)


def fit_trial_models(
    df: pd.DataFrame,
    z_col: str,
    categories: Sequence[str] = ("negative", "neutral", "positive"),
    min_n: int = 5,
    subject_col: str = "subject",
    pair_col: str = "pair_label",
    pair_vc: bool = True,
) -> tuple[TrialModelBundle, pd.DataFrame]:
    require_columns(df, [subject_col, "rpe_cat", z_col])
    work = df.copy()
    work["rpe_cat"] = pd.Categorical(work["rpe_cat"], categories=list(categories), ordered=False)
    if pair_vc:
        require_columns(work, [pair_col])
        work = with_pair_id(work, subject_col=subject_col, pair_col=pair_col)
    work = work.dropna(subset=[z_col, "rpe_cat"])
    work, keep_subjects = filter_min_per_category(
        work, group_col=subject_col, category_col="rpe_cat", categories=categories, min_n=min_n
    )
    if len(work) == 0:
        raise RuntimeError("no rows remain after minimum-per-category filtering")

    full_formula = f"{z_col} ~ C(rpe_cat, Treatment('positive'))"
    null_formula = f"{z_col} ~ 1"
    full = _fit_mixedlm(full_formula, work, subject_col, pair_vc=pair_vc)
    null = _fit_mixedlm(null_formula, work, subject_col, pair_vc=pair_vc)

    lr = 2.0 * (float(full.llf) - float(null.llf))
    df_diff = int(len(full.fe_params) - len(null.fe_params))
    p_omni = float(chi2.sf(lr, df=max(df_diff, 1)))

    diag = pd.DataFrame(
        [
            {
                "n_rows": int(len(work)),
                "n_subjects": int(work[subject_col].nunique()),
                "n_pairs": int(work[pair_col].nunique()) if pair_col in work.columns else 1,
                "n_subjects_complete": int(len(keep_subjects)),
                "min_trials_per_category": int(min_n),
                "pair_variance_component": bool(pair_vc),
                "full_aic": float(full.aic),
                "full_bic": float(full.bic),
                "null_aic": float(null.aic),
                "null_bic": float(null.bic),
                "llf_full": float(full.llf),
                "llf_null": float(null.llf),
                "lr_stat": float(lr),
                "lr_df": int(df_diff),
                "lr_pvalue": float(p_omni),
                "converged_full": bool(getattr(full, "converged", False)),
                "converged_null": bool(getattr(null, "converged", False)),
            }
        ]
    )

    ci = full.conf_int()
    fe_rows = []
    for term, coef in full.params.items():
        fe_rows.append(
            {
                "term": str(term),
                "coef": float(coef),
                "std_err": float(full.bse.get(term, np.nan)),
                "z_value": float(full.tvalues.get(term, np.nan)),
                "p_value": float(full.pvalues.get(term, np.nan)),
                "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
                "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
            }
        )
    fe = pd.DataFrame(fe_rows)

    pairwise = paired_subject_contrasts(work, z_col=z_col, categories=categories, subject_col=subject_col)
    return TrialModelBundle(full, null, diag, fe, pairwise), work


def paired_subject_contrasts(
    df: pd.DataFrame,
    z_col: str,
    categories: Sequence[str] = ("negative", "neutral", "positive"),
    subject_col: str = "subject",
) -> pd.DataFrame:
    require_columns(df, [subject_col, "rpe_cat", z_col])
    subj = (
        df.groupby([subject_col, "rpe_cat"], observed=True)[z_col]
        .mean()
        .unstack("rpe_cat")
        .reindex(columns=list(categories))
    )
    pairs = [("negative", "positive"), ("negative", "neutral"), ("positive", "neutral")]
    rows = []
    for a, b in pairs:
        sub = subj[[a, b]].dropna()
        if len(sub) < 2:
            rows.append(
                {
                    "contrast": f"{a}-{b}",
                    "n_subjects": int(len(sub)),
                    "w_stat": np.nan,
                    "p_value": np.nan,
                    "mean_diff": np.nan,
                }
            )
            continue
        stat = wilcoxon(sub[a], sub[b], zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "contrast": f"{a}-{b}",
                "n_subjects": int(len(sub)),
                "w_stat": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "mean_diff": float((sub[a] - sub[b]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["pval_fdr"] = np.nan
    mask = out["p_value"].notna()
    if mask.any():
        _, qvals, _, _ = multipletests(out.loc[mask, "p_value"].to_numpy(float), method="fdr_bh")
        out.loc[mask, "pval_fdr"] = qvals
    return out


def plot_subject_overlay(
    df: pd.DataFrame,
    z_col: str,
    categories: Sequence[str],
    out_png: str | Path,
    out_pdf: str | Path,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    subject_col: str = "subject",
) -> None:
    require_columns(df, [subject_col, "rpe_cat", z_col])
    subj_means = (
        df.groupby([subject_col, "rpe_cat"], observed=True)[z_col]
        .mean()
        .unstack("rpe_cat")
        .reindex(columns=list(categories))
    )
    x = np.arange(len(categories), dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for _, row in subj_means.iterrows():
        ax.plot(x, row.to_numpy(float), color="0.75", lw=0.8, alpha=0.8)
    grand = subj_means.mean(axis=0)
    sem = subj_means.std(axis=0, ddof=1) / np.sqrt(max(1, len(subj_means)))
    for i, cat in enumerate(categories):
        ax.errorbar(
            x[i],
            float(grand[cat]),
            yerr=float(sem[cat]) if np.isfinite(sem[cat]) else np.nan,
            fmt="o",
            color=colors.get(str(cat), "k"),
            ms=8,
            capsize=3,
            label=str(cat),
            zorder=5,
        )
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_distribution(
    df: pd.DataFrame,
    z_col: str,
    categories: Sequence[str],
    out_png: str | Path,
    out_pdf: str | Path,
    title: str,
    ylabel: str,
    colors: dict[str, str],
) -> None:
    require_columns(df, ["rpe_cat", z_col, "subject"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    plot_df = df.copy()
    plot_df["rpe_cat"] = pd.Categorical(plot_df["rpe_cat"], categories=list(categories), ordered=False)
    sns.boxplot(
        data=plot_df,
        x="rpe_cat",
        y=z_col,
        hue="rpe_cat",
        order=list(categories),
        palette=[colors.get(str(c), "0.5") for c in categories],
        dodge=False,
        legend=False,
        width=0.55,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="rpe_cat",
        y=z_col,
        order=list(categories),
        hue="subject",
        dodge=False,
        alpha=0.25,
        size=2.2,
        linewidth=0,
        ax=ax,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("rpe_cat")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_pdf), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_trial_stats_and_plots(
    df: pd.DataFrame,
    z_col: str,
    metric_name: str,
    event: str,
    save_dir: str | Path,
    plot_dir: str | Path,
    categories: Sequence[str] = ("negative", "neutral", "positive"),
    min_n: int = 5,
    pair_vc: bool = True,
    colors: dict[str, str] | None = None,
    ylabel: str | None = None,
    title_prefix: str | None = None,
) -> TrialModelBundle:
    colors = colors or {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    ylabel = ylabel or f"{metric_name} z"
    title_prefix = title_prefix or metric_name
    save_dir = Path(save_dir)
    plot_dir = Path(plot_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    bundle, df_keep = fit_trial_models(
        df=df,
        z_col=z_col,
        categories=categories,
        min_n=min_n,
        pair_vc=pair_vc,
    )

    df_keep.to_csv(save_dir / f"{metric_name}_trialwise_{event}.csv", index=False)
    bundle.fixed_effects.to_csv(save_dir / f"{metric_name}_trial_model_coeffs_{event}.csv", index=False)
    bundle.diagnostics.to_csv(save_dir / f"{metric_name}_trial_model_diag_{event}.csv", index=False)
    bundle.pairwise_subject.to_csv(save_dir / f"{metric_name}_trial_pairwise_{event}.csv", index=False)

    base = f"{metric_name}_trial_{event}"
    plot_subject_overlay(
        df_keep,
        z_col=z_col,
        categories=categories,
        out_png=plot_dir / f"{base}_subject_overlay.png",
        out_pdf=plot_dir / f"{base}_subject_overlay.pdf",
        title=f"{title_prefix} · {event} · subject means",
        ylabel=ylabel,
        colors=colors,
    )
    if pair_vc:
        plot_pair_distribution(
            df_keep,
            z_col=z_col,
            categories=categories,
            out_png=plot_dir / f"{base}_pair_distribution.png",
            out_pdf=plot_dir / f"{base}_pair_distribution.pdf",
            title=f"{title_prefix} · {event} · pair distribution",
            ylabel=ylabel,
            colors=colors,
        )
    return bundle


__all__ = [
    "TrialModelBundle",
    "require_columns",
    "with_pair_id",
    "filter_min_per_category",
    "fit_trial_models",
    "paired_subject_contrasts",
    "plot_subject_overlay",
    "plot_pair_distribution",
    "run_trial_stats_and_plots",
]

