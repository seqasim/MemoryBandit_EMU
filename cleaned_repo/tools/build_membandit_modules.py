#!/usr/bin/env python3
"""
Build partitioned membandit modules from Notebooks/ source monoliths.

Strategy:
- Keep full source logic in membandit/_impl/ (patched imports + Subject delegates to behavior).
- Public modules re-export the symbols belonging to each use-case partition.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "Notebooks"
PKG = ROOT / "cleaned_repo" / "membandit"
IMPL = PKG / "_impl"

IO_FROM_MV = [
    "Subject",
    "load_subject_data",
    "load_trial_level",
    "load_mv_activation_and_mni",
    "coords_values_to_heatmap_nifti",
    "load_all_trial_level_per_roi",
    "load_subject_crossregion_hfa",
    "load_univariate_trial_level",
    "load_z_diff_stack_from_pair_dir",
]
IO_FROM_GC = [
    "get_roi_channels",
    "select_top_electrode_per_roi",
    "load_subject_epochs",
    "make_multivariate_indices",
]

REGRESSION_SYMBOLS = [
    "fit_encoding_model",
    "compute_time_resolved_reactivation",
    "compute_subject_reactivation",
    "permutation_regression_zscore_local",
    "process_subject_smoothed_regression",
    "run_smoothed_time_resolved_regression_pipeline",
    "fit_residualizer",
    "apply_residualizer",
    "residualize_predictor",
    "tune_alpha_pooled_subject_roi_predictor",
    "run_pooled_alpha_tuning_pipeline",
    "tune_alpha_nested_cv_subject_roi",
    "run_nested_alpha_tuning_pipeline",
    "process_subject_smoothed_multivariate_regression",
    "run_smoothed_time_resolved_multivariate_regression_pipeline",
    "compute_subject_multivariate_reactivation",
    "run_multivariate_reactivation_pipeline",
    "build_multivariate_csv_path",
    "build_rpe_pair_contrast_regressors",
    "compute_subject_pairwise_multivariate_reactivation",
    "run_pairwise_multivariate_reactivation_pipeline",
    "collapse_reactivation_feedback_windows",
    "extract_theta_power_from_cue_tfr",
    "extract_theta_phase_from_cue_epochs",
    "extract_theta_features_from_cue_epochs",
    "compute_time_resolved_theta_power_reactivation_beta",
    "compute_time_resolved_theta_phase_reactivation_mi",
    "compute_subject_univariate_reactivation",
    "run_univariate_reactivation_pipeline",
    "build_legacy_single_elec_csv_path",
    "summarize_univariate_hits_misses",
    "run_reactivation_pipeline",
]

RSA_SYMBOLS = [
    "build_crossregion_rsa_paths",
    "compute_cross_temporal_rsa_from_hfa",
    "build_rpe_category_model_rdm",
    "compute_cross_temporal_rsa_partial_from_hfa",
    "compute_cross_temporal_reinstatement_category_maps",
    "load_pooled_crossregion_hfa_for_rpe_cat",
    "run_subject_crossregion_rsa_permutation",
    "run_pooled_crossregion_rsa_permutation",
    "run_group_crossregion_rsa_cluster_test",
    "build_cluster_mask_from_clusters",
    "compute_acc_hpc_phase_rsa_at_cue",
    "run_subject_acc_hpc_phase_rsa_permutation",
    "cluster_hpc_phase_states",
    "compute_reactivation_by_phase_state",
    "compute_trial_reinstatement_index",
    "compute_cross_temporal_reinstatement_maps",
    "loso_cluster_summary",
    "wilcoxon_paired_ri_by_subject",
    "run_group_crossregion_rsa_pipeline",
]

TFR_GLM_SYMBOLS = [
    "Cluster",
    "TFRClusterPermutationGLM",
    "process_subject_tfr_cluster_glm",
    "process_subject_crossregion_pac_cluster_glm",
    "run_batch_crossregion_pac_cluster_glm_pipeline",
    "process_subject_withinregion_pac_cluster_glm",
    "run_batch_withinregion_pac_cluster_glm_pipeline",
    "run_batch_tfr_cluster_glm_pipeline",
]

OSCILLATION_SYMBOLS = [
    "get_subject_roi_event_spectra",
    "compute_subject_roi_oscillation_features",
    "summarize_oscillation_electrodes",
    "build_oscillation_matched_pairs",
    "summarize_oscillation_property_stats",
    "run_roi_oscillation_comparison",
    "plot_roi_oscillation_psd_summary",
    "plot_roi_peak_frequency_histograms",
    "plot_roi_property_summaries",
    "plot_roi_property_deltas",
]

STATS_FROM_MV = [
    "run_cluster_test_per_roi",
    "run_2sample_cluster_test_per_roi",
    "run_subject_level_1sample_test",
    "run_subject_wise_2sample_cluster_test",
    "compute_all_subjects_cluster_statistics",
    "run_per_electrode_2sample_cluster_test",
    "run_per_electrode_2sample_cluster_test_with_t_obs",
    "compute_per_electrode_cluster_counts",
    "run_encoding_weight_cluster_test",
    "multinomial_regression",
    "compute_subject_lrt_map",
    "compute_all_subjects_lrt_maps",
    "compute_all_subjects_lrt_maps_from_npz",
    "plot_electrode_cluster_counts",
    "plot_encoding_weight_clusters",
    "plot_cluster_statistics_bar_chart",
]

PLOTTING_FROM_MV = [
    "plot_electrode_cluster_counts",
    "plot_encoding_weight_clusters",
    "plot_cluster_statistics_bar_chart",
    "plot_roi_oscillation_psd_summary",
    "plot_roi_peak_frequency_histograms",
    "plot_roi_property_summaries",
    "plot_roi_property_deltas",
]

PLOTTING_FROM_GC = [
    "plot_mixed_effects_hit_miss",
    "plot_aggregate_hit_miss",
    "plot_mixed_effects_reward",
    "plot_aggregate_reward",
    "plot_mixed_effects_memory_outcome",
    "plot_aggregate_memory_outcome",
    "plot_mixed_effects_rpe",
    "plot_aggregate_rpe",
    "plot_smoke_reward",
    "plot_smoke_memory_outcome",
    "plot_smoke_hit_miss",
    "plot_smoke_rpe",
]

GC_DROP = {
    "load_behavior_tables",
    "preprocess_learn_df",
    "preprocess_combined_df",
    "_gelman_standardize",
}


def _replace_lines(lines: list[str], start: int, end: int, new_block: str) -> list[str]:
    """Replace 1-indexed inclusive line range with new_block (may be multi-line)."""
    new_lines = new_block.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1] + "\n"
    return lines[: start - 1] + new_lines + lines[end:]


def _func_span(src: str, name: str) -> tuple[int, int]:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node.lineno, node.end_lineno
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return item.lineno, item.end_lineno
    raise KeyError(name)


def _write_facade(path: Path, docstring: str, imports: list[tuple[str, list[str]]]) -> None:
    lines = ['"""' + docstring + '"""', "", "from __future__ import annotations", ""]
    all_names: list[str] = []
    for mod, names in imports:
        if not names:
            continue
        lines.append(f"from {mod} import (")
        for n in names:
            lines.append(f"    {n},")
            all_names.append(n)
        lines.append(")")
        lines.append("")
    lines.append("__all__ = [")
    for n in all_names:
        lines.append(f'    "{n}",')
    lines.append("]")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _patch_multivariate(src: str) -> str:
    inject = (
        "\n# membandit: canonical behavior helpers\n"
        "from membandit.behavior import (\n"
        "    preprocess_learn_df as _canonical_preprocess_learn_df,\n"
        "    preprocess_combined_df as _canonical_preprocess_combined_df,\n"
        "    normalize_rpe_cat_labels,\n"
        "    resolve_trial_for_event,\n"
        ")\n"
    )
    if "from membandit.behavior import" not in src:
        src = src.replace(
            "warnings.filterwarnings('ignore')\n",
            "warnings.filterwarnings('ignore')\n" + inject,
            1,
        )

    lines = src.splitlines(keepends=True)

    # Replace Subject preprocess methods (line numbers from ORIGINAL; adjust after inject)
    # Recompute spans on current text
    text = "".join(lines)
    a, b = _func_span(text, "_preprocess_learn_df")
    lines = _replace_lines(
        lines,
        a,
        b,
        "    def _preprocess_learn_df(self, learn_df: pd.DataFrame) -> pd.DataFrame:\n"
        '        """Delegate to membandit.behavior.preprocess_learn_df."""\n'
        "        return _canonical_preprocess_learn_df(learn_df)\n",
    )
    text = "".join(lines)
    a, b = _func_span(text, "_preprocess_combined_df")
    lines = _replace_lines(
        lines,
        a,
        b,
        "    def _preprocess_combined_df(self, combined_df: pd.DataFrame) -> pd.DataFrame:\n"
        '        """Delegate to membandit.behavior.preprocess_combined_df."""\n'
        "        return _canonical_preprocess_combined_df(combined_df)\n",
    )
    text = "".join(lines)
    a, b = _func_span(text, "_resolve_trial_var_for_event")
    lines = _replace_lines(
        lines,
        a,
        b,
        "def _resolve_trial_var_for_event(df: pd.DataFrame, event: str) -> str:\n"
        "    return resolve_trial_for_event(df, event)\n",
    )
    text = "".join(lines)
    a, b = _func_span(text, "_canonicalize_rpe_cat_label")
    lines = _replace_lines(
        lines,
        a,
        b,
        "def _canonicalize_rpe_cat_label(label: Any) -> Optional[str]:\n"
        "    if label is None:\n"
        "        return None\n"
        "    s = normalize_rpe_cat_labels(pd.Series([label]))\n"
        "    val = s.iloc[0]\n"
        "    return None if pd.isna(val) else str(val)\n",
    )
    return "".join(lines)


def _patch_granger(src: str) -> str:
    inject = (
        "\nfrom membandit.behavior import (\n"
        "    load_behavior_tables,\n"
        "    preprocess_learn_df,\n"
        "    preprocess_combined_df,\n"
        "    normalize_rpe_cat_labels,\n"
        "    resolve_trial_for_event,\n"
        ")\n"
    )
    if "from membandit.behavior import" not in src:
        src = src.replace(
            'warnings.filterwarnings("ignore")\n',
            'warnings.filterwarnings("ignore")\n' + inject,
            1,
        )

    lines = src.splitlines(keepends=True)
    text = "".join(lines)

    # Remove original behavior defs (bottom-up so line numbers stay valid)
    for name in ("preprocess_combined_df", "preprocess_learn_df", "load_behavior_tables", "_gelman_standardize"):
        text = "".join(lines)
        a, b = _func_span(text, name)
        lines = _replace_lines(lines, a, b, f"# {name}: provided by membandit.behavior\n")

    text = "".join(lines)
    a, b = _func_span(text, "_resolve_trial_col_for_event")
    lines = _replace_lines(
        lines,
        a,
        b,
        "def _resolve_trial_col_for_event(df: pd.DataFrame, event: str) -> str:\n"
        "    return resolve_trial_for_event(df, event)\n",
    )
    text = "".join(lines)
    a, b = _func_span(text, "_normalize_rpe_cat_series")
    lines = _replace_lines(
        lines,
        a,
        b,
        "def _normalize_rpe_cat_series(s: pd.Series) -> pd.Series:\n"
        "    return normalize_rpe_cat_labels(s)\n",
    )
    return "".join(lines)


def _patch_bosc(src: str) -> str:
    # Replace top-level LFPAnalysis + mv_granger imports
    src = src.replace(
        "from LFPAnalysis.oscillation_utils import eBOSC_wrapper\n",
        "# LFPAnalysis.eBOSC_wrapper imported lazily inside compute_exact_bosc_channel\n",
    )
    src = src.replace(
        "from mv_granger_utils import (\n"
        "    _resolve_trial_col_for_event,\n"
        "    load_behavior_tables,\n"
        "    preprocess_combined_df,\n"
        "    preprocess_learn_df,\n"
        ")\n",
        "from membandit.behavior import (\n"
        "    load_behavior_tables,\n"
        "    preprocess_combined_df,\n"
        "    preprocess_learn_df,\n"
        "    resolve_trial_for_event,\n"
        ")\n"
        "from membandit.behavior import resolve_trial_for_event as _resolve_trial_col_for_event\n",
    )

    # Lazy-import eBOSC at the start of the function body (after signature / docstring)
    lines = src.splitlines(keepends=True)
    text = "".join(lines)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_exact_bosc_channel":
            # First statement in body
            first = node.body[0]
            insert_at = first.lineno  # 1-based; insert before first statement
            # If first statement is a docstring Expr, insert after it
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), (ast.Str, ast.Constant))
            ):
                insert_at = first.end_lineno + 1
            lazy = "    from LFPAnalysis.oscillation_utils import eBOSC_wrapper  # lazy lab dependency\n"
            lines = lines[: insert_at - 1] + [lazy] + lines[insert_at - 1 :]
            break
    return "".join(lines)


def _patch_mvpa(src: str) -> str:
    src = src.replace(
        "from mv_regression_utils import (\n"
        "    Subject,\n"
        "    _build_lrt_trial_axis,\n"
        "    _build_subject_regressors,\n"
        "    _encode_multiclass_target,\n"
        "    _get_smoothed_event_signal,\n"
        ")\n",
        "from membandit._impl.multivariate import (\n"
        "    Subject,\n"
        "    _build_lrt_trial_axis,\n"
        "    _build_subject_regressors,\n"
        "    _encode_multiclass_target,\n"
        "    _get_smoothed_event_signal,\n"
        ")\n",
    )
    return src


def main() -> None:
    IMPL.mkdir(parents=True, exist_ok=True)
    (IMPL / "__init__.py").write_text(
        '"""Internal implementation modules (source monoliths, lightly patched)."""\n',
        encoding="utf-8",
    )

    mv_src = (NOTEBOOKS / "mv_regression_utils.py").read_text(encoding="utf-8")
    mv_src = _patch_multivariate(mv_src)
    ast.parse(mv_src)  # validate
    (IMPL / "multivariate.py").write_text(mv_src, encoding="utf-8")

    gc_src = (NOTEBOOKS / "mv_granger_utils.py").read_text(encoding="utf-8")
    gc_src = _patch_granger(gc_src)
    ast.parse(gc_src)
    (IMPL / "granger.py").write_text(gc_src, encoding="utf-8")

    lrt = (NOTEBOOKS / "likelihood_ratio_test.py").read_text(encoding="utf-8")
    (IMPL / "likelihood_ratio_test.py").write_text(lrt, encoding="utf-8")

    gc_tree = ast.parse(gc_src)
    gc_public = [
        n.name
        for n in gc_tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not n.name.startswith("_")
        and n.name not in GC_DROP
    ]
    # Deduplicate while preserving order
    seen = set()
    gc_public = [x for x in gc_public if not (x in seen or seen.add(x))]

    gc_public_extra = [
        "load_behavior_tables",
        "preprocess_learn_df",
        "preprocess_combined_df",
    ]

    _write_facade(
        PKG / "io.py",
        "\nSubject class, epoch/electrode/TFR loading, and ROI helpers.\n",
        [
            ("membandit._impl.multivariate", IO_FROM_MV),
            ("membandit._impl.granger", IO_FROM_GC),
        ],
    )
    _write_facade(
        PKG / "regression.py",
        "\nEncoding, reactivation, and time-resolved MV/UV regression pipelines.\n",
        [("membandit._impl.multivariate", REGRESSION_SYMBOLS)],
    )
    _write_facade(
        PKG / "rsa.py",
        "\nCross-region RSA, reinstatement, and phase-state analyses.\n",
        [("membandit._impl.multivariate", RSA_SYMBOLS)],
    )
    _write_facade(
        PKG / "tfr_glm.py",
        "\nTFR / PAC cluster-permutation GLM classes and batch pipelines.\n",
        [("membandit._impl.multivariate", TFR_GLM_SYMBOLS)],
    )
    _write_facade(
        PKG / "oscillation.py",
        "\nROI PSD / FOOOF oscillation comparison utilities.\n",
        [("membandit._impl.multivariate", OSCILLATION_SYMBOLS)],
    )
    _write_facade(
        PKG / "granger.py",
        "\nMultivariate Granger causality, trial maps, surrogates, caching, and mixed-effects.\n",
        [
            ("membandit.behavior", gc_public_extra),
            ("membandit._impl.granger", gc_public),
        ],
    )
    _write_facade(
        PKG / "stats.py",
        "\nCluster tests, LRT maps, and electrode-level nested model comparison.\n",
        [
            ("membandit._impl.multivariate", STATS_FROM_MV),
            ("membandit._impl.likelihood_ratio_test", ["electrode_likelihood_ratio_test"]),
            ("membandit._impl.granger", ["compute_null_stats_one_sided"]),
            ("mne.stats", ["permutation_cluster_test", "permutation_cluster_1samp_test"]),
        ],
    )
    _write_facade(
        PKG / "plotting.py",
        "\nShared plotting helpers re-exported from analysis modules.\n",
        [
            ("membandit._impl.multivariate", PLOTTING_FROM_MV),
            ("membandit._impl.granger", PLOTTING_FROM_GC),
        ],
    )

    plot_cfg = (NOTEBOOKS / "plot_config.py").read_text(encoding="utf-8")
    (PKG / "plotting_style.py").write_text(
        '"""Matplotlib / seaborn style defaults for MemoryBandit figures."""\n\n' + plot_cfg,
        encoding="utf-8",
    )

    bosc = (NOTEBOOKS / "bosc_theta_alpha_rpe_utils.py").read_text(encoding="utf-8")
    bosc = _patch_bosc(bosc)
    ast.parse(bosc)
    (PKG / "extras" / "bosc.py").write_text(bosc, encoding="utf-8")

    mvpa = (NOTEBOOKS / "rpe_cat_mvpa_utils.py").read_text(encoding="utf-8")
    mvpa = _patch_mvpa(mvpa)
    ast.parse(mvpa)
    (PKG / "extras" / "mvpa.py").write_text(mvpa, encoding="utf-8")
    (PKG / "extras" / "__init__.py").write_text(
        '"""Optional / heavy-dependency extras (BOSC, specialized MVPA)."""\n',
        encoding="utf-8",
    )

    print("Built membandit modules into", PKG)
    print("  multivariate lines:", len(mv_src.splitlines()))
    print("  granger public symbols:", len(gc_public))


if __name__ == "__main__":
    main()
