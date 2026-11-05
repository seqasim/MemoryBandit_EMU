"""
Cluster tests, LRT maps, and electrode-level nested model comparison.
"""

from __future__ import annotations

from membandit._impl.multivariate import (
    run_cluster_test_per_roi,
    run_2sample_cluster_test_per_roi,
    run_subject_level_1sample_test,
    run_subject_wise_2sample_cluster_test,
    compute_all_subjects_cluster_statistics,
    run_per_electrode_2sample_cluster_test,
    run_per_electrode_2sample_cluster_test_with_t_obs,
    compute_per_electrode_cluster_counts,
    run_encoding_weight_cluster_test,
    multinomial_regression,
    compute_subject_lrt_map,
    compute_all_subjects_lrt_maps,
    compute_all_subjects_lrt_maps_from_npz,
    plot_electrode_cluster_counts,
    plot_encoding_weight_clusters,
    plot_cluster_statistics_bar_chart,
)

from membandit._impl.likelihood_ratio_test import (
    electrode_likelihood_ratio_test,
)

from membandit._impl.granger import (
    compute_null_stats_one_sided,
)

from mne.stats import (
    permutation_cluster_test,
    permutation_cluster_1samp_test,
)

__all__ = [
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
    "electrode_likelihood_ratio_test",
    "compute_null_stats_one_sided",
    "permutation_cluster_test",
    "permutation_cluster_1samp_test",
]
