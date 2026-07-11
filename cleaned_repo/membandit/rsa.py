"""
Cross-region RSA, reinstatement, and phase-state analyses.
"""

from __future__ import annotations

from membandit._impl.multivariate import (
    build_crossregion_rsa_paths,
    compute_cross_temporal_rsa_from_hfa,
    build_rpe_category_model_rdm,
    compute_cross_temporal_rsa_partial_from_hfa,
    compute_cross_temporal_reinstatement_category_maps,
    load_pooled_crossregion_hfa_for_rpe_cat,
    run_subject_crossregion_rsa_permutation,
    run_pooled_crossregion_rsa_permutation,
    run_group_crossregion_rsa_cluster_test,
    build_cluster_mask_from_clusters,
    compute_acc_hpc_phase_rsa_at_cue,
    run_subject_acc_hpc_phase_rsa_permutation,
    cluster_hpc_phase_states,
    compute_reactivation_by_phase_state,
    compute_trial_reinstatement_index,
    compute_cross_temporal_reinstatement_maps,
    loso_cluster_summary,
    wilcoxon_paired_ri_by_subject,
    run_group_crossregion_rsa_pipeline,
)

__all__ = [
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
