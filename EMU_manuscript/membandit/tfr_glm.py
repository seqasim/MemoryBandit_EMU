"""
TFR / PAC cluster-permutation GLM classes and batch pipelines.
"""

from __future__ import annotations

from membandit._impl.multivariate import (
    Cluster,
    TFRClusterPermutationGLM,
    process_subject_tfr_cluster_glm,
    process_subject_crossregion_pac_cluster_glm,
    run_batch_crossregion_pac_cluster_glm_pipeline,
    process_subject_withinregion_pac_cluster_glm,
    run_batch_withinregion_pac_cluster_glm_pipeline,
    run_batch_tfr_cluster_glm_pipeline,
)

__all__ = [
    "Cluster",
    "TFRClusterPermutationGLM",
    "process_subject_tfr_cluster_glm",
    "process_subject_crossregion_pac_cluster_glm",
    "run_batch_crossregion_pac_cluster_glm_pipeline",
    "process_subject_withinregion_pac_cluster_glm",
    "run_batch_withinregion_pac_cluster_glm_pipeline",
    "run_batch_tfr_cluster_glm_pipeline",
]
