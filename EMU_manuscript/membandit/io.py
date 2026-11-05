"""
Subject class, epoch/electrode/TFR loading, and ROI helpers.
"""

from __future__ import annotations

from membandit._impl.multivariate import (
    Subject,
    load_subject_data,
    load_trial_level,
    load_mv_activation_and_mni,
    coords_values_to_heatmap_nifti,
    load_all_trial_level_per_roi,
    load_subject_crossregion_hfa,
    load_univariate_trial_level,
    load_z_diff_stack_from_pair_dir,
)

from membandit._impl.granger import (
    get_roi_channels,
    select_top_electrode_per_roi,
    load_subject_epochs,
    make_multivariate_indices,
)

__all__ = [
    "Subject",
    "load_subject_data",
    "load_trial_level",
    "load_mv_activation_and_mni",
    "coords_values_to_heatmap_nifti",
    "load_all_trial_level_per_roi",
    "load_subject_crossregion_hfa",
    "load_univariate_trial_level",
    "load_z_diff_stack_from_pair_dir",
    "get_roi_channels",
    "select_top_electrode_per_roi",
    "load_subject_epochs",
    "make_multivariate_indices",
]
