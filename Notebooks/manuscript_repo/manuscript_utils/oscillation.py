"""
Oscillation-related analysis: BOSC (theta/alpha bursts) and ROI PSD / FOOOF comparison.

BOSC helpers live in ``bosc_theta_alpha_rpe_utils``; ROI spectral summaries in ``multivariate``.
"""

from __future__ import annotations

from .bosc_theta_alpha_rpe_utils import (
    apply_behavior_to_epochs,
    assign_exact_band,
    build_period_plot_df,
    build_time_plot_df,
    compute_exact_bosc_channel,
    compute_period_stats,
    get_aligned_behavior_for_event,
    get_aligned_behavior_from_combined_df,
    get_aligned_behavior_from_learn_df,
    get_subject_electrodes,
    load_and_preprocess_behavior,
    load_event_epochs,
    load_group_bosc_outputs,
    normalize_rpe_cat_labels,
    plot_period_averaged_bosc,
    plot_time_resolved_bosc,
    prepare_bosc_channel_dataframe,
    print_pairwise_rpe_ttests_period_plot,
    run_subject_event_bosc,
    validate_required_columns,
)
from .multivariate import (
    compute_subject_roi_oscillation_features,
    get_subject_roi_event_spectra,
    plot_roi_oscillation_psd_summary,
    plot_roi_peak_frequency_histograms,
    plot_roi_property_deltas,
    plot_roi_property_summaries,
    run_roi_oscillation_comparison,
    summarize_oscillation_electrodes,
    summarize_oscillation_property_stats,
)

__all__ = [
    "apply_behavior_to_epochs",
    "assign_exact_band",
    "build_period_plot_df",
    "build_time_plot_df",
    "compute_exact_bosc_channel",
    "compute_period_stats",
    "compute_subject_roi_oscillation_features",
    "get_aligned_behavior_for_event",
    "get_aligned_behavior_from_combined_df",
    "get_aligned_behavior_from_learn_df",
    "get_subject_electrodes",
    "get_subject_roi_event_spectra",
    "load_and_preprocess_behavior",
    "load_event_epochs",
    "load_group_bosc_outputs",
    "normalize_rpe_cat_labels",
    "plot_period_averaged_bosc",
    "plot_roi_oscillation_psd_summary",
    "plot_roi_peak_frequency_histograms",
    "plot_roi_property_deltas",
    "plot_roi_property_summaries",
    "plot_time_resolved_bosc",
    "prepare_bosc_channel_dataframe",
    "print_pairwise_rpe_ttests_period_plot",
    "run_roi_oscillation_comparison",
    "run_subject_event_bosc",
    "summarize_oscillation_electrodes",
    "summarize_oscillation_property_stats",
    "validate_required_columns",
]
