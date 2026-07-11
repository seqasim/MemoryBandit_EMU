"""
ROI PSD / FOOOF oscillation comparison utilities.
"""

from __future__ import annotations

from membandit._impl.multivariate import (
    get_subject_roi_event_spectra,
    compute_subject_roi_oscillation_features,
    summarize_oscillation_electrodes,
    build_oscillation_matched_pairs,
    summarize_oscillation_property_stats,
    run_roi_oscillation_comparison,
    plot_roi_oscillation_psd_summary,
    plot_roi_peak_frequency_histograms,
    plot_roi_property_summaries,
    plot_roi_property_deltas,
)

__all__ = [
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
