"""
Figure helpers re-exported from ``granger`` (MVGC / mixed-effects panels) and
``multivariate`` (encoding clusters, ROI oscillation summaries).
"""

from __future__ import annotations

from .granger import (
    plot_aggregate_hit_miss,
    plot_aggregate_memory_outcome,
    plot_aggregate_reward,
    plot_aggregate_rpe,
    plot_mixed_effects_hit_miss,
    plot_mixed_effects_memory_outcome,
    plot_mixed_effects_reward,
    plot_mixed_effects_rpe,
    plot_smoke_hit_miss,
    plot_smoke_memory_outcome,
    plot_smoke_reward,
    plot_smoke_rpe,
)
from .multivariate import (
    plot_cluster_statistics_bar_chart,
    plot_electrode_cluster_counts,
    plot_encoding_weight_clusters,
    plot_roi_oscillation_psd_summary,
    plot_roi_peak_frequency_histograms,
    plot_roi_property_deltas,
    plot_roi_property_summaries,
)

__all__ = [
    "plot_aggregate_hit_miss",
    "plot_aggregate_memory_outcome",
    "plot_aggregate_reward",
    "plot_aggregate_rpe",
    "plot_cluster_statistics_bar_chart",
    "plot_electrode_cluster_counts",
    "plot_encoding_weight_clusters",
    "plot_mixed_effects_hit_miss",
    "plot_mixed_effects_memory_outcome",
    "plot_mixed_effects_reward",
    "plot_mixed_effects_rpe",
    "plot_roi_oscillation_psd_summary",
    "plot_roi_peak_frequency_histograms",
    "plot_roi_property_deltas",
    "plot_roi_property_summaries",
    "plot_smoke_hit_miss",
    "plot_smoke_memory_outcome",
    "plot_smoke_reward",
    "plot_smoke_rpe",
]
