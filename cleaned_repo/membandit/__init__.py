"""
membandit — MemoryBandit EMU analysis package.

Canonical entry points for behavior loading, subject I/O, and plotting style.
Heavier analysis APIs live in submodules (``regression``, ``rsa``, ``granger``, etc.).
"""

from __future__ import annotations

from membandit import paths
from membandit.behavior import (
    load_behavior_tables,
    normalize_rpe_cat_labels,
    preprocess_combined_df,
    preprocess_learn_df,
    resolve_trial_for_event,
)
from membandit.plotting_style import set_plot_style

# Subject is imported lazily-friendly via io; re-export for convenience.
from membandit.io import Subject

__all__ = [
    "paths",
    "Subject",
    "load_behavior_tables",
    "preprocess_learn_df",
    "preprocess_combined_df",
    "normalize_rpe_cat_labels",
    "resolve_trial_for_event",
    "set_plot_style",
]

__version__ = "0.1.0"
