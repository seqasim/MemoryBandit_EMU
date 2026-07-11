"""
Manuscript analysis utilities for MemoryBandit EMU (Python).

Heavy dependencies (MNE, sklearn, etc.) are loaded only when you import
submodules such as ``multivariate`` or ``granger``::

    from manuscript_utils.multivariate import Subject, load_subject_data
    from manuscript_utils.granger import compute_multivariate_gc

Lightweight imports (pandas/numpy only)::

    from manuscript_utils.behavior import load_behavior_tables, preprocess_learn_df
    from manuscript_utils import paths
"""

from __future__ import annotations

from .behavior import (
    load_behavior_tables,
    normalize_rpe_cat_labels,
    preprocess_combined_df,
    preprocess_learn_df,
    resolve_trial_for_event,
)
from . import paths

__all__ = [
    "load_behavior_tables",
    "normalize_rpe_cat_labels",
    "paths",
    "preprocess_combined_df",
    "preprocess_learn_df",
    "resolve_trial_for_event",
]
