"""
Path configuration for manuscript analyses.

Set ``MEMORYBANDIT_DATA_DIR`` to override the default data root (e.g. external drive path).
"""

from __future__ import annotations

import os
from typing import Optional

# Default matches common lab layout; override via env for portability.
_DEFAULT_BASE = "/Volumes/T7 Shield"


def get_base_dir() -> str:
    return os.environ.get("MEMORYBANDIT_DATA_DIR", _DEFAULT_BASE).rstrip("/")


def get_emu_behavior_dir(base_dir: Optional[str] = None) -> str:
    """Directory containing ``learn_df_RWH.csv`` and ``full_df_RWH.csv``."""
    b = base_dir if base_dir is not None else get_base_dir()
    return os.path.join(b, "work", "qasims01", "MemoryBanditData", "EMU")


def get_subjects_xlsx_path(base_dir: Optional[str] = None) -> str:
    b = base_dir if base_dir is not None else get_base_dir()
    return os.path.join(b, "work", "qasims01", "MemoryBanditsDatabase.xlsx")


# Aliases for notebook boilerplate
load_dir = get_emu_behavior_dir
subjects_table_path = get_subjects_xlsx_path


def analysis_output_root(repo_relative: bool = True) -> str:
    """Default folder for analysis exports (CSVs, intermediate arrays)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if repo_relative:
        return os.path.join(here, "..", "analysis_outputs")
    return os.path.join(get_base_dir(), "scratch", "MemoryBandit", "analysis_outputs")


def figure_output_root(repo_relative: bool = True) -> str:
    """Default folder for publication figures."""
    here = os.path.dirname(os.path.abspath(__file__))
    if repo_relative:
        return os.path.join(here, "..", "figures")
    return os.path.join(get_base_dir(), "scratch", "MemoryBandit", "figures")
