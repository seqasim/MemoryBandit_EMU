"""
Path configuration for MemoryBandit EMU manuscript analyses.

Set ``MEMORYBANDIT_DATA_DIR`` to point at the root of the data directory
(see ``DATA_LAYOUT.md`` for expected directory structure and file schemas).
"""

from __future__ import annotations

import os
from typing import Optional

# Override via environment variable MEMORYBANDIT_DATA_DIR.
_DEFAULT_BASE = os.path.expanduser("~/MemoryBanditData")


def repo_root() -> str:
    """Return the root directory of the EMU_manuscript package."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_base_dir() -> str:
    """Return the root of the external data directory."""
    return os.environ.get("MEMORYBANDIT_DATA_DIR", _DEFAULT_BASE).rstrip("/")


def get_emu_behavior_dir(base_dir: Optional[str] = None) -> str:
    """Directory containing ``learn_df_RWH.csv`` and ``full_df_RWH.csv``."""
    b = base_dir if base_dir is not None else get_base_dir()
    return os.path.join(b, "work", "qasims01", "MemoryBanditData", "EMU")


def get_prolific_dir(base_dir: Optional[str] = None) -> str:
    """Directory containing online (Prolific) cohort behavior CSVs."""
    b = base_dir if base_dir is not None else get_base_dir()
    primary = os.path.join(b, "work", "qasims01", "MemoryBanditData", "Prolific", "DelayedMemoryTest")
    alt = os.path.join(b, "projects", "guLab", "Salman", "Prolific", "DelayedMemoryTest")
    if os.path.isdir(primary) or not os.path.isdir(alt):
        return primary
    return alt


def get_ephys_dir(base_dir: Optional[str] = None) -> str:
    """Directory containing per-subject ``{subj}/neural/Day1/*-epo.fif`` and ``scratch/TFR/*``."""
    b = base_dir if base_dir is not None else get_base_dir()
    return os.path.join(b, "projects", "guLab", "Salman", "EphysAnalyses")


def get_emu_raw_dir(base_dir: Optional[str] = None) -> str:
    """Directory containing EMU metadata (e.g. ``MemoryBanditsDatabase.xlsx``)."""
    b = base_dir if base_dir is not None else get_base_dir()
    return os.path.join(b, "projects", "guLab", "Salman", "EMU")


def get_subjects_xlsx_path(base_dir: Optional[str] = None) -> str:
    """Path to ``MemoryBanditsDatabase.xlsx`` participant/electrode roster."""
    b = base_dir if base_dir is not None else get_base_dir()
    p1 = os.path.join(b, "projects", "guLab", "Salman", "EMU", "MemoryBanditsDatabase.xlsx")
    p2 = os.path.join(b, "work", "qasims01", "MemoryBanditsDatabase.xlsx")
    if os.path.isfile(p1) or not os.path.isfile(p2):
        return p1
    return p2


# Aliases for notebook boilerplate
load_dir = get_emu_behavior_dir
subjects_table_path = get_subjects_xlsx_path


def results_root(repo_relative: bool = True) -> str:
    """Default folder for analysis exports (CSVs, intermediate arrays)."""
    if repo_relative:
        return os.path.join(repo_root(), "results")
    return os.path.join(get_base_dir(), "scratch", "MemoryBandit", "analysis_outputs")


def analysis_output_root(repo_relative: bool = True) -> str:
    """Alias for ``results_root``."""
    return results_root(repo_relative=repo_relative)


def figure_output_root(repo_relative: bool = True) -> str:
    """Default folder for publication figures."""
    if repo_relative:
        return os.path.join(repo_root(), "figures")
    return os.path.join(get_base_dir(), "scratch", "MemoryBandit", "figures")
