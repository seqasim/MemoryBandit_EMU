#!/usr/bin/env python3
"""
Convert per-subject MNE EpochsTFR HDF5 files (``{event}-tfr.h5``) into the
MATLAB ``.mat`` structure expected by the MVPA-Light decoding scripts in ``matlab/``.

Each output ``{event}-tfr.mat`` contains:
  - ``powspctrm`` : float32/float64 array of shape (n_trials, n_channels, n_freqs, n_times)
  - ``freqs``     : 1D frequency vector (Hz)
  - ``times``     : 1D time vector (s)
  - ``ch_names``  : cell array of channel name strings
  - ``sfreq``     : scalar sampling frequency (Hz)

Usage:
  export MEMORYBANDIT_DATA_DIR="/path/to/data"
  python tools/export_tfr_to_mat.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
from scipy.io import savemat

from membandit import paths

DEFAULT_SUBJECTS: tuple[str, ...] = (
    "MS012", "UI001", "MS016", "MS017", "UI002", "MS019", "MS020", "MS022",
    "MS023", "UI003", "MS025", "MS026", "MS028", "MS030", "UI004", "MS035",
    "MS036", "UI006", "UI007",
)
DEFAULT_EVENTS: tuple[str, ...] = ("feedback_start", "cue_on")


def export_subject_event_tfr(
    ephys_dir: Path,
    subject: str,
    event: str,
    *,
    overwrite: bool = False,
) -> Path | None:
    tfr_dir = ephys_dir / subject / "scratch" / "TFR"
    h5_path = tfr_dir / f"{event}-tfr.h5"
    mat_path = tfr_dir / f"{event}-tfr.mat"

    if not h5_path.is_file():
        print(f"[skip] {subject} {event}: missing {h5_path}")
        return None
    if mat_path.is_file() and not overwrite:
        print(f"[exists] {subject} {event}: {mat_path}")
        return mat_path

    tfr_obj = mne.time_frequency.read_tfrs(str(h5_path))
    if isinstance(tfr_obj, list):
        tfr_obj = tfr_obj[0]

    ch_arr = np.empty((len(tfr_obj.ch_names),), dtype=object)
    for i, ch in enumerate(tfr_obj.ch_names):
        ch_arr[i] = str(ch)

    payload = {
        "powspctrm": np.asarray(tfr_obj.data, dtype=np.float32),
        "freqs": np.asarray(tfr_obj.freqs, dtype=np.float64),
        "times": np.asarray(tfr_obj.times, dtype=np.float64),
        "ch_names": ch_arr,
        "sfreq": float(tfr_obj.info["sfreq"]),
    }
    savemat(str(mat_path), payload, do_compression=True)
    print(f"[wrote] {mat_path}  shape={payload['powspctrm'].shape}")
    return mat_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="Subject IDs to export (default: 19 manuscript subjects)",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=list(DEFAULT_EVENTS),
        help="Events to export (default: feedback_start cue_on)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .mat files",
    )
    args = parser.parse_args(argv)

    ephys_dir = Path(paths.get_ephys_dir())
    if not ephys_dir.is_dir():
        print(f"Ephys directory not found: {ephys_dir}")
        print("Set MEMORYBANDIT_DATA_DIR to the root of your data directory.")
        return 1

    for subj in args.subjects:
        for ev in args.events:
            export_subject_event_tfr(ephys_dir, subj, ev, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
