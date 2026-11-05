"""
Multivariate Gaussian-copula phase-amplitude coupling (Tensorpac idpac=6).

Requires a patched tensorpac checkout when phase and amplitude channel counts differ
(cross-region ROI-hemi groups).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from membandit._impl.granger import build_rpe_trial_map, get_roi_channels

DEFAULT_TENSORPAC_ROOT = os.path.expanduser("~/Documents/GitRepos/tensorpac")


def alignment_valid_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask for aligned trials; NaN from left merges treated as invalid."""
    if "alignment_valid" not in df.columns:
        return pd.Series(True, index=df.index)
    col = df["alignment_valid"]
    if col.dtype == bool:
        return col
    return col.astype("boolean").fillna(False).astype(bool)


def effective_filter_edges(
    n_times: int,
    sfreq: float,
    buf_ms: int,
    *,
    min_keep_samples: int = 64,
) -> Tuple[int, int]:
    """Cap edge trim so tensorpac retains at least ``min_keep_samples`` time points."""
    edges_req = int(round(float(buf_ms) * (float(sfreq) / 1000.0)))
    if n_times <= 0:
        return 0, edges_req
    max_edges = max(0, (int(n_times) - int(min_keep_samples)) // 2)
    edges_used = min(edges_req, max_edges)
    return edges_used, edges_req


@dataclass(frozen=True)
class GcpacParams:
    """Configuration for multivariate GC-PAC estimation."""

    f_pha_start: float = 4.0
    f_pha_end: float = 8.0
    f_pha_nb: int = 5
    f_amp_start: float = 70.0
    f_amp_end: float = 200.0
    f_amp_nb: int = 5
    idpac: Tuple[int, int, int] = (6, 2, 0)
    dcomplex: str = "wavelet"
    buf_ms: int = 1000
    n_perm: int = 250
    random_state: int = 42
    n_jobs: int = -1

    @property
    def band_tag(self) -> str:
        return f"theta{self.f_pha_start:g}-{self.f_pha_end:g}_hfa{self.f_amp_start:g}-{self.f_amp_end:g}"

    def frequency_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        f_pha_mid = np.linspace(self.f_pha_start, self.f_pha_end, self.f_pha_nb)
        f_pha = np.c_[f_pha_mid - f_pha_mid / 4.0, f_pha_mid + f_pha_mid / 4.0]
        f_amp_mid = np.linspace(self.f_amp_start, self.f_amp_end, self.f_amp_nb)
        f_amp = np.c_[f_amp_mid - f_amp_mid / 8.0, f_amp_mid + f_amp_mid / 8.0]
        return f_pha, f_amp

    def params_hash(self) -> str:
        payload = asdict(self)
        payload["idpac"] = list(self.idpac)
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def ensure_tensorpac(
    tensorpac_root: Optional[str] = None,
    *,
    require_patched: bool = True,
) -> Any:
    """Import tensorpac, optionally enforcing the multivariate filterfit patch."""
    root = os.path.expanduser(tensorpac_root or os.environ.get("TENSORPAC_ROOT", DEFAULT_TENSORPAC_ROOT))
    if root and root not in sys.path:
        sys.path.insert(0, root)
    elif root and sys.path[0] != root:
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)

    stale_keys = [k for k in list(sys.modules) if k == "tensorpac" or k.startswith("tensorpac.")]
    for key in stale_keys:
        del sys.modules[key]

    import tensorpac  # noqa: WPS433
    from tensorpac import Pac  # noqa: WPS433

    if require_patched:
        src = inspect.getsource(Pac.filterfit)
        if "x_pha.shape == x_amp.shape" in src:
            raise RuntimeError(
                "Unpatched tensorpac detected (strict x_pha/x_amp shape guard). "
                f"Install patched checkout at {root!r} or set TENSORPAC_ROOT. "
                f"Loaded from {getattr(tensorpac, '__file__', '?')}."
            )
    return Pac


def make_pac_object(
    params: GcpacParams,
    tensorpac_root: Optional[str] = None,
    *,
    require_patched: bool = True,
):
    Pac = ensure_tensorpac(tensorpac_root, require_patched=require_patched)
    f_pha, f_amp = params.frequency_bins()
    return Pac(idpac=params.idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=params.dcomplex)


def compute_group_gcpac(
    pac_obj,
    ph_sig: np.ndarray,
    amp_sig: np.ndarray,
    sfreq: float,
    params: GcpacParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multivariate GC-PAC for one ROI-hemi group.

    Parameters
    ----------
    ph_sig, amp_sig
        Arrays shaped (n_trials, n_channels, n_times).
    """
    ph_sig = np.asarray(ph_sig, dtype=float)
    amp_sig = np.asarray(amp_sig, dtype=float)
    if ph_sig.ndim != 3 or amp_sig.ndim != 3:
        raise ValueError("ph_sig and amp_sig must be (n_trials, n_channels, n_times)")
    if ph_sig.shape[0] != amp_sig.shape[0] or ph_sig.shape[-1] != amp_sig.shape[-1]:
        raise ValueError("ph_sig and amp_sig must share n_trials and n_times")

    n_times = int(ph_sig.shape[-1])
    edges, edges_req = effective_filter_edges(n_times, sfreq, params.buf_ms)
    if edges_req > 0 and edges < edges_req:
        warnings.warn(
            f"GCPAC edge trim capped from {edges_req} to {edges} samples "
            f"(n_times={n_times}, sfreq={sfreq}). Load longer epochs or reduce buf_ms.",
            stacklevel=2,
        )
    if n_times <= 2 * edges:
        raise ValueError(
            f"Not enough time samples for GCPAC after edge trim: n_times={n_times}, "
            f"edges={edges} (requested {edges_req}). "
            "Load uncropped epochs or reduce buf_ms."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epoch_pac = pac_obj.filterfit(
            float(sfreq),
            ph_sig,
            amp_sig,
            n_perm=int(params.n_perm),
            n_jobs=int(params.n_jobs),
            edges=edges if edges > 0 else None,
            random_state=int(params.random_state),
        )

    if (params.n_perm <= 0) or (getattr(pac_obj, "surrogates", None) is None):
        z_scored = np.asarray(epoch_pac, dtype=float).copy()
    else:
        surr_mean = pac_obj.surrogates.mean(axis=0)
        surr_std = pac_obj.surrogates.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_scored = (epoch_pac - surr_mean) / surr_std

    z_scored = np.asarray(z_scored, dtype=float)
    trial_pac_z = np.nanmean(z_scored, axis=(0, 1))
    return z_scored, np.asarray(epoch_pac, dtype=float), trial_pac_z


def build_trial_table(
    trial_pac_z: np.ndarray,
    trial_map: pd.DataFrame,
    *,
    subject: str,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    phase_channels: Sequence[str],
    amp_channels: Sequence[str],
) -> pd.DataFrame:
    """Attach trial-level PAC z-scores to an epoch-aligned RPE trial map."""
    n_trials = int(len(trial_pac_z))
    base = pd.DataFrame(
        {
            "subject": str(subject),
            "event": str(event),
            "phase_roi": str(phase_roi),
            "amp_roi": str(amp_roi),
            "hemi": str(hemi),
            "epoch_idx": np.arange(n_trials, dtype=int),
            "trial_pac_z": np.asarray(trial_pac_z, dtype=float),
            "pair_label": f"{phase_roi}->{amp_roi}:{hemi}",
            "phase_channels": "|".join(map(str, phase_channels)),
            "amp_channels": "|".join(map(str, amp_channels)),
            "n_phase_channels": int(len(phase_channels)),
            "n_amp_channels": int(len(amp_channels)),
        }
    )

    if len(trial_map) == 0:
        out = base.copy()
        out["alignment_valid"] = False
        return out

    tmap = trial_map.copy()
    if "epoch_idx" not in tmap.columns:
        raise ValueError("trial_map missing epoch_idx")
    merge_cols = [c for c in tmap.columns if c not in base.columns or c == "epoch_idx"]
    out = base.merge(tmap[merge_cols], on="epoch_idx", how="left")
    if "alignment_valid" in out.columns:
        out["alignment_valid"] = alignment_valid_mask(out)
    else:
        out["alignment_valid"] = out["rpe_cat"].notna()
    return out


def subject_cache_path(
    cache_dir: str,
    subject: str,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    params_hash: str,
) -> str:
    fname = f"{subject}_{phase_roi}-{amp_roi}_{event}_{hemi}_mv_gcpac_{params_hash}.npz"
    return os.path.join(cache_dir, "subject_cache", event, fname)


def skip_cache_path(
    cache_dir: str,
    subject: str,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    params_hash: str,
    reason: str,
) -> str:
    fname = (
        f"{subject}_{phase_roi}-{amp_roi}_{event}_{hemi}_skip_{params_hash}.npz"
    )
    return os.path.join(cache_dir, "subject_cache", event, "skip", fname)


def save_skip_record(path: str, **meta) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: np.asarray(v) for k, v in meta.items()}
    np.savez_compressed(path, **payload)


def save_subject_cache(
    path: str,
    z_scored_pac: np.ndarray,
    raw_pac: np.ndarray,
    trial_pac_z: np.ndarray,
    *,
    phase_channels: Sequence[str],
    amp_channels: Sequence[str],
    sfreq: float,
    params: GcpacParams,
    n_epochs: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f_pha, f_amp = params.frequency_bins()
    n_ep = int(n_epochs) if n_epochs is not None else int(len(trial_pac_z))
    np.savez_compressed(
        path,
        z_scored_pac=z_scored_pac,
        raw_pac=raw_pac,
        trial_pac_z=trial_pac_z,
        phase_channels=np.asarray(list(phase_channels), dtype=str),
        amp_channels=np.asarray(list(amp_channels), dtype=str),
        sfreq=np.array(float(sfreq)),
        params_hash=np.array(params.params_hash()),
        f_pha=f_pha,
        f_amp=f_amp,
        n_epochs=np.array(n_ep, dtype=int),
    )


def record_from_subject_cache(
    cached: Dict[str, Any],
    *,
    subject: str,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    behavior_df: pd.DataFrame,
    day: str = "Day1",
    n_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Rebuild a process_subject_gcpac_unit return dict from an NPZ cache hit."""
    trial_pac_z = np.asarray(cached["trial_pac_z"], dtype=float)
    if n_epochs is None:
        if "n_epochs" in cached:
            n_epochs = int(np.asarray(cached["n_epochs"]))
        else:
            n_epochs = int(len(trial_pac_z))
    trial_map = build_rpe_trial_map(
        behavior_df, subject, n_epochs=int(n_epochs), day=day, event=event
    )
    trial_df = build_trial_table(
        trial_pac_z,
        trial_map,
        subject=subject,
        event=event,
        phase_roi=phase_roi,
        amp_roi=amp_roi,
        hemi=hemi,
        phase_channels=np.asarray(cached["phase_channels"], dtype=str),
        amp_channels=np.asarray(cached["amp_channels"], dtype=str),
    )
    return {
        "subject": subject,
        "event": event,
        "phase_roi": phase_roi,
        "amp_roi": amp_roi,
        "hemi": hemi,
        "z_scored_pac": np.asarray(cached["z_scored_pac"], dtype=float),
        "raw_pac": np.asarray(cached["raw_pac"], dtype=float),
        "trial_pac_z": trial_pac_z,
        "trial_df": trial_df,
        "phase_channels": list(np.asarray(cached["phase_channels"], dtype=str)),
        "amp_channels": list(np.asarray(cached["amp_channels"], dtype=str)),
        "sfreq": float(np.asarray(cached["sfreq"])),
        "from_cache": True,
    }


def try_load_subject_gcpac_cache(
    *,
    cache_dir: str,
    subject: str,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    params: GcpacParams,
    behavior_df: pd.DataFrame,
    overwrite_cache: bool = False,
    day: str = "Day1",
) -> Optional[Dict[str, Any]]:
    """Return cached unit result without loading epochs, or None on miss."""
    if overwrite_cache:
        return None
    path = subject_cache_path(cache_dir, subject, event, phase_roi, amp_roi, hemi, params.params_hash())
    if not os.path.exists(path):
        return None
    return record_from_subject_cache(
        load_subject_cache(path),
        subject=subject,
        event=event,
        phase_roi=phase_roi,
        amp_roi=amp_roi,
        hemi=hemi,
        behavior_df=behavior_df,
        day=day,
    )


def load_subject_cache(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_elec_df(base_dir: str, subject: str, day: str = "Day1") -> pd.DataFrame:
    path = os.path.join(
        base_dir,
        "projects",
        "guLab",
        "Salman",
        "EphysAnalyses",
        str(subject),
        f"{day}_reref_elec_df",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def get_hemi_channel_sets(
    elec_df: pd.DataFrame,
    epochs,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    phase_info = get_roi_channels(elec_df, epochs, phase_roi, hemisphere=hemi)
    amp_info = get_roi_channels(elec_df, epochs, amp_roi, hemisphere=hemi)
    phase_ch = list(phase_info["channels"])
    amp_ch = list(amp_info["channels"])
    return phase_ch, amp_ch, np.asarray(phase_info["indices"], dtype=int), np.asarray(amp_info["indices"], dtype=int)


def process_subject_gcpac_unit(
    *,
    subject: str,
    epoch_data: Optional[Dict[str, Any]],
    elec_df: Optional[pd.DataFrame],
    behavior_df: pd.DataFrame,
    event: str,
    phase_roi: str,
    amp_roi: str,
    hemi: str,
    params: GcpacParams,
    pac_obj=None,
    cache_dir: Optional[str] = None,
    overwrite_cache: bool = False,
    min_trials_per_category: int = 5,
    day: str = "Day1",
) -> Optional[Dict[str, Any]]:
    """
    Compute MV-GCPAC for one subject × event × ROI pair × hemisphere.

    Returns dict with z_scored_pac, trial_df, and metadata; None if skipped.
    ``epoch_data`` / ``elec_df`` may be None when a subject NPZ cache hit is expected;
    a cache miss with ``epoch_data is None`` raises ValueError.
    """
    if pac_obj is None:
        pac_obj = make_pac_object(params)

    params_hash = params.params_hash()
    cache_path = None
    if cache_dir is not None:
        cache_path = subject_cache_path(
            cache_dir, subject, event, phase_roi, amp_roi, hemi, params_hash
        )
        if os.path.exists(cache_path) and not overwrite_cache:
            n_epochs = len(epoch_data["epochs"]) if epoch_data is not None else None
            return record_from_subject_cache(
                load_subject_cache(cache_path),
                subject=subject,
                event=event,
                phase_roi=phase_roi,
                amp_roi=amp_roi,
                hemi=hemi,
                behavior_df=behavior_df,
                day=day,
                n_epochs=n_epochs,
            )

    if epoch_data is None:
        raise ValueError("epoch_data is required when subject cache is missing")

    epochs = epoch_data["epochs"]
    phase_ch, amp_ch, phase_idx, amp_idx = get_hemi_channel_sets(
        elec_df, epochs, phase_roi, amp_roi, hemi
    )
    if len(phase_ch) == 0 or len(amp_ch) == 0:
        if cache_dir is not None:
            save_skip_record(
                skip_cache_path(cache_dir, subject, event, phase_roi, amp_roi, hemi, params_hash, "no_channels"),
                reason=np.array("no_channels"),
                n_phase=np.array(len(phase_ch)),
                n_amp=np.array(len(amp_ch)),
            )
        return None

    ph_sig = epochs.get_data(copy=True)[:, phase_idx, :]
    amp_sig = epochs.get_data(copy=True)[:, amp_idx, :]
    sfreq = float(epoch_data["sfreq"])

    z_scored, raw_pac, trial_pac_z = compute_group_gcpac(
        pac_obj, ph_sig, amp_sig, sfreq, params
    )

    trial_map = build_rpe_trial_map(
        behavior_df, subject, n_epochs=len(epochs), day=day, event=event
    )
    trial_df = build_trial_table(
        trial_pac_z,
        trial_map,
        subject=subject,
        event=event,
        phase_roi=phase_roi,
        amp_roi=amp_roi,
        hemi=hemi,
        phase_channels=phase_ch,
        amp_channels=amp_ch,
    )

    valid = trial_df[alignment_valid_mask(trial_df)].copy()
    counts = valid.groupby("rpe_cat", observed=False).size()
    if not all(int(counts.get(c, 0)) >= min_trials_per_category for c in ("negative", "neutral", "positive")):
        if cache_dir is not None:
            save_skip_record(
                skip_cache_path(cache_dir, subject, event, phase_roi, amp_roi, hemi, params_hash, "min_trials"),
                reason=np.array("min_trials"),
                n_negative=np.array(int(counts.get("negative", 0))),
                n_neutral=np.array(int(counts.get("neutral", 0))),
                n_positive=np.array(int(counts.get("positive", 0))),
            )
        return None

    if cache_path is not None:
        save_subject_cache(
            cache_path,
            z_scored,
            raw_pac,
            trial_pac_z,
            phase_channels=phase_ch,
            amp_channels=amp_ch,
            sfreq=sfreq,
            params=params,
            n_epochs=len(epochs),
        )

    return {
        "subject": subject,
        "event": event,
        "phase_roi": phase_roi,
        "amp_roi": amp_roi,
        "hemi": hemi,
        "z_scored_pac": z_scored,
        "raw_pac": raw_pac,
        "trial_pac_z": trial_pac_z,
        "trial_df": trial_df,
        "phase_channels": phase_ch,
        "amp_channels": amp_ch,
        "sfreq": sfreq,
        "from_cache": False,
    }


def cluster_perm_pos_vs_neutral(
    z_scored_pac: np.ndarray,
    trial_df: pd.DataFrame,
    *,
    n_perm: int = 500,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Supplementary comodulogram cluster test: positive vs neutral trials (unpaired).

    z_scored_pac shape: (n_amp_bins, n_phase_bins, n_trials)
    """
    from mne.stats import permutation_cluster_test

    df = trial_df.dropna(subset=["rpe_cat"]).copy()
    df = df[alignment_valid_mask(df)].copy()
    pos_idx = df.loc[df["rpe_cat"] == "positive", "epoch_idx"].astype(int).to_numpy()
    neu_idx = df.loc[df["rpe_cat"] == "neutral", "epoch_idx"].astype(int).to_numpy()
    pos_idx = pos_idx[(pos_idx >= 0) & (pos_idx < z_scored_pac.shape[-1])]
    neu_idx = neu_idx[(neu_idx >= 0) & (neu_idx < z_scored_pac.shape[-1])]

    out: Dict[str, Any] = {
        "n_pos": int(len(pos_idx)),
        "n_neutral": int(len(neu_idx)),
        "cluster_ok": False,
        "min_cluster_p": np.nan,
        "n_sig_clusters": 0,
    }
    if len(pos_idx) < 2 or len(neu_idx) < 2:
        return out

    z_trials = np.moveaxis(z_scored_pac, -1, 0)
    X_pos = z_trials[pos_idx, :, :]
    X_neu = z_trials[neu_idx, :, :]
    try:
        _, _, cluster_pv, _ = permutation_cluster_test(
            [X_pos, X_neu],
            n_permutations=int(n_perm),
            tail=0,
            out_type="mask",
            verbose=False,
        )
        if len(cluster_pv) > 0:
            out["min_cluster_p"] = float(np.nanmin(cluster_pv))
            out["n_sig_clusters"] = int(np.sum(np.asarray(cluster_pv) < alpha))
        out["cluster_ok"] = True
    except Exception:
        out["cluster_ok"] = False
    return out
