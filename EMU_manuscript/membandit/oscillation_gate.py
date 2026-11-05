"""
Reusable oscillation-gating helpers for connectivity pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from membandit._impl.multivariate import _fit_fooof_spectrum


def fit_channel_fooof(
    epochs: Any,
    channels: Sequence[str],
    fooof_kwargs: Optional[Dict[str, Any]] = None,
    psd_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit FOOOF per channel from pooled trials and return spectra + fit stats.
    """
    psd_settings = dict(psd_kwargs or {})
    psd_settings.setdefault("method", "multitaper")
    psd_settings.setdefault("verbose", False)

    epochs_pick = epochs.copy().pick(list(channels), verbose=False)
    psd = epochs_pick.compute_psd(**psd_settings)
    freqs = np.asarray(psd.freqs, dtype=np.float64)
    data = np.asarray(psd.get_data(), dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Unexpected PSD shape: {data.shape}")
    trial_mean = np.nanmean(data, axis=0)
    psd_ch_names = [str(ch) for ch in getattr(psd, "ch_names", epochs_pick.ch_names)]
    n_psd_spectra = int(trial_mean.shape[0])
    n_psd_names = int(len(psd_ch_names))
    n_keep = int(min(n_psd_spectra, n_psd_names))
    if n_keep <= 0:
        raise ValueError("No usable PSD channels for FOOOF fitting")
    if n_keep < n_psd_spectra or n_keep < n_psd_names:
        trial_mean = trial_mean[:n_keep, :]
        psd_ch_names = psd_ch_names[:n_keep]

    channel_results: List[Dict[str, Any]] = []
    for ch_idx, ch_name in enumerate(psd_ch_names):
        raw_psd = np.asarray(trial_mean[ch_idx], dtype=np.float64)
        fooof_res = _fit_fooof_spectrum(
            freqs=freqs,
            power_spectrum=raw_psd,
            fooof_kwargs=fooof_kwargs,
        )
        corrected = np.asarray(fooof_res["corrected_psd"], dtype=np.float64)
        channel_results.append(
            {
                "channel": str(ch_name),
                "raw_psd": raw_psd,
                "flattened_psd": corrected,
                "peak_freq": float(fooof_res.get("peak_freq", np.nan)),
                "peak_height": float(fooof_res.get("peak_height", np.nan)),
                "peak_bandwidth": float(fooof_res.get("peak_bandwidth", np.nan)),
                "n_peaks": int(fooof_res.get("n_peaks", 0)),
                "aperiodic_offset": float(fooof_res.get("aperiodic_offset", np.nan)),
                "aperiodic_exponent": float(fooof_res.get("aperiodic_exponent", np.nan)),
                "r_squared": float(fooof_res.get("r_squared", np.nan)),
                "fit_error": float(fooof_res.get("fit_error", np.nan)),
            }
        )

    return {
        "freqs": freqs,
        "n_trials": int(len(epochs_pick)),
        "channels": [str(x) for x in epochs_pick.ch_names],
        "channel_results": channel_results,
    }


def gate_channels(
    fooof_results: Dict[str, Any],
    band: Tuple[float, float] = (2.0, 10.0),
    min_height: float = 0.1,
    channels: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Keep channels whose max flattened PSD in `band` is >= `min_height`.
    """
    freqs = np.asarray(fooof_results["freqs"], dtype=float)
    lo, hi = float(band[0]), float(band[1])
    band_mask = (freqs >= lo) & (freqs <= hi)
    allowed = None if channels is None else {str(ch) for ch in channels}

    kept: List[str] = []
    dropped: List[str] = []
    rows: List[Dict[str, Any]] = []
    for rec in fooof_results["channel_results"]:
        ch = str(rec["channel"])
        if allowed is not None and ch not in allowed:
            continue
        flat = np.asarray(rec["flattened_psd"], dtype=float)
        if flat.shape[0] != freqs.shape[0]:
            raise ValueError(f"flattened_psd length mismatch for channel {ch}")
        band_vals = flat[band_mask]
        max_flat = float(np.nanmax(band_vals)) if np.any(np.isfinite(band_vals)) else np.nan
        keep = bool(np.isfinite(max_flat) and max_flat >= float(min_height))
        if keep:
            kept.append(ch)
        else:
            dropped.append(ch)
        rows.append(
            {
                "channel": ch,
                "max_flat_in_band": max_flat,
                "kept": keep,
                "n_peaks": int(rec.get("n_peaks", 0)),
                "peak_freq": float(rec.get("peak_freq", np.nan)),
                "peak_height": float(rec.get("peak_height", np.nan)),
                "peak_bandwidth": float(rec.get("peak_bandwidth", np.nan)),
                "r_squared": float(rec.get("r_squared", np.nan)),
                "fit_error": float(rec.get("fit_error", np.nan)),
            }
        )
    return kept, dropped, pd.DataFrame(rows)


def pair_oscillation_weights(
    flat_src: np.ndarray,
    flat_tgt: np.ndarray,
    psd_freqs: np.ndarray,
    conn_freqs: np.ndarray,
    band: Tuple[float, float] = (2.0, 10.0),
) -> Optional[np.ndarray]:
    """
    Compute normalized pair weights from geometric mean flattened spectra.
    """
    src = np.asarray(flat_src, dtype=float)
    tgt = np.asarray(flat_tgt, dtype=float)
    psd_f = np.asarray(psd_freqs, dtype=float)
    conn_f = np.asarray(conn_freqs, dtype=float)
    if src.shape != tgt.shape or src.shape != psd_f.shape:
        raise ValueError("flat_src, flat_tgt, and psd_freqs must share shape")

    src_i = np.interp(conn_f, psd_f, np.nan_to_num(src, nan=0.0), left=0.0, right=0.0)
    tgt_i = np.interp(conn_f, psd_f, np.nan_to_num(tgt, nan=0.0), left=0.0, right=0.0)
    src_i = np.clip(src_i, 0.0, None)
    tgt_i = np.clip(tgt_i, 0.0, None)
    w = np.sqrt(src_i * tgt_i)

    lo, hi = float(band[0]), float(band[1])
    band_mask = (conn_f >= lo) & (conn_f <= hi)
    w = np.where(band_mask, w, 0.0)
    denom = float(np.nansum(w))
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    return w / denom


def roi_oscillation_weights(
    flat_src_list: Sequence[np.ndarray],
    flat_tgt_list: Sequence[np.ndarray],
    psd_freqs: np.ndarray,
    conn_freqs: np.ndarray,
    band: Tuple[float, float] = (2.0, 10.0),
) -> Optional[np.ndarray]:
    """
    Build ROI-level normalized weights from channel flattened spectra.
    """
    src_rows = [np.asarray(x, dtype=float) for x in flat_src_list]
    tgt_rows = [np.asarray(x, dtype=float) for x in flat_tgt_list]
    if len(src_rows) == 0 or len(tgt_rows) == 0:
        return None
    src_stack = np.vstack(src_rows)
    tgt_stack = np.vstack(tgt_rows)
    src_mean = np.nanmean(np.clip(src_stack, 0.0, None), axis=0)
    tgt_mean = np.nanmean(np.clip(tgt_stack, 0.0, None), axis=0)
    return pair_oscillation_weights(
        flat_src=src_mean,
        flat_tgt=tgt_mean,
        psd_freqs=psd_freqs,
        conn_freqs=conn_freqs,
        band=band,
    )


def weighted_spectrum_average(spec: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted average of a spectrum using normalized per-frequency weights.
    """
    y = np.asarray(spec, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if y.shape != w.shape:
        raise ValueError("spec and weights must have matching shape")
    return float(np.nansum(w * y))


def edge_weights_from_pair_weights(weights: np.ndarray) -> Optional[np.ndarray]:
    """
    Build normalized adjacent-bin edge weights from per-bin pair weights.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size < 2:
        return None
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)
    edge = np.sqrt(w[:-1] * w[1:])
    denom = float(np.nansum(edge))
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    return edge / denom


def weighted_psi_from_cohy(
    cohy_data: np.ndarray,
    freqs: np.ndarray,
    pair_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted PSI from per-bin coherency and per-bin pair weights.
    """
    del freqs  # included for API symmetry and future checks
    cohy = np.asarray(cohy_data)
    if cohy.ndim == 1:
        cohy = cohy[np.newaxis, :]
    if cohy.ndim != 2:
        raise ValueError("cohy_data must be shape (n_pairs, n_freqs)")

    w = np.asarray(pair_weights, dtype=float)
    if w.ndim == 1:
        w = w[np.newaxis, :]
    if w.ndim != 2 or w.shape != cohy.shape:
        raise ValueError("pair_weights must match cohy_data shape")

    out = np.full(cohy.shape[0], np.nan, dtype=float)
    for i in range(cohy.shape[0]):
        ew = edge_weights_from_pair_weights(w[i])
        if ew is None:
            continue
        prod = np.conj(cohy[i, :-1]) * cohy[i, 1:]
        out[i] = float(np.imag(np.nansum(ew * prod)))
    return out


def save_fooof_results(path: str, results: Dict[str, Any]) -> None:
    """Persist FOOOF results payload as a compressed npz object."""
    np.savez_compressed(path, payload=np.array([results], dtype=object))


def load_fooof_results(path: str) -> Dict[str, Any]:
    """Load FOOOF results payload saved by `save_fooof_results`."""
    with np.load(path, allow_pickle=True) as data:
        return dict(data["payload"][0])


def fooof_results_to_table(
    fooof_results: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Expand channel-level FOOOF results into a long DataFrame.
    """
    extra = dict(extra or {})
    freqs = np.asarray(fooof_results["freqs"], dtype=float)
    rows: List[Dict[str, Any]] = []
    for rec in fooof_results["channel_results"]:
        base = {
            "channel": str(rec["channel"]),
            "peak_freq": float(rec.get("peak_freq", np.nan)),
            "peak_height": float(rec.get("peak_height", np.nan)),
            "peak_bandwidth": float(rec.get("peak_bandwidth", np.nan)),
            "n_peaks": int(rec.get("n_peaks", 0)),
            "aperiodic_offset": float(rec.get("aperiodic_offset", np.nan)),
            "aperiodic_exponent": float(rec.get("aperiodic_exponent", np.nan)),
            "r_squared": float(rec.get("r_squared", np.nan)),
            "fit_error": float(rec.get("fit_error", np.nan)),
        }
        flat = np.asarray(rec["flattened_psd"], dtype=float)
        raw = np.asarray(rec["raw_psd"], dtype=float)
        for fi, freq in enumerate(freqs):
            row = dict(extra)
            row.update(base)
            row["frequency"] = float(freq)
            row["psd_raw"] = float(raw[fi]) if fi < raw.shape[0] else np.nan
            row["psd_flattened"] = float(flat[fi]) if fi < flat.shape[0] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "fit_channel_fooof",
    "gate_channels",
    "pair_oscillation_weights",
    "roi_oscillation_weights",
    "weighted_spectrum_average",
    "edge_weights_from_pair_weights",
    "weighted_psi_from_cohy",
    "save_fooof_results",
    "load_fooof_results",
    "fooof_results_to_table",
]
