"""
Phase-synchrony helpers shared by connectivity notebooks.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed


def _as_array(data: np.ndarray) -> np.ndarray:
    """Return data as (n_epochs, n_channels, n_times) float array."""
    if hasattr(data, "get_data"):
        arr = np.asarray(data.get_data(), dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
    if arr.ndim != 3:
        raise ValueError("data must have shape (n_epochs, n_channels, n_times)")
    return arr


def _get_make_surrogate_evoked_data():
    """
    Cut-and-reverse evoked surrogates.

    Prefer ``mne_connectivity.make_surrogate_evoked_data`` (0.9+); otherwise use
    the local fallback matching that implementation.
    """
    try:
        from mne_connectivity import make_surrogate_evoked_data as _fn

        return _fn
    except Exception:
        # ImportError on older mne-connectivity, or RuntimeError if numba cache
        # is unavailable in restricted environments.
        pass

    from mne import EpochsArray

    def make_surrogate_evoked_data(data, n_shuffles=1000, rng_seed=None, return_generator=True):
        data_arr = data.get_data()
        rng = np.random.default_rng(rng_seed)

        def _gen():
            n_epochs, n_chans, n_times = data_arr.shape
            for _ in range(int(n_shuffles)):
                surrogate_arr = np.empty_like(data_arr)
                cutpoints = rng.integers(1, n_times, size=(n_epochs, n_chans))
                for ei in range(n_epochs):
                    for ci in range(n_chans):
                        parts = np.array_split(
                            data_arr[ei, ci], [int(cutpoints[ei, ci])], axis=-1
                        )
                        parts.reverse()
                        surrogate_arr[ei, ci] = np.concatenate(parts, axis=-1)
                yield EpochsArray(
                    data=surrogate_arr,
                    info=data.info,
                    tmin=float(data.tmin),
                    verbose=False,
                )

        gen = _gen()
        return gen if return_generator else list(gen)

    return make_surrogate_evoked_data


def shuffle_target_trials(
    data: np.ndarray,
    target_picks: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shuffle trial order of target channels, keeping seed channels fixed.

    Parameters
    ----------
    data
        Epoch array of shape (n_epochs, n_channels, n_times).
    target_picks
        Channel indices whose trial order is permuted.
    rng
        NumPy random generator.
    """
    arr = _as_array(data)
    out = arr.copy()
    n_epochs = out.shape[0]
    for ch_idx in target_picks:
        perm = rng.permutation(n_epochs)
        out[:, int(ch_idx), :] = out[perm, int(ch_idx), :]
    return out


def _surrogate_once(
    conn_fn: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    target_picks: Sequence[int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = shuffle_target_trials(data=data, target_picks=target_picks, rng=rng)
    return np.asarray(conn_fn(shuffled), dtype=float)


def zscore_vs_trial_shuffle(
    conn_fn: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    target_picks: Sequence[int],
    n_surrogates: int = 50,
    n_jobs: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score connectivity against target-trial-shuffle surrogates.

    Parameters
    ----------
    conn_fn
        Callable that maps array (n_epochs, n_channels, n_times) -> metric array.
    data
        Epoch data as ndarray or object with ``get_data()``.
    target_picks
        Target-channel indices to shuffle across trials.
    n_surrogates
        Number of surrogate shuffles to compute.
    n_jobs
        Number of parallel jobs over surrogates.
    seed
        Base RNG seed.
    """
    if int(n_surrogates) < 2:
        raise ValueError("n_surrogates must be >= 2")

    arr = _as_array(data)
    true_val = np.asarray(conn_fn(arr), dtype=float)
    surr = Parallel(n_jobs=int(n_jobs))(
        delayed(_surrogate_once)(conn_fn, arr, target_picks, int(seed) + i)
        for i in range(int(n_surrogates))
    )
    surr = np.asarray(surr, dtype=float)
    surr_mean = np.nanmean(surr, axis=0)
    surr_std = np.nanstd(surr, axis=0, ddof=1)
    surr_std = np.where(surr_std < 1e-12, np.nan, surr_std)
    z = (true_val - surr_mean) / surr_std
    return z, surr_mean, surr_std


def _evoked_surrogate_once(
    conn_fn: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    info,
    n_surrogates: int,
    seed: int,
) -> np.ndarray:
    from mne import EpochsArray

    make_surrogate_evoked_data = _get_make_surrogate_evoked_data()

    ep = EpochsArray(data, info.copy(), tmin=0.0, verbose=False)
    gen = make_surrogate_evoked_data(
        ep,
        n_shuffles=int(n_surrogates),
        rng_seed=int(seed),
        return_generator=True,
    )
    out = []
    for surr in gen:
        out.append(np.asarray(conn_fn(surr.get_data()), dtype=float))
    return np.asarray(out, dtype=float)


def zscore_vs_evoked_surrogate(
    conn_fn: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    info,
    n_surrogates: int = 50,
    n_jobs: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score connectivity against cut-and-reverse evoked-data surrogates.

    Surrogates are generated using ``mne_connectivity.make_surrogate_evoked_data``.
    """
    if int(n_surrogates) < 2:
        raise ValueError("n_surrogates must be >= 2")

    arr = _as_array(data)
    true_val = np.asarray(conn_fn(arr), dtype=float)

    # Split shuffles into chunks so joblib parallelizes over surrogate generation.
    n_jobs = max(1, int(n_jobs))
    counts = np.full(n_jobs, int(n_surrogates) // n_jobs, dtype=int)
    counts[: int(n_surrogates) % n_jobs] += 1
    counts = counts[counts > 0]

    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_evoked_surrogate_once)(
            conn_fn=conn_fn,
            data=arr,
            info=info,
            n_surrogates=int(count),
            seed=int(seed) + i,
        )
        for i, count in enumerate(counts.tolist())
    )
    surr = np.concatenate(chunks, axis=0)
    surr_mean = np.nanmean(surr, axis=0)
    surr_std = np.nanstd(surr, axis=0, ddof=1)
    surr_std = np.where(surr_std < 1e-12, np.nan, surr_std)
    z = (true_val - surr_mean) / surr_std
    return z, surr_mean, surr_std


def zscore_epoch_conn_vs_evoked_surrogate(
    conn_fn: Callable,
    epochs,
    n_surrogates: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-epoch z-score against cut-and-reverse evoked-data surrogates.

    Parameters
    ----------
    conn_fn
        Callable mapping an MNE Epochs-like object to an array with the
        epoch axis first, i.e. ``(n_epochs, ...)``.
    epochs
        Epochs-like object used for both true and surrogate connectivity.
    n_surrogates
        Number of surrogates from ``make_surrogate_evoked_data``.
    seed
        RNG seed for surrogate generation.
    """
    make_surrogate_evoked_data = _get_make_surrogate_evoked_data()

    if int(n_surrogates) < 2:
        raise ValueError("n_surrogates must be >= 2")

    true_val = np.asarray(conn_fn(epochs), dtype=float)
    if true_val.ndim < 1:
        raise ValueError("conn_fn output must include an epoch axis")

    n_epochs = len(epochs)
    if true_val.shape[0] != n_epochs:
        raise ValueError(
            f"conn_fn epoch axis mismatch: got {true_val.shape[0]}, expected {n_epochs}"
        )

    running_mean = None
    running_m2 = None
    n = 0
    gen = make_surrogate_evoked_data(
        epochs,
        n_shuffles=int(n_surrogates),
        rng_seed=int(seed),
        return_generator=True,
    )
    for surr in gen:
        val = np.asarray(conn_fn(surr), dtype=float)
        if val.shape != true_val.shape:
            raise ValueError("conn_fn output shape changed across surrogates")
        n += 1
        if running_mean is None:
            running_mean = val.copy()
            running_m2 = np.zeros_like(val, dtype=float)
        else:
            delta = val - running_mean
            running_mean = running_mean + delta / n
            running_m2 = running_m2 + delta * (val - running_mean)

    if n < 2 or running_mean is None or running_m2 is None:
        raise RuntimeError("insufficient surrogates generated for z-score")

    surr_mean = running_mean
    surr_std = np.sqrt(running_m2 / (n - 1))
    surr_std = np.where(surr_std < 1e-12, np.nan, surr_std)
    z = (true_val - surr_mean) / surr_std
    return z, surr_mean, surr_std


def compute_epoch_psi(
    epochs,
    indices,
    sfreq: float,
    *,
    freqs,
    fmin: float,
    fmax: float,
    n_cycles=7.0,
    padding: float = 0.0,
    n_jobs: int = 1,
):
    """
    Trial-wise HPC→ACC PSI via ``phase_slope_index_time`` (band-averaged 4–8 Hz).

    Returns array of shape ``(n_epochs, n_pairs)``. Positive values indicate the
    seed time series leads the target (see MNE-Connectivity PSI docs).
    """
    from mne_connectivity import phase_slope_index_time

    conn = phase_slope_index_time(
        epochs,
        freqs=freqs,
        indices=indices,
        sfreq=float(sfreq),
        mode="cwt_morlet",
        fmin=float(fmin),
        fmax=float(fmax),
        n_cycles=n_cycles,
        padding=float(padding),
        average=False,
        n_jobs=int(n_jobs),
        verbose=False,
    )
    data = np.asarray(conn.get_data(), dtype=float)
    if data.ndim == 2:
        data = data[:, :, None]
    return np.nanmean(data, axis=2).astype(float)


def zscore_epoch_psi(
    psi_true: np.ndarray,
    epochs,
    indices,
    sfreq: float,
    *,
    freqs,
    fmin: float,
    fmax: float,
    n_cycles=7.0,
    padding: float = 0.0,
    n_surrogates: int = 50,
    seed: int = 42,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-epoch PSI z-score vs cut-and-reverse evoked surrogates.

    Each surrogate shuffle yields one PSI surface; Welford accumulation gives
    per-(epoch, pair) surrogate mean/std for z-scoring ``psi_true``.
    """
    make_surrogate_evoked_data = _get_make_surrogate_evoked_data()

    if int(n_surrogates) < 2:
        raise ValueError("n_surrogates must be >= 2")

    psi_true = np.asarray(psi_true, dtype=float)

    def _psi_conn(ep):
        return compute_epoch_psi(
            ep,
            indices,
            sfreq,
            freqs=freqs,
            fmin=fmin,
            fmax=fmax,
            n_cycles=n_cycles,
            padding=padding,
            n_jobs=n_jobs,
        )

    running_mean = None
    running_m2 = None
    n = 0
    gen = make_surrogate_evoked_data(
        epochs,
        n_shuffles=int(n_surrogates),
        rng_seed=int(seed),
        return_generator=True,
    )
    for surr in gen:
        psi_s = _psi_conn(surr)
        n += 1
        if running_mean is None:
            running_mean = psi_s.astype(float).copy()
            running_m2 = np.zeros_like(running_mean, dtype=float)
        else:
            delta = psi_s - running_mean
            running_mean = running_mean + delta / n
            running_m2 = running_m2 + delta * (psi_s - running_mean)

    if n < 2 or running_mean is None or running_m2 is None:
        raise RuntimeError("Need at least 2 surrogates to z-score epoch PSI")

    surr_std = np.sqrt(running_m2 / (n - 1))
    surr_std = np.where(surr_std < 1e-12, np.nan, surr_std)
    z = (psi_true - running_mean) / surr_std
    return z, running_mean, surr_std


def _hilbert_phase(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Bandpass + Hilbert instantaneous phase for ``(n_epochs, n_channels, n_times)``."""
    from scipy.signal import butter, filtfilt, hilbert

    arr = _as_array(data)
    n_epochs, n_channels, n_times = arr.shape
    nyq = 0.5 * float(sfreq)
    lo = max(float(fmin) / nyq, 1e-6)
    hi = min(float(fmax) / nyq, 0.999)
    if not (0.0 < lo < hi < 1.0):
        raise ValueError(f"invalid bandpass relative to Nyquist: fmin={fmin}, fmax={fmax}, sfreq={sfreq}")
    b, a = butter(4, [lo, hi], btype="band")
    flat = arr.reshape(n_epochs * n_channels, n_times)
    filtered = filtfilt(b, a, flat, axis=-1)
    phase = np.angle(hilbert(filtered, axis=-1)).reshape(n_epochs, n_channels, n_times)
    return phase.astype(float)


def window_pair_locking_and_dphi(
    data: np.ndarray,
    seed_idx: Sequence[int],
    target_idx: Sequence[int],
    sfreq: float,
    *,
    fmin: float = 4.0,
    fmax: float = 8.0,
    times: np.ndarray | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trial-wise HPC–ACC locking and phase difference in a time window.

    Parameters
    ----------
    data
        Array ``(n_epochs, n_channels, n_times)``.
    seed_idx, target_idx
        Channel indices for seed (HPC) and target (ACC).
    sfreq
        Sampling rate (Hz).
    fmin, fmax
        Theta band for FIR/IIR bandpass before Hilbert.
    times
        Optional sample times (s). When provided with ``tmin``/``tmax``,
        metrics are averaged only inside that window.
    tmin, tmax
        Inclusive analysis window in seconds (relative to ``times``).

    Returns
    -------
    R_lock : (n_epochs,)
        Mean circular resultant length of ``exp(i (φ_seed − φ_target))``
        across pairs and time (single-trial PPC analogue).
    dphi : (n_epochs,)
        Circular mean of ``φ_seed − φ_target`` across pairs and time (radians).
    seed_phase : (n_epochs,)
        Circular mean seed (HPC) phase in the window.
    target_phase : (n_epochs,)
        Circular mean target (ACC) phase in the window.
    """
    arr = _as_array(data)
    seed_idx = [int(i) for i in seed_idx]
    target_idx = [int(i) for i in target_idx]
    if len(seed_idx) == 0 or len(target_idx) == 0:
        raise ValueError("seed_idx and target_idx must be non-empty")

    phase = _hilbert_phase(arr, sfreq=sfreq, fmin=fmin, fmax=fmax)
    n_times = phase.shape[-1]
    if times is None:
        tmask = np.ones(n_times, dtype=bool)
    else:
        times = np.asarray(times, dtype=float).reshape(-1)
        if times.shape[0] != n_times:
            raise ValueError("times length must match n_times")
        t0 = -np.inf if tmin is None else float(tmin)
        t1 = np.inf if tmax is None else float(tmax)
        tmask = (times >= t0) & (times <= t1)
        if not np.any(tmask):
            raise ValueError(f"empty time mask for window [{tmin}, {tmax}]")

    n_epochs = phase.shape[0]
    R_lock = np.full(n_epochs, np.nan, dtype=float)
    dphi = np.full(n_epochs, np.nan, dtype=float)
    seed_phase = np.full(n_epochs, np.nan, dtype=float)
    target_phase = np.full(n_epochs, np.nan, dtype=float)

    for ei in range(n_epochs):
        comps = []
        seed_comps = []
        target_comps = []
        for s in seed_idx:
            ph_s = phase[ei, s, tmask]
            seed_comps.append(np.exp(1j * ph_s))
            for t in target_idx:
                ph_t = phase[ei, t, tmask]
                comps.append(np.exp(1j * (ph_s - ph_t)))
                target_comps.append(np.exp(1j * ph_t))
        if not comps:
            continue
        stack = np.concatenate(comps, axis=0)
        mean_c = np.nanmean(stack)
        R_lock[ei] = float(np.abs(mean_c))
        dphi[ei] = float(np.angle(mean_c))
        seed_phase[ei] = float(np.angle(np.nanmean(np.concatenate(seed_comps, axis=0))))
        target_phase[ei] = float(np.angle(np.nanmean(np.concatenate(target_comps, axis=0))))

    return R_lock, dphi, seed_phase, target_phase


def trial_item_ri_from_patterns(
    encoding_patterns: np.ndarray,
    cue_patterns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Item-specific reinstatement from window-mean multivariate patterns.

    encoding_patterns, cue_patterns
        Arrays ``(n_trials, n_electrodes)``. Electrode counts may differ;
        rows are zero-padded, z-scored, and L2-normalized before cosine.

    Returns
    -------
    match_sim : (n_trials,) diagonal of S
    mismatch_sim : (n_trials,) mean off-diagonal per row
    ri : (n_trials,) match − mismatch
    """
    E = np.asarray(encoding_patterns, dtype=float)
    R = np.asarray(cue_patterns, dtype=float)
    if E.ndim != 2 or R.ndim != 2:
        raise ValueError("encoding_patterns and cue_patterns must be 2D")
    if E.shape[0] != R.shape[0]:
        raise ValueError("trial count mismatch between encoding and cue patterns")
    n, p = E.shape
    _, q = R.shape
    d = max(int(p), int(q))
    if p < d:
        E = np.pad(E, ((0, 0), (0, d - p)), mode="constant")
    if q < d:
        R = np.pad(R, ((0, 0), (0, d - q)), mode="constant")
    E_z = E - E.mean(axis=1, keepdims=True)
    E_z = E_z / (np.std(E_z, axis=1, keepdims=True) + 1e-12)
    R_z = R - R.mean(axis=1, keepdims=True)
    R_z = R_z / (np.std(R_z, axis=1, keepdims=True) + 1e-12)
    E_n = E_z / (np.linalg.norm(E_z, axis=1, keepdims=True) + 1e-12)
    R_n = R_z / (np.linalg.norm(R_z, axis=1, keepdims=True) + 1e-12)
    S = E_n @ R_n.T
    match = np.diag(S).astype(float)
    if n > 1:
        mismatch = (S.sum(axis=1) - match) / float(n - 1)
    else:
        mismatch = np.zeros(n, dtype=float)
    ri = match - mismatch
    return match, mismatch, ri


__all__ = [
    "shuffle_target_trials",
    "zscore_vs_trial_shuffle",
    "zscore_vs_evoked_surrogate",
    "zscore_epoch_conn_vs_evoked_surrogate",
    "compute_epoch_psi",
    "zscore_epoch_psi",
    "window_pair_locking_and_dphi",
    "trial_item_ri_from_patterns",
]
