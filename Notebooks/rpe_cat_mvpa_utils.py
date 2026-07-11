"""
Time-resolved multivariate RPE-category (3-class) analysis utilities.

Implements crossnobis distance with Freedman–Lane nulls, LDA filters
(pairwise + canonical), Haufe activation patterns, cue-time projection
with surrogate z-scoring, and per-electrode trial-level Haufe contributions.

Reuses only: Subject, _get_smoothed_event_signal, _build_subject_regressors,
_build_lrt_trial_axis from mv_regression_utils.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

from mv_regression_utils import (
    Subject,
    _build_lrt_trial_axis,
    _build_subject_regressors,
    _encode_multiclass_target,
    _get_smoothed_event_signal,
)

# ---------------------------------------------------------------------------
# Design / residualization
# ---------------------------------------------------------------------------


def _design_matrix(C: Optional[np.ndarray], n_rows: int) -> np.ndarray:
    """Z = [1, C] with shape (n_rows, 1+q). C may be None (intercept only)."""
    if C is None or (isinstance(C, np.ndarray) and C.size == 0):
        return np.ones((n_rows, 1), dtype=np.float64)
    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.ndim == 1:
        C_arr = C_arr.reshape(-1, 1)
    if C_arr.shape[0] != n_rows:
        raise ValueError(f"C rows {C_arr.shape[0]} != n_rows {n_rows}")
    return np.column_stack([np.ones((n_rows, 1), dtype=np.float64), C_arr])


def residualize_rows(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return Y - Z @ pinv(Z) @ Y (rows residualized on columns of Z)."""
    Y_arr = np.asarray(Y, dtype=np.float64)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Y_arr.shape[0] != Z_arr.shape[0]:
        raise ValueError("Y and Z row mismatch")
    pinv = np.linalg.pinv(Z_arr)
    return Y_arr - (Z_arr @ (pinv @ Y_arr))


def residualize_train_test(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Z_train: np.ndarray,
    Z_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only fit: residualize Y_train and Y_test using coefficients from Z_train."""
    Z_tr = np.asarray(Z_train, dtype=np.float64)
    Z_te = np.asarray(Z_test, dtype=np.float64)
    Y_tr = np.asarray(Y_train, dtype=np.float64)
    Y_te = np.asarray(Y_test, dtype=np.float64)
    pinv = np.linalg.pinv(Z_tr)
    beta = pinv @ Y_tr
    return Y_tr - (Z_tr @ beta), Y_te - (Z_te @ beta)


def one_hot_from_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    """y integer 0..n_classes-1 -> (n, n_classes) one-hot."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    D = np.zeros((y.size, n_classes), dtype=np.float64)
    valid = (y >= 0) & (y < n_classes)
    D[valid, y[valid]] = 1.0
    return D


def freedman_lane_pseudo_onehot(
    D_hat: np.ndarray,
    resid: np.ndarray,
    perm_ix: np.ndarray,
) -> np.ndarray:
    """D* = D_hat + P @ resid (same permutation on rows)."""
    return D_hat + resid[np.asarray(perm_ix, dtype=np.int64)]


def labels_from_fl_onehot(D_star: np.ndarray) -> np.ndarray:
    """Map FL-reconstructed one-hot rows to class indices via argmax."""
    return np.argmax(D_star, axis=1).astype(np.int64)


def prepare_fl_onehot_state(
    y_labels: np.ndarray,
    Z: np.ndarray,
    n_classes: int,
) -> Dict[str, np.ndarray]:
    """OLS fit of one-hot D on Z; returns D_hat, resid, Z, pinv_Z, D_obs."""
    D = one_hot_from_labels(y_labels, n_classes)
    pinv_Z = np.linalg.pinv(Z)
    D_hat = Z @ (pinv_Z @ D)
    resid = D - D_hat
    return {"D_hat": D_hat, "resid": resid, "Z": Z, "pinv_Z": pinv_Z, "D_obs": D}


# ---------------------------------------------------------------------------
# Ledoit-Wolf + LDA filters
# ---------------------------------------------------------------------------


def pooled_ledoit_cov(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of rows of X (n x p)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return np.eye(X.shape[1], dtype=np.float64)
    lw = LedoitWolf().fit(X)
    return np.asarray(lw.covariance_, dtype=np.float64)


def inv_sym_psd(S: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    S = S + ridge * np.eye(S.shape[0], dtype=np.float64)
    return np.linalg.inv(S)


def pairwise_lda_filters(
    X: np.ndarray,
    y: np.ndarray,
    class_labels: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    For each ordered pair (i,j) with i<j, w_ij = Sigma^{-1} (mu_i - mu_j).
    Sigma = Ledoit-Wolf on all rows of X.
    Returns dict pair_name -> (p,) and Sigma.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    Sigma = pooled_ledoit_cov(X)
    S_inv = inv_sym_psd(Sigma)
    n_classes = len(class_labels)
    weights: Dict[str, np.ndarray] = {}
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            m_i = X[y == i].mean(axis=0) if np.any(y == i) else np.zeros(X.shape[1])
            m_j = X[y == j].mean(axis=0) if np.any(y == j) else np.zeros(X.shape[1])
            d = m_i - m_j
            key = f"{class_labels[i]}_vs_{class_labels[j]}"
            weights[key] = S_inv @ d
    return weights, Sigma


def canonical_lda_filters(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn LDA shrinkage='auto'; returns W (p, n_classes-1) from scalings_.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    uniq = np.unique(y[y >= 0])
    p = X.shape[1]
    n_comp = n_classes - 1
    if uniq.size < 2:
        return np.zeros((p, n_comp)), pooled_ledoit_cov(X)
    clf = LinearDiscriminantAnalysis(
        solver="eigen",
        shrinkage="auto",
        store_covariance=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        clf.fit(X, y)
    sc = getattr(clf, "scalings_", None)
    if sc is not None and sc.size > 0:
        W = np.asarray(sc, dtype=np.float64)[:, :n_comp]
    else:
        W = np.asarray(clf.coef_, dtype=np.float64).T[:, :n_comp]
    Sigma = pooled_ledoit_cov(X)
    return W, Sigma


def haufe_pattern(w: np.ndarray, Sigma_X: np.ndarray) -> np.ndarray:
    """A = Sigma_X @ w (vector or matrix p x k)."""
    return np.asarray(Sigma_X, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


def compute_omnibus_reactivation(z_canonical: np.ndarray) -> np.ndarray:
    """
    Omnibus reactivation strength from surrogate-z canonical projections.

    For each time cell and trial, sum z_k^2 over canonical LDA axes k.
    Under an ideal null, each z_k ~ N(0,1) so the sum is ~ Chi^2(n_axes).

    Parameters
    ----------
    z_canonical : ndarray
        Shape (..., n_axes), typically (n_fb, n_cue, n_trials, 2).

    Returns
    -------
    omnibus : ndarray
        Same leading dimensions as input without the last axis.
    """
    z = np.asarray(z_canonical, dtype=np.float64)
    if z.shape[-1] < 2:
        raise ValueError(
            "z_canonical must have at least 2 canonical axes in the last dimension"
        )
    return np.sum(z * z, axis=-1)


# ---------------------------------------------------------------------------
# Crossnobis (Walther-style cross-validated Mahalanobis product)
# ---------------------------------------------------------------------------


def crossnobis_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[np.ndarray],
    n_folds: int,
    seed: int,
    class_labels: List[str],
) -> Tuple[float, np.ndarray]:
    """
    Mean pairwise crossnobis statistic across CV folds.

    Per pair (i,j) and fold: c = (m_te_i - m_te_j).T @ S_inv_tr @ (m_tr_i - m_tr_j)
    Averaged over folds and pairs. Returns (mean_scalar, per_pair_values averaged over folds).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    n, p = X.shape
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    C = C[valid] if C is not None else None
    n = X.shape[0]
    if n < 4:
        return float("nan"), np.full(len(class_labels) * (len(class_labels) - 1) // 2, np.nan)

    Z = _design_matrix(C, n)
    n_classes = len(class_labels)
    counts = np.array([(y == k).sum() for k in range(n_classes)], dtype=int)
    if np.any(counts < 2):
        return float("nan"), np.full(n_classes * (n_classes - 1) // 2, np.nan)

    max_splits = int(counts.min())
    n_splits = max(2, min(n_folds, max_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pair_keys = [
        f"{class_labels[i]}_vs_{class_labels[j]}"
        for i in range(n_classes)
        for j in range(i + 1, n_classes)
    ]
    pair_sums = {k: 0.0 for k in pair_keys}
    pair_counts = {k: 0 for k in pair_keys}

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        Z_tr, Z_te = Z[tr], Z[te]
        X_tr_r, X_te_r = residualize_train_test(X_tr, X_te, Z_tr, Z_te)
        Sigma_tr = pooled_ledoit_cov(X_tr_r)
        S_inv = inv_sym_psd(Sigma_tr)
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                key = f"{class_labels[i]}_vs_{class_labels[j]}"
                tr_i = y_tr == i
                tr_j = y_tr == j
                te_i = y_te == i
                te_j = y_te == j
                if tr_i.sum() < 1 or tr_j.sum() < 1 or te_i.sum() < 1 or te_j.sum() < 1:
                    continue
                m_tr_i = X_tr_r[tr_i].mean(axis=0)
                m_tr_j = X_tr_r[tr_j].mean(axis=0)
                m_te_i = X_te_r[te_i].mean(axis=0)
                m_te_j = X_te_r[te_j].mean(axis=0)
                d_tr = m_tr_i - m_tr_j
                d_te = m_te_i - m_te_j
                c = float(d_te @ S_inv @ d_tr)
                pair_sums[key] += c
                pair_counts[key] += 1

    vals = []
    for k in pair_keys:
        if pair_counts[k] > 0:
            vals.append(pair_sums[k] / pair_counts[k])
        else:
            vals.append(np.nan)
    vals_arr = np.asarray(vals, dtype=np.float64)
    mean_stat = float(np.nanmean(vals_arr)) if np.any(np.isfinite(vals_arr)) else float("nan")
    return mean_stat, vals_arr


def permutation_p_value_greater(
    observed: float,
    null_samples: np.ndarray,
) -> float:
    """(sum(null >= obs) + 1) / (n + 1)."""
    null_samples = null_samples[np.isfinite(null_samples)]
    if null_samples.size == 0 or not np.isfinite(observed):
        return float("nan")
    return (float(np.sum(null_samples >= observed)) + 1.0) / (null_samples.size + 1.0)


def zscore_vs_null(observed: float, null_samples: np.ndarray) -> float:
    null_samples = null_samples[np.isfinite(null_samples)]
    if null_samples.size == 0 or not np.isfinite(observed):
        return float("nan")
    m = float(np.mean(null_samples))
    s = float(np.std(null_samples, ddof=0))
    s = max(s, 1e-12)
    return (float(observed) - m) / s


# ---------------------------------------------------------------------------
# Single subject encoding
# ---------------------------------------------------------------------------


@dataclass
class EncodingConfig:
    event: str = "feedback_start"
    predictor: str = "rpe_cat"
    covariates: Tuple[str, ...] = ("zpm",)
    freq_range: Tuple[float, float] = (70.0, 201.0)
    win_ms: int = 200
    downsample_factor: int = 5
    n_folds: int = 10
    n_perm: int = 250
    min_trials: int = 10
    seed: int = 42
    regressor_source: str = "learn"


def process_subject_encoding(
    subject: str,
    base_dir: str,
    load_dir: str,
    roi: str,
    cfg: EncodingConfig,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Per-feedback-time crossnobis + Freedman–Lane permutation null;
    Haufe per-electrode contributions saved in sidecar dict (caller writes NPZ).
    """
    subj_obj = Subject(subject, base_dir, load_dir, min_trials=cfg.min_trials, verbose=False)
    rows: List[Dict[str, Any]] = []
    sidecar: Optional[Dict[str, Any]] = None

    try:
        signal, times_sec, elec_labels = _get_smoothed_event_signal(
            subject_obj=subj_obj,
            roi=roi,
            event=cfg.event,
            freq_range=cfg.freq_range,
            downsample_factor=cfg.downsample_factor,
            win_ms=cfg.win_ms,
        )
        regressors, _ = _build_subject_regressors(
            subject_obj=subj_obj,
            event=cfg.event,
            predictor=cfg.predictor,
            covariates=list(cfg.covariates),
            regressor_source=cfg.regressor_source,
        )
    except Exception as exc:
        if verbose:
            print(f"[encoding] skip {subject} {roi}: {exc}")
        return pd.DataFrame(), None

    trial_ix = regressors["trial"].to_numpy(dtype=np.int64) - 1
    in_range = (trial_ix >= 0) & (trial_ix < signal.shape[0])
    dropped = int((~in_range).sum())
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]

    pred_series = regressors[cfg.predictor]
    cov_cols = [c for c in cfg.covariates if c in regressors.columns and c != cfg.predictor]
    C = regressors[cov_cols].to_numpy(dtype=np.float64) if cov_cols else None

    y_enc, class_labels = _encode_multiclass_target(pred_series.to_numpy(dtype=object))
    n_classes = len(class_labels)
    if C is None:
        core_valid = y_enc >= 0
    else:
        core_valid = (y_enc >= 0) & np.all(np.isfinite(C), axis=1)

    if int(core_valid.sum()) < max(3, cfg.min_trials) or n_classes < 2:
        return pd.DataFrame(), None

    n_trials_align = trial_ix.size
    n_elec = signal.shape[1]
    n_fb = signal.shape[2]

    # Preallocate sidecar tensors (only if we will fill)
    contribs_pw = np.full((n_fb, n_elec, n_trials_align, n_classes * (n_classes - 1) // 2), np.nan, dtype=np.float32)
    contribs_ca = np.full((n_fb, n_elec, n_trials_align, n_classes - 1), np.nan, dtype=np.float32)
    pair_key_order = [
        f"{class_labels[i]}_vs_{class_labels[j]}"
        for i in range(n_classes)
        for j in range(i + 1, n_classes)
    ]

    for t_idx, ts_sec in enumerate(times_sec):
        X_t = np.asarray(signal[trial_ix, :, t_idx], dtype=np.float64)
        valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
        n_valid = int(valid_t.sum())
        status = "ok"
        metric = np.nan
        z_m = np.nan
        raw_p = np.nan

        if n_valid < max(3, cfg.min_trials):
            status = "insufficient_trials"
        else:
            X_v = X_t[valid_t]
            y_v = y_enc[valid_t]
            C_v = C[valid_t] if C is not None else None
            Z_v = _design_matrix(C_v, n_valid)
            idx_valid = np.flatnonzero(valid_t)

            # FL reduced model on the same rows used for this timepoint
            fl_t = prepare_fl_onehot_state(y_v, Z_v, n_classes)

            # Observed crossnobis
            try:
                metric, _pair_vals = crossnobis_multiclass(
                    X_v, y_v, C_v, cfg.n_folds, cfg.seed + t_idx, class_labels
                )
            except Exception:
                status = "failed"
                metric = np.nan

            if status == "ok" and np.isfinite(metric):
                rng = np.random.default_rng(cfg.seed + t_idx)
                perm_stats = np.full(cfg.n_perm, np.nan, dtype=np.float64)
                for p_ix in range(cfg.n_perm):
                    perm_ix = rng.permutation(fl_t["resid"].shape[0])
                    D_star = freedman_lane_pseudo_onehot(
                        fl_t["D_hat"], fl_t["resid"], perm_ix
                    )
                    y_perm = labels_from_fl_onehot(D_star)
                    try:
                        perm_stats[p_ix], _ = crossnobis_multiclass(
                            X_v,
                            y_perm,
                            C_v,
                            cfg.n_folds,
                            cfg.seed + t_idx + 1000 + p_ix,
                            class_labels,
                        )
                    except Exception:
                        perm_stats[p_ix] = np.nan
                perm_stats = perm_stats[np.isfinite(perm_stats)]
                if perm_stats.size > 0:
                    z_m = zscore_vs_null(metric, perm_stats)
                    raw_p = permutation_p_value_greater(metric, perm_stats)
                else:
                    status = "failed"

            # Residualized X for filters + Haufe (same valid rows)
            if status == "ok" and np.isfinite(metric):
                X_res = residualize_rows(X_v, Z_v)
                Sigma_X = np.cov(X_res, rowvar=False)
                try:
                    w_dict, _Sig = pairwise_lda_filters(X_res, y_v, class_labels)
                    W_can, _ = canonical_lda_filters(X_res, y_v, n_classes)
                    for pi, key in enumerate(pair_key_order):
                        a_col = haufe_pattern(w_dict[key], Sigma_X)
                        for r in range(X_res.shape[0]):
                            tr_idx = int(idx_valid[r])
                            contribs_pw[t_idx, :, tr_idx, pi] = (
                                a_col * X_res[r, :]
                            ).astype(np.float32)
                    for ci in range(W_can.shape[1]):
                        a_col = haufe_pattern(W_can[:, ci], Sigma_X)
                        for r in range(X_res.shape[0]):
                            tr_idx = int(idx_valid[r])
                            contribs_ca[t_idx, :, tr_idx, ci] = (
                                a_col * X_res[r, :]
                            ).astype(np.float32)
                except Exception:
                    pass

        rows.append(
            {
                "predictor": cfg.predictor,
                "metric": metric,
                "metric_name": "crossnobis_mean_pairwise",
                "z_metric": z_m,
                "raw_p": raw_p,
                "ts": float(ts_sec * 1000.0),
                "subj": subject,
                "region": roi,
                "n_trials": n_valid,
                "n_electrodes": n_elec,
                "n_classes": n_classes,
                "class_labels_json": json.dumps(class_labels),
                "cv_folds": cfg.n_folds,
                "event": cfg.event,
                "covariates": ",".join(cov_cols),
                "n_perm": cfg.n_perm,
                "status": status,
                "dropped_alignment": dropped,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        sidecar = {
            "contribs_pairwise_haufe": contribs_pw,
            "contribs_canonical_haufe": contribs_ca,
            "pair_labels": np.array(pair_key_order, dtype=object),
            "feedback_times": np.asarray(times_sec, dtype=np.float32),
            "electrode_labels": np.asarray(elec_labels, dtype=str),
            "class_labels": np.array(class_labels, dtype=object),
            "trial_index": trial_ix.astype(np.int64),
            "meta": {
                "predictor": cfg.predictor,
                "event": cfg.event,
                "freq_range": cfg.freq_range,
                "win_ms": cfg.win_ms,
                "downsample_factor": cfg.downsample_factor,
                "covariates": list(cfg.covariates),
                "n_perm": cfg.n_perm,
                "seed": cfg.seed,
            },
        }
    return df, sidecar


def run_encoding_pipeline(
    subjects: List[str],
    rois: List[str],
    base_dir: str,
    load_dir: str,
    cfg: EncodingConfig,
    out_csv: str,
    contribs_dir: str,
    n_jobs: int = 1,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoints: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Subject×ROI encoding with optional per-subject CSV checkpoints."""
    os.makedirs(contribs_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _one(subj: str) -> Tuple[str, pd.DataFrame, List[Tuple[str, Optional[Dict[str, Any]]]]]:
        frames = []
        sidecars: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        for roi in rois:
            df_roi, sc = process_subject_encoding(subj, base_dir, load_dir, roi, cfg, verbose=verbose)
            if len(df_roi) > 0:
                frames.append(df_roi)
            sidecars.append((roi, sc))
        if not frames:
            return subj, pd.DataFrame(), sidecars
        return subj, pd.concat(frames, ignore_index=True), sidecars

    results: List[pd.DataFrame] = []
    if checkpoint_dir and not overwrite_checkpoints:
        done = {f.replace("_encoding.csv", "") for f in os.listdir(checkpoint_dir) if f.endswith("_encoding.csv")}
        subjects = [s for s in subjects if s not in done]
    if len(subjects) == 0:
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "_all_encoding.csv")):
            return pd.read_csv(os.path.join(checkpoint_dir, "_all_encoding.csv"))
        return pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()

    if n_jobs == 1:
        outs = [_one(s) for s in subjects]
    else:
        outs = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_one)(s) for s in subjects
        )

    for subj, df_sub, sidecars in outs:
        if len(df_sub) > 0:
            results.append(df_sub)
        for roi, sc in sidecars:
            if sc is not None:
                path_npz = os.path.join(contribs_dir, f"{subj}_{roi}_rpe_cat_electrode_contribs.npz")
                np.savez_compressed(
                    path_npz,
                    contribs_pairwise_haufe=sc["contribs_pairwise_haufe"],
                    contribs_canonical_haufe=sc["contribs_canonical_haufe"],
                    pair_labels=sc["pair_labels"],
                    feedback_times=sc["feedback_times"],
                    electrode_labels=sc["electrode_labels"],
                    class_labels=sc["class_labels"],
                    trial_index=sc["trial_index"],
                    meta_json=json.dumps(sc["meta"]),
                )
        if checkpoint_dir:
            df_sub.to_csv(os.path.join(checkpoint_dir, f"{subj}_encoding.csv"), index=False)

    all_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    all_df.to_csv(out_csv, index=False)
    if checkpoint_dir:
        all_df.to_csv(os.path.join(checkpoint_dir, "_all_encoding.csv"), index=False)
    return all_df


# ---------------------------------------------------------------------------
# Projection / reactivation
# ---------------------------------------------------------------------------


@dataclass
class ProjectionConfig:
    event_encoding: str = "feedback_start"
    event_projection: str = "cue_on"
    predictor: str = "rpe_cat"
    covariates: Tuple[str, ...] = ("zpm",)
    freq_range: Tuple[float, float] = (70.0, 201.0)
    win_ms: int = 200
    downsample_factor: int = 5
    min_trials: int = 10
    n_surrogates: int = 250
    seed: int = 42
    regressor_source: str = "learn"


def _fb_global_standardize(
    X_fb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-electrode mean/std from all trials × time on feedback stack."""
    # X_fb: (n, e, t)
    n, e, t = X_fb.shape
    flat = X_fb.transpose(1, 0, 2).reshape(e, n * t)
    mu = np.nanmean(flat, axis=1, keepdims=True).astype(np.float64)
    sd = np.nanstd(flat, axis=1, keepdims=True).astype(np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xn = (X_fb.astype(np.float64) - mu.reshape(1, e, 1)) / sd.reshape(1, e, 1)
    return Xn.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def process_subject_projection(
    subject: str,
    base_dir: str,
    load_dir: str,
    roi: str,
    cfg: ProjectionConfig,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Train LDA filters at each feedback time; project cue HFA; surrogate z."""
    subj_obj = Subject(subject, base_dir, load_dir, min_trials=cfg.min_trials, verbose=False)
    try:
        fb_hfa, fb_times, elec_labels = _get_smoothed_event_signal(
            subj_obj, roi, cfg.event_encoding, cfg.freq_range, cfg.downsample_factor, cfg.win_ms
        )
        cue_hfa, cue_times, _ = _get_smoothed_event_signal(
            subj_obj, roi, cfg.event_projection, cfg.freq_range, cfg.downsample_factor, cfg.win_ms
        )
        regressors, _ = _build_subject_regressors(
            subj_obj,
            cfg.event_encoding,
            cfg.predictor,
            list(cfg.covariates),
            regressor_source=cfg.regressor_source,
        )
    except Exception as exc:
        if verbose:
            print(f"[projection] skip {subject} {roi}: {exc}")
        return None

    trial_ix = regressors["trial"].to_numpy(dtype=np.int64) - 1
    in_range = (trial_ix >= 0) & (trial_ix < fb_hfa.shape[0]) & (trial_ix < cue_hfa.shape[0])
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]

    pred_series = regressors[cfg.predictor]
    cov_cols = [c for c in cfg.covariates if c in regressors.columns and c != cfg.predictor]
    C = regressors[cov_cols].to_numpy(dtype=np.float64) if cov_cols else None
    y_enc, class_labels = _encode_multiclass_target(pred_series.to_numpy(dtype=object))
    n_classes = len(class_labels)
    if C is None:
        core_valid = y_enc >= 0
    else:
        core_valid = (y_enc >= 0) & np.all(np.isfinite(C), axis=1)

    if int(core_valid.sum()) < cfg.min_trials or n_classes < 2:
        return None

    Z_core = _design_matrix(C[core_valid] if C is not None else None, int(core_valid.sum()))
    fl = prepare_fl_onehot_state(y_enc[core_valid], Z_core, n_classes)

    fb_aligned = np.asarray(fb_hfa[trial_ix], dtype=np.float32)
    cue_aligned = np.asarray(cue_hfa[trial_ix], dtype=np.float32)
    n_trials, n_elec, n_fb = fb_aligned.shape
    n_cue = cue_aligned.shape[2]

    # Global standardize using feedback
    fb_n, mu, sd = _fb_global_standardize(fb_aligned)
    cue_n = (cue_aligned.astype(np.float64) - mu.reshape(1, n_elec, 1)) / sd.reshape(1, n_elec, 1)
    cue_n = cue_n.astype(np.float32)

    pair_key_order = [
        f"{class_labels[i]}_vs_{class_labels[j]}"
        for i in range(n_classes)
        for j in range(i + 1, n_classes)
    ]

    W_pw = np.zeros((n_fb, n_elec, len(pair_key_order)), dtype=np.float32)
    W_ca = np.zeros((n_fb, n_elec, n_classes - 1), dtype=np.float32)
    P_pw = np.zeros_like(W_pw)
    P_ca = np.zeros_like(W_ca)

    re_pw = np.zeros((n_fb, n_cue, n_trials, len(pair_key_order)), dtype=np.float32)
    re_ca = np.zeros((n_fb, n_cue, n_trials, n_classes - 1), dtype=np.float32)

    C_full = C if C is not None else np.zeros((n_trials, 0))
    Z_full = _design_matrix(C_full, n_trials)

    for t_idx in range(n_fb):
        X_t = np.asarray(fb_n[:, :, t_idx], dtype=np.float64)
        valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
        if int(valid_t.sum()) < cfg.min_trials:
            continue
        X_v = X_t[valid_t]
        y_v = y_enc[valid_t]
        Z_v = Z_full[valid_t]
        X_res = residualize_rows(X_v, Z_v)
        Sigma_X = np.cov(X_res, rowvar=False)
        try:
            w_dict, _ = pairwise_lda_filters(X_res, y_v, class_labels)
            W_can, _ = canonical_lda_filters(X_res, y_v, n_classes)
            for pi, key in enumerate(pair_key_order):
                w = w_dict[key].astype(np.float32)
                W_pw[t_idx, :, pi] = w
                P_pw[t_idx, :, pi] = haufe_pattern(w.astype(np.float64), Sigma_X).astype(np.float32)
            for ci in range(W_can.shape[1]):
                w = W_can[:, ci].astype(np.float32)
                W_ca[t_idx, :, ci] = w
                P_ca[t_idx, :, ci] = haufe_pattern(w.astype(np.float64), Sigma_X).astype(np.float32)
            # projection: for each cue time, score = dot(x_cue, w)
            w_mat_pw = np.stack([w_dict[k] for k in pair_key_order], axis=1).astype(np.float64)
            for c_idx in range(n_cue):
                Xc = np.asarray(cue_n[:, :, c_idx], dtype=np.float64)
                Xc_r = residualize_rows(Xc, Z_full)
                # (n, n_pair) = Xc_r @ w_mat
                re_pw[t_idx, c_idx, :, :] = (Xc_r @ w_mat_pw).astype(np.float32)
            w_mat_ca = W_can.astype(np.float64)
            for c_idx in range(n_cue):
                Xc = np.asarray(cue_n[:, :, c_idx], dtype=np.float64)
                Xc_r = residualize_rows(Xc, Z_full)
                re_ca[t_idx, c_idx, :, :] = (Xc_r @ w_mat_ca).astype(np.float32)
        except Exception:
            continue

    # Surrogates
    rng = np.random.default_rng(cfg.seed)
    sum_pw = np.zeros_like(re_pw, dtype=np.float64)
    sum2_pw = np.zeros_like(re_pw, dtype=np.float64)
    sum_ca = np.zeros_like(re_ca, dtype=np.float64)
    sum2_ca = np.zeros_like(re_ca, dtype=np.float64)

    resid = fl["resid"]
    D_hat = fl["D_hat"]
    for _ in range(cfg.n_surrogates):
        perm_ix = rng.permutation(resid.shape[0])
        D_star = freedman_lane_pseudo_onehot(D_hat, resid, perm_ix)
        y_perm_full = np.full_like(y_enc, -1)
        y_perm_full[core_valid] = labels_from_fl_onehot(D_star)

        Wp_s = np.zeros_like(W_pw)
        Wc_s = np.zeros_like(W_ca)
        Rp_s = np.zeros_like(re_pw)
        Rc_s = np.zeros_like(re_ca)
        for t_idx in range(n_fb):
            X_t = np.asarray(fb_n[:, :, t_idx], dtype=np.float64)
            valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
            if int(valid_t.sum()) < cfg.min_trials:
                continue
            X_v = X_t[valid_t]
            y_v = y_perm_full[valid_t]
            if len(np.unique(y_v[y_v >= 0])) < 2:
                continue
            Z_v = Z_full[valid_t]
            X_res = residualize_rows(X_v, Z_v)
            Sigma_X = np.cov(X_res, rowvar=False)
            try:
                w_dict, _ = pairwise_lda_filters(X_res, y_v, class_labels)
                W_can, _ = canonical_lda_filters(X_res, y_v, n_classes)
                for pi, key in enumerate(pair_key_order):
                    w = w_dict[key].astype(np.float32)
                    Wp_s[t_idx, :, pi] = w
                for ci in range(W_can.shape[1]):
                    Wc_s[t_idx, :, ci] = W_can[:, ci].astype(np.float32)
                w_mat_pw = np.stack([w_dict[k] for k in pair_key_order], axis=1).astype(np.float64)
                for c_idx in range(n_cue):
                    Xc = np.asarray(cue_n[:, :, c_idx], dtype=np.float64)
                    Xc_r = residualize_rows(Xc, Z_full)
                    Rp_s[t_idx, c_idx, :, :] = (Xc_r @ w_mat_pw).astype(np.float32)
                w_mat_ca = W_can.astype(np.float64)
                for c_idx in range(n_cue):
                    Xc = np.asarray(cue_n[:, :, c_idx], dtype=np.float64)
                    Xc_r = residualize_rows(Xc, Z_full)
                    Rc_s[t_idx, c_idx, :, :] = (Xc_r @ w_mat_ca).astype(np.float32)
            except Exception:
                continue
        sum_pw += Rp_s
        sum2_pw += Rp_s.astype(np.float64) ** 2
        sum_ca += Rc_s
        sum2_ca += Rc_s.astype(np.float64) ** 2

    n_s = float(cfg.n_surrogates)
    mean_pw = (sum_pw / n_s).astype(np.float64)
    std_pw = np.sqrt(np.maximum(sum2_pw / n_s - mean_pw ** 2, 0.0))
    std_pw = np.maximum(std_pw, 1e-8)
    z_pw = ((re_pw.astype(np.float64) - mean_pw) / std_pw).astype(np.float32)

    mean_ca = (sum_ca / n_s).astype(np.float64)
    std_ca = np.sqrt(np.maximum(sum2_ca / n_s - mean_ca ** 2, 0.0))
    std_ca = np.maximum(std_ca, 1e-8)
    z_ca = ((re_ca.astype(np.float64) - mean_ca) / std_ca).astype(np.float32)

    # LRT axis for memory outcomes (same trials order as encoding trial_ix)
    lrt = _build_lrt_trial_axis(subj_obj, cfg.event_projection, cue_hfa.shape[0])
    trial_ix_lrt = lrt["trial_index_lrt"]
    mem_out = lrt["memory_outcomes_lrt"]
    cue_lrt = np.asarray(cue_hfa[trial_ix_lrt], dtype=np.float32)
    cue_lrt_n = (cue_lrt.astype(np.float64) - mu.reshape(1, n_elec, 1)) / sd.reshape(1, n_elec, 1)
    cue_lrt_n = cue_lrt_n.astype(np.float32)
    re_pw_lrt = np.full(
        (n_fb, n_cue, len(trial_ix_lrt), len(pair_key_order)), np.nan, dtype=np.float32
    )
    re_ca_lrt = np.full(
        (n_fb, n_cue, len(trial_ix_lrt), n_classes - 1), np.nan, dtype=np.float32
    )
    # If LRT n matches encoding n and same trial indexing, use encoding Z; else intercept only
    if len(trial_ix_lrt) == n_trials and np.array_equal(trial_ix_lrt, trial_ix):
        Z_lrt_full = Z_full
    else:
        Z_lrt_full = np.ones((cue_lrt_n.shape[0], 1), dtype=np.float64)

    for t_idx in range(n_fb):
        w_mat_pw = W_pw[t_idx].astype(np.float64)
        w_mat_ca = W_ca[t_idx].astype(np.float64)
        if not np.any(np.isfinite(w_mat_pw)):
            continue
        for c_idx in range(n_cue):
            Xc = np.asarray(cue_lrt_n[:, :, c_idx], dtype=np.float64)
            Xc_r = residualize_rows(Xc, Z_lrt_full)
            re_pw_lrt[t_idx, c_idx, :, :] = (Xc_r @ w_mat_pw).astype(np.float32)
            Xc_r2 = residualize_rows(Xc, Z_lrt_full)
            re_ca_lrt[t_idx, c_idx, :, :] = (Xc_r2 @ w_mat_ca).astype(np.float32)

    return {
        "reactivation_pairwise": re_pw,
        "reactivation_canonical": re_ca,
        "z_reactivation_pairwise": z_pw,
        "z_reactivation_canonical": z_ca,
        "filters_pairwise": W_pw,
        "filters_canonical": W_ca,
        "patterns_pairwise": P_pw,
        "patterns_canonical": P_ca,
        "feedback_times": fb_times.astype(np.float32),
        "cue_times": cue_times.astype(np.float32),
        "electrode_labels": np.asarray(elec_labels, dtype=str),
        "class_labels": np.array(class_labels, dtype=object),
        "pair_labels": np.array(pair_key_order, dtype=object),
        "memory_outcomes": mem_out,
        "trial_index": trial_ix.astype(np.int64),
        "trial_index_lrt": trial_ix_lrt.astype(np.int64),
        "reactivation_pairwise_lrt": re_pw_lrt,
        "reactivation_canonical_lrt": re_ca_lrt,
        "subject": subject,
        "roi": roi,
        "n_surrogates": cfg.n_surrogates,
        "covariates": ",".join(cov_cols),
        "freq_range": cfg.freq_range,
        "win_ms": cfg.win_ms,
        "downsample_factor": cfg.downsample_factor,
    }


def save_projection_npz(path: str, out: Dict[str, Any], proj_cfg: ProjectionConfig) -> None:
    """Write projection dict to compressed NPZ (same schema as run_projection_pipeline)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        reactivation_pairwise=out["reactivation_pairwise"],
        reactivation_canonical=out["reactivation_canonical"],
        z_reactivation_pairwise=out["z_reactivation_pairwise"],
        z_reactivation_canonical=out["z_reactivation_canonical"],
        filters_pairwise=out["filters_pairwise"],
        filters_canonical=out["filters_canonical"],
        patterns_pairwise=out["patterns_pairwise"],
        patterns_canonical=out["patterns_canonical"],
        feedback_times=out["feedback_times"],
        cue_times=out["cue_times"],
        electrode_labels=out["electrode_labels"],
        class_labels=out["class_labels"],
        pair_labels=out["pair_labels"],
        memory_outcomes=out["memory_outcomes"],
        trial_index=out["trial_index"],
        trial_index_lrt=out["trial_index_lrt"],
        reactivation_pairwise_lrt=out["reactivation_pairwise_lrt"],
        reactivation_canonical_lrt=out["reactivation_canonical_lrt"],
        meta_json=json.dumps(
            {
                "subject": out["subject"],
                "roi": out["roi"],
                "n_surrogates": out["n_surrogates"],
                "covariates": out["covariates"],
                "freq_range": out["freq_range"],
                "win_ms": out["win_ms"],
                "downsample_factor": out["downsample_factor"],
                "event_encoding": proj_cfg.event_encoding,
                "event_projection": proj_cfg.event_projection,
            }
        ),
    )


def run_projection_pipeline(
    subjects: List[str],
    rois: List[str],
    base_dir: str,
    load_dir: str,
    cfg: ProjectionConfig,
    out_dir: str,
    n_jobs: int = 1,
    verbose: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _one(sub_roi: Tuple[str, str]) -> None:
        s, r = sub_roi
        out = process_subject_projection(s, base_dir, load_dir, r, cfg, verbose=verbose)
        if out is None:
            return
        path = os.path.join(out_dir, r, f"{s}_rpe_cat_projection.npz")
        save_projection_npz(path, out, cfg)

    jobs = [(s, r) for s in subjects for r in rois]
    if n_jobs == 1:
        for j in jobs:
            _one(j)
    else:
        Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(delayed(_one)(j) for j in jobs)


# ---------------------------------------------------------------------------
# Load / summarize electrode contributions
# ---------------------------------------------------------------------------


def load_electrode_contribs(
    subj: str,
    roi: str,
    save_dir: str,
) -> Dict[str, Any]:
    path = os.path.join(save_dir, f"{subj}_{roi}_rpe_cat_electrode_contribs.npz")
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    return {
        "contribs_pairwise": z["contribs_pairwise_haufe"],
        "contribs_canonical": z["contribs_canonical_haufe"],
        "pair_labels": z["pair_labels"],
        "feedback_times": z["feedback_times"],
        "electrode_labels": z["electrode_labels"],
        "class_labels": z["class_labels"],
        "trial_index": z["trial_index"],
        "meta": meta,
    }


def summarize_electrode_contribs(
    subj: str,
    roi: str,
    loaded: Dict[str, Any],
    t_lo_ms: float,
    t_hi_ms: float,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Long-form (subj, roi, electrode, trial, pair, contribution) from Haufe pairwise tensor.
    Times are in ms (same convention as encoding CSV ts).
    """
    cp = np.asarray(loaded["contribs_pairwise"], dtype=np.float64)
    ft = np.asarray(loaded["feedback_times"], dtype=np.float64) * 1000.0
    labels = [str(x) for x in np.asarray(loaded["pair_labels"]).ravel()]
    elec = [str(x) for x in np.asarray(loaded["electrode_labels"]).ravel()]
    trial_ix = np.asarray(loaded["trial_index"], dtype=np.int64)
    mask = (ft >= t_lo_ms) & (ft <= t_hi_ms)
    if not np.any(mask):
        return pd.DataFrame(
            columns=["subj", "roi", "electrode", "trial_index", "pair", "contribution"]
        )
    sl = cp[mask]  # (n_t_win, n_elec, n_trials, n_pairs)
    if agg == "mean":
        v = np.nanmean(sl, axis=0)
    elif agg == "sum":
        v = np.nansum(sl, axis=0)
    else:
        raise ValueError("agg must be mean or sum")
    # v: (n_elec, n_trials, n_pairs)
    n_e, n_tr, n_p = v.shape
    rows = []
    for ei in range(n_e):
        for ti in range(n_tr):
            for pi in range(n_p):
                rows.append(
                    {
                        "subj": subj,
                        "roi": roi,
                        "electrode": elec[ei],
                        "trial_index": int(trial_ix[ti]) if ti < trial_ix.size else ti,
                        "pair": labels[pi],
                        "contribution": float(v[ei, ti, pi]),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "Subject",
    "EncodingConfig",
    "ProjectionConfig",
    "process_subject_encoding",
    "process_subject_projection",
    "run_encoding_pipeline",
    "run_projection_pipeline",
    "save_projection_npz",
    "load_electrode_contribs",
    "summarize_electrode_contribs",
    "crossnobis_multiclass",
    "residualize_rows",
    "compute_omnibus_reactivation",
]
