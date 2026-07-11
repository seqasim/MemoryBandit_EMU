from pathlib import Path
import inspect

import nbformat as nbf


NOTEBOOK_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-ACC_Cue_MultivariateMI_MultiRegion.ipynb"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text) + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(inspect.cleandoc(text) + "\n")


cells = [
    md(
        """
        # ACC Cue-Period Multivariate MI Across Target Regions

        This notebook computes cue-period HFA multivariate mutual information (MI)
        between `ACC` and each of:

        - `OFC`
        - `dmPFC`
        - `dlPFC`
        - `AMY`
        - `HPC`

        The primary inferential target is the cue-period window:

        - `(0.37, 0.60)` seconds

        The main question is whether window-averaged ACC↔ROI MI differs by:

        - `region_pair`
        - `hit_miss`
        - `rpe_cat`
        - `hit_miss × rpe_cat`
        - and whether those effects vary across `region_pair`
        """
    ),
    code(
        """
        %matplotlib inline
        %reload_ext autoreload
        %autoreload 2

        import os
        import warnings
        from itertools import combinations
        from pathlib import Path
        from typing import Any, Dict, List, Tuple

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import statsmodels.formula.api as smf
        from IPython.display import display
        from joblib import Parallel, delayed
        from scipy.stats import chi2, ttest_rel
        from statsmodels.stats.multitest import multipletests

        from plot_config import set_plot_style
        from mv_regression_utils import Subject
        from mv_granger_utils import build_hit_miss_trial_map, build_rpe_trial_map

        warnings.filterwarnings("ignore")
        set_plot_style()
        sns.set_style("whitegrid")
        np.random.seed(42)
        """
    ),
    code(
        """
        # -----------------------------
        # Config
        # -----------------------------
        base_dir = "/Volumes/T7 Shield"
        load_dir = f"{base_dir}/work/qasims01/MemoryBanditData/EMU"

        EVENT = "cue_on"
        DAY = "Day1"
        SEED = 42

        ACC_ROI = "ACC"
        TARGET_ROIS = ["OFC", "dmPFC", "dlPFC", "AMY", "HPC"]
        REGION_PAIR_ORDER = [f"{ACC_ROI}-{roi}" for roi in TARGET_ROIS]

        FREQ_RANGE = (70.0, 201.0)
        DOWNSAMPLE_FACTOR = 5
        WIN_MS = 200
        HFA_SMOOTH_METHOD = "movmean"
        HFA_ARTIFACT_ABS_THRESH = 10.0

        REG_EPS = 1e-6
        PCA_VAR_KEEP = 0.99
        PCA_MAX_COMPONENTS = 20
        N_PERM = 200

        WINDOW = (0.37, 0.60)
        MIN_TRIALS_PER_MODEL = 10

        N_JOBS = -1
        JOBLIB_BACKEND = "loky"

        output_root = Path(
            "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/analysis_outputs/"
            "acc_cue_multivariate_mi_multiregion"
        )
        fig_root = output_root / "figures"
        output_root.mkdir(parents=True, exist_ok=True)
        fig_root.mkdir(parents=True, exist_ok=True)

        print("Output root:", output_root)
        print("Target ROIs:", TARGET_ROIS)
        print("Cue analysis window:", WINDOW)
        """
    ),
    code(
        """
        # -----------------------------
        # Helpers
        # -----------------------------
        def _get_smoothed_event_signal(
            subject_obj: Subject,
            roi: str,
            event: str,
            freq_range: Tuple[float, float],
            downsample_factor: int,
            win_ms: int,
            smooth_method: str = "movmean",
            artifact_abs_thresh: float = 10.0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            neural = subject_obj._get_or_build_event_neural_data(
                event=event,
                roi=roi,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
                smooth_method=smooth_method,
                artifact_abs_thresh=artifact_abs_thresh,
            )
            signal = np.asarray(neural["hfa"], dtype=np.float64)
            times = np.asarray(neural["times"], dtype=np.float64)
            labels = np.asarray(neural.get("electrode_labels", []), dtype=str)
            if labels.size == 0:
                raise ValueError(f"Missing electrode_labels for {subject_obj.subject} {event}/{roi}")
            return signal, times, labels


        def _resolve_trial_var_for_event(df: pd.DataFrame, event: str) -> str:
            if str(event).strip().lower() == "cue_on":
                for candidate in ("trials_mem", "trials_gamble", "trials"):
                    if candidate in df.columns:
                        return candidate
            for candidate in ("trials", "trials_gamble", "trials_mem"):
                if candidate in df.columns:
                    return candidate
            raise ValueError(f"Could not determine trial variable for event={event!r}")


        def _build_cue_trial_axis(subject_obj: Subject, n_epochs: int, event: str = "cue_on") -> Dict[str, Any]:
            cdf = subject_obj.combined_df[
                (subject_obj.combined_df.participant == subject_obj.subject)
                & (subject_obj.combined_df.condition == DAY)
            ].copy()
            if len(cdf) == 0:
                raise ValueError("no_day1_trials")

            trial_col = _resolve_trial_var_for_event(cdf, event)
            cdf = cdf.dropna(subset=[trial_col]).copy()
            trial_vals = pd.to_numeric(cdf[trial_col], errors="coerce").to_numpy(dtype=np.float64)
            in_range = np.isfinite(trial_vals) & ((trial_vals - 1) >= 0) & ((trial_vals - 1) < int(n_epochs))
            if int(np.sum(in_range)) == 0:
                raise ValueError("no_in_range_trials")

            trial_ix = trial_vals[in_range].astype(np.int64) - 1
            return {
                "trial_index": trial_ix.astype(np.int64, copy=False),
                "trial_number": trial_vals[in_range].astype(np.int64, copy=False),
                "trial_col": trial_col,
                "n_dropped": int((~in_range).sum()),
            }


        def _build_behavior_labels_for_subject(subject_obj: Subject, n_epochs: int) -> pd.DataFrame:
            hm_map = build_hit_miss_trial_map(
                subject_obj._combined_df,
                subj_id=subject_obj.subject,
                n_epochs=int(n_epochs),
                day=DAY,
                event=EVENT,
            )
            rpe_map = build_rpe_trial_map(
                subject_obj._combined_df,
                subj_id=subject_obj.subject,
                n_epochs=int(n_epochs),
                day=DAY,
                event=EVENT,
            )

            if len(hm_map):
                hm_map = hm_map[hm_map["alignment_valid"]].copy()
                hm = hm_map[["epoch_idx", "hit_miss"]].copy()
            else:
                hm = pd.DataFrame(columns=["epoch_idx", "hit_miss"])

            if len(rpe_map):
                rpe_map = rpe_map[rpe_map["alignment_valid"]].copy()
                rp = rpe_map[["epoch_idx", "rpe_cat"]].copy()
            else:
                rp = pd.DataFrame(columns=["epoch_idx", "rpe_cat"])

            lab = hm.merge(rp, on="epoch_idx", how="outer")
            if len(lab) == 0:
                lab = pd.DataFrame({"epoch_idx": np.arange(int(n_epochs), dtype=int)})
            lab["subject"] = subject_obj.subject
            return lab


        def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
            X = np.asarray(X, dtype=np.float64)
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True)
            sd = np.where(np.isfinite(sd) & (sd > eps), sd, np.nan)
            Z = (X - mu) / sd
            finite_col = np.isfinite(Z).all(axis=0)
            Z = Z[:, finite_col]
            return Z, finite_col


        def _pca_reduce(
            X: np.ndarray,
            var_keep: float = 0.99,
            max_components: int = 20,
            eps: float = 1e-10,
        ) -> np.ndarray:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
                return X
            U, s, _ = np.linalg.svd(X, full_matrices=False)
            var = s ** 2
            if not np.any(var > eps):
                return X[:, :1]
            csum = np.cumsum(var) / np.sum(var)
            k = int(np.searchsorted(csum, float(var_keep)) + 1)
            k = max(1, min(k, int(max_components), X.shape[1], X.shape[0] - 1))
            return U[:, :k] * s[:k]


        def _inv_sqrtm_spd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
            w, V = np.linalg.eigh(S)
            w = np.where(w > eps, w, eps)
            return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


        def _fit_gaussian_cca_local_model(X: np.ndarray, Y: np.ndarray, reg_eps: float = 1e-6) -> Dict[str, Any]:
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
                return {"ok": False, "reason": "bad_input"}

            n = int(X.shape[0])
            px = int(X.shape[1])
            py = int(Y.shape[1])
            if n < 4 or px < 1 or py < 1:
                return {"ok": False, "reason": "insufficient_shape"}

            Xc = X - np.mean(X, axis=0, keepdims=True)
            Yc = Y - np.mean(Y, axis=0, keepdims=True)

            Sxx = (Xc.T @ Xc) / max(1, n - 1)
            Syy = (Yc.T @ Yc) / max(1, n - 1)
            Sxy = (Xc.T @ Yc) / max(1, n - 1)

            Sxx = Sxx + (float(reg_eps) * np.eye(px))
            Syy = Syy + (float(reg_eps) * np.eye(py))

            Wx = _inv_sqrtm_spd(Sxx, eps=reg_eps)
            Wy = _inv_sqrtm_spd(Syy, eps=reg_eps)

            C = Wx @ Sxy @ Wy
            U, s, Vt = np.linalg.svd(C, full_matrices=False)

            rho = np.asarray(s, dtype=np.float64)
            rho = np.clip(rho, 0.0, 1.0 - 1e-10)
            keep = rho > 1e-10
            if not np.any(keep):
                return {"ok": False, "reason": "no_valid_cc"}

            rho = rho[keep]
            U = U[:, keep]
            V = Vt.T[:, keep]

            Ax = Wx @ U
            By = Wy @ V

            Uscore = Xc @ Ax
            Vscore = Yc @ By

            su = np.std(Uscore, axis=0, ddof=1)
            sv = np.std(Vscore, axis=0, ddof=1)
            su = np.where(np.isfinite(su) & (su > 1e-12), su, 1.0)
            sv = np.where(np.isfinite(sv) & (sv > 1e-12), sv, 1.0)
            Uscore = Uscore / su
            Vscore = Vscore / sv

            global_mi = float(-0.5 * np.sum(np.log(1.0 - (rho ** 2))))
            return {
                "ok": True,
                "rho": rho,
                "Uscore": Uscore,
                "Vscore": Vscore,
                "global_mi": global_mi,
                "n_cc": int(rho.size),
            }


        def _local_mi_from_canonical(Uscore: np.ndarray, Vscore: np.ndarray, rho: np.ndarray) -> np.ndarray:
            Uscore = np.asarray(Uscore, dtype=np.float64)
            Vscore = np.asarray(Vscore, dtype=np.float64)
            rho = np.asarray(rho, dtype=np.float64)
            one_minus = np.clip(1.0 - rho ** 2, 1e-12, None)
            const = -0.5 * np.log(one_minus)
            quad = -((rho ** 2) * (Uscore ** 2 + Vscore ** 2) - (2.0 * rho * Uscore * Vscore)) / (2.0 * one_minus)
            local_per_cc = const.reshape(1, -1) + quad
            return np.sum(local_per_cc, axis=1)


        def _prepare_subject_pair(subject_obj: Subject, target_roi: str) -> Dict[str, Any]:
            acc_hfa, cue_times_acc, acc_labels = _get_smoothed_event_signal(
                subject_obj=subject_obj,
                roi=ACC_ROI,
                event=EVENT,
                freq_range=FREQ_RANGE,
                downsample_factor=DOWNSAMPLE_FACTOR,
                win_ms=WIN_MS,
                smooth_method=HFA_SMOOTH_METHOD,
                artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
            )
            roi_hfa, cue_times_roi, roi_labels = _get_smoothed_event_signal(
                subject_obj=subject_obj,
                roi=target_roi,
                event=EVENT,
                freq_range=FREQ_RANGE,
                downsample_factor=DOWNSAMPLE_FACTOR,
                win_ms=WIN_MS,
                smooth_method=HFA_SMOOTH_METHOD,
                artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
            )

            if not np.allclose(np.asarray(cue_times_acc), np.asarray(cue_times_roi), atol=1e-6, rtol=0):
                raise ValueError("cue_time_vectors_mismatch")

            n_epochs = int(min(acc_hfa.shape[0], roi_hfa.shape[0]))
            axis_pack = _build_cue_trial_axis(subject_obj=subject_obj, n_epochs=n_epochs, event=EVENT)
            trial_index = np.asarray(axis_pack["trial_index"], dtype=np.int64)
            trial_number = np.asarray(axis_pack["trial_number"], dtype=np.int64)

            in_range = (trial_index >= 0) & (trial_index < n_epochs)
            trial_index = trial_index[in_range]
            trial_number = trial_number[in_range]
            if trial_index.size == 0:
                raise ValueError("no_valid_trial_index_after_range_filter")

            acc = np.asarray(acc_hfa[trial_index, :, :], dtype=np.float64)
            roi = np.asarray(roi_hfa[trial_index, :, :], dtype=np.float64)
            labels_df = _build_behavior_labels_for_subject(subject_obj=subject_obj, n_epochs=n_epochs)

            return {
                "acc": acc,
                "roi": roi,
                "cue_times": np.asarray(cue_times_acc, dtype=np.float64),
                "acc_labels": np.asarray(acc_labels, dtype=str),
                "roi_labels": np.asarray(roi_labels, dtype=str),
                "trial_index": trial_index,
                "trial_number": trial_number,
                "labels_df": labels_df,
            }


        def _run_subject_target_mi(subject_id: str, target_roi: str) -> Dict[str, Any]:
            region_pair = f"{ACC_ROI}-{target_roi}"
            try:
                subj = Subject(subject_id, base_dir, load_dir, min_trials=10, verbose=False)
                pack = _prepare_subject_pair(subj, target_roi=target_roi)

                acc = pack["acc"]
                roi = pack["roi"]
                cue_times = pack["cue_times"]
                trial_index = pack["trial_index"]
                trial_number = pack["trial_number"]

                n_trials, n_acc, n_t = acc.shape
                n_trials_r, n_roi, n_t_r = roi.shape
                n_t_use = int(min(n_t, n_t_r))
                if n_trials_r != n_trials:
                    raise ValueError("trial_count_mismatch")
                if n_acc < 1 or n_roi < 1:
                    raise ValueError("zero_electrodes")

                rows: List[Dict[str, Any]] = []
                rng = np.random.default_rng(SEED + abs(hash((subject_id, target_roi))) % (2 ** 20))

                for t in range(n_t_use):
                    X = np.asarray(acc[:, :, t], dtype=np.float64)
                    Y = np.asarray(roi[:, :, t], dtype=np.float64)

                    Xz, _ = _zscore_cols(X)
                    Yz, _ = _zscore_cols(Y)
                    if Xz.ndim != 2 or Yz.ndim != 2 or Xz.shape[0] != n_trials or Yz.shape[0] != n_trials:
                        continue
                    if Xz.shape[1] < 1 or Yz.shape[1] < 1:
                        continue

                    Xr = _pca_reduce(Xz, var_keep=PCA_VAR_KEEP, max_components=PCA_MAX_COMPONENTS)
                    Yr = _pca_reduce(Yz, var_keep=PCA_VAR_KEEP, max_components=PCA_MAX_COMPONENTS)
                    if Xr.shape[1] < 1 or Yr.shape[1] < 1:
                        continue

                    model = _fit_gaussian_cca_local_model(Xr, Yr, reg_eps=REG_EPS)
                    if not model.get("ok", False):
                        continue

                    local_mi = _local_mi_from_canonical(model["Uscore"], model["Vscore"], model["rho"])
                    if local_mi.shape[0] != n_trials:
                        continue

                    null_local = np.full((int(N_PERM), n_trials), np.nan, dtype=np.float64)
                    for pidx in range(int(N_PERM)):
                        yp = Yr[rng.permutation(n_trials), :]
                        pm = _fit_gaussian_cca_local_model(Xr, yp, reg_eps=REG_EPS)
                        if pm.get("ok", False):
                            plocal = _local_mi_from_canonical(pm["Uscore"], pm["Vscore"], pm["rho"])
                            if plocal.shape[0] == n_trials:
                                null_local[pidx, :] = plocal

                    mi_local_null_mean = np.nanmean(null_local, axis=0)
                    mi_local_null_std = np.nanstd(null_local, axis=0)
                    mi_local_null_std = np.where(
                        np.isfinite(mi_local_null_std) & (mi_local_null_std > 1e-12),
                        mi_local_null_std,
                        np.nan,
                    )
                    z_mi_perm_local = (local_mi - mi_local_null_mean) / mi_local_null_std

                    for i in range(n_trials):
                        rows.append(
                            {
                                "subject": subject_id,
                                "target_roi": target_roi,
                                "region_pair": region_pair,
                                "band": "hfa",
                                "trial_row": int(i),
                                "trial_index_epoch0": int(trial_index[i]),
                                "trial_number_epoch1": int(trial_number[i]),
                                "time_idx": int(t),
                                "time_s": float(cue_times[t]),
                                "mi_local": float(local_mi[i]),
                                "mi_local_null_mean": float(mi_local_null_mean[i]) if np.isfinite(mi_local_null_mean[i]) else np.nan,
                                "mi_local_null_std": float(mi_local_null_std[i]) if np.isfinite(mi_local_null_std[i]) else np.nan,
                                "z_mi_perm": float(z_mi_perm_local[i]) if np.isfinite(z_mi_perm_local[i]) else np.nan,
                                "global_mi": float(model["global_mi"]),
                                "n_cc": int(model["n_cc"]),
                                "n_acc_electrodes_raw": int(n_acc),
                                "n_target_electrodes_raw": int(n_roi),
                            }
                        )

                audit_row = {
                    "subject": subject_id,
                    "target_roi": target_roi,
                    "region_pair": region_pair,
                    "status": "ok" if len(rows) else "no_rows",
                    "error": "",
                    "n_rows": int(len(rows)),
                    "n_trials_aligned": int(n_trials),
                    "n_acc_electrodes": int(n_acc),
                    "n_target_electrodes": int(n_roi),
                    "n_timepoints": int(n_t_use),
                }
                return {
                    "subject": subject_id,
                    "target_roi": target_roi,
                    "region_pair": region_pair,
                    "status": audit_row["status"],
                    "error": "",
                    "rows": rows,
                    "audit_row": audit_row,
                    "labels_df": pack["labels_df"],
                }
            except Exception as exc:
                audit_row = {
                    "subject": subject_id,
                    "target_roi": target_roi,
                    "region_pair": region_pair,
                    "status": "error",
                    "error": str(exc),
                    "n_rows": 0,
                    "n_trials_aligned": 0,
                    "n_acc_electrodes": np.nan,
                    "n_target_electrodes": np.nan,
                    "n_timepoints": np.nan,
                }
                return {
                    "subject": subject_id,
                    "target_roi": target_roi,
                    "region_pair": region_pair,
                    "status": "error",
                    "error": str(exc),
                    "rows": [],
                    "audit_row": audit_row,
                    "labels_df": pd.DataFrame(columns=["subject", "epoch_idx", "hit_miss", "rpe_cat"]),
                }


        def _term_mask(names: List[str], include: List[str], exact_colons: int) -> np.ndarray:
            out = []
            for name in names:
                if name == "Intercept":
                    out.append(False)
                    continue
                if name.count(":") != exact_colons:
                    out.append(False)
                    continue
                out.append(all(token in name for token in include))
            return np.asarray(out, dtype=bool)


        def wald_test_from_mask(result, mask: np.ndarray, label: str) -> Dict[str, Any]:
            names = list(result.fe_params.index)
            if mask.sum() == 0:
                return {"term": label, "df": 0, "chi2": np.nan, "p_value": np.nan}

            beta = np.asarray(result.fe_params, dtype=float)
            cov = np.asarray(result.cov_params().loc[names, names], dtype=float)
            idx = np.flatnonzero(mask)
            b = beta[idx]
            V = cov[np.ix_(idx, idx)]
            try:
                stat = float(b.T @ np.linalg.pinv(V) @ b)
                p_value = float(chi2.sf(stat, df=len(idx)))
            except Exception:
                stat = np.nan
                p_value = np.nan
            return {"term": label, "df": int(len(idx)), "chi2": stat, "p_value": p_value}


        def fit_primary_mixedlm(data: pd.DataFrame):
            formula = "mi_window_mean ~ C(region_pair) * C(hit_miss) * C(rpe_cat)"
            methods = ["lbfgs", "powell", "cg"]
            last_error = None
            for method in methods:
                try:
                    model = smf.mixedlm(formula=formula, data=data, groups=data["subject"])
                    result = model.fit(reml=False, method=method, maxiter=500, disp=False)
                    return result, method
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"MixedLM failed for all methods. Last error: {last_error}")


        def build_omnibus_table(result) -> pd.DataFrame:
            names = list(result.fe_params.index)
            tests = [
                ("region_pair", _term_mask(names, ["C(region_pair)"], exact_colons=0)),
                ("hit_miss", _term_mask(names, ["C(hit_miss)"], exact_colons=0)),
                ("rpe_cat", _term_mask(names, ["C(rpe_cat)"], exact_colons=0)),
                ("region_pair:hit_miss", _term_mask(names, ["C(region_pair)", "C(hit_miss)"], exact_colons=1)),
                ("region_pair:rpe_cat", _term_mask(names, ["C(region_pair)", "C(rpe_cat)"], exact_colons=1)),
                ("hit_miss:rpe_cat", _term_mask(names, ["C(hit_miss)", "C(rpe_cat)"], exact_colons=1)),
                (
                    "region_pair:hit_miss:rpe_cat",
                    _term_mask(names, ["C(region_pair)", "C(hit_miss)", "C(rpe_cat)"], exact_colons=2),
                ),
            ]
            rows = [wald_test_from_mask(result, mask, label) for label, mask in tests]
            out = pd.DataFrame(rows)
            valid = np.isfinite(out["p_value"].to_numpy(dtype=float))
            if int(np.sum(valid)) > 0:
                out.loc[valid, "p_fdr"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
            else:
                out["p_fdr"] = np.nan
            return out


        def build_pairwise_region_contrasts(window_df: pd.DataFrame) -> pd.DataFrame:
            subj_region = (
                window_df.groupby(["subject", "region_pair"], observed=True)["mi_window_mean"]
                .mean()
                .reset_index()
                .pivot(index="subject", columns="region_pair", values="mi_window_mean")
            )
            rows = []
            for a, b in combinations(REGION_PAIR_ORDER, 2):
                if a not in subj_region.columns or b not in subj_region.columns:
                    continue
                pair_df = subj_region[[a, b]].dropna()
                n_subjects = int(len(pair_df))
                if n_subjects < 2:
                    rows.append(
                        {
                            "region_a": a,
                            "region_b": b,
                            "mean_diff_a_minus_b": np.nan,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "n_subjects": n_subjects,
                        }
                    )
                    continue
                diff = pair_df[a] - pair_df[b]
                t_stat, p_value = ttest_rel(pair_df[a], pair_df[b], nan_policy="omit")
                rows.append(
                    {
                        "region_a": a,
                        "region_b": b,
                        "mean_diff_a_minus_b": float(np.nanmean(diff)),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                        "n_subjects": n_subjects,
                    }
                )
            out = pd.DataFrame(rows)
            if len(out):
                valid = np.isfinite(out["p_value"].to_numpy(dtype=float))
                out["p_fdr"] = np.nan
                if int(np.sum(valid)) > 0:
                    out.loc[valid, "p_fdr"] = multipletests(out.loc[valid, "p_value"], method="fdr_bh")[1]
            return out.sort_values(["p_fdr", "p_value", "region_a", "region_b"], na_position="last").reset_index(drop=True)
        """
    ),
    code(
        """
        # -----------------------------
        # Subject list + parallel run
        # -----------------------------
        demo_file = f"{base_dir}/projects/guLab/Salman/EMU/MemoryBanditsDatabase.xlsx"
        demo_data = pd.read_excel(demo_file, engine="openpyxl")
        subj_df = demo_data[(demo_data.LongVersion == 1) & (demo_data.Minerva == 1)].copy()
        subj_df = subj_df[~subj_df["MSSMCode"].isin(["MS015", "MS024", "MS034", "MS038"])].copy()
        subjects = sorted(subj_df["MSSMCode"].dropna().astype(str).unique().tolist())

        print("Eligible subjects:", len(subjects))
        print("First 10 subjects:", subjects[:10])

        job_args = [(sid, roi) for sid in subjects for roi in TARGET_ROIS]
        results = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND, verbose=5)(
            delayed(_run_subject_target_mi)(sid, roi) for sid, roi in job_args
        )

        audit_df = pd.DataFrame([r["audit_row"] for r in results]).sort_values(["subject", "target_roi"]).reset_index(drop=True)
        frames = [pd.DataFrame(r["rows"]) for r in results if len(r["rows"]) > 0]
        hfa_mi_long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        label_frames = [
            r["labels_df"]
            for r in results
            if isinstance(r.get("labels_df"), pd.DataFrame) and len(r["labels_df"]) > 0
        ]
        labels_df_all = (
            pd.concat(label_frames, ignore_index=True)
            .drop_duplicates(subset=["subject", "epoch_idx"])
            .reset_index(drop=True)
            if label_frames else
            pd.DataFrame(columns=["subject", "epoch_idx", "hit_miss", "rpe_cat"])
        )

        if len(hfa_mi_long_df):
            n_before = len(hfa_mi_long_df)
            lab = labels_df_all.rename(columns={"epoch_idx": "trial_index_epoch0"}).copy()
            lab["trial_index_epoch0"] = pd.to_numeric(lab["trial_index_epoch0"], errors="coerce")
            lab = lab.dropna(subset=["trial_index_epoch0"])
            lab["trial_index_epoch0"] = lab["trial_index_epoch0"].astype(int)

            hfa_mi_long_df = hfa_mi_long_df.merge(
                lab[["subject", "trial_index_epoch0", "hit_miss", "rpe_cat"]].drop_duplicates(),
                on=["subject", "trial_index_epoch0"],
                how="left",
            )
            hfa_mi_long_df["label_available"] = hfa_mi_long_df["hit_miss"].notna() | hfa_mi_long_df["rpe_cat"].notna()
            print("Row count unchanged after label merge:", n_before == len(hfa_mi_long_df))

        audit_df.to_csv(output_root / "subject_target_audit.csv", index=False)
        if len(hfa_mi_long_df):
            hfa_mi_long_df.to_csv(output_root / "hfa_mi_trial_time_long.csv", index=False)
            try:
                hfa_mi_long_df.to_parquet(output_root / "hfa_mi_trial_time_long.parquet", index=False)
            except Exception:
                pass

        print("Successful subject-target fits:", int((audit_df["status"] == "ok").sum()), "of", len(audit_df))
        print("Long MI rows:", len(hfa_mi_long_df))
        display(audit_df.head(10))
        if len(hfa_mi_long_df):
            display(hfa_mi_long_df.head(10))
        """
    ),
    code(
        """
        # -----------------------------
        # Validation + coverage
        # -----------------------------
        if len(hfa_mi_long_df) == 0:
            raise ValueError("No MI rows were generated.")

        required_cols = [
            "subject",
            "target_roi",
            "region_pair",
            "trial_index_epoch0",
            "trial_number_epoch1",
            "time_idx",
            "time_s",
            "mi_local",
            "z_mi_perm",
            "hit_miss",
            "rpe_cat",
        ]
        missing_cols = [c for c in required_cols if c not in hfa_mi_long_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required MI columns: {missing_cols}")

        key_cols = ["subject", "region_pair", "trial_index_epoch0", "time_idx"]
        dup_n = int(hfa_mi_long_df.duplicated(subset=key_cols).sum())
        print("Duplicate key rows:", dup_n)

        coverage_df = (
            hfa_mi_long_df.groupby(["subject", "region_pair"], observed=True)
            .agg(
                label_available_rate=("label_available", "mean"),
                n_trials=("trial_index_epoch0", "nunique"),
                n_timepoints=("time_idx", "nunique"),
            )
            .reset_index()
        )
        display(coverage_df.head(15))

        pair_counts = (
            audit_df.groupby("region_pair", observed=True)
            .agg(
                n_subject_targets=("subject", "nunique"),
                n_ok=("status", lambda x: int(np.sum(np.asarray(x) == "ok"))),
            )
            .reset_index()
        )
        display(pair_counts)
        """
    ),
    code(
        """
        # -----------------------------
        # Time-resolved summaries
        # -----------------------------
        plot_df = hfa_mi_long_df.copy()
        plot_df["region_pair"] = plot_df["region_pair"].astype(str)

        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        sns.lineplot(
            data=plot_df,
            x="time_s",
            y="z_mi_perm",
            hue="region_pair",
            hue_order=REGION_PAIR_ORDER,
            errorbar="se",
            ax=ax,
        )
        ax.axvspan(WINDOW[0], WINDOW[1], color="k", alpha=0.08)
        ax.set_title("Cue-period local MI by ACC-target pair")
        ax.set_xlabel("Cue time (s)")
        ax.set_ylabel("Permutation-z local MI")
        fig.tight_layout()
        fig.savefig(fig_root / "mi_timecourse_by_region_pair.png", dpi=200, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        """
        # -----------------------------
        # Window averaging in 0.37-0.60 s
        # -----------------------------
        time_mask = (
            (hfa_mi_long_df["time_s"].to_numpy(dtype=float) >= WINDOW[0]) &
            (hfa_mi_long_df["time_s"].to_numpy(dtype=float) <= WINDOW[1])
        )
        print("Rows inside analysis window:", int(np.sum(time_mask)))
        if not np.any(time_mask):
            raise ValueError(f"No rows fell inside window {WINDOW}")

        window_long_df = hfa_mi_long_df.loc[time_mask].copy()
        window_long_df["region_pair"] = window_long_df["region_pair"].astype(str)

        window_df = (
            window_long_df.groupby(
                ["subject", "target_roi", "region_pair", "trial_index_epoch0", "trial_number_epoch1", "hit_miss", "rpe_cat"],
                as_index=False,
                observed=True,
            )
            .agg(
                mi_window_mean=("mi_local", "mean"),
                z_mi_window_mean=("z_mi_perm", "mean"),
                n_window_timepoints=("time_idx", "nunique"),
                n_acc_electrodes_raw=("n_acc_electrodes_raw", "max"),
                n_target_electrodes_raw=("n_target_electrodes_raw", "max"),
            )
        )

        window_df = window_df.replace([np.inf, -np.inf], np.nan)
        window_df = window_df.dropna(subset=["mi_window_mean", "hit_miss", "rpe_cat"]).copy()
        window_df = window_df[window_df["hit_miss"].isin(["hit", "miss"])].copy()
        window_df["rpe_cat"] = window_df["rpe_cat"].astype(str)
        window_df["hit_miss"] = window_df["hit_miss"].astype(str)
        window_df["region_pair"] = window_df["region_pair"].astype(str)

        dup_window = int(
            window_df.duplicated(subset=["subject", "region_pair", "trial_index_epoch0"]).sum()
        )
        print("Duplicate subject-region-trial rows after averaging:", dup_window)
        print("Window-averaged rows:", len(window_df))

        window_df.to_csv(output_root / "hfa_mi_window_trial_level.csv", index=False)
        try:
            window_df.to_parquet(output_root / "hfa_mi_window_trial_level.parquet", index=False)
        except Exception:
            pass

        validation_df = (
            window_df.groupby(["subject", "region_pair"], observed=True)
            .agg(
                n_trials=("trial_index_epoch0", "nunique"),
                n_rows=("trial_index_epoch0", "size"),
                n_hits=("hit_miss", lambda s: int(np.sum(np.asarray(s) == "hit"))),
                n_misses=("hit_miss", lambda s: int(np.sum(np.asarray(s) == "miss"))),
            )
            .reset_index()
        )
        display(validation_df.head(15))
        display(window_df.head(10))
        """
    ),
    code(
        """
        # -----------------------------
        # Window plots
        # -----------------------------
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sns.pointplot(
            data=window_df,
            x="region_pair",
            y="mi_window_mean",
            order=REGION_PAIR_ORDER,
            errorbar="se",
            dodge=False,
            ax=ax,
        )
        ax.set_title("Window-averaged MI by region pair")
        ax.set_xlabel("Region pair")
        ax.set_ylabel("Mean local MI")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_root / "window_mi_by_region_pair.png", dpi=200, bbox_inches="tight")
        plt.show()

        fig = sns.catplot(
            data=window_df,
            x="hit_miss",
            y="mi_window_mean",
            hue="rpe_cat",
            col="region_pair",
            kind="point",
            order=["miss", "hit"],
            hue_order=["neg", "neutral", "pos"],
            col_order=REGION_PAIR_ORDER,
            errorbar="se",
            sharey=True,
            height=3.6,
            aspect=0.95,
        )
        fig.set_titles("{col_name}")
        fig.set_axis_labels("Memory outcome", "Mean local MI")
        fig.figure.tight_layout()
        fig.figure.savefig(fig_root / "window_mi_by_region_hit_rpe.png", dpi=200, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        """
        # -----------------------------
        # Primary mixed-effects model
        # -----------------------------
        model_df = window_df.copy()
        model_df = model_df.dropna(subset=["mi_window_mean", "region_pair", "hit_miss", "rpe_cat", "subject"]).copy()

        counts_df = (
            model_df.groupby("region_pair", observed=True)
            .agg(
                n_subjects=("subject", "nunique"),
                n_trials=("trial_index_epoch0", "nunique"),
                n_rows=("trial_index_epoch0", "size"),
            )
            .reset_index()
        )
        display(counts_df)

        if len(model_df) < MIN_TRIALS_PER_MODEL:
            raise ValueError(f"Insufficient rows for mixed model: {len(model_df)}")

        mixedlm_result, fit_method = fit_primary_mixedlm(model_df)
        print("MixedLM fit method:", fit_method)
        print(mixedlm_result.summary())

        summary_txt = mixedlm_result.summary().as_text()
        (output_root / "mixedlm_summary.txt").write_text(summary_txt)

        fe_table = pd.DataFrame(
            {
                "term": mixedlm_result.fe_params.index,
                "beta": mixedlm_result.fe_params.values,
                "se": mixedlm_result.bse_fe.values,
                "z_value": mixedlm_result.tvalues.loc[mixedlm_result.fe_params.index].values,
                "p_value": mixedlm_result.pvalues.loc[mixedlm_result.fe_params.index].values,
            }
        )
        fe_table.to_csv(output_root / "mixedlm_fixed_effects.csv", index=False)
        display(fe_table)
        """
    ),
    code(
        """
        # -----------------------------
        # Omnibus tests
        # -----------------------------
        omnibus_df = build_omnibus_table(mixedlm_result)
        omnibus_df["n_rows"] = int(len(model_df))
        omnibus_df["n_subjects"] = int(model_df["subject"].nunique())
        omnibus_df.to_csv(output_root / "mixedlm_omnibus_tests.csv", index=False)
        display(omnibus_df)
        """
    ),
    code(
        """
        # -----------------------------
        # Pairwise region contrasts on subject-level marginal means
        # -----------------------------
        pairwise_df = build_pairwise_region_contrasts(window_df)
        pairwise_df.to_csv(output_root / "pairwise_region_contrasts.csv", index=False)
        display(pairwise_df)
        """
    ),
    code(
        """
        # -----------------------------
        # Condition-stratified summaries
        # -----------------------------
        condition_summary_df = (
            window_df.groupby(["region_pair", "hit_miss", "rpe_cat"], observed=True)
            .agg(
                mean_mi=("mi_window_mean", "mean"),
                sem_mi=("mi_window_mean", lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
                mean_z_mi=("z_mi_window_mean", "mean"),
                n_rows=("mi_window_mean", "size"),
                n_subjects=("subject", "nunique"),
            )
            .reset_index()
        )
        condition_summary_df.to_csv(output_root / "condition_stratified_summary.csv", index=False)
        display(condition_summary_df.head(20))
        """
    ),
    code(
        """
        # -----------------------------
        # Quick text summary
        # -----------------------------
        print("Analysis window:", WINDOW)
        print("Window rows:", len(window_df))
        print("Subjects contributing to mixed model:", model_df["subject"].nunique())
        print("Region pairs in mixed model:", sorted(model_df["region_pair"].astype(str).unique().tolist()))
        print("Saved files:")
        for path in sorted(output_root.glob("*")):
            print(" -", path.name)
        print("Figure files:")
        for path in sorted(fig_root.glob("*")):
            print(" -", path.name)
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.12",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote notebook to {NOTEBOOK_PATH}")
