"""
Multivariate Regression and Projection Utilities

This module contains functions for:
- Loading and preprocessing subject data
- Fitting RPE encoding models and computing reactivation
- Running statistical cluster tests
- Performing multinomial regression analysis
- Visualizing results

All functions are designed to work with MNE-Python data structures and support
parallel processing for efficient computation.
"""

import os
import glob
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Literal
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullLocator
import seaborn as sns
import mne
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, balanced_accuracy_score
from scipy.signal import filtfilt, hilbert
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed
from tqdm import tqdm
from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

from .behavior import preprocess_combined_df, preprocess_learn_df, resolve_trial_for_event


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================


from dataclasses import dataclass
from scipy.stats import t as t_dist, ttest_rel, rankdata, wilcoxon
from scipy.ndimage import label as cc_label


@dataclass
class Cluster:
    sign: str                 # "pos" or "neg"
    mass: float               # signed sum of t-values (pos > 0, neg < 0)
    abs_mass: float           # |mass|
    pvalue: float             # cluster-level p-value vs max-|cluster| null
    freq_idx: tuple           # (min_f, max_f)
    time_idx: tuple           # (min_t, max_t)
    n_pixels: int
    label_id: int             # connected-component id within that sign map


class TFRClusterPermutationGLM:
    """
    Pixel-wise GLM on TFR (epochs x freqs x times) with cluster-based permutation correction.

    Key fixes vs original:
    - Uses reshape (never np.resize).
    - Never mutates the stored design matrix during permutations.
    - Vectorized OLS across all pixels (fast, consistent df).
    - Valid covariate-adjusted permutation via Freedman–Lane (permute residuals of reduced model).
    - Consistent handling of missingness: predictors must be complete; TFR must contain no NaNs.
    - Two-sided handled via a null of max-|cluster mass| across both signs per permutation (robust, no sign bookkeeping).
    - df_resid computed from matrix rank (not hand-wavy).
    """

    def __init__(
        self,
        tfr_data: np.ndarray,               # (n_epochs, n_freqs, n_times)
        predictor_data: pd.DataFrame,       # rows = epochs
        target_var: str,                    # must be numeric column
        ch_name: str = "",
        alternative: str = "two-sided",     # "two-sided", "greater", "less"
        alpha_cluster: float = 0.05,        # cluster-forming threshold (uncorrected)
        alpha_test: float = 0.05,           # cluster-level significance threshold
        connectivity: np.ndarray = None,    # 3x3 structure for connected components
        standardize_X: bool = False,        # optional: z-score numeric predictors (except intercept)
        random_state: int | None = None,
    ):
        self.tfr_data = np.asarray(tfr_data)
        if self.tfr_data.ndim != 3:
            raise ValueError("tfr_data must be 3D: (epochs, freqs, times).")

        self.n_epochs, self.n_freqs, self.n_times = self.tfr_data.shape
        self.n_pixels = self.n_freqs * self.n_times
        self.ch_name = ch_name

        if np.isnan(self.tfr_data).any():
            raise ValueError(
                "tfr_data contains NaNs. For valid, consistent df and thresholding, "
                "please pre-mask/remove bad epochs globally (same epochs for all pixels)."
            )

        self.predictor_data = predictor_data.copy()
        if len(self.predictor_data) != self.n_epochs:
            raise ValueError("predictor_data must have the same number of rows as tfr_data has epochs.")

        if target_var not in self.predictor_data.columns:
            raise ValueError(f"target_var='{target_var}' not found in predictor_data.")

        # Only support a single numeric target regressor (your original indexing breaks for categorical).
        if not pd.api.types.is_numeric_dtype(self.predictor_data[target_var]):
            raise ValueError(
                f"target_var='{target_var}' must be numeric. If you need categorical targets, "
                "treat them as explicit dummy columns and pass a numeric column as target_var."
            )

        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less' (got {alternative}).")

        self.target_var = target_var
        self.alternative = alternative
        self.alpha_cluster = float(alpha_cluster)
        self.alpha_test = float(alpha_test)

        if connectivity is None:
            connectivity = np.ones((3, 3), dtype=int)  # allow diagonals
        connectivity = np.asarray(connectivity)
        if connectivity.shape != (3, 3):
            raise ValueError("connectivity must be a 3x3 array.")
        self.connectivity = connectivity

        self.rng = np.random.default_rng(random_state)

        # Build X using get_dummies for any categorical covariates, but target must stay numeric.
        X_df = pd.get_dummies(self.predictor_data, drop_first=True, dtype=float)

        # Ensure target is present and unique (if user had a categorical 'target_var', it would have been removed above)
        if target_var not in X_df.columns:
            raise ValueError(
                f"After dummy-coding, target_var='{target_var}' is not a single column. "
                "Make sure it's a numeric column in predictor_data."
            )

        # Drop any rows with NaNs in predictors (global mask for consistency)
        good = np.isfinite(X_df.to_numpy()).all(axis=1)
        if good.sum() < 5:
            raise ValueError("Too few valid epochs after dropping NaNs in predictors.")

        self.good_mask = good
        X_df = X_df.loc[good].reset_index(drop=True)

        # Optional standardization of numeric predictors (except target handling stays numeric; intercept added later)
        if standardize_X:
            # Standardize all columns except those that are 0/1 dummies (std==0 or close),
            # and do not standardize the target specially—just treat like others.
            Xn = X_df.to_numpy()
            means = Xn.mean(axis=0)
            stds = Xn.std(axis=0, ddof=1)
            keep = stds > 1e-12
            Xn[:, keep] = (Xn[:, keep] - means[keep]) / stds[keep]
            X_df = pd.DataFrame(Xn, columns=X_df.columns)

        self.X_df = X_df
        self.target_idx = int(np.where(self.X_df.columns.to_numpy() == target_var)[0][0])

        # Precompute Y as (n_good_epochs, n_pixels)
        Y = self.tfr_data[self.good_mask].reshape(good.sum(), -1).astype(np.float64, copy=False)
        self.Y = Y

        # Precompute full X and reduced X (both with intercept)
        self.X_full = self._add_intercept(self.X_df.to_numpy(dtype=np.float64))
        self.X_reduced = self._add_intercept(np.delete(self.X_df.to_numpy(dtype=np.float64), self.target_idx, axis=1))

        # Precompute linear algebra factors for speed
        self._prep_design_mats()

        # Cluster-forming threshold in t-space
        self.tcrit = self._tcritical()

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0],), dtype=X.dtype), X])

    def _prep_design_mats(self):
        # Full design
        self.rank_full = np.linalg.matrix_rank(self.X_full)
        self.df_resid = self.X_full.shape[0] - self.rank_full
        if self.df_resid <= 1:
            raise ValueError(f"Nonpositive/too small df_resid={self.df_resid}. Check design matrix rank.")

        XtX = self.X_full.T @ self.X_full
        self.XtX_inv = np.linalg.pinv(XtX)  # robust to near-collinearity
        self.P_full = self.XtX_inv @ self.X_full.T  # (p_full x n)

        # Reduced design
        XtX_r = self.X_reduced.T @ self.X_reduced
        XtX_r_inv = np.linalg.pinv(XtX_r)
        self.P_reduced = XtX_r_inv @ self.X_reduced.T

        # For target coefficient SE: need its diagonal element in (X'X)^-1
        # In params vector: [intercept, predictors...], so target param index = 1 + target_idx
        self.target_param_index = 1 + self.target_idx
        self.target_var_coef_var_factor = float(self.XtX_inv[self.target_param_index, self.target_param_index])
        if self.target_var_coef_var_factor <= 0:
            raise ValueError("Computed negative/zero variance factor for target coefficient; check design matrix.")

    def _tcritical(self) -> float:
        if self.alternative == "two-sided":
            return float(t_dist.ppf(1 - self.alpha_cluster / 2.0, self.df_resid))
        elif self.alternative == "greater":
            return float(t_dist.ppf(1 - self.alpha_cluster, self.df_resid))
        else:  # "less"
            # we still use a positive threshold internally and apply sign logic in thresholding
            return float(t_dist.ppf(1 - self.alpha_cluster, self.df_resid))

    # ---------- Vectorized OLS ----------
    def _fit_full_get_beta_t(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit full model to Y (n x m), return target beta and t-stat per pixel (length m).
        """
        B = self.P_full @ Y  # (p_full x m)
        resid = Y - (self.X_full @ B)  # (n x m)
        # sigma^2 per pixel
        s2 = (resid * resid).sum(axis=0) / self.df_resid  # (m,)
        se_beta = np.sqrt(s2 * self.target_var_coef_var_factor)  # (m,)
        beta = B[self.target_param_index, :]
        tvals = beta / se_beta
        return beta, tvals

    def _fit_reduced_get_fitted_resid(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit reduced model to Y, return fitted values and residuals.
        """
        Br = self.P_reduced @ Y
        fitted = self.X_reduced @ Br
        resid = Y - fitted
        return fitted, resid

    # ---------- Public API ----------
    def tfr_regression(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            beta_map: (freqs, times)
            t_map:    (freqs, times)
        """
        beta, tvals = self._fit_full_get_beta_t(self.Y)
        beta_map = beta.reshape(self.n_freqs, self.n_times)
        t_map = tvals.reshape(self.n_freqs, self.n_times)
        return beta_map, t_map

    def _threshold_maps(self, t_map: np.ndarray) -> dict[str, np.ndarray]:
        """
        Returns binary maps for clustering depending on alternative.
        """
        if self.alternative == "two-sided":
            return {
                "pos": (t_map >= self.tcrit).astype(np.int8),
                "neg": (t_map <= -self.tcrit).astype(np.int8),
            }
        elif self.alternative == "greater":
            return {"pos": (t_map >= self.tcrit).astype(np.int8)}
        else:  # "less"
            return {"neg": (t_map <= -self.tcrit).astype(np.int8)}

    def _extract_clusters(self, t_map: np.ndarray) -> list[dict]:
        """
        Extract all suprathreshold clusters (not just max) with signed mass.
        Returns list of dicts: {sign, mass, abs_mass, freq_idx, time_idx, n_pixels, label_id}
        """
        out = []
        binmaps = self._threshold_maps(t_map)
        for sign, bm in binmaps.items():
            if bm.sum() == 0:
                continue
            lab, nlab = cc_label(bm, structure=self.connectivity)
            for k in range(1, nlab + 1):
                idx = np.where(lab == k)
                mass = float(t_map[idx].sum())  # signed (pos>0, neg<0)
                abs_mass = float(abs(mass))
                fmin, fmax = int(idx[0].min()), int(idx[0].max())
                tmin, tmax = int(idx[1].min()), int(idx[1].max())
                out.append(
                    dict(
                        sign=sign,
                        mass=mass,
                        abs_mass=abs_mass,
                        freq_idx=(fmin, fmax),
                        time_idx=(tmin, tmax),
                        n_pixels=int(len(idx[0])),
                        label_id=int(k),
                    )
                )
        return out

    def _max_abs_cluster_mass(self, t_map: np.ndarray) -> float:
        """
        Single number: max absolute cluster mass across both signs for a t_map.
        If no clusters, returns 0.
        """
        clusters = self._extract_clusters(t_map)
        if not clusters:
            return 0.0
        return float(max(c["abs_mass"] for c in clusters))

    def _freedman_lane_permutation_tmap(self) -> np.ndarray:
        """
        Freedman–Lane permutation:
        - Fit reduced model: Y = Xr b + e
        - Permute residuals across epochs (rows)
        - Y* = fitted + permuted_resid
        - Fit full model to Y*, return t-map for target
        """
        fitted, resid = self._fit_reduced_get_fitted_resid(self.Y)
        perm = self.rng.permutation(resid.shape[0])
        Yp = fitted + resid[perm, :]
        _, tvals = self._fit_full_get_beta_t(Yp)
        return tvals.reshape(self.n_freqs, self.n_times)

    def compute_null_max_abs_cluster_masses(self, n_permutations: int, n_jobs: int = 1, verbose: int = 0) -> np.ndarray:
        """
        Returns:
            null_max_abs_masses: (n_permutations,)
        """
        if n_permutations < 100:
            # not an error, but typically too small for stable cluster p-values
            pass

        if n_jobs == 1:
            arr = np.empty((n_permutations,), dtype=np.float64)
            for i in range(n_permutations):
                tmap = self._freedman_lane_permutation_tmap()
                arr[i] = self._max_abs_cluster_mass(tmap)
            return arr

        # Parallel (note: each worker needs independent RNG stream)
        seeds = self.rng.integers(0, 2**32 - 1, size=n_permutations, dtype=np.uint32)

        def _one(seed):
            local = np.random.default_rng(int(seed))
            # temporarily swap RNG for this call
            old = self.rng
            try:
                self.rng = local
                tmap = self._freedman_lane_permutation_tmap()
                return self._max_abs_cluster_mass(tmap)
            finally:
                self.rng = old

        return np.asarray(
            Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_one)(s) for s in seeds),
            dtype=np.float64,
        )

    def cluster_significance(self, t_map: np.ndarray, null_max_abs_masses: np.ndarray) -> list[Cluster]:
        """
        Cluster-level inference using max-|cluster mass| null distribution.

        p(cluster) = ( #{null >= |cluster_mass|} + 1 ) / (n_perm + 1 )
        (the +1 is a standard finite-sample correction)
        """
        clusters = self._extract_clusters(t_map)
        if len(null_max_abs_masses) == 0:
            raise ValueError("null_max_abs_masses is empty.")

        nperm = len(null_max_abs_masses)
        out: list[Cluster] = []
        for c in clusters:
            p = (np.sum(null_max_abs_masses >= c["abs_mass"]) + 1.0) / (nperm + 1.0)
            out.append(
                Cluster(
                    sign=c["sign"],
                    mass=c["mass"],
                    abs_mass=c["abs_mass"],
                    pvalue=float(p),
                    freq_idx=c["freq_idx"],
                    time_idx=c["time_idx"],
                    n_pixels=c["n_pixels"],
                    label_id=c["label_id"],
                )
            )
        # sort by abs_mass descending
        out.sort(key=lambda z: z.abs_mass, reverse=True)
        return out

    def run(
        self,
        n_permutations: int = 2000,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> dict:
        """
        Convenience end-to-end run.

        Returns dict with:
          - beta_map, t_map
          - tcrit, df_resid
          - null_max_abs_masses
          - clusters (all suprathreshold) with p-values
          - significant_clusters (p < alpha_test)
        """
        beta_map, t_map = self.tfr_regression()
        null_max = self.compute_null_max_abs_cluster_masses(n_permutations, n_jobs=n_jobs, verbose=verbose)
        clusters = self.cluster_significance(t_map, null_max)
        sig = [c for c in clusters if c.pvalue < self.alpha_test]

        return dict(
            ch_name=self.ch_name,
            target_var=self.target_var,
            alternative=self.alternative,
            alpha_cluster=self.alpha_cluster,
            alpha_test=self.alpha_test,
            df_resid=int(self.df_resid),
            tcrit=float(self.tcrit),
            beta_map=beta_map,
            t_map=t_map,
            null_max_abs_masses=null_max,
            clusters=clusters,
            significant_clusters=sig,
        )


# ------------------------ Example usage ------------------------
# tester = TFRClusterPermutationGLM(
#     tfr_data=tfr_data,                         # (epochs, freqs, times)
#     predictor_data=predictor_df,               # rows=epochs
#     target_var="zrpe",                         # numeric regressor
#     ch_name="OFC3",
#     alternative="two-sided",
#     alpha_cluster=0.05,                        # cluster-forming threshold
#     alpha_test=0.05,                           # cluster-level alpha
#     standardize_X=True,
#     random_state=0,
# )
# res = tester.run(n_permutations=2000, n_jobs=8, verbose=5)
# print([(c.sign, c.pvalue, c.freq_idx, c.time_idx, c.abs_mass) for c in res["significant_clusters"][:5]])

class Subject:
    """
    Subject class for efficient loading and caching of neural and behavioral data.
    
    Lazily loads behavioral and neural resources on first access, then caches
    processed neural data per (event, ROI, freq_range, downsample_factor, win_ms)
    combination to avoid redundant work.
    """
    
    def __init__(self, subject: str, base_dir: str, load_dir: str,
                 min_trials: int = 10, verbose: bool = True,
                 neural_cache_dir: Optional[str] = None,
                 cache_mode: str = "recompute"):
        """
        Initialize Subject with lazy data loading.
        
        No behavioral CSVs or TFR files are loaded at construction time.
        Resources are loaded on demand when accessed by analysis helpers.
        
        Parameters:
        -----------
        subject : str
            Subject identifier (e.g., 'MS012')
        base_dir : str
            Base directory containing project files
        load_dir : str
            Directory containing behavioral CSV files
        min_trials : int, default=10
            Minimum number of trials required
        verbose : bool, default=True
            Print progress messages
        neural_cache_dir : str or None, default=None
            Directory to save/load processed neural data cache files.
            If None, no disk caching is performed (only in-memory caching).
            If provided, processed neural data will be saved and reloaded from disk
            to avoid reprocessing across sessions. Default location if None but caching
            desired: {base_dir}/projects/guLab/Salman/EphysAnalyses/{subject}/scratch/neural_cache/
        """
        self.subject = subject
        self.base_dir = base_dir
        self.load_dir = load_dir
        self.min_trials = min_trials
        self.verbose = verbose
        self.cache_mode = str(cache_mode).lower()
        if self.cache_mode not in {"reuse", "migrate", "recompute"}:
            raise ValueError(
                f"Invalid cache_mode='{cache_mode}'. Expected one of: reuse, migrate, recompute."
            )
        self._learn_df = None
        self._combined_df = None
        self._elec_df = None
        self._tfr_cache: Dict[str, Any] = {}
        self._epoch_cache: Dict[str, Any] = {}
        
        # Set up disk cache directory
        if neural_cache_dir is None:
            # Default cache directory
            self.neural_cache_dir = os.path.join(
                base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', 
                subject, 'scratch', 'neural_cache'
            )
        else:
            self.neural_cache_dir = neural_cache_dir
        
        # Create cache directory and all necessary parent directories if they don't exist
        if self.neural_cache_dir is not None:
            try:
                os.makedirs(self.neural_cache_dir, exist_ok=True)
                if verbose:
                    print(f"Neural cache directory: {self.neural_cache_dir}")
            except OSError as e:
                if verbose:
                    print(f"Warning: Could not create cache directory {self.neural_cache_dir}: {e}")
                # Continue without disk caching if directory creation fails
                self.neural_cache_dir = None
        
        # Cache for processed neural data (in-memory)
        # Key: (event, roi, freq_range_tuple, downsample_factor, win_ms)
        self._neural_cache = {}
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"INITIALIZING SUBJECT (LAZY): {subject}")
            print(f"{'='*50}")
            print("No behavioral or neural data loaded yet (lazy mode).")

    def _ensure_behavior_loaded(self) -> None:
        """
        Lazily load and preprocess behavioral tables for this subject only.
        """
        if self._learn_df is not None and self._combined_df is not None:
            return
        try:
            learn_df = pd.read_csv(f'{self.load_dir}/learn_df_RWH.csv')
            combined_df = pd.read_csv(f'{self.load_dir}/full_df_RWH.csv')
        except FileNotFoundError as e:
            if self.verbose:
                print(f"Behavioral data not found: {e}")
            raise FileNotFoundError(f"Behavioral data not found: {e}")

        learn_df = learn_df.loc[learn_df.participant == self.subject].copy()
        combined_df = combined_df.loc[combined_df.participant == self.subject].copy()

        if self.verbose:
            print(f"Preprocessing behavioral data for {self.subject}...")
        self._learn_df = preprocess_learn_df(learn_df)
        self._combined_df = preprocess_combined_df(combined_df)

    def _ensure_electrodes_loaded(self) -> None:
        """
        Lazily load electrode table for this subject.
        """
        if self._elec_df is not None:
            return
        elec_file = (
            f'{self.base_dir}/projects/guLab/Salman/EphysAnalyses/'
            f'{self.subject}/Day1_reref_elec_df'
        )
        if not os.path.exists(elec_file):
            if self.verbose:
                print(f"Electrode file not found for subject {self.subject}")
            raise FileNotFoundError(f"Electrode file not found for subject {self.subject}")
        self._elec_df = pd.read_csv(elec_file)

    def _ensure_tfr_loaded(self, event: str) -> Any:
        """
        Lazily load requested event TFR.
        """
        if event in self._tfr_cache:
            return self._tfr_cache[event]
        filepath = (
            f'{self.base_dir}/projects/guLab/Salman/EphysAnalyses/'
            f'{self.subject}/scratch/TFR'
        )
        tfr_file = f'{filepath}/{event}-tfr.h5'
        if not os.path.exists(tfr_file):
            if self.verbose:
                print(f"TFR file not found for subject {self.subject}, event {event}")
            raise FileNotFoundError(
                f"TFR file not found for subject {self.subject}, event {event}"
            )
        if self.verbose:
            print(f"Loading TFR: {event} for {self.subject}")
        self._tfr_cache[event] = mne.time_frequency.read_tfrs(tfr_file)
        return self._tfr_cache[event]

    def _ensure_epochs_loaded(self, event: str) -> Any:
        """
        Lazily load requested event epochs.
        """
        if event in self._epoch_cache:
            return self._epoch_cache[event]
        epoch_file = (
            f'{self.base_dir}/projects/guLab/Salman/EphysAnalyses/'
            f'{self.subject}/neural/Day1/{event}-epo.fif'
        )
        if not os.path.exists(epoch_file):
            if self.verbose:
                print(f"Epochs file not found for subject {self.subject}, event {event}")
            raise FileNotFoundError(
                f"Epochs file not found for subject {self.subject}, event {event}"
            )
        if self.verbose:
            print(f"Loading epochs: {event} for {self.subject}")
        self._epoch_cache[event] = mne.read_epochs(epoch_file, preload=True, verbose=False)
        return self._epoch_cache[event]

    @property
    def learn_df(self) -> pd.DataFrame:
        self._ensure_behavior_loaded()
        return self._learn_df

    @property
    def combined_df(self) -> pd.DataFrame:
        self._ensure_behavior_loaded()
        return self._combined_df

    @property
    def elec_df(self) -> pd.DataFrame:
        self._ensure_electrodes_loaded()
        return self._elec_df

    @property
    def feedback_tfr(self) -> Any:
        return self._ensure_tfr_loaded('feedback_start')

    @property
    def cue_tfr(self) -> Any:
        return self._ensure_tfr_loaded('cue_on')
    
    def _get_cache_filepath(self, cache_key: Tuple) -> str:
        """
        Generate cache file path for a given cache key.
        
        Parameters:
        -----------
        cache_key : tuple
            New format: (event, roi, freq_range_tuple, downsample_factor, win_ms)
            Legacy format: (roi, freq_range_tuple, downsample_factor, win_ms)
        
        Returns:
        --------
        str
            Full path to cache file
        """
        if len(cache_key) == 5:
            event, roi, freq_range_tuple, downsample_factor, win_ms = cache_key
            event_token = event.replace('/', '_')
            prefix = f"{self.subject}_{event_token}_{roi}"
        elif len(cache_key) == 4:
            roi, freq_range_tuple, downsample_factor, win_ms = cache_key
            prefix = f"{self.subject}_{roi}"
        else:
            raise ValueError(f"Unsupported cache key length: {len(cache_key)}")
        freq_str = f"{freq_range_tuple[0]:.1f}_{freq_range_tuple[1]:.1f}"
        filename = f"{prefix}_freq{freq_str}_ds{downsample_factor}_win{win_ms}.npz"
        return os.path.join(self.neural_cache_dir, filename)
    
    def _load_neural_data_from_disk(self, cache_key: Tuple) -> Optional[Dict[str, Any]]:
        """
        Load processed event neural data from disk cache.
        
        Parameters:
        -----------
        cache_key : tuple
            (event, roi, freq_range_tuple, downsample_factor, win_ms)
        
        Returns:
        --------
        dict or None
            Cached neural data if found, None otherwise
        """
        if self.neural_cache_dir is None:
            return None

        if len(cache_key) != 5:
            return None

        event, roi, freq_range_tuple, downsample_factor, win_ms = cache_key
        cache_file = self._get_cache_filepath((event, roi, freq_range_tuple, downsample_factor, win_ms))

        # New cache format (event-specific).
        if os.path.exists(cache_file):
            try:
                with np.load(cache_file, allow_pickle=True) as npz:
                    files = set(npz.files)
                    if {'hfa', 'times'}.issubset(files):
                        return {
                            'hfa': np.asarray(npz['hfa'], dtype=np.float32),
                            'times': np.asarray(npz['times'], dtype=np.float32),
                            'electrode_labels': (
                                np.asarray(npz['electrode_labels'], dtype=object)
                                if 'electrode_labels' in files else None
                            ),
                            'electrode_region': (
                                np.asarray(npz['electrode_region'], dtype=object)
                                if 'electrode_region' in files else None
                            ),
                            'electrode_hemi': (
                                np.asarray(npz['electrode_hemi'], dtype=object)
                                if 'electrode_hemi' in files else None
                            ),
                            'event': event,
                        }
                    # Graceful handling if event-specific file accidentally contains legacy keys.
                    if event == 'cue_on' and {'cue_hfa', 'cue_times'}.issubset(files):
                        return {
                            'hfa': np.asarray(npz['cue_hfa'], dtype=np.float32),
                            'times': np.asarray(npz['cue_times'], dtype=np.float32),
                            'electrode_labels': (
                                np.asarray(npz['electrode_labels'], dtype=object)
                                if 'electrode_labels' in files else None
                            ),
                            'electrode_region': None,
                            'electrode_hemi': None,
                            'event': event,
                        }
                    if event == 'feedback_start' and {'feedback_hfa', 'feedback_times'}.issubset(files):
                        return {
                            'hfa': np.asarray(npz['feedback_hfa'], dtype=np.float32),
                            'times': np.asarray(npz['feedback_times'], dtype=np.float32),
                            'electrode_labels': (
                                np.asarray(npz['electrode_labels'], dtype=object)
                                if 'electrode_labels' in files else None
                            ),
                            'electrode_region': None,
                            'electrode_hemi': None,
                            'event': event,
                        }
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache file {cache_file}: {e}")

        # Legacy cache format fallback (combined feedback+cue in one file).
        legacy_file = self._get_cache_filepath((roi, freq_range_tuple, downsample_factor, win_ms))
        if not os.path.exists(legacy_file):
            return None
        try:
            with np.load(legacy_file, allow_pickle=True) as npz:
                files = set(npz.files)
                electrode_labels = (
                    np.asarray(npz['electrode_labels'], dtype=object)
                    if 'electrode_labels' in files else None
                )
                if event == 'cue_on' and {'cue_hfa', 'cue_times'}.issubset(files):
                    return {
                        'hfa': np.asarray(npz['cue_hfa'], dtype=np.float32),
                        'times': np.asarray(npz['cue_times'], dtype=np.float32),
                        'electrode_labels': electrode_labels,
                        'electrode_region': None,
                        'electrode_hemi': None,
                        'event': event,
                    }
                if event == 'feedback_start' and {'feedback_hfa', 'feedback_times'}.issubset(files):
                    return {
                        'hfa': np.asarray(npz['feedback_hfa'], dtype=np.float32),
                        'times': np.asarray(npz['feedback_times'], dtype=np.float32),
                        'electrode_labels': electrode_labels,
                        'electrode_region': None,
                        'electrode_hemi': None,
                        'event': event,
                    }
        except Exception as e:
            if self.verbose:
                print(f"Error loading legacy cache file {legacy_file}: {e}")
        return None
    
    def _save_neural_data_to_disk(self, cache_key: Tuple, neural_data: Dict[str, Any]) -> None:
        """
        Save processed neural data to disk cache.
        
        Parameters:
        -----------
        cache_key : tuple
            (event, roi, freq_range_tuple, downsample_factor, win_ms)
        neural_data : dict
            Dictionary containing processed neural data
        """
        if self.neural_cache_dir is None:
            return

        if len(cache_key) != 5:
            return
        event, roi, freq_range_tuple, downsample_factor, win_ms = cache_key
        cache_file = self._get_cache_filepath((event, roi, freq_range_tuple, downsample_factor, win_ms))

        try:
            np.savez_compressed(
                cache_file,
                hfa=np.asarray(neural_data['hfa'], dtype=np.float32),
                times=np.asarray(neural_data['times'], dtype=np.float32),
                electrode_labels=neural_data.get('electrode_labels'),
                electrode_region=neural_data.get('electrode_region'),
                electrode_hemi=neural_data.get('electrode_hemi'),
                event=np.asarray(event),
                roi=np.asarray(roi),
                freq_range=np.asarray(freq_range_tuple, dtype=np.float32),
                downsample_factor=np.int32(downsample_factor),
                win_ms=np.int32(win_ms),
                subject=np.asarray(self.subject),
            )
        except Exception as e:
            if self.verbose:
                print(f"Error saving cache file {cache_file}: {e}")

    def _reconstruct_electrode_metadata(
        self,
        roi: str,
        n_channels: int,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct electrode label/region/hemisphere metadata from elec_df.
        """
        elec_roi = self.elec_df[self.elec_df.salman_region == roi][
            ['label', 'salman_region', 'hemisphere']
        ].copy()
        elec_roi['label'] = elec_roi['label'].astype(str)
        if labels is not None and len(labels) > 0:
            label_list = [str(x) for x in np.asarray(labels, dtype=object).tolist()]
            elec_roi = elec_roi.set_index('label')
            out_labels = []
            out_region = []
            out_hemi = []
            for ch in label_list:
                if ch not in elec_roi.index:
                    raise ValueError(
                        f"Missing electrode metadata for channel '{ch}' in ROI '{roi}' ({self.subject})"
                    )
                row = elec_roi.loc[ch]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                out_labels.append(ch)
                out_region.append(str(row['salman_region']))
                out_hemi.append(str(row['hemisphere']).lower())
            return (
                np.asarray(out_labels, dtype=object),
                np.asarray(out_region, dtype=object),
                np.asarray(out_hemi, dtype=object),
            )

        if len(elec_roi) < n_channels:
            raise ValueError(
                f"Cannot reconstruct metadata: ROI '{roi}' has {len(elec_roi)} electrodes "
                f"but signal has {n_channels} channels ({self.subject})."
            )
        elec_roi = elec_roi.iloc[:n_channels]
        return (
            elec_roi['label'].astype(str).to_numpy(dtype=object),
            elec_roi['salman_region'].astype(str).to_numpy(dtype=object),
            elec_roi['hemisphere'].astype(str).str.lower().to_numpy(dtype=object),
        )

    def _get_or_build_event_neural_data(
        self,
        event: str,
        roi: str,
        freq_range: Tuple[float, float],
        downsample_factor: int,
        win_ms: int,
    ) -> Dict[str, Any]:
        """
        Resolve event neural data based on cache_mode: reuse, migrate, or recompute.
        """
        cache_key = (event, roi, tuple(freq_range), int(downsample_factor), int(win_ms))
        if cache_key in self._neural_cache:
            return self._neural_cache[cache_key]

        def _ensure_metadata(payload: Dict[str, Any], allow_reconstruct: bool) -> Dict[str, Any]:
            data = dict(payload)
            labels = data.get('electrode_labels', None)
            region = data.get('electrode_region', None)
            hemi = data.get('electrode_hemi', None)
            n_channels = int(np.asarray(data['hfa']).shape[1])

            labels_ok = labels is not None and len(np.asarray(labels).reshape(-1)) == n_channels
            region_ok = region is not None and len(np.asarray(region).reshape(-1)) == n_channels
            hemi_ok = hemi is not None and len(np.asarray(hemi).reshape(-1)) == n_channels

            if labels_ok and region_ok and hemi_ok:
                data['electrode_labels'] = np.asarray(labels, dtype=object)
                data['electrode_region'] = np.asarray(region, dtype=object)
                data['electrode_hemi'] = np.asarray(hemi, dtype=object)
                return data

            if not allow_reconstruct:
                raise ValueError(
                    f"Cached neural metadata incomplete for {self.subject} {event}/{roi}. "
                    "Use cache_mode='migrate' or 'recompute'."
                )

            rec_labels, rec_region, rec_hemi = self._reconstruct_electrode_metadata(
                roi=roi, n_channels=n_channels, labels=labels
            )
            data['electrode_labels'] = rec_labels
            data['electrode_region'] = rec_region
            data['electrode_hemi'] = rec_hemi
            return data

        if self.cache_mode in {"reuse", "migrate"} and self.neural_cache_dir is not None:
            cached = self._load_neural_data_from_disk(cache_key)
            if cached is not None:
                enriched = _ensure_metadata(cached, allow_reconstruct=(self.cache_mode == "migrate"))
                self._neural_cache[cache_key] = enriched
                if self.cache_mode == "migrate":
                    self._save_neural_data_to_disk(cache_key, enriched)
                return enriched

        processed = self._process_event_neural_data(
            roi=roi,
            event=event,
            freq_range=freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
        )
        processed = _ensure_metadata(processed, allow_reconstruct=True)
        self._neural_cache[cache_key] = processed
        if self.neural_cache_dir is not None:
            self._save_neural_data_to_disk(cache_key, processed)
        return processed

    def _process_event_neural_data(
        self,
        roi: str,
        event: str,
        freq_range: Tuple[float, float],
        downsample_factor: int,
        win_ms: int
    ) -> Dict[str, Any]:
        """
        Process neural data for one event/ROI/parameter setting.
        """
        roi_electrodes = self.elec_df[self.elec_df.salman_region == roi]['label'].tolist()
        if len(roi_electrodes) == 0:
            raise ValueError(f"No electrodes found in target ROI {roi} for subject {self.subject}")

        tfr = self._ensure_tfr_loaded(event).copy().pick(roi_electrodes)
        picked_labels = list(tfr.ch_names)

        elec_sub = self.elec_df[
            self.elec_df.label.astype(str).isin([str(x) for x in picked_labels])
        ][['label', 'salman_region', 'hemisphere']].copy()
        elec_sub['label'] = elec_sub['label'].astype(str)
        elec_sub = elec_sub.set_index('label')

        electrode_region = []
        electrode_hemi = []
        for ch in picked_labels:
            ch_key = str(ch)
            if ch_key in elec_sub.index:
                row = elec_sub.loc[ch_key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                electrode_region.append(str(row.get('salman_region', roi)))
                hemi_val = row.get('hemisphere', np.nan)
                electrode_hemi.append(str(hemi_val).lower() if pd.notna(hemi_val) else np.nan)
            else:
                electrode_region.append(roi)
                electrode_hemi.append(np.nan)

        freq_mask = (tfr.freqs >= freq_range[0]) & (tfr.freqs <= freq_range[1])
        hfa = np.nanmean(tfr.data[:, :, freq_mask, :], axis=2).astype(np.float32)

        if win_ms > 0:
            win_size = max(1, int(tfr.info['sfreq'] * (win_ms / 1000)))
            window = np.ones(win_size, dtype=np.float32) / float(win_size)
            hfa_smooth = np.zeros_like(hfa)
            for electrode_idx in range(hfa.shape[1]):
                hfa_smooth[:, electrode_idx, :] = filtfilt(window, [1], hfa[:, electrode_idx, :])
        else:
            hfa_smooth = hfa

        hfa_downsampled = hfa_smooth[:, :, ::int(downsample_factor)]

        if hasattr(tfr, 'times'):
            times = np.asarray(tfr.times[::int(downsample_factor)], dtype=np.float32)
        else:
            times = np.arange(hfa_downsampled.shape[2], dtype=np.float32)

        return {
            'hfa': hfa_downsampled,
            'times': times,
            'electrode_labels': np.asarray(picked_labels, dtype=object),
            'electrode_region': np.asarray(electrode_region, dtype=object),
            'electrode_hemi': np.asarray(electrode_hemi, dtype=object),
            'event': event,
        }

    def _process_neural_data(
        self,
        roi: str,
        freq_range: Tuple[float, float],
        downsample_factor: int,
        win_ms: int
    ) -> Dict[str, Any]:
        """
        Legacy combined-event neural processing wrapper.
        """
        feedback = self._process_event_neural_data(
            roi=roi,
            event='feedback_start',
            freq_range=freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
        )
        cue = self._process_event_neural_data(
            roi=roi,
            event='cue_on',
            freq_range=freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
        )
        subj_combined_df = self.combined_df[
            (self.combined_df.condition == 'Day1') &
            (self.combined_df.participant == self.subject)
        ].copy()
        return {
            'feedback_hfa': feedback['hfa'],
            'cue_hfa': cue['hfa'],
            'feedback_times': feedback['times'],
            'cue_times': cue['times'],
            'memory_outcomes': subj_combined_df.memory_outcome.values,
            'electrode_labels': feedback.get('electrode_labels'),
        }
    
    def get_data_for_roi_and_dependent_var(self, roi: str, dependent_var: str = 'rpe',
                                           freq_range: Tuple[float, float] = (70., 201.),
                                           downsample_factor: int = 5, 
                                           win_ms: int = 200,
                                           verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get processed data for a specific ROI, dependent variable, and processing parameters.
        
        Caches processed neural data per (event, ROI, freq_range, downsample_factor, win_ms)
        to avoid recomputation.
        
        Parameters:
        -----------
        roi : str
            Region of interest
        dependent_var : str, default='rpe'
            Dependent variable name
        freq_range : tuple, default=(70., 201.)
            Frequency range to extract (Hz)
        downsample_factor : int, default=5
            Downsampling factor
        win_ms : int, default=200
            Smoothing window size (ms)
        verbose : bool, default=False
            Print progress messages
        
        Returns:
        --------
        dict or None
            Dictionary compatible with load_subject_data output:
            - feedback_hfa, cue_hfa (processed neural data for freq_range)
            - feedback_times, cue_times
            - memory_outcomes
            - dependent_var (values)
            - rpe_category
            - subject, roi
            Returns None if subject cannot be analyzed.
        """
        try:
            feedback_hfa, feedback_times, electrode_labels = _get_smoothed_event_signal(
                subject_obj=self,
                roi=roi,
                event='feedback_start',
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
            )
            cue_hfa, cue_times, _ = _get_smoothed_event_signal(
                subject_obj=self,
                roi=roi,
                event='cue_on',
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
            )
        except (ValueError, FileNotFoundError) as e:
            if verbose:
                print(f"Error processing neural data: {e}")
            return None
        
        # Get subject-specific dataframes
        subj_learn_df = self.learn_df[self.learn_df.participant == self.subject].copy()
        subj_combined_df = self.combined_df[
            (self.combined_df.condition == 'Day1') & 
            (self.combined_df.participant == self.subject)
        ].copy()
        
        if len(subj_learn_df) == 0 or len(subj_combined_df) == 0:
            if verbose:
                print(f"No data found for subject {self.subject}")
            return None
        
        # Check minimum trials
        n_trials_cue = cue_hfa.shape[0]
        if n_trials_cue < self.min_trials:
            if verbose:
                print(f"Insufficient trials for subject {self.subject}: {n_trials_cue} < {self.min_trials}")
            return None
        
        # Get dependent variable values
        if dependent_var not in subj_learn_df.columns:
            if verbose:
                print(f"Dependent variable '{dependent_var}' not found in learn_df")
            return None
        
        dependent_var_values = subj_learn_df[dependent_var].values
        
        # Compute category using tercile split
        terciles = np.percentile(dependent_var_values, [33.33, 66.67])
        if verbose:
            print(f"{dependent_var} terciles: {terciles}")
        
        rpe_category = []
        for val in dependent_var_values:
            if val <= terciles[0]:
                rpe_category.append('negative')
            elif val <= terciles[1]:
                rpe_category.append('neutral')
            else:
                rpe_category.append('positive')
        
        subj_learn_df['rpe_category'] = rpe_category
        
        if verbose:
            print(f"\n{dependent_var} category distribution:")
            print(subj_learn_df['rpe_category'].value_counts())
        
        # Add category to combined_df
        subj_combined_df['rpe_category'] = np.nan
        for idx, row in subj_learn_df.iterrows():
            mask = (
                (subj_combined_df.img_path == row.img_path) & 
                (subj_combined_df.participant == row.participant)
            )
            subj_combined_df.loc[mask, 'rpe_category'] = row.rpe_category
        
        rpe_category = subj_combined_df.rpe_category.dropna().values
        
        # Combine neural data with dependent variable data
        result = {
            'feedback_hfa': feedback_hfa,
            'cue_hfa': cue_hfa,
            'memory_outcomes': subj_combined_df.memory_outcome.values,
            'rpe_category': rpe_category,
            'dependent_var': dependent_var_values,
            'feedback_times': feedback_times,
            'cue_times': cue_times,
            'electrode_labels': electrode_labels,
            'n_trials': n_trials_cue,
            'subject': self.subject,
            'roi': roi,
        }
        
        return result

def load_subject_data(
    subject: str,
    base_dir: str,
    load_dir: str,
    roi: str = 'OFC',
    downsample_factor: int = 5,
    win_ms: int = 200,
    hfa_freq_range: Tuple[float, float] = (70., 201.),
    min_trials: int = 10,
    verbose: bool = True,
    dependent_var: str = 'rpe'
) -> Optional[Dict[str, Any]]:
    """
    Load and preprocess all data for a single subject.
    
    This function now uses the Subject class internally for efficient caching.
    Maintains backward compatibility with existing code.
    
    Parameters:
    -----------
    subject : str
        Subject identifier (e.g., 'MS012')
    base_dir : str
        Base directory containing project files
    load_dir : str
        Directory containing behavioral CSV files
    roi : str, default='OFC'
        Region of interest to analyze
    downsample_factor : int, default=5
        Factor by which to downsample timepoints
    win_ms : int, default=200
        Smoothing window size in milliseconds
    hfa_freq_range : tuple, default=(70., 201.)
        Frequency range for high-frequency activity (Hz)
    min_trials : int, default=10
        Minimum number of trials required
    verbose : bool, default=True
        Print progress messages
    dependent_var : str, default='rpe'
        Name of the dependent variable column in learn_df to use for encoding models
        
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'feedback_hfa': array, shape (n_trials, n_electrodes, n_timepoints)
        - 'cue_hfa': array, shape (n_trials, n_electrodes, n_timepoints)
        - 'memory_outcomes': array, shape (n_trials,)
        - 'rpe_category': array, shape (n_trials,)
        - 'dependent_var': array, shape (n_trials,)
        - 'feedback_times': array, shape (n_timepoints,)
        - 'cue_times': array, shape (n_timepoints,)
        - 'n_trials': int
        - 'subject': str
        - 'roi': str
        Returns None if subject cannot be analyzed.
    """
    try:
        subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=verbose)
        return subject_obj.get_data_for_roi_and_dependent_var(
            roi=roi,
            dependent_var=dependent_var,
            freq_range=hfa_freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
            verbose=verbose
        )
    except (FileNotFoundError, ValueError) as e:
        if verbose:
            print(f"Error loading subject data: {e}")
        return None


def load_trial_level(
    roi: str,
    subject: str,
    trial_save_dir: str,
    dependent_var: str = 'rpe'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load trial-level z-scored reactivation data for a subject/ROI.
    
    Automatically detects whether data is vector-level (3D) or electrode-level (4D).
    
    Parameters:
    -----------
    roi : str
        Region of interest
    subject : str
        Subject identifier
    trial_save_dir : str
        Directory containing saved trial-level data
    dependent_var : str, default='rpe'
        Name of the dependent variable used for encoding models
        
    Returns:
    --------
    z_reactivation : array
        If vector-level: shape (n_feedback_t, n_cue_t, n_trials)
        If electrode-level: shape (n_feedback_t, n_cue_t, n_electrodes, n_trials)
        Z-scored reactivation values
    feedback_times : array, shape (n_feedback_t,)
        Time vector for feedback period
    cue_times : array, shape (n_cue_t,)
        Time vector for cue period
    meta : dict
        Metadata dictionary, including 'electrode_level' key indicating mode
    """
    path = os.path.join(trial_save_dir, roi, f"{subject}_{dependent_var}_zreactivation.npz")
    npz = np.load(path, allow_pickle=True)
    
    z_reactivation = npz['z_reactivation']
    feedback_times = npz['feedback_times']
    cue_times = npz['cue_times']
    
    meta = {k: npz[k] for k in npz.files if k not in {
        'z_reactivation', 'feedback_times', 'cue_times'
    }}
    
    # Detect electrode_level from shape or metadata
    if 'electrode_level' in meta:
        meta['electrode_level'] = bool(np.asarray(meta['electrode_level']).item())
    else:
        # Auto-detect from shape: 4D = electrode-level, 3D = vector-level
        meta['electrode_level'] = (z_reactivation.ndim == 4)
    
    return z_reactivation, feedback_times, cue_times, meta


def load_mv_activation_and_mni(
    trial_save_dir: str,
    base_dir: str,
    load_dir: str,
    rois: List[str],
    predictors: List[str],
    subjects: List[str],
    mni_columns: Optional[Tuple[str, str, str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Load time-averaged activation patterns from MV NPZ files and merge with MNI coordinates from Subject elec_df.

    For each (roi, subject, predictor) where the NPZ exists, loads activation_patterns, time-averages
    over the feedback-time axis (axis=0), matches electrode_labels to elec_df MNI columns, and
    returns a long-format table suitable for glass-brain heatmap interpolation (one row per
    subject/roi/predictor/electrode with MNI and time_avg_activation).

    Parameters
    ----------
    trial_save_dir : str
        Directory containing roi subdirs with NPZ files (multivariate pipeline).
    base_dir : str
        Base directory for project (used for elec_df path).
    load_dir : str
        Load directory (for Subject; not used for elec_df, which uses base_dir).
    rois : list of str
        Region names to consider.
    predictors : list of str
        Predictor names (NPZ filename is {subject}_{predictor}_zreactivation.npz).
    subjects : list of str
        Subject identifiers.
    mni_columns : tuple of (str, str, str) or None
        (x, y, z) column names in elec_df for MNI. Default ('mni_x', 'mni_y', 'mni_z').
    verbose : bool
        If True, log skipped files/reasons.

    Returns
    -------
    df : pandas.DataFrame
        Columns: subject, roi, predictor, label, mni_x, mni_y, mni_z, time_avg_activation.
        Rows with missing MNI (electrode not in elec_df or MNI cols missing) are dropped.
    skipped : list of dict
        Each entry has keys like 'subject', 'roi', 'predictor', 'reason' for files not loaded or rows skipped.
    """
    if mni_columns is None:
        mni_columns = ('mni_x', 'mni_y', 'mni_z')
    mni_x, mni_y, mni_z = mni_columns

    rows = []
    skipped = []

    # Cache elec_df per subject to avoid repeated disk reads
    elec_cache: Dict[str, pd.DataFrame] = {}

    def get_elec_df(subject: str) -> Optional[pd.DataFrame]:
        if subject in elec_cache:
            return elec_cache[subject]
        elec_file = os.path.join(
            base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subject, 'Day1_reref_elec_df'
        )
        if not os.path.exists(elec_file):
            if verbose:
                skipped.append({'subject': subject, 'roi': '', 'predictor': '', 'reason': 'elec_df not found'})
            return None
        try:
            edf = pd.read_csv(elec_file)
            elec_cache[subject] = edf
            return edf
        except Exception as e:
            if verbose:
                skipped.append({'subject': subject, 'roi': '', 'predictor': '', 'reason': f'elec_df read error: {e}'})
            return None

    for roi in rois:
        roi_dir = os.path.join(trial_save_dir, roi)
        if not os.path.exists(roi_dir):
            if verbose:
                skipped.append({'subject': '', 'roi': roi, 'predictor': '', 'reason': 'roi_dir not found'})
            continue
        for subject in subjects:
            edf = get_elec_df(subject)
            if edf is None:
                continue
            if mni_x not in edf.columns or mni_y not in edf.columns or mni_z not in edf.columns:
                if verbose:
                    skipped.append({'subject': subject, 'roi': roi, 'predictor': '', 'reason': f'elec_df missing {mni_x}/{mni_y}/{mni_z}'})
                continue
            for predictor in predictors:
                path = os.path.join(roi_dir, f"{subject}_{predictor}_zreactivation.npz")
                if not os.path.exists(path):
                    continue
                try:
                    npz = np.load(path, allow_pickle=True)
                except Exception as e:
                    if verbose:
                        skipped.append({'subject': subject, 'roi': roi, 'predictor': predictor, 'reason': f'NPZ load error: {e}'})
                    continue
                if 'activation_patterns' not in npz.files:
                    if verbose:
                        skipped.append({'subject': subject, 'roi': roi, 'predictor': predictor, 'reason': 'missing activation_patterns'})
                    continue
                ap = npz['activation_patterns']
                if 'electrode_labels' not in npz.files:
                    if verbose:
                        skipped.append({'subject': subject, 'roi': roi, 'predictor': predictor, 'reason': 'missing electrode_labels'})
                    continue
                labels = np.asarray(npz['electrode_labels']).tolist()
                if isinstance(labels[0], (list, np.ndarray)):
                    labels = [str(l) for l in labels]
                else:
                    labels = [str(l) for l in labels]
                # time-average over axis=0 (feedback time)
                time_avg = np.nanmean(ap, axis=0)
                if len(labels) != len(time_avg):
                    if verbose:
                        skipped.append({'subject': subject, 'roi': roi, 'predictor': predictor, 'reason': 'labels len != activation len'})
                    continue
                for idx, label in enumerate(labels):
                    match = edf[edf['label'].astype(str) == str(label)]
                    if match.empty:
                        continue
                    row = match.iloc[0]
                    mx, my, mz = row[mni_x], row[mni_y], row[mni_z]
                    if pd.isna(mx) or pd.isna(my) or pd.isna(mz):
                        continue
                    rows.append({
                        'subject': subject,
                        'roi': roi,
                        'predictor': predictor,
                        'label': label,
                        'mni_x': float(mx),
                        'mni_y': float(my),
                        'mni_z': float(mz),
                        'time_avg_activation': float(time_avg[idx]),
                    })
                npz.close()
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=['subject', 'roi', 'predictor', 'label', 'mni_x', 'mni_y', 'mni_z', 'time_avg_activation'])
    return df, skipped


def coords_values_to_heatmap_nifti(
    mni_coords: np.ndarray,
    values: np.ndarray,
    template_img: Any,
    sigma_mm: float = 10.0,
) -> Any:
    """
    Build a 3D Nifti heatmap by Gaussian kernel smoothing of scattered (MNI coords, values) onto the template grid.

    Each electrode contributes a Gaussian blob (in mm) to the volume; voxel values are the sum of
    value * exp(-distance² / (2 * sigma_mm²)) over all electrodes. Suitable for glass-brain plotting.

    Parameters
    ----------
    mni_coords : np.ndarray, shape (N, 3)
        MNI coordinates in mm (x, y, z) for each of N points.
    values : np.ndarray, shape (N,)
        Value (e.g. time-averaged activation) at each point.
    template_img : image-like
        Template volume with .affine and .shape or .dataobj / get_fdata() (e.g. nibabel or nilearn image).
    sigma_mm : float
        Gaussian kernel width in mm (smoothing).

    Returns
    -------
    heatmap_img : nibabel.Nifti1Image
        Heatmap volume with same affine and shape as template. Caller can pass to nilearn.plot_glass_brain.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("coords_values_to_heatmap_nifti requires nibabel. Install with: pip install nibabel")

    mni_coords = np.asarray(mni_coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if mni_coords.ndim != 2 or mni_coords.shape[1] != 3:
        raise ValueError("mni_coords must be shape (N, 3)")
    if values.size != mni_coords.shape[0]:
        raise ValueError("values length must match mni_coords shape[0]")

    if hasattr(template_img, 'dataobj'):
        template_data = np.asarray(template_img.dataobj).astype(np.float64)
    elif hasattr(template_img, 'get_fdata'):
        template_data = template_img.get_fdata().astype(np.float64)
    else:
        template_data = np.asarray(template_img).astype(np.float64)
    affine = np.asarray(template_img.affine, dtype=np.float64)
    shape = template_data.shape
    inv_affine = np.linalg.inv(affine)

    out = np.zeros(shape, dtype=np.float64)
    voxel_coords = np.zeros(3, dtype=np.float64)
    sigma_sq = sigma_mm * sigma_mm
    # Approximate voxel size (mm) from affine for neighborhood size
    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    half_vox = int(np.ceil(3.0 * sigma_mm / voxel_size.min())) + 1

    for i in range(mni_coords.shape[0]):
        x, y, z = mni_coords[i]
        v = values[i]
        vox_center = nib.affines.apply_affine(inv_affine, [x, y, z])
        cx, cy, cz = int(round(vox_center[0])), int(round(vox_center[1])), int(round(vox_center[2]))
        for dx in range(-half_vox, half_vox + 1):
            for dy in range(-half_vox, half_vox + 1):
                for dz in range(-half_vox, half_vox + 1):
                    vi, vj, vk = cx + dx, cy + dy, cz + dz
                    if vi < 0 or vi >= shape[0] or vj < 0 or vj >= shape[1] or vk < 0 or vk >= shape[2]:
                        continue
                    world = nib.affines.apply_affine(affine, [vi, vj, vk])
                    d_sq = (world[0] - x) ** 2 + (world[1] - y) ** 2 + (world[2] - z) ** 2
                    w = np.exp(-d_sq / (2.0 * sigma_sq))
                    out[vi, vj, vk] += v * w
    return nib.Nifti1Image(out, affine)


def load_all_trial_level_per_roi(
    roi: str,
    trial_save_dir: str,
    dependent_var: Optional[str] = None,
    aggregate_electrodes: bool = True,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load all trial-level z-reactivation data for an ROI across all subjects and
    split into hits/misses trial arrays.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    trial_save_dir : str
        Directory containing saved trial-level data
        
    dependent_var : str or None, default=None
        Optional dependent variable name filter. If None, load all files.
    aggregate_electrodes : bool, default=True
        If file contains electrode-level 4D array, aggregate across electrodes.
    verbose : bool, default=True
        Print warnings for skipped files.

    Returns:
    --------
    hits_data, misses_data, feedback_times_list, cue_times_list, subjects
        `hits_data` and `misses_data` each contain subject arrays of shape
        (n_feedback_t, n_cue_t, n_trials_in_group).
    """
    roi_dir = os.path.join(trial_save_dir, roi)
    if not os.path.exists(roi_dir):
        return [], [], [], [], []
    
    # Find all subject files
    if dependent_var is None:
        pattern = os.path.join(roi_dir, "*_zreactivation.npz")
    else:
        pattern = os.path.join(roi_dir, f"*_{dependent_var}_zreactivation.npz")
    files = glob.glob(pattern)

    hits_data = []
    misses_data = []
    feedback_times_list = []
    cue_times_list = []
    subjects = []

    for file in sorted(files):
        basename = os.path.basename(file)
        if not basename.endswith("_zreactivation.npz"):
            continue
        stem = basename[:-len("_zreactivation.npz")]
        parts = stem.split("_", 1)
        if len(parts) != 2:
            if verbose:
                print(f"Skipping malformed trial file name: {basename}")
            continue
        subject, file_dep_var = parts
        if dependent_var is not None and file_dep_var != dependent_var:
            continue

        try:
            z_reactivation, fb_times, cue_times, _ = load_trial_level(
                roi, subject, trial_save_dir, dependent_var=file_dep_var
            )

            if z_reactivation.ndim == 4:
                if aggregate_electrodes:
                    z_reactivation = np.mean(z_reactivation, axis=2)
                else:
                    if verbose:
                        print(f"Skipping {subject}: electrode-level data requires aggregation.")
                    continue
            elif z_reactivation.ndim != 3:
                if verbose:
                    print(f"Skipping {subject}: unexpected z_reactivation shape {z_reactivation.shape}")
                continue

            npz = np.load(file, allow_pickle=True)
            if 'memory_outcomes' not in npz.files:
                if verbose:
                    print(f"Skipping {subject}: missing memory_outcomes in {basename}")
                continue
            memory_outcomes = np.asarray(npz['memory_outcomes'])
            if len(memory_outcomes) != z_reactivation.shape[-1]:
                if verbose:
                    print(
                        f"Skipping {subject}: memory_outcomes length ({len(memory_outcomes)}) "
                        f"!= trial count ({z_reactivation.shape[-1]})"
                    )
                continue

            hits_mask = memory_outcomes == 'hits'
            misses_mask = memory_outcomes == 'misses'
            if np.sum(hits_mask) < 2 or np.sum(misses_mask) < 2:
                if verbose:
                    print(f"Skipping {subject}: insufficient hits/misses trials")
                continue

            hits_data.append(z_reactivation[:, :, hits_mask])
            misses_data.append(z_reactivation[:, :, misses_mask])
            feedback_times_list.append(fb_times)
            cue_times_list.append(cue_times)
            subjects.append(subject)
        except Exception as e:
            if verbose:
                print(f"Error loading {subject}: {e}")
            continue

    return hits_data, misses_data, feedback_times_list, cue_times_list, subjects


# =============================================================================
# ENCODING & REACTIVATION
# =============================================================================

def fit_encoding_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Fit encoding model using Ridge regression.
    
    Parameters:
    -----------
    X : array, shape (n_trials, n_electrodes)
        HFA data for all electrodes at a single timepoint
    y : array, shape (n_trials,)
        Dependent variable values
    alpha : float, default=1.0
        Ridge regression regularization parameter
        
    Returns:
    --------
    coef : array, shape (n_electrodes,)
        Regression coefficients (encoding weights)
    intercept : float
        Regression intercept
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model.coef_.astype(np.float32), np.float32(model.intercept_)


def compute_time_resolved_reactivation(
    cue_hfa: np.ndarray,
    w: np.ndarray,
    b: float,
    electrode_level: bool = False
) -> np.ndarray:
    """
    Project cue HFA through encoding weights to compute reactivation.
    
    Parameters:
    -----------
    cue_hfa : array, shape (n_trials, n_electrodes, n_timepoints)
        Cue period HFA data
    w : array, shape (n_electrodes,)
        Encoding weights from encoding model
    b : float
        Intercept from encoding model
    electrode_level : bool, default=False
        If True, compute reactivation at electrode level (element-wise).
        Returns shape (n_trials, n_electrodes, n_timepoints).
        If False, compute vector-level reactivation by summing over electrodes (axis 1),
        not by matmul on the last axis. Returns shape (n_trials, n_timepoints).
        
    Returns:
    --------
    reactivation : array
        If electrode_level=False: shape (n_trials, n_timepoints)
        If electrode_level=True: shape (n_trials, n_electrodes, n_timepoints)
        Reactivation values at each timepoint (and electrode if electrode_level=True)
    """
    n_trials, n_electrodes, n_times = cue_hfa.shape

    if electrode_level:
        # Electrode-level reactivation: (n_trials, n_electrodes, n_timepoints), vectorized
        reactivation = cue_hfa * w[np.newaxis, :, np.newaxis] + b
    else:
        # Vector-level reactivation: (n_trials, n_timepoints), sum over electrodes (axis 1)
        reactivation = np.tensordot(cue_hfa, w, axes=(1, 0)) + b

    return reactivation


def _compute_single_surrogate(
    fb_hfa: np.ndarray,
    cue_hfa: np.ndarray,
    y: np.ndarray,
    n_feedback_t: int,
    n_cue_t: int,
    n_trials_cue: int,
    surrogate_idx: int,
    base_seed: int,
    ridge_alpha: float,
    electrode_level: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Helper function to compute a single surrogate using dependent variable shuffling.
    
    This function is designed to be called in parallel. It shuffles dependent variable values
    to break the relationship between the dependent variable and HFA patterns, then refits the
    encoding model and computes null reactivation.
    
    Parameters:
    -----------
    fb_hfa : array, shape (n_trials, n_electrodes, n_feedback_t)
        Feedback period HFA data
    cue_hfa : array, shape (n_trials, n_electrodes, n_cue_t)
        Cue period HFA data
    y : array, shape (n_trials,)
        Original dependent variable values
    n_feedback_t : int
        Number of feedback timepoints
    n_cue_t : int
        Number of cue timepoints
    n_trials_cue : int
        Number of trials
    surrogate_idx : int
        Index of this surrogate iteration
    base_seed : int
        Base random seed
    ridge_alpha : float
        Ridge regression regularization parameter
    electrode_level : bool, default=False
        If True, compute reactivation at electrode level.
        Returns shape (n_feedback_t, n_cue_t, n_electrodes, n_trials_cue).
        If False, compute vector-level reactivation.
        Returns shape (n_feedback_t, n_cue_t, n_trials_cue).
        
    Returns:
    --------
    reactivation_surrogate : array
        If electrode_level=False: shape (n_feedback_t, n_cue_t, n_trials_cue)
        If electrode_level=True: shape (n_feedback_t, n_cue_t, n_electrodes, n_trials_cue)
        Null reactivation values
    null_weights : list of arrays
        Null encoding weights for each feedback timepoint
    null_intercepts : list of floats
        Null intercepts for each feedback timepoint
    """
    # Set random seed for this surrogate
    np.random.seed(base_seed + surrogate_idx)
    
    # Shuffle dependent variable values
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)
    
    # Store null weights and compute null reactivation
    null_weights = []
    null_intercepts = []
    
    n_electrodes = fb_hfa.shape[1]
    
    if electrode_level:
        reactivation_surrogate = np.zeros((n_feedback_t, n_cue_t, n_electrodes, n_trials_cue), dtype=np.float32)
    else:
        reactivation_surrogate = np.zeros((n_feedback_t, n_cue_t, n_trials_cue), dtype=np.float32)
    
    for t in range(n_feedback_t):
        # Refit Ridge regression with shuffled dependent variable
        X = fb_hfa[:, :, t]  # (n_trials, n_electrodes)
        w_null, b_null = fit_encoding_model(X, y_shuffled, alpha=ridge_alpha)
        null_weights.append(w_null)
        null_intercepts.append(b_null)
        
        # Compute reactivation with null weights
        reactivation_t = compute_time_resolved_reactivation(
            cue_hfa, w_null, b_null, electrode_level=electrode_level
        )
        
        if electrode_level:
            # Shape: (n_trials, n_electrodes, n_cue_t) -> (n_cue_t, n_electrodes, n_trials)
            reactivation_surrogate[t] = np.transpose(reactivation_t, (2, 1, 0))
        else:
            # Shape: (n_trials, n_cue_t) -> (n_cue_t, n_trials)
            reactivation_surrogate[t] = reactivation_t.T
    
    return reactivation_surrogate, null_weights, null_intercepts


def compute_subject_reactivation(
    data: Dict[str, Any],
    n_surrogates: int = 500,
    use_surrogates: bool = True,
    ridge_alpha: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = -1,
    trial_save_dtype: type = np.float32,
    electrode_level: bool = False,
    surrogate_chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    """
    Compute z-scored reactivation for a single subject.

    Fits encoding models at each feedback timepoint, computes reactivation during
    cue periods, and optionally z-scores against a surrogate null distribution.
    Surrogate null is computed with Welford's algorithm so the full surrogate
    array is never allocated (avoids OOM when electrode_level=True and n_electrodes
    is large). Use surrogate_chunk_size=1 (default) for minimal memory, or set
    to 25-50 with small n_jobs for some speed. When electrode_level=True,
    use low n_jobs (e.g. 2) to limit worker memory.

    Parameters:
    -----------
    data : dict
        Output from load_subject_data containing feedback_hfa, cue_hfa, dependent_var, etc.
    n_surrogates : int, default=500
        Number of surrogate iterations for null distribution
    use_surrogates : bool, default=True
        Whether to compute surrogates and z-score
    ridge_alpha : float, default=1.0
        Ridge regression regularization parameter
    seed : int, default=42
        Random seed for surrogate dependent variable shuffling
    verbose : bool, default=True
        Print progress messages
    n_jobs : int, default=-1
        Number of parallel jobs for surrogate computation (use low value when electrode_level=True)
    trial_save_dtype : type, default=np.float32
        Data type for saving trial-level reactivation
    electrode_level : bool, default=False
        If True, compute reactivation at electrode level.
        Returns shape (n_feedback_t, n_cue_t, n_electrodes, n_trials).
        If False, compute vector-level reactivation.
        Returns shape (n_feedback_t, n_cue_t, n_trials).
    surrogate_chunk_size : int or None, default=None
        If None, process surrogates one at a time (Welford, minimal memory).
        If set (e.g. 50), process in chunks and merge into Welford state for some parallelism.

    Returns:
    --------
    reactivation_real : array, float32
        Real reactivation values (before z-scoring). Shape as in electrode_level.
    z_reactivation : array
        Z-scored reactivation (trial_save_dtype, typically float32).
    feedback_times : array, float32
        Time vector for feedback period
    cue_times : array, float32
        Time vector for cue period
    weights : list of arrays
        Actual encoding weights for each feedback timepoint
    activation_patterns : list of arrays
        Haufe et al. activation patterns (Sigma_x @ w) for each feedback timepoint
    null_weights : array or None, shape (n_surrogates, n_feedback_t, n_electrodes), float32
        Null encoding weights (None if use_surrogates=False)
    """
    fb_hfa = np.asarray(data['feedback_hfa'], dtype=np.float32)
    cue_hfa = np.asarray(data['cue_hfa'], dtype=np.float32)
    y = data['dependent_var']
    memory_outcomes = data['memory_outcomes']
    feedback_times = np.asarray(data['feedback_times'], dtype=np.float32)
    cue_times = np.asarray(data['cue_times'], dtype=np.float32)

    n_feedback_t = fb_hfa.shape[-1]
    n_cue_t = cue_hfa.shape[-1]
    n_trials_cue = cue_hfa.shape[0]
    n_electrodes = fb_hfa.shape[1]

    weights = []
    activation_patterns = []
    intercepts = []

    if verbose:
        print(f"Fitting encoding models for {n_feedback_t} feedback timepoints...")

    X_all = fb_hfa.transpose(0, 2, 1).reshape(-1, fb_hfa.shape[1])
    Sigma = np.cov(X_all, rowvar=False)  # (n_electrodes, n_electrodes)

    for t in range(n_feedback_t):
        X = fb_hfa[:, :, t]
        w, b = fit_encoding_model(X, y, alpha=ridge_alpha)
        weights.append(w)
        # sigma_x_t = np.cov(X, rowvar=False)
        # activation_patterns.append((sigma_x_t @ w).astype(np.float32))
        activation_patterns.append((Sigma @ w).astype(np.float32))
        intercepts.append(b)

    if verbose:
        print("Computing real reactivation...")

    if electrode_level:
        reactivation_real = np.zeros((n_feedback_t, n_cue_t, n_electrodes, n_trials_cue), dtype=np.float32)
        for t in range(n_feedback_t):
            reactivation_t = compute_time_resolved_reactivation(
                cue_hfa, weights[t], intercepts[t], electrode_level=electrode_level
            )
            reactivation_real[t] = np.transpose(reactivation_t, (2, 1, 0))
    else:
        reactivation_real = np.zeros((n_feedback_t, n_cue_t, n_trials_cue), dtype=np.float32)
        for t in range(n_feedback_t):
            reactivation_real[t] = compute_time_resolved_reactivation(
                cue_hfa, weights[t], intercepts[t], electrode_level=electrode_level
            ).T

    if not use_surrogates:
        reactivation_to_use = reactivation_real
        null_weights = None
    else:
        if verbose:
            print(f"Computing {n_surrogates} surrogates (Welford)...")
        chunk_size = surrogate_chunk_size if surrogate_chunk_size is not None else 1
        if electrode_level:
            running_mean = np.zeros((n_feedback_t, n_cue_t, n_electrodes, n_trials_cue), dtype=np.float32)
            running_M2 = np.zeros((n_feedback_t, n_cue_t, n_electrodes, n_trials_cue), dtype=np.float32)
        else:
            running_mean = np.zeros((n_feedback_t, n_cue_t, n_trials_cue), dtype=np.float32)
            running_M2 = np.zeros((n_feedback_t, n_cue_t, n_trials_cue), dtype=np.float32)
        n_total = 0
        null_weights_list = []

        for chunk_start in tqdm(range(0, n_surrogates, chunk_size), desc="Surrogates", disable=not verbose):
            chunk_end = min(chunk_start + chunk_size, n_surrogates)
            chunk_indices = range(chunk_start, chunk_end)
            if chunk_size == 1:
                s = chunk_start
                reactivation_s, null_w_list, _ = _compute_single_surrogate(
                    fb_hfa, cue_hfa, y, n_feedback_t, n_cue_t, n_trials_cue, s, seed,
                    ridge_alpha, electrode_level
                )
                null_weights_list.append(null_w_list)
                x = np.asarray(reactivation_s, dtype=np.float32)
                chunk_n = 1
                chunk_mean = x
                chunk_M2 = np.zeros_like(x, dtype=np.float32)
            else:
                chunk_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(_compute_single_surrogate)(
                        fb_hfa, cue_hfa, y, n_feedback_t, n_cue_t, n_trials_cue, s, seed,
                        ridge_alpha, electrode_level
                    )
                    for s in chunk_indices
                )
                chunk_arrays = [np.asarray(r[0], dtype=np.float32) for r in chunk_results]
                null_weights_list.extend([r[1] for r in chunk_results])
                chunk_array = np.stack(chunk_arrays, axis=0)
                chunk_n = chunk_array.shape[0]
                chunk_mean = np.mean(chunk_array, axis=0).astype(np.float32)
                chunk_M2 = np.sum((chunk_array - chunk_mean) ** 2, axis=0).astype(np.float32)

            # Merge chunk into Welford state (Chan et al. parallel merge)
            if n_total == 0:
                running_mean = chunk_mean.copy()
                running_M2 = chunk_M2.copy()
                n_total = chunk_n
            else:
                n_new = n_total + chunk_n
                delta = chunk_mean - running_mean
                running_mean = running_mean + delta * (np.float32(chunk_n) / np.float32(n_new))
                running_M2 = running_M2 + chunk_M2 + delta ** 2 * np.float32(n_total * chunk_n) / np.float32(n_new)
                n_total = n_new

        surrogate_mean = running_mean
        surrogate_std = np.sqrt(running_M2 / n_total)
        surrogate_std = np.maximum(surrogate_std, 1e-8)
        reactivation_to_use = ((reactivation_real - surrogate_mean) / surrogate_std).astype(np.float32)

        null_weights = np.stack([
            np.stack(null_w_list, axis=0).astype(np.float32)
            for null_w_list in null_weights_list
        ], axis=0)

    z_reactivation = reactivation_to_use.astype(trial_save_dtype, copy=False)
    return reactivation_real, z_reactivation, feedback_times, cue_times, weights, activation_patterns, null_weights


def permutation_regression_zscore_local(
    df: pd.DataFrame,
    formula: str,
    n_permutations: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Local Freedman-Lane permutation z-scoring for OLS formulas.

    Fits OLS once on observed data, then for each non-intercept term builds a
    term-specific reduced model and computes permutation p-values using
    Freedman-Lane pseudo-responses.
    """
    y_df, x_df = dmatrices(formula, data=df, return_type='dataframe')
    y = np.asarray(y_df).ravel()
    X = np.asarray(x_df, dtype=np.float64)
    predictor_names = list(x_df.columns)

    if X.shape[0] < 3:
        return pd.DataFrame(columns=['predictor', 'beta', 'z_beta', 'raw_p'])

    hat = np.linalg.pinv(X)
    beta_obs = hat @ y

    rng = np.random.default_rng(seed)
    n_terms = X.shape[1]
    z_beta = np.full(n_terms, np.nan, dtype=np.float64)
    raw_p = np.full(n_terms, np.nan, dtype=np.float64)

    is_intercept = np.array([
        (str(name).strip().lower() == 'intercept') or np.allclose(X[:, idx], X[0, idx])
        for idx, name in enumerate(predictor_names)
    ], dtype=bool)

    for term_idx in range(n_terms):
        if is_intercept[term_idx]:
            continue

        X_reduced = np.delete(X, term_idx, axis=1)
        y_hat0, resid0 = _fit_ols_reduced_components(y, X_reduced)
        y_hat0 = y_hat0.reshape(-1)
        resid0 = resid0.reshape(-1)

        perm_betas_term = np.full(max(0, int(n_permutations)), np.nan, dtype=np.float64)
        for p_idx in range(perm_betas_term.size):
            perm_ix = rng.permutation(y.shape[0])
            y_star = y_hat0 + resid0[perm_ix]
            beta_perm = hat @ y_star
            perm_betas_term[p_idx] = float(beta_perm[term_idx])

        perm_betas_term = perm_betas_term[np.isfinite(perm_betas_term)]
        if perm_betas_term.size == 0:
            continue

        perm_mean = float(np.mean(perm_betas_term))
        perm_std = max(float(np.std(perm_betas_term, ddof=0)), 1e-8)
        z_beta[term_idx] = (float(beta_obs[term_idx]) - perm_mean) / perm_std
        raw_p[term_idx] = (
            np.sum(np.abs(perm_betas_term) >= np.abs(float(beta_obs[term_idx]))) + 1.0
        ) / (perm_betas_term.size + 1.0)

    return pd.DataFrame({
        'predictor': predictor_names,
        'beta': beta_obs.astype(np.float64),
        'z_beta': z_beta,
        'raw_p': raw_p,
    })


def _resolve_trial_var_for_event(df: pd.DataFrame, event: str) -> str:
    """Resolve trial index column for an event-specific dataframe."""
    return resolve_trial_for_event(df, event)


def _build_lrt_trial_axis(
    subject_obj: Subject,
    event_projection: str,
    cue_n_trials: int,
) -> Dict[str, Any]:
    """
    Build all-trial projection axis for LRT using Day1 combined_df rows.

    Returns aligned cue-trial indices and 4-class memory outcomes
    (hits/misses/false_alarms/correct_rejections).
    """
    memory_df = subject_obj.combined_df[
        (subject_obj.combined_df.participant == subject_obj.subject) &
        (subject_obj.combined_df.condition == 'Day1')
    ].copy()
    if len(memory_df) == 0:
        raise ValueError("no_day1_trials_for_lrt_axis")

    if 'memory_outcome' not in memory_df.columns:
        needed = ['hits', 'misses', 'false_alarms', 'correct_rejections']
        missing = [c for c in needed if c not in memory_df.columns]
        if len(missing) > 0:
            raise ValueError(f"missing_memory_columns_for_lrt_axis:{missing}")
        memory_df['memory_outcome'] = memory_df[needed].idxmax(axis=1)

    trial_col = _resolve_trial_var_for_event(memory_df, event_projection)
    memory_df = memory_df.dropna(subset=[trial_col, 'memory_outcome']).copy()
    if len(memory_df) == 0:
        raise ValueError("no_valid_trials_after_dropna_for_lrt_axis")

    trial_vals = pd.to_numeric(memory_df[trial_col], errors='coerce').to_numpy(dtype=np.float64)
    in_range = (
        np.isfinite(trial_vals) &
        ((trial_vals - 1) >= 0) &
        ((trial_vals - 1) < cue_n_trials)
    )
    if int(np.sum(in_range)) == 0:
        raise ValueError("no_in_range_trials_for_lrt_axis")

    trial_ix = (trial_vals[in_range].astype(np.int64) - 1)
    outcomes = memory_df.loc[in_range, 'memory_outcome'].astype(object).to_numpy()
    return {
        'trial_index_lrt': trial_ix.astype(np.int64, copy=False),
        'memory_outcomes_lrt': outcomes,
        'n_dropped_alignment': int((~in_range).sum()),
    }


def _build_subject_regressors(
    subject_obj: Subject,
    event: str,
    predictor: str,
    covariates: Optional[List[str]] = None,
    include_columns: Optional[List[str]] = None,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Build subject-specific regressor dataframe for a given event.

    When regressor_source is 'combined', regressors are built from combined_df and
    the trial variable defaults to trials_gamble (so rows align to feedback-period
    neural data indexed by gamble trial). Use this for predictors that live only
    in combined_df (e.g. phit, hits) while still encoding at feedback_start.
    """
    covariates = covariates or ['zpm']
    if regressor_source == 'combined':
        src_df = subject_obj.combined_df[
            (subject_obj.combined_df.participant == subject_obj.subject) &
            (subject_obj.combined_df.condition == 'Day1')
        ].copy()
        if trial_var_override is not None:
            trial_var = trial_var_override
        elif 'trials_gamble' in src_df.columns:
            trial_var = 'trials_gamble'
        else:
            trial_var = _resolve_trial_var_for_event(src_df, 'cue_on')
    elif regressor_source == 'learn':
        src_df = subject_obj.learn_df[
            subject_obj.learn_df.participant == subject_obj.subject
        ].copy()
        if trial_var_override is not None:
            trial_var = trial_var_override
        else:
            trial_var = _resolve_trial_var_for_event(src_df, event)
    else:
        if event == 'cue_on':
            src_df = subject_obj.combined_df[
                (subject_obj.combined_df.participant == subject_obj.subject) &
                (subject_obj.combined_df.condition == 'Day1')
            ].copy()
        else:
            src_df = subject_obj.learn_df[
                subject_obj.learn_df.participant == subject_obj.subject
            ].copy()
        if trial_var_override is not None:
            trial_var = trial_var_override
        else:
            trial_var = _resolve_trial_var_for_event(src_df, event)

    if len(src_df) == 0:
        raise ValueError(f"No regressors available for subject {subject_obj.subject}, event={event}")

    if trial_var not in src_df.columns:
        raise ValueError(f"Trial variable '{trial_var}' not in regressor dataframe columns")
    include_columns = include_columns or []
    required = [predictor, trial_var] + [c for c in covariates if c in src_df.columns]
    optional = [c for c in include_columns if c in src_df.columns]
    missing = [c for c in (predictor,) if c not in src_df.columns]
    if missing:
        raise ValueError(f"Missing required regressor columns: {missing}")

    reg_df = src_df[required + optional].copy()
    reg_df = reg_df.dropna(subset=[predictor, trial_var])
    reg_df = reg_df.sort_values(by=[trial_var]).rename(columns={trial_var: 'trial'})
    reg_df['trial'] = reg_df['trial'].astype(int)
    return reg_df, trial_var


def _get_smoothed_event_signal(
    subject_obj: Subject,
    roi: str,
    event: str,
    freq_range: Tuple[float, float],
    downsample_factor: int,
    win_ms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get smoothed/downsampled event signal from Subject cache for one ROI.
    """
    cache_key = (event, roi, tuple(freq_range), int(downsample_factor), int(win_ms))
    neural = subject_obj._get_or_build_event_neural_data(
        event=event,
        roi=roi,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
    )
    subject_obj._neural_cache[cache_key] = neural
    signal = np.asarray(neural['hfa'], dtype=np.float32)
    times = np.asarray(neural['times'], dtype=np.float32)

    electrode_labels = neural.get('electrode_labels')
    if electrode_labels is None:
        raise ValueError(
            f"Missing electrode_labels for {subject_obj.subject} {event}/{roi}; "
            "refusing synthetic label fallback."
        )
    electrode_labels = np.asarray(electrode_labels, dtype=str)
    return signal, times, electrode_labels


def _compute_pairwise_dissimilarity(
    X: np.ndarray,
    method: str = 'euclidean',
) -> np.ndarray:
    """
    Compute upper-triangle dissimilarity vector for trial patterns.
    """
    from scipy.spatial.distance import pdist

    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] < 3:
        return np.array([], dtype=np.float64)
    if method == 'euclidean':
        return pdist(X_arr, metric='euclidean')
    if method in ('correlation', 'corr'):
        return pdist(X_arr, metric='correlation')
    if method == 'mahalanobis':
        cov = np.cov(X_arr, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        return pdist(X_arr, metric='mahalanobis', VI=cov_inv)
    raise ValueError(f"Unsupported rdm_method: {method}")


def _compare_rdm_vectors(
    v1: np.ndarray,
    v2: np.ndarray,
    method: str = 'rho-a',
) -> float:
    """
    Compare two vectorized RDMs and return scalar similarity.
    """
    from scipy.stats import spearmanr, pearsonr

    a = np.asarray(v1, dtype=np.float64).reshape(-1)
    b = np.asarray(v2, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    if method in ('rho-a', 'spearman'):
        return float(spearmanr(a, b)[0])
    if method in ('corr', 'pearson'):
        return float(pearsonr(a, b)[0])
    raise ValueError(f"Unsupported rdm_compare_method: {method}")


def _compute_pairwise_circular_phase_dissimilarity(P: np.ndarray) -> np.ndarray:
    """
    Upper-triangle vector of multivariate circular phase dissimilarity.

    For input ``P`` of shape ``(n_trials, n_elec)`` with phases in radians, the
    pairwise dissimilarity between trials ``i`` and ``j`` is

        d(i, j) = 1 - | (1 / n_elec) * sum_e exp( i * (P[i, e] - P[j, e]) ) |

    which is one minus the cross-trial mean resultant length of the per-electrode
    phase differences (a population-PLV distance). The result lives in [0, 1],
    is zero on the diagonal, and is invariant to constant per-electrode phase
    offsets shared by both trials.

    NaN-safe: non-finite phase entries are excluded per-pair via an
    electrode-count denominator that counts only entries finite in both trials.
    Pairs with zero shared finite electrodes are returned as NaN.

    Returns
    -------
    upper_tri : (n_trials * (n_trials - 1) / 2,) float64
        Condensed upper-triangle vector in row-major order (matching
        ``scipy.spatial.distance.pdist``). Empty array if fewer than 3 trials.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    if P_arr.ndim != 2 or P_arr.shape[0] < 3 or P_arr.shape[1] < 1:
        return np.array([], dtype=np.float64)

    n_trials, n_elec = P_arr.shape
    finite_mask = np.isfinite(P_arr)
    P_safe = np.where(finite_mask, P_arr, 0.0)

    Z = finite_mask.astype(np.complex128) * np.exp(1j * P_safe)
    counts = finite_mask.astype(np.float64) @ finite_mask.astype(np.float64).T
    inner = Z @ np.conj(Z).T

    with np.errstate(invalid='ignore', divide='ignore'):
        prl = np.abs(inner) / counts

    dist = 1.0 - prl
    dist[counts <= 0] = np.nan
    np.fill_diagonal(dist, 0.0)

    iu = np.triu_indices(n_trials, k=1)
    return dist[iu].astype(np.float64, copy=False)


def build_crossregion_rsa_paths(
    trial_save_dir: str,
    group_save_dir: str,
    encoding_roi: str,
    cue_roi: str,
    subject: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build deterministic cross-region RSA output/checkpoint paths.
    """
    pair_name = f"{encoding_roi}_to_{cue_roi}"
    pair_dir = os.path.join(trial_save_dir, pair_name)
    checkpoint_dir = os.path.join(pair_dir, "checkpoints")
    subject_npz = (
        os.path.join(pair_dir, f"{subject}_rsa.npz")
        if subject is not None else ""
    )
    group_npz = os.path.join(group_save_dir, f"{pair_name}_group_rsa.npz")
    subject_csv = os.path.join(group_save_dir, f"{pair_name}_subject_summary.csv")
    group_csv = os.path.join(group_save_dir, "group_summary_crossregion_rsa.csv")
    return {
        'pair_name': pair_name,
        'pair_dir': pair_dir,
        'checkpoint_dir': checkpoint_dir,
        'subject_npz': subject_npz,
        'group_npz': group_npz,
        'subject_csv': subject_csv,
        'group_csv': group_csv,
    }


def load_subject_crossregion_hfa(
    subject_obj: Subject,
    encoding_roi: str,
    cue_roi: str = 'HPC',
    event_encoding: str = 'feedback_start',
    event_cue: str = 'cue_on',
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
) -> Dict[str, Any]:
    """
    Load aligned cross-region encoding/cue HFA using legacy combined_df trial mapping.
    """
    enc_hfa_all, fb_times, enc_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=encoding_roi,
        event=event_encoding,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
    )
    cue_hfa_all, cue_times, cue_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=cue_roi,
        event=event_cue,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
    )

    subj_learn_df = subject_obj.learn_df[
        subject_obj.learn_df.participant == subject_obj.subject
    ].copy()
    subj_combined_df = subject_obj.combined_df[
        (subject_obj.combined_df.condition == 'Day1') &
        (subject_obj.combined_df.participant == subject_obj.subject)
    ].copy()

    if len(subj_learn_df) == 0:
        raise ValueError("no_day1_learn_df")
    if len(subj_combined_df) == 0:
        raise ValueError("no_day1_trials")

    needed = ['trials_gamble', 'trials_mem']
    missing_cols = [c for c in needed if c not in subj_combined_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"missing_trial_columns:{missing_cols}")

    subj_combined_df = subj_combined_df.dropna(subset=['trials_gamble', 'trials_mem']).copy()
    if len(subj_combined_df) == 0:
        raise ValueError("no_day1_trials_after_dropna")

    trial_ix_encoding = subj_combined_df['trials_gamble'].to_numpy(dtype=np.int64) - 1
    trial_ix_cue = subj_combined_df['trials_mem'].to_numpy(dtype=np.int64) - 1
    valid = (
        (trial_ix_encoding >= 0) & (trial_ix_encoding < enc_hfa_all.shape[0]) &
        (trial_ix_cue >= 0) & (trial_ix_cue < cue_hfa_all.shape[0])
    )
    n_dropped_alignment = int((~valid).sum())
    if np.sum(valid) < min_trials:
        raise ValueError(
            f"Insufficient aligned trials ({np.sum(valid)} < {min_trials}) "
            f"for {subject_obj.subject}, {encoding_roi}->{cue_roi}"
        )

    trial_ix_encoding = trial_ix_encoding[valid]
    trial_ix_cue = trial_ix_cue[valid]
    combined_sel = subj_combined_df.loc[valid].reset_index(drop=True)

    if 'hits' not in combined_sel.columns:
        raise ValueError("missing_hits_column")
    hits_col = pd.to_numeric(combined_sel['hits'], errors='coerce').to_numpy()
    if np.any(~np.isfinite(hits_col)):
        raise ValueError("invalid_hits_values")
    mem = np.where(hits_col.astype(bool), 'remembered', 'forgotten')

    enc_hfa = np.asarray(enc_hfa_all[trial_ix_encoding], dtype=np.float32)
    cue_hfa = np.asarray(cue_hfa_all[trial_ix_cue], dtype=np.float32)

    return {
        'encoding_hfa': enc_hfa,
        'cue_hfa': cue_hfa,
        'feedback_times': np.asarray(fb_times, dtype=np.float32),
        'cue_times': np.asarray(cue_times, dtype=np.float32),
        'memory_outcomes': np.asarray(mem),
        'trial_index': np.asarray(trial_ix_encoding, dtype=np.int64),
        'retrieval_trial_index': np.asarray(trial_ix_cue, dtype=np.int64),
        'electrode_labels_encoding': np.asarray(enc_labels, dtype=str),
        'electrode_labels_cue': np.asarray(cue_labels, dtype=str),
        'subject': subject_obj.subject,
        'encoding_roi': encoding_roi,
        'cue_roi': cue_roi,
        'freq_range': np.asarray(freq_range, dtype=np.float32),
        'win_ms': int(win_ms),
        'downsample_factor': int(downsample_factor),
        'n_dropped_alignment': np.int32(n_dropped_alignment),
        'alignment_mode': 'legacy_combined_df_trial_map',
    }


def compute_cross_temporal_rsa_from_hfa(
    encoding_hfa: np.ndarray,
    cue_hfa: np.ndarray,
    memory_outcomes: np.ndarray,
    rdm_method: str = 'euclidean',
    rdm_compare_method: str = 'rho-a',
    remembered_label: str = 'remembered',
    forgotten_label: str = 'forgotten',
) -> Dict[str, Any]:
    """
    Compute remembered/forgotten cross-temporal RSA heatmaps from aligned HFA.
    """
    enc = np.asarray(encoding_hfa, dtype=np.float64)
    cue = np.asarray(cue_hfa, dtype=np.float64)
    mem = np.asarray(memory_outcomes)
    if enc.ndim != 3 or cue.ndim != 3:
        raise ValueError("encoding_hfa and cue_hfa must be 3D: trials x electrodes x time")
    if enc.shape[0] != cue.shape[0]:
        raise ValueError("encoding_hfa and cue_hfa must have same n_trials")
    if mem.shape[0] != enc.shape[0]:
        raise ValueError("memory_outcomes length mismatch")

    n_fb = enc.shape[2]
    n_cue = cue.shape[2]

    def _compute_for_mask(mask: np.ndarray) -> np.ndarray:
        if np.sum(mask) < 3:
            return np.full((n_fb, n_cue), np.nan, dtype=np.float64)
        heat = np.full((n_fb, n_cue), np.nan, dtype=np.float64)
        enc_m = enc[mask]
        cue_m = cue[mask]
        for t_fb in range(n_fb):
            rdm_fb = _compute_pairwise_dissimilarity(enc_m[:, :, t_fb], method=rdm_method)
            if rdm_fb.size == 0:
                continue
            for t_cue in range(n_cue):
                rdm_c = _compute_pairwise_dissimilarity(cue_m[:, :, t_cue], method=rdm_method)
                if rdm_c.size == 0:
                    continue
                heat[t_fb, t_cue] = _compare_rdm_vectors(
                    rdm_fb, rdm_c, method=rdm_compare_method
                )
        return heat

    remembered_mask = mem == remembered_label
    forgotten_mask = mem == forgotten_label
    remembered_heat = _compute_for_mask(remembered_mask)
    forgotten_heat = _compute_for_mask(forgotten_mask)
    diff_heat = remembered_heat - forgotten_heat
    return {
        'remembered_heatmap': remembered_heat.astype(np.float32),
        'forgotten_heatmap': forgotten_heat.astype(np.float32),
        'diff_heatmap': diff_heat.astype(np.float32),
        'n_remembered': int(np.sum(remembered_mask)),
        'n_forgotten': int(np.sum(forgotten_mask)),
        'rdm_method': rdm_method,
        'rdm_compare_method': rdm_compare_method,
    }


def run_subject_crossregion_rsa_permutation(
    subject_obj: Subject,
    encoding_roi: str,
    cue_roi: str = 'HPC',
    event_encoding: str = 'feedback_start',
    event_cue: str = 'cue_on',
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
    n_permutations: int = 200,
    shuffle_type: str = 'trial_labels',
    seed: int = 42,
    rdm_method: str = 'euclidean',
    rdm_compare_method: str = 'rho-a',
) -> Dict[str, Any]:
    """
    Subject-level cross-region RSA with permutation-based null summary.
    """
    data = load_subject_crossregion_hfa(
        subject_obj=subject_obj,
        encoding_roi=encoding_roi,
        cue_roi=cue_roi,
        event_encoding=event_encoding,
        event_cue=event_cue,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
        min_trials=min_trials,
    )

    obs = compute_cross_temporal_rsa_from_hfa(
        encoding_hfa=data['encoding_hfa'],
        cue_hfa=data['cue_hfa'],
        memory_outcomes=data['memory_outcomes'],
        rdm_method=rdm_method,
        rdm_compare_method=rdm_compare_method,
    )

    obs_diff = np.asarray(obs['diff_heatmap'], dtype=np.float32)
    null_mean = np.zeros_like(obs_diff, dtype=np.float32)
    null_M2 = np.zeros_like(obs_diff, dtype=np.float32)
    extreme_counts = np.zeros_like(obs_diff, dtype=np.int32)

    mem = np.asarray(data['memory_outcomes'])
    cue = np.asarray(data['cue_hfa'], dtype=np.float32)
    n_done = 0
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            if shuffle_type == 'trial_labels':
                mem_perm = rng.permutation(mem)
                cue_perm = cue
            elif shuffle_type == 'trial_pairing':
                mem_perm = mem
                cue_perm = cue[rng.permutation(cue.shape[0])]
            else:
                raise ValueError(f"Unsupported shuffle_type: {shuffle_type}")

            perm = compute_cross_temporal_rsa_from_hfa(
                encoding_hfa=data['encoding_hfa'],
                cue_hfa=cue_perm,
                memory_outcomes=mem_perm,
                rdm_method=rdm_method,
                rdm_compare_method=rdm_compare_method,
            )
            x = np.asarray(perm['diff_heatmap'], dtype=np.float32)
            n_done += 1
            delta = x - null_mean
            null_mean = null_mean + delta / np.float32(n_done)
            null_M2 = null_M2 + delta * (x - null_mean)
            extreme_counts += (np.abs(x) >= np.abs(obs_diff)).astype(np.int32)

    if n_done > 0:
        null_std = np.sqrt(null_M2 / np.float32(n_done))
        null_std = np.maximum(null_std, 1e-8)
        z_diff = (obs_diff - null_mean) / null_std
        p_diff = (extreme_counts.astype(np.float32) + 1.0) / np.float32(n_done + 1)
    else:
        null_std = np.full_like(obs_diff, np.nan, dtype=np.float32)
        z_diff = np.full_like(obs_diff, np.nan, dtype=np.float32)
        p_diff = np.full_like(obs_diff, np.nan, dtype=np.float32)

    return {
        'subject': subject_obj.subject,
        'encoding_roi': encoding_roi,
        'cue_roi': cue_roi,
        'feedback_times': data['feedback_times'],
        'cue_times': data['cue_times'],
        'memory_outcomes': data['memory_outcomes'],
        'trial_index': data['trial_index'],
        'retrieval_trial_index': data['retrieval_trial_index'],
        'electrode_labels_encoding': data['electrode_labels_encoding'],
        'electrode_labels_cue': data['electrode_labels_cue'],
        'remembered_heatmap': obs['remembered_heatmap'],
        'forgotten_heatmap': obs['forgotten_heatmap'],
        'diff_heatmap': obs_diff,
        'z_diff_heatmap': z_diff.astype(np.float32),
        'p_diff_heatmap': p_diff.astype(np.float32),
        'null_mean': null_mean.astype(np.float32) if n_done > 0 else null_mean,
        'null_std': null_std.astype(np.float32),
        'n_trials': int(data['encoding_hfa'].shape[0]),
        'n_remembered': int(obs['n_remembered']),
        'n_forgotten': int(obs['n_forgotten']),
        'rdm_method': rdm_method,
        'rdm_compare_method': rdm_compare_method,
        'shuffle_type': shuffle_type,
        'n_permutations': int(n_permutations),
        'n_dropped_alignment': int(data.get('n_dropped_alignment', 0)),
        'alignment_mode': str(data.get('alignment_mode', 'legacy_combined_df_trial_map')),
        'status': 'ok',
    }


def run_group_crossregion_rsa_cluster_test(
    observed_diffs_stack: np.ndarray,
    n_permutations: int = 1024,
    tail: int = 1,
    threshold: Union[float, None] = 0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Cluster test on stacked subject diff maps: shape (n_subjects, n_fb, n_cue).
    """
    D = np.asarray(observed_diffs_stack, dtype=np.float64)
    if D.ndim != 3:
        raise ValueError(f"Expected 3D observed_diffs_stack, got {D.ndim}D")
    if D.shape[0] < 2:
        return {
            't_obs': None,
            'clusters': [],
            'cluster_pvals': np.array([]),
            'sig_mask': np.zeros(D.shape[1:], dtype=bool),
        }

    t_obs, clusters, cluster_pvals, h0 = permutation_cluster_1samp_test(
        D,
        n_permutations=n_permutations,
        tail=tail,
        threshold=threshold,
        out_type='mask',
        seed=seed,
        verbose=verbose,
    )
    sig_mask = np.zeros(D.shape[1:], dtype=bool)
    for c_mask, p in zip(clusters, cluster_pvals):
        if p <= 0.05:
            sig_mask |= c_mask.astype(bool)
    return {
        't_obs': np.asarray(t_obs),
        'clusters': clusters,
        'cluster_pvals': np.asarray(cluster_pvals),
        'sig_mask': sig_mask,
    }


def build_cluster_mask_from_clusters(
    clusters: List[np.ndarray],
    cluster_pvals: np.ndarray,
    shape: Tuple[int, int],
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Union of cluster masks whose cluster p-value is <= alpha.

    Parameters
    ----------
    clusters : list of 2D boolean/0-1 arrays matching `shape`
    cluster_pvals : 1D array, one p-value per cluster
    shape : (n_fb, n_cue)
    """
    sig_mask = np.zeros(shape, dtype=bool)
    for c_mask, p in zip(clusters, np.asarray(cluster_pvals).ravel()):
        if p <= alpha:
            sig_mask |= np.asarray(c_mask, dtype=bool)
    return sig_mask


def compute_acc_hpc_phase_rsa_at_cue(
    acc_hfa: np.ndarray,
    hpc_phase: np.ndarray,
    rdm_compare_method: str = 'rho-a',
    acc_method: str = 'euclidean',
    hpc_method: str = 'circular_plv',
) -> Dict[str, Any]:
    """
    Time-resolved trial-RDM RSA between ACC HFA and HPC theta phase on a single
    (shared) event grid (e.g., the cue window).

    Parameters
    ----------
    acc_hfa : (n_trials, n_acc, n_t)
        ACC HFA pattern per trial and time.
    hpc_phase : (n_trials, n_hpc, n_t)
        HPC theta phase per trial and time, in radians.
    rdm_compare_method : str
        ``'rho-a'`` / ``'spearman'`` (default) or ``'pearson'``.
    acc_method : str
        Distance for the ACC pattern RDM; passed to
        :func:`_compute_pairwise_dissimilarity` (default ``'euclidean'``).
    hpc_method : str
        ``'circular_plv'`` (default) for the multivariate circular phase RDM,
        or any metric supported by :func:`_compute_pairwise_dissimilarity`
        (e.g. ``'euclidean'``) to use the same Euclidean pipeline as the
        HFA-only negative control.

    Returns
    -------
    dict with keys
        rho     : (n_t,) float32 — observed RDM-RDM similarity per timepoint
        n_trials, n_acc, n_hpc, n_t
        acc_method, hpc_method, rdm_compare_method
    """
    A = np.asarray(acc_hfa, dtype=np.float64)
    P = np.asarray(hpc_phase, dtype=np.float64)
    if A.ndim != 3 or P.ndim != 3:
        raise ValueError("acc_hfa and hpc_phase must be 3D: trials x electrodes x time")
    if A.shape[0] != P.shape[0]:
        raise ValueError(
            f"trial-count mismatch: acc_hfa={A.shape[0]}, hpc_phase={P.shape[0]}"
        )
    if A.shape[2] != P.shape[2]:
        raise ValueError(
            f"time-count mismatch: acc_hfa={A.shape[2]}, hpc_phase={P.shape[2]}"
        )

    n_trials, n_acc, n_t = A.shape
    n_hpc = P.shape[1]
    rho = np.full(n_t, np.nan, dtype=np.float64)

    if n_trials < 3 or n_acc < 1 or n_hpc < 1:
        return {
            'rho': rho.astype(np.float32),
            'n_trials': int(n_trials),
            'n_acc': int(n_acc),
            'n_hpc': int(n_hpc),
            'n_t': int(n_t),
            'acc_method': str(acc_method),
            'hpc_method': str(hpc_method),
            'rdm_compare_method': str(rdm_compare_method),
        }

    for t_idx in range(n_t):
        acc_pat = A[:, :, t_idx]
        if acc_method == 'euclidean':
            acc_rdm = _compute_pairwise_dissimilarity(acc_pat, method='euclidean')
        else:
            acc_rdm = _compute_pairwise_dissimilarity(acc_pat, method=acc_method)
        if acc_rdm.size == 0:
            continue

        hpc_pat = P[:, :, t_idx]
        if hpc_method == 'circular_plv':
            hpc_rdm = _compute_pairwise_circular_phase_dissimilarity(hpc_pat)
        else:
            hpc_rdm = _compute_pairwise_dissimilarity(hpc_pat, method=hpc_method)
        if hpc_rdm.size == 0:
            continue

        rho[t_idx] = _compare_rdm_vectors(acc_rdm, hpc_rdm, method=rdm_compare_method)

    return {
        'rho': rho.astype(np.float32),
        'n_trials': int(n_trials),
        'n_acc': int(n_acc),
        'n_hpc': int(n_hpc),
        'n_t': int(n_t),
        'acc_method': str(acc_method),
        'hpc_method': str(hpc_method),
        'rdm_compare_method': str(rdm_compare_method),
    }


def run_subject_acc_hpc_phase_rsa_permutation(
    subject_obj: Subject,
    acc_roi: str = 'ACC',
    hpc_roi: str = 'HPC',
    event: str = 'cue_on',
    freq_range: Tuple[float, float] = (70., 201.),
    theta_band: Tuple[float, float] = (4., 8.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    smooth_method: str = 'movmean',
    artifact_abs_thresh: Optional[float] = 10.0,
    min_trials: int = 15,
    min_electrodes: int = 3,
    n_permutations: int = 500,
    seed: int = 42,
    rdm_compare_method: str = 'rho-a',
    acc_method: str = 'euclidean',
    hpc_method: str = 'circular_plv',
) -> Dict[str, Any]:
    """
    Subject-level time-resolved ACC HFA × HPC theta phase RSA on the cue grid,
    with HPC-side trial-label shuffle for a permutation-based z-score per time.

    ACC HFA is loaded with :func:`_get_smoothed_event_signal` (MVPA-Light
    parity via ``smooth_method`` / ``artifact_abs_thresh``); HPC theta phase
    is loaded with :func:`extract_theta_phase_from_cue_epochs` on the same
    cue time grid.

    Permutation: at each iteration the HPC trial axis is randomly permuted
    (preserves the marginal HPC RDM but breaks ACC-HPC trial correspondence),
    the time-resolved RSA is recomputed, and a streaming Welford pass yields
    the null mean / std per cue timepoint. The observed RSA is then z-scored
    against that null.

    Set ``hpc_method='euclidean'`` to repeat the same machinery as an
    HFA-only negative control (HPC HFA passed in as ``hpc_phase``).
    """
    if min_electrodes < 1:
        raise ValueError("min_electrodes must be >= 1")
    if min_trials < 3:
        raise ValueError("min_trials must be >= 3 to build an RDM")

    acc_signal, cue_times, acc_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=acc_roi,
        event=event,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
        smooth_method=smooth_method,
        artifact_abs_thresh=artifact_abs_thresh,
    )
    acc_signal = np.asarray(acc_signal, dtype=np.float32)
    if acc_signal.ndim != 3:
        raise ValueError(f"Expected ACC HFA shape (trials, electrodes, time), got {acc_signal.shape}")
    n_trials_acc, n_acc, n_cue_t = acc_signal.shape
    if n_acc < min_electrodes:
        raise ValueError(
            f"insufficient_acc_electrodes: have {n_acc}, need {min_electrodes}"
        )
    if n_trials_acc < min_trials:
        raise ValueError(
            f"insufficient_acc_trials: have {n_trials_acc}, need {min_trials}"
        )

    trial_index = np.arange(n_trials_acc, dtype=np.int64)
    phase_res = extract_theta_phase_from_cue_epochs(
        subject_obj=subject_obj,
        roi=hpc_roi,
        event=event,
        freq_band=theta_band,
        target_times=cue_times,
        trial_index=trial_index,
    )
    hpc_phase_raw = np.asarray(phase_res['phase'], dtype=np.float64)
    if hpc_phase_raw.ndim != 3:
        raise ValueError(
            f"Expected HPC phase shape (electrodes, time, trials), got {hpc_phase_raw.shape}"
        )
    hpc_phase = np.transpose(hpc_phase_raw, (2, 0, 1))
    n_hpc = hpc_phase.shape[1]
    if n_hpc < min_electrodes:
        raise ValueError(
            f"insufficient_hpc_electrodes: have {n_hpc}, need {min_electrodes}"
        )

    n_t = min(int(acc_signal.shape[2]), int(hpc_phase.shape[2]))
    n_tr = min(int(acc_signal.shape[0]), int(hpc_phase.shape[0]))
    acc_signal = acc_signal[:n_tr, :, :n_t]
    hpc_phase = hpc_phase[:n_tr, :, :n_t]
    cue_times = np.asarray(cue_times, dtype=np.float32)[:n_t]

    obs = compute_acc_hpc_phase_rsa_at_cue(
        acc_hfa=acc_signal,
        hpc_phase=hpc_phase,
        rdm_compare_method=rdm_compare_method,
        acc_method=acc_method,
        hpc_method=hpc_method,
    )
    rho_obs = np.asarray(obs['rho'], dtype=np.float64)

    null_mean = np.zeros_like(rho_obs, dtype=np.float64)
    null_M2 = np.zeros_like(rho_obs, dtype=np.float64)
    extreme_counts = np.zeros_like(rho_obs, dtype=np.int64)
    n_done = 0

    if int(n_permutations) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_permutations)):
            perm_ix = rng.permutation(n_tr)
            hpc_perm = hpc_phase[perm_ix, :, :]
            perm = compute_acc_hpc_phase_rsa_at_cue(
                acc_hfa=acc_signal,
                hpc_phase=hpc_perm,
                rdm_compare_method=rdm_compare_method,
                acc_method=acc_method,
                hpc_method=hpc_method,
            )
            x = np.asarray(perm['rho'], dtype=np.float64)
            n_done += 1
            delta = x - null_mean
            null_mean = null_mean + delta / float(n_done)
            null_M2 = null_M2 + delta * (x - null_mean)
            valid = np.isfinite(x) & np.isfinite(rho_obs)
            extreme_counts[valid] += (np.abs(x[valid]) >= np.abs(rho_obs[valid])).astype(np.int64)

    if n_done > 0:
        null_std = np.sqrt(null_M2 / float(n_done))
        null_std = np.maximum(null_std, 1e-8)
        z_rsa = (rho_obs - null_mean) / null_std
        p_rsa = (extreme_counts.astype(np.float64) + 1.0) / float(n_done + 1)
    else:
        null_std = np.full_like(rho_obs, np.nan)
        z_rsa = np.full_like(rho_obs, np.nan)
        p_rsa = np.full_like(rho_obs, np.nan)

    return {
        'subject': subject_obj.subject,
        'acc_roi': str(acc_roi),
        'hpc_roi': str(hpc_roi),
        'event': str(event),
        'cue_times': cue_times.astype(np.float32),
        'rho_obs': rho_obs.astype(np.float32),
        'null_mean': null_mean.astype(np.float32),
        'null_std': null_std.astype(np.float32),
        'z_rsa': z_rsa.astype(np.float32),
        'p_rsa': p_rsa.astype(np.float32),
        'n_trials': int(n_tr),
        'n_acc_electrodes': int(acc_signal.shape[1]),
        'n_hpc_electrodes': int(hpc_phase.shape[1]),
        'electrode_labels_acc': np.asarray(acc_labels, dtype=str),
        'electrode_labels_hpc': np.asarray(phase_res['electrode_labels'], dtype=str),
        'rdm_compare_method': str(rdm_compare_method),
        'acc_method': str(acc_method),
        'hpc_method': str(hpc_method),
        'n_permutations': int(n_permutations),
        'theta_band': np.asarray(theta_band, dtype=np.float32),
        'freq_range': np.asarray(freq_range, dtype=np.float32),
        'win_ms': int(win_ms),
        'downsample_factor': int(downsample_factor),
        'smooth_method': str(smooth_method),
        'artifact_abs_thresh': (
            float(artifact_abs_thresh) if artifact_abs_thresh is not None else np.nan
        ),
        'status': 'ok',
    }


def cluster_hpc_phase_states(
    hpc_phase_window: np.ndarray,
    n_states_range: Tuple[int, int] = (4, 8),
    random_state: int = 42,
    n_init: int = 10,
) -> Dict[str, Any]:
    """
    K-means cluster trials by HPC multi-electrode theta phase configuration.

    Phases are linearized via per-electrode sin/cos features (a standard trick
    to embed angular data in Euclidean space for K-means). When the input is
    3D, sin and cos are averaged across the time window before clustering, so
    each trial contributes one (2 * n_elec)-vector. The chosen ``k`` is the
    silhouette-maximizing value in ``n_states_range``.

    Parameters
    ----------
    hpc_phase_window : (n_trials, n_elec) or (n_trials, n_elec, n_t_window)
        HPC theta phase in radians.
    n_states_range : (int, int)
        Inclusive range of K to scan; defaults to ``(4, 8)``.

    Returns
    -------
    dict with keys
        status              : 'ok' | 'too_few_trials' | 'no_valid_k'
        labels              : (n_trials,) int64, with -1 for trials that had
                              non-finite features after sin/cos averaging.
        centroids           : (chosen_k, 2 * n_elec) cluster centers in
                              sin/cos space.
        chosen_k            : int — silhouette-maximizing K.
        silhouette          : float — silhouette at chosen_k.
        silhouette_by_k     : dict {k: silhouette}
        inertia_by_k        : dict {k: kmeans inertia}
        features            : (n_trials, 2 * n_elec) sin/cos features (NaN-safe
                              entries preserved as-is; invalid rows masked out
                              of the K-means fit).
        valid_mask          : (n_trials,) bool — rows that participated in fit.
        n_elec              : int
        n_trials            : int
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    phi = np.asarray(hpc_phase_window, dtype=np.float64)
    if phi.ndim == 2:
        sin_feat = np.sin(phi)
        cos_feat = np.cos(phi)
    elif phi.ndim == 3:
        sin_feat = np.nanmean(np.sin(phi), axis=2)
        cos_feat = np.nanmean(np.cos(phi), axis=2)
    else:
        raise ValueError(
            f"hpc_phase_window must be 2D or 3D, got {phi.ndim}D"
        )

    n_trials, n_elec = sin_feat.shape
    features = np.concatenate([sin_feat, cos_feat], axis=1)
    valid_mask = np.all(np.isfinite(features), axis=1)
    n_valid = int(valid_mask.sum())

    k_min, k_max = int(n_states_range[0]), int(n_states_range[1])
    if k_max < k_min:
        raise ValueError(f"n_states_range[1] >= n_states_range[0] required, got {n_states_range}")
    if n_valid < k_min + 1:
        return {
            'status': 'too_few_trials',
            'labels': np.full(n_trials, -1, dtype=np.int64),
            'centroids': np.empty((0, features.shape[1]), dtype=np.float64),
            'chosen_k': 0,
            'silhouette': float('nan'),
            'silhouette_by_k': {},
            'inertia_by_k': {},
            'features': features,
            'valid_mask': valid_mask,
            'n_elec': int(n_elec),
            'n_trials': int(n_trials),
        }

    feats_valid = features[valid_mask]

    sil_by_k: Dict[int, float] = {}
    inertia_by_k: Dict[int, float] = {}
    labels_by_k: Dict[int, np.ndarray] = {}
    centers_by_k: Dict[int, np.ndarray] = {}
    best_k, best_sil = -1, -np.inf
    for k in range(k_min, k_max + 1):
        if n_valid < k + 1:
            continue
        km = KMeans(n_clusters=k, random_state=int(random_state), n_init=int(n_init))
        labels_k = km.fit_predict(feats_valid)
        try:
            sil = float(silhouette_score(feats_valid, labels_k)) if k > 1 else float('nan')
        except ValueError:
            sil = float('nan')
        sil_by_k[k] = sil
        inertia_by_k[k] = float(km.inertia_)
        labels_by_k[k] = labels_k
        centers_by_k[k] = np.asarray(km.cluster_centers_, dtype=np.float64)
        if np.isfinite(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k < 0:
        return {
            'status': 'no_valid_k',
            'labels': np.full(n_trials, -1, dtype=np.int64),
            'centroids': np.empty((0, features.shape[1]), dtype=np.float64),
            'chosen_k': 0,
            'silhouette': float('nan'),
            'silhouette_by_k': sil_by_k,
            'inertia_by_k': inertia_by_k,
            'features': features,
            'valid_mask': valid_mask,
            'n_elec': int(n_elec),
            'n_trials': int(n_trials),
        }

    labels_full = np.full(n_trials, -1, dtype=np.int64)
    labels_full[valid_mask] = labels_by_k[best_k].astype(np.int64, copy=False)

    return {
        'status': 'ok',
        'labels': labels_full,
        'centroids': centers_by_k[best_k],
        'chosen_k': int(best_k),
        'silhouette': float(best_sil),
        'silhouette_by_k': sil_by_k,
        'inertia_by_k': inertia_by_k,
        'features': features,
        'valid_mask': valid_mask,
        'n_elec': int(n_elec),
        'n_trials': int(n_trials),
    }


def compute_reactivation_by_phase_state(
    reactivation: np.ndarray,
    state_labels: np.ndarray,
    rpe_cat: Optional[np.ndarray] = None,
    focus_rpe: str = 'pos',
) -> Dict[str, Any]:
    """
    Mean ACC reactivation timecourse per HPC phase state + across-state test.

    Parameters
    ----------
    reactivation : (n_trials, n_t)
        Trial-by-time scalar reactivation (e.g., cue HFA · w_z from the
        PositiveRPE-style pipeline).
    state_labels : (n_trials,)
        Integer labels in {0, ..., K-1}; ``-1`` marks invalid trials and is
        excluded from all summaries and tests.
    rpe_cat : (n_trials,) optional
        Categorical trial labels (e.g., ``'pos'`` / ``'neg'`` / ``'neutral'``).
        When given, the analysis is repeated within trials matching
        ``focus_rpe``.
    focus_rpe : str
        Label in ``rpe_cat`` used for the focal sub-analysis (default ``'pos'``).

    Returns
    -------
    dict with keys
        states              : list[int] of state ids in sorted order
        per_state_mean      : dict {state: (n_t,)} mean reactivation timecourse
        per_state_sem       : dict {state: (n_t,)} SEM across trials in state
        per_state_n         : dict {state: int} trial count
        kruskal_h_per_t     : (n_t,) Kruskal-Wallis H across states per time
        kruskal_p_per_t     : (n_t,) p-value per time
        (if rpe_cat provided)
        focus_rpe           : str
        focus_per_state_mean: dict {state: (n_t,)}
        focus_per_state_n   : dict {state: int}
        focus_kruskal_h_per_t / focus_kruskal_p_per_t : (n_t,) within-focus tests
    """
    from scipy.stats import kruskal

    R = np.asarray(reactivation, dtype=np.float64)
    L = np.asarray(state_labels, dtype=np.int64)
    if R.ndim != 2:
        raise ValueError("reactivation must be 2D (n_trials, n_t)")
    if L.shape[0] != R.shape[0]:
        raise ValueError(
            f"trial count mismatch: reactivation={R.shape[0]}, state_labels={L.shape[0]}"
        )

    states = sorted(int(s) for s in np.unique(L) if s >= 0)
    n_t = R.shape[1]

    per_state_mean: Dict[int, np.ndarray] = {}
    per_state_sem: Dict[int, np.ndarray] = {}
    per_state_n: Dict[int, int] = {}
    for s in states:
        mask = L == s
        per_state_n[s] = int(mask.sum())
        if mask.sum() > 0:
            per_state_mean[s] = np.nanmean(R[mask], axis=0).astype(np.float32)
            denom = max(int(mask.sum()), 1)
            per_state_sem[s] = (np.nanstd(R[mask], axis=0) / np.sqrt(denom)).astype(np.float32)
        else:
            per_state_mean[s] = np.full(n_t, np.nan, dtype=np.float32)
            per_state_sem[s] = np.full(n_t, np.nan, dtype=np.float32)

    h_per_t = np.full(n_t, np.nan, dtype=np.float64)
    p_per_t = np.full(n_t, np.nan, dtype=np.float64)
    for t_idx in range(n_t):
        samples: List[np.ndarray] = []
        for s in states:
            vals = R[L == s, t_idx]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                samples.append(vals)
        if len(samples) >= 2:
            try:
                res = kruskal(*samples)
                h_per_t[t_idx] = float(res.statistic)
                p_per_t[t_idx] = float(res.pvalue)
            except ValueError:
                pass

    out: Dict[str, Any] = {
        'states': states,
        'per_state_mean': per_state_mean,
        'per_state_sem': per_state_sem,
        'per_state_n': per_state_n,
        'kruskal_h_per_t': h_per_t.astype(np.float32),
        'kruskal_p_per_t': p_per_t.astype(np.float32),
    }

    if rpe_cat is not None:
        rpe_arr = np.asarray(rpe_cat)
        focus_mask = rpe_arr == focus_rpe
        focus_per_state_mean: Dict[int, np.ndarray] = {}
        focus_per_state_n: Dict[int, int] = {}
        for s in states:
            mask = (L == s) & focus_mask
            focus_per_state_n[s] = int(mask.sum())
            if mask.sum() > 0:
                focus_per_state_mean[s] = np.nanmean(R[mask], axis=0).astype(np.float32)
            else:
                focus_per_state_mean[s] = np.full(n_t, np.nan, dtype=np.float32)

        h_focus = np.full(n_t, np.nan, dtype=np.float64)
        p_focus = np.full(n_t, np.nan, dtype=np.float64)
        for t_idx in range(n_t):
            samples = []
            for s in states:
                vals = R[(L == s) & focus_mask, t_idx]
                vals = vals[np.isfinite(vals)]
                if vals.size >= 2:
                    samples.append(vals)
            if len(samples) >= 2:
                try:
                    res = kruskal(*samples)
                    h_focus[t_idx] = float(res.statistic)
                    p_focus[t_idx] = float(res.pvalue)
                except ValueError:
                    pass

        out.update({
            'focus_rpe': str(focus_rpe),
            'focus_per_state_mean': focus_per_state_mean,
            'focus_per_state_n': focus_per_state_n,
            'focus_kruskal_h_per_t': h_focus.astype(np.float32),
            'focus_kruskal_p_per_t': p_focus.astype(np.float32),
        })

    return out


def _padded_row_zscore_cosine_similarity(E: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Cross-trial similarity S[i, j] between encoding trial i and cue trial j.

    E: (n_trials, n_elec_enc), R: (n_trials, n_elec_cue). When electrode counts
    differ, both rows are zero-padded to max(n_elec_enc, n_elec_cue), then each
    row is z-scored across electrodes and L2-normalized; S = E_n @ R_n.T (cosine
    between the two multivariate patterns in a common padded space).
    """
    E = np.asarray(E, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n, p = E.shape
    _, q = R.shape
    d = max(int(p), int(q))
    if p < d:
        E = np.pad(E, ((0, 0), (0, d - p)), mode='constant')
    if q < d:
        R = np.pad(R, ((0, 0), (0, d - q)), mode='constant')
    E_z = E - E.mean(axis=1, keepdims=True)
    E_z = E_z / (np.std(E_z, axis=1, keepdims=True) + 1e-12)
    R_z = R - R.mean(axis=1, keepdims=True)
    R_z = R_z / (np.std(R_z, axis=1, keepdims=True) + 1e-12)
    E_n = E_z / (np.linalg.norm(E_z, axis=1, keepdims=True) + 1e-12)
    R_n = R_z / (np.linalg.norm(R_z, axis=1, keepdims=True) + 1e-12)
    return E_n @ R_n.T


def compute_trial_reinstatement_index(
    encoding_hfa: np.ndarray,
    cue_hfa: np.ndarray,
    cluster_mask: np.ndarray,
    group_z_for_peak: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Item-specific reinstatement index per trial, averaged over cluster timepoints.

    At each (t_fb, t_cue) in ``cluster_mask``, compute cross-trial similarity
    matrix S with :func:`_padded_row_zscore_cosine_similarity`, then per trial i:
      match_sim[i] = S[i, i]
      mismatch_sim[i] = mean_{j != i} S[i, j]
      RI[i] = match_sim[i] - mismatch_sim[i]
      match_rank_norm[i] = (n - rank_i) / (n - 1) where rank_i is the rank of
        S[i, i] among row S[i, :] with rank 1 = largest (best match).

    Returns cluster-mean aggregates plus the same quantities at a single peak
    voxel (within-mask argmax of |group_z_for_peak| if provided, else first mask voxel).
    """
    enc = np.asarray(encoding_hfa, dtype=np.float64)
    cue = np.asarray(cue_hfa, dtype=np.float64)
    mask = np.asarray(cluster_mask, dtype=bool)
    if enc.ndim != 3 or cue.ndim != 3:
        raise ValueError('encoding_hfa and cue_hfa must be 3D')
    if enc.shape[0] != cue.shape[0]:
        raise ValueError('trial count mismatch')
    if mask.shape != (enc.shape[2], cue.shape[2]):
        raise ValueError('cluster_mask shape must match (n_feedback_times, n_cue_times)')

    n_trials = enc.shape[0]
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        nanv = np.full(n_trials, np.nan, dtype=np.float64)
        return {
            'match_sim': nanv.copy(),
            'mismatch_sim': nanv.copy(),
            'ri': nanv.copy(),
            'match_rank_norm': nanv.copy(),
            'match_sim_peak': nanv.copy(),
            'mismatch_sim_peak': nanv.copy(),
            'ri_peak': nanv.copy(),
            'match_rank_norm_peak': nanv.copy(),
            'peak_fb_idx': -1,
            'peak_cue_idx': -1,
            'n_cluster_voxels': 0,
        }

    n_vox = idxs.shape[0]
    match_acc = np.zeros(n_trials, dtype=np.float64)
    mismatch_acc = np.zeros(n_trials, dtype=np.float64)
    rank_acc = np.zeros(n_trials, dtype=np.float64)

    for t_fb, t_cue in idxs:
        E = enc[:, :, int(t_fb)]
        R = cue[:, :, int(t_cue)]
        S = _padded_row_zscore_cosine_similarity(E, R)
        diag = np.diag(S).astype(np.float64, copy=False)
        if n_trials > 1:
            off = (S.sum(axis=1) - diag) / float(n_trials - 1)
        else:
            off = np.zeros_like(diag)
        for i in range(n_trials):
            rd = rankdata(-S[i, :], method='min')
            r_i = float(rd[i])
            rank_acc[i] += (n_trials - r_i) / float(n_trials - 1) if n_trials > 1 else 1.0
        match_acc += diag
        mismatch_acc += off

    match_sim = match_acc / float(n_vox)
    mismatch_sim = mismatch_acc / float(n_vox)
    ri = match_sim - mismatch_sim
    match_rank_norm = rank_acc / float(n_vox)

    if group_z_for_peak is not None:
        gz = np.asarray(group_z_for_peak, dtype=np.float64)
        if gz.shape != mask.shape:
            raise ValueError('group_z_for_peak shape must match cluster_mask')
        abs_z = np.abs(np.nan_to_num(gz, nan=0.0))
        masked_scores = np.where(mask, abs_z, -np.inf)
        peak_fb_idx, peak_cue_idx = (int(x) for x in np.unravel_index(
            int(np.argmax(masked_scores)), masked_scores.shape
        ))
    else:
        peak_fb_idx, peak_cue_idx = int(idxs[0][0]), int(idxs[0][1])

    E_p = enc[:, :, peak_fb_idx]
    R_p = cue[:, :, peak_cue_idx]
    S_p = _padded_row_zscore_cosine_similarity(E_p, R_p)
    diag_p = np.diag(S_p)
    if n_trials > 1:
        off_p = (S_p.sum(axis=1) - diag_p) / float(n_trials - 1)
    else:
        off_p = np.zeros_like(diag_p)
    match_rank_peak = np.zeros(n_trials, dtype=np.float64)
    for i in range(n_trials):
        rd = rankdata(-S_p[i, :], method='min')
        r_i = float(rd[i])
        match_rank_peak[i] = (n_trials - r_i) / float(n_trials - 1) if n_trials > 1 else 1.0

    return {
        'match_sim': match_sim.astype(np.float64),
        'mismatch_sim': mismatch_sim.astype(np.float64),
        'ri': ri.astype(np.float64),
        'match_rank_norm': match_rank_norm.astype(np.float64),
        'match_sim_peak': diag_p.astype(np.float64),
        'mismatch_sim_peak': off_p.astype(np.float64),
        'ri_peak': (diag_p - off_p).astype(np.float64),
        'match_rank_norm_peak': match_rank_peak.astype(np.float64),
        'peak_fb_idx': peak_fb_idx,
        'peak_cue_idx': peak_cue_idx,
        'n_cluster_voxels': int(n_vox),
    }


def _ri_match_mismatch_means(
    E: np.ndarray,
    R: np.ndarray,
) -> Tuple[float, float, float]:
    """
    E, R: (n_trials, n_elec) aligned rows. Returns (mean_diag, mean_offdiag, ri).
    """
    E = np.asarray(E, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n = E.shape[0]
    if n < 3:
        return float('nan'), float('nan'), float('nan')
    S = _padded_row_zscore_cosine_similarity(E, R)
    diag = np.diag(S).astype(np.float64, copy=False)
    if n > 1:
        off = (S.sum(axis=1) - diag) / float(n - 1)
    else:
        off = np.zeros_like(diag)
    return float(np.mean(diag)), float(np.mean(off)), float(np.mean(diag - off))


def compute_cross_temporal_reinstatement_maps(
    encoding_hfa: np.ndarray,
    cue_hfa: np.ndarray,
    memory_outcomes: np.ndarray,
    remembered_label: str = 'remembered',
    forgotten_label: str = 'forgotten',
) -> Dict[str, Any]:
    """
    Cross-temporal item-specific reinstatement (RI) heatmaps, analogous layout to RSA.

    For each memory condition separately, at each (t_feedback, t_cue), build the
    cross-trial similarity matrix S between aligned encoding and cue patterns
    (same trials, same item), then:

      match = mean_i S[i, i]
      mismatch = mean_i mean_{j != i} S[i, j]
      RI = match - mismatch

    Returns remembered/forgotten RI maps and their difference, plus separate
    match/mismatch maps for plotting.
    """
    enc = np.asarray(encoding_hfa, dtype=np.float64)
    cue = np.asarray(cue_hfa, dtype=np.float64)
    mem = np.asarray(memory_outcomes)
    if enc.ndim != 3 or cue.ndim != 3:
        raise ValueError('encoding_hfa and cue_hfa must be 3D')
    if enc.shape[0] != cue.shape[0]:
        raise ValueError('trial count mismatch')
    if mem.shape[0] != enc.shape[0]:
        raise ValueError('memory_outcomes length mismatch')

    remembered_mask = mem == remembered_label
    forgotten_mask = mem == forgotten_label
    n_fb = enc.shape[2]
    n_cue = cue.shape[2]

    def _maps_for_condition(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_h = np.full((n_fb, n_cue), np.nan, dtype=np.float64)
        mismatch_h = np.full((n_fb, n_cue), np.nan, dtype=np.float64)
        ri_h = np.full((n_fb, n_cue), np.nan, dtype=np.float64)
        if np.sum(mask) < 3:
            return match_h, mismatch_h, ri_h
        for t_fb in range(n_fb):
            for t_cue in range(n_cue):
                E = enc[mask, :, t_fb]
                R = cue[mask, :, t_cue]
                m, mm, ri = _ri_match_mismatch_means(E, R)
                match_h[t_fb, t_cue] = m
                mismatch_h[t_fb, t_cue] = mm
                ri_h[t_fb, t_cue] = ri
        return match_h, mismatch_h, ri_h

    rem_m, rem_mm, rem_ri = _maps_for_condition(remembered_mask)
    forg_m, forg_mm, forg_ri = _maps_for_condition(forgotten_mask)
    diff_ri = rem_ri - forg_ri

    return {
        'remembered_heatmap': rem_ri.astype(np.float32),
        'forgotten_heatmap': forg_ri.astype(np.float32),
        'diff_heatmap': diff_ri.astype(np.float32),
        'remembered_match_heatmap': rem_m.astype(np.float32),
        'remembered_mismatch_heatmap': rem_mm.astype(np.float32),
        'forgotten_match_heatmap': forg_m.astype(np.float32),
        'forgotten_mismatch_heatmap': forg_mm.astype(np.float32),
        'n_remembered': int(np.sum(remembered_mask)),
        'n_forgotten': int(np.sum(forgotten_mask)),
    }


def load_z_diff_stack_from_pair_dir(
    pair_dir: str,
    subjects: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Stack per-subject z_diff_heatmap from ``<pair_dir>/<subject>_rsa.npz``.

    Non-finite entries are replaced with 0.0. Subjects missing a file are skipped.

    Returns
    -------
    stack : (n_loaded, n_fb, n_cue)
    loaded_subjects : list of subject IDs in row order
    """
    rows: List[np.ndarray] = []
    loaded: List[str] = []
    for subj in subjects:
        path = os.path.join(pair_dir, f'{subj}_rsa.npz')
        if not os.path.isfile(path):
            continue
        with np.load(path, allow_pickle=True) as npz:
            if 'z_diff_heatmap' not in npz.files:
                continue
            z = np.asarray(npz['z_diff_heatmap'], dtype=np.float64)
        z = np.where(np.isfinite(z), z, 0.0)
        rows.append(z)
        loaded.append(subj)
    if len(rows) == 0:
        return np.empty((0, 0, 0), dtype=np.float64), []
    return np.stack(rows, axis=0).astype(np.float64), loaded


def loso_cluster_summary(
    z_diff_stack: np.ndarray,
    subjects: List[str],
    fixed_cluster_mask: np.ndarray,
    n_permutations: int = 1000,
    tail: int = 1,
    threshold: Union[float, None] = 0,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Leave-one-subject-out cluster-test summaries on stacked z_diff maps.

    For each left-out subject, re-runs :func:`permutation_cluster_1samp_test` on
    the remaining subjects and records:
      - min_cluster_p, n_sig_clusters (p <= 0.05)
      - peak_t (max of t_obs map)
      - fixed_mask_t_mass: sum of t_obs over ``fixed_cluster_mask`` (full-cohort mask)
      - fixed_mask_mean_t: mean t_obs inside mask
    """
    D = np.asarray(z_diff_stack, dtype=np.float64)
    if D.ndim != 3:
        raise ValueError('z_diff_stack must be 3D')
    n_subj = D.shape[0]
    if n_subj != len(subjects):
        raise ValueError('len(subjects) must match z_diff_stack.shape[0]')
    mask = np.asarray(fixed_cluster_mask, dtype=bool)
    if mask.shape != D.shape[1:]:
        raise ValueError('fixed_cluster_mask must match spatial dims of z_diff_stack')

    # Full-cohort reference t-map and fixed-mask mass
    full_res = run_group_crossregion_rsa_cluster_test(
        D, n_permutations=n_permutations, tail=tail,
        threshold=threshold, seed=seed, verbose=verbose,
    )
    t_full = np.asarray(full_res['t_obs'], dtype=np.float64)
    full_mass = float(np.nansum(t_full[mask]))
    full_mean = float(np.nanmean(t_full[mask])) if np.any(mask) else np.nan

    records: List[Dict[str, Any]] = []
    for k in range(n_subj):
        left_out = subjects[k]
        D_loo = np.delete(D, k, axis=0)
        if D_loo.shape[0] < 2:
            records.append({
                'left_out_subject': left_out,
                'min_cluster_p': np.nan,
                'n_sig_clusters': 0,
                'peak_t': np.nan,
                'fixed_mask_t_mass': np.nan,
                'fixed_mask_mean_t': np.nan,
                'full_cohort_fixed_mask_t_mass': full_mass,
                'full_cohort_fixed_mask_mean_t': full_mean,
            })
            continue
        res = run_group_crossregion_rsa_cluster_test(
            D_loo, n_permutations=n_permutations, tail=tail,
            threshold=threshold, seed=seed, verbose=False,
        )
        t_obs = np.asarray(res['t_obs'], dtype=np.float64)
        cp = np.asarray(res['cluster_pvals'], dtype=float)
        peak_t = float(np.nanmax(t_obs))
        min_p = float(np.min(cp)) if cp.size > 0 else np.nan
        n_sig = int(np.sum(cp <= 0.05)) if cp.size > 0 else 0
        lo_mass = float(np.nansum(t_obs[mask]))
        lo_mean = float(np.nanmean(t_obs[mask])) if np.any(mask) else np.nan
        records.append({
            'left_out_subject': left_out,
            'min_cluster_p': min_p,
            'n_sig_clusters': n_sig,
            'peak_t': peak_t,
            'fixed_mask_t_mass': lo_mass,
            'fixed_mask_mean_t': lo_mean,
            'full_cohort_fixed_mask_t_mass': full_mass,
            'full_cohort_fixed_mask_mean_t': full_mean,
        })
    return pd.DataFrame(records)


def wilcoxon_paired_ri_by_subject(
    reinstatement_df: pd.DataFrame,
    ri_col: str = 'ri',
    subject_col: str = 'participant',
    hit_col: str = 'hits',
) -> Dict[str, Any]:
    """
    Per-subject mean RI for hits vs misses; paired Wilcoxon across subjects.
    """
    subj_means_hit: List[float] = []
    subj_means_miss: List[float] = []
    subj_ids: List[str] = []
    for subj, g in reinstatement_df.groupby(subject_col):
        gh = g[g[hit_col] == 1][ri_col].dropna()
        gm = g[g[hit_col] == 0][ri_col].dropna()
        if len(gh) == 0 or len(gm) == 0:
            continue
        subj_means_hit.append(float(np.mean(gh)))
        subj_means_miss.append(float(np.mean(gm)))
        subj_ids.append(str(subj))
    if len(subj_ids) < 3:
        return {'n_subjects': len(subj_ids), 'statistic': np.nan, 'pvalue': np.nan, 'subj_ids': subj_ids}
    stat, p = wilcoxon(subj_means_hit, subj_means_miss, alternative='two-sided')
    return {
        'n_subjects': len(subj_ids),
        'statistic': float(stat),
        'pvalue': float(p),
        'subj_ids': subj_ids,
        'mean_ri_hit': subj_means_hit,
        'mean_ri_miss': subj_means_miss,
    }


def run_group_crossregion_rsa_pipeline(
    subjects: List[str],
    roi_pairs: List[Tuple[str, str]],
    base_dir: str,
    load_dir: str,
    trial_save_dir: str,
    group_save_dir: str,
    event_encoding: str = 'feedback_start',
    event_cue: str = 'cue_on',
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
    n_permutations: int = 200,
    shuffle_type: str = 'trial_labels',
    seed: int = 42,
    rdm_method: str = 'euclidean',
    rdm_compare_method: str = 'rho-a',
    run_group_cluster: bool = True,
    group_cluster_n_permutations: int = 5000,
    group_cluster_tail: int = 1,
    group_cluster_threshold: Union[float, None] = 0,
    n_jobs_outer: int = 1,
    overwrite_trial_level: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Subject-first batch cross-region RSA pipeline writing NPZ + CSV manifests.
    """
    os.makedirs(trial_save_dir, exist_ok=True)
    os.makedirs(group_save_dir, exist_ok=True)

    def _subject_worker(subject: str, encoding_roi: str, cue_roi: str) -> Dict[str, Any]:
        paths = build_crossregion_rsa_paths(
            trial_save_dir=trial_save_dir,
            group_save_dir=group_save_dir,
            encoding_roi=encoding_roi,
            cue_roi=cue_roi,
            subject=subject,
        )
        os.makedirs(paths['pair_dir'], exist_ok=True)
        os.makedirs(paths['checkpoint_dir'], exist_ok=True)
        out = {
            'subject': subject,
            'encoding_roi': encoding_roi,
            'cue_roi': cue_roi,
            'status': 'failed',
            'npz_path': paths['subject_npz'],
        }
        if (not overwrite_trial_level) and os.path.exists(paths['subject_npz']):
            try:
                npz = np.load(paths['subject_npz'], allow_pickle=True)
                status_val = np.asarray(npz['status']).item() if 'status' in npz.files else 'loaded_existing'
                out.update({
                    'status': str(status_val),
                    'n_trials': int(npz['n_trials']),
                    'n_remembered': int(npz['n_remembered']),
                    'n_forgotten': int(npz['n_forgotten']),
                    'min_diff': float(np.nanmin(npz['diff_heatmap'])),
                    'max_diff': float(np.nanmax(npz['diff_heatmap'])),
                    'mean_diff': float(np.nanmean(npz['diff_heatmap'])),
                })
                return out
            except Exception:
                pass

        try:
            subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
            res = run_subject_crossregion_rsa_permutation(
                subject_obj=subject_obj,
                encoding_roi=encoding_roi,
                cue_roi=cue_roi,
                event_encoding=event_encoding,
                event_cue=event_cue,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
                min_trials=min_trials,
                n_permutations=n_permutations,
                shuffle_type=shuffle_type,
                seed=seed,
                rdm_method=rdm_method,
                rdm_compare_method=rdm_compare_method,
            )
            np.savez_compressed(
                paths['subject_npz'],
                remembered_heatmap=res['remembered_heatmap'],
                forgotten_heatmap=res['forgotten_heatmap'],
                diff_heatmap=res['diff_heatmap'],
                z_diff_heatmap=res['z_diff_heatmap'],
                p_diff_heatmap=res['p_diff_heatmap'],
                feedback_times=res['feedback_times'],
                cue_times=res['cue_times'],
                memory_outcomes=res['memory_outcomes'],
                trial_index=res['trial_index'],
                retrieval_trial_index=res['retrieval_trial_index'],
                subject=res['subject'],
                encoding_roi=res['encoding_roi'],
                cue_roi=res['cue_roi'],
                n_trials=np.int32(res['n_trials']),
                n_remembered=np.int32(res['n_remembered']),
                n_forgotten=np.int32(res['n_forgotten']),
                rdm_method=np.asarray(res['rdm_method']),
                rdm_compare_method=np.asarray(res['rdm_compare_method']),
                shuffle_type=np.asarray(res['shuffle_type']),
                n_permutations=np.int32(res['n_permutations']),
                n_dropped_alignment=np.int32(res.get('n_dropped_alignment', 0)),
                alignment_mode=np.asarray(res.get('alignment_mode', 'legacy_combined_df_trial_map')),
                status=np.asarray(res['status']),
            )
            out.update({
                'status': res['status'],
                'n_trials': int(res['n_trials']),
                'n_remembered': int(res['n_remembered']),
                'n_forgotten': int(res['n_forgotten']),
                'min_diff': float(np.nanmin(res['diff_heatmap'])),
                'max_diff': float(np.nanmax(res['diff_heatmap'])),
                'mean_diff': float(np.nanmean(res['diff_heatmap'])),
            })
            return out
        except Exception as e:
            out.update({'status': f'failed:{type(e).__name__}'})
            return out

    all_group_rows: List[Dict[str, Any]] = []
    all_subject_rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    for encoding_roi, cue_roi in roi_pairs:
        pair_paths = build_crossregion_rsa_paths(
            trial_save_dir=trial_save_dir,
            group_save_dir=group_save_dir,
            encoding_roi=encoding_roi,
            cue_roi=cue_roi,
            subject=None,
        )
        pair_name = pair_paths['pair_name']
        if verbose:
            print(f"\n=== Cross-region RSA: {pair_name} ===")

        if n_jobs_outer == 1:
            subject_rows = [_subject_worker(s, encoding_roi, cue_roi) for s in subjects]
        else:
            subject_rows = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
                delayed(_subject_worker)(s, encoding_roi, cue_roi) for s in subjects
            )
        subject_df = pd.DataFrame(subject_rows)
        subject_df.to_csv(pair_paths['subject_csv'], index=False)
        all_subject_rows.extend(subject_rows)

        included = []
        failed = []
        diff_list = []
        fb_times = None
        cue_times = None
        for row in subject_rows:
            if str(row.get('status', '')).startswith('failed'):
                failed.append(row['subject'])
                continue
            try:
                npz = np.load(row['npz_path'], allow_pickle=True)
                d = np.asarray(npz['diff_heatmap'], dtype=np.float32)
                if not np.any(np.isfinite(d)):
                    failed.append(row['subject'])
                    continue
                diff_list.append(d)
                if fb_times is None:
                    fb_times = np.asarray(npz['feedback_times'])
                    cue_times = np.asarray(npz['cue_times'])
                included.append(row['subject'])
            except Exception:
                failed.append(row['subject'])

        cluster_res = {
            't_obs': None,
            'clusters': [],
            'cluster_pvals': np.array([]),
            'sig_mask': None,
        }
        observed_stack = np.empty((0, 0, 0), dtype=np.float32)
        group_mean_diff = np.empty((0, 0), dtype=np.float32)
        group_mean_z = np.empty((0, 0), dtype=np.float32)
        if len(diff_list) > 0:
            observed_stack = np.stack(diff_list, axis=0).astype(np.float32)
            group_mean_diff = np.nanmean(observed_stack, axis=0).astype(np.float32)
            z_list = []
            for row in subject_rows:
                if row['subject'] not in included:
                    continue
                npz = np.load(row['npz_path'], allow_pickle=True)
                z_list.append(np.asarray(npz['z_diff_heatmap'], dtype=np.float32))
            if len(z_list) > 0:
                group_mean_z = np.nanmean(np.stack(z_list, axis=0), axis=0).astype(np.float32)
            if run_group_cluster:
                cluster_res = run_group_crossregion_rsa_cluster_test(
                    observed_diffs_stack=observed_stack,
                    n_permutations=group_cluster_n_permutations,
                    tail=group_cluster_tail,
                    threshold=group_cluster_threshold,
                    seed=seed,
                    verbose=verbose,
                )

        np.savez_compressed(
            pair_paths['group_npz'],
            observed_diffs_stack=observed_stack,
            group_mean_diff=group_mean_diff,
            group_mean_z=group_mean_z,
            feedback_times=np.asarray(fb_times if fb_times is not None else np.array([])),
            cue_times=np.asarray(cue_times if cue_times is not None else np.array([])),
            subjects_included=np.asarray(included, dtype=str),
            subjects_failed=np.asarray(failed, dtype=str),
            cluster_t_obs=np.asarray(cluster_res['t_obs']) if cluster_res['t_obs'] is not None else np.array([]),
            cluster_masks=np.asarray(cluster_res['clusters'], dtype=object),
            cluster_pvals=np.asarray(cluster_res['cluster_pvals']),
            cluster_sig_mask=np.asarray(cluster_res['sig_mask']) if cluster_res['sig_mask'] is not None else np.array([]),
        )

        sig_p = np.asarray(cluster_res['cluster_pvals'])
        sig_p = sig_p[sig_p <= 0.05] if sig_p.size > 0 else np.array([])
        g_row = {
            'encoding_roi': encoding_roi,
            'cue_roi': cue_roi,
            'n_subjects': int(len(included)),
            'n_failed': int(len(failed)),
            'n_permutations': int(n_permutations),
            'n_clusters': int(len(cluster_res['clusters'])),
            'n_sig_clusters': int(sig_p.size),
            'min_cluster_p': float(np.min(sig_p)) if sig_p.size > 0 else np.nan,
            'sig_cluster_pvals_json': json.dumps(sig_p.astype(float).tolist()),
            'group_npz_path': pair_paths['group_npz'],
        }
        all_group_rows.append(g_row)
        results[pair_name] = {
            'subject_summary': subject_df,
            'group_summary': g_row,
            'group_npz_path': pair_paths['group_npz'],
        }

    group_df = pd.DataFrame(all_group_rows)
    if len(group_df) > 0:
        group_df.to_csv(
            os.path.join(group_save_dir, "group_summary_crossregion_rsa.csv"),
            index=False,
        )
    all_subject_df = pd.DataFrame(all_subject_rows)
    return {
        'pairs': results,
        'group_summary': group_df,
        'subject_summary_all': all_subject_df,
    }


def process_subject_smoothed_regression(
    subject: str,
    event: str,
    predictor: str,
    formula: str,
    base_dir: str,
    load_dir: str,
    rois: List[str],
    freq_range: Tuple[float, float] = (70., 201.),
    win_ms: int = 200,
    downsample_factor: int = 5,
    n_permutations: int = 100,
    covariates: Optional[List[str]] = None,
    min_trials: int = 10,
    seed: int = 42,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Subject-class-backed version of smoothed single-electrode timewise regression.
    """
    subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
    subject_rows = []

    for roi in rois:
        try:
            signal, times_sec, labels = _get_smoothed_event_signal(
                subject_obj=subject_obj,
                roi=roi,
                event=event,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms
            )
            regressors, _ = _build_subject_regressors(
                subject_obj=subject_obj,
                event=event,
                predictor=predictor,
                covariates=covariates,
                regressor_source=regressor_source,
                trial_var_override=trial_var_override,
            )
            cache_key = (event, roi, tuple(freq_range), int(downsample_factor), int(win_ms))
            neural_meta = subject_obj._neural_cache.get(cache_key, {})
            hemi_raw = neural_meta.get('electrode_hemi', None)
            region_raw = neural_meta.get('electrode_region', None)
            hemi_arr = np.asarray(hemi_raw, dtype=object) if hemi_raw is not None else np.array([], dtype=object)
            region_arr = np.asarray(region_raw, dtype=object) if region_raw is not None else np.array([], dtype=object)
            if len(hemi_arr) == 0 or len(region_arr) == 0:
                elec_sub = subject_obj.elec_df[
                    subject_obj.elec_df.salman_region == roi
                ][['label', 'salman_region', 'hemisphere']].copy()
                elec_sub['label'] = elec_sub['label'].astype(str)
                elec_sub = elec_sub.set_index('label')
                region_fill = []
                hemi_fill = []
                for ch in labels:
                    ch_key = str(ch)
                    if ch_key in elec_sub.index:
                        row = elec_sub.loc[ch_key]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        region_fill.append(str(row.get('salman_region', roi)))
                        hemi_val = row.get('hemisphere', np.nan)
                        hemi_fill.append(str(hemi_val).lower() if pd.notna(hemi_val) else np.nan)
                    else:
                        region_fill.append(roi)
                        hemi_fill.append(np.nan)
                region_arr = np.asarray(region_fill, dtype=object)
                hemi_arr = np.asarray(hemi_fill, dtype=object)
        except Exception:
            continue

        for e_idx, label in enumerate(labels):
            for t_idx, ts_sec in enumerate(times_sec):
                ch_df = regressors.copy()
                trial_ix = ch_df['trial'].values - 1
                valid = (trial_ix >= 0) & (trial_ix < signal.shape[0])
                if not np.any(valid):
                    continue
                ch_df = ch_df.loc[valid].copy()
                trial_ix = trial_ix[valid]
                ch_df['tfr'] = signal[trial_ix, e_idx, t_idx]
                results = permutation_regression_zscore_local(
                    ch_df, formula=formula, n_permutations=n_permutations, seed=seed
                )
                if len(results) == 0:
                    continue
                results['ts'] = float(ts_sec * 1000.0)
                results['subj'] = subject
                results['unique_label'] = label
                results['region'] = (
                    region_arr[e_idx] if e_idx < len(region_arr) else roi
                )
                results['hemi'] = (
                    hemi_arr[e_idx] if e_idx < len(hemi_arr) else np.nan
                )
                subject_rows.append(results)

    if len(subject_rows) == 0:
        return pd.DataFrame()
    return pd.concat(subject_rows, ignore_index=True)


def run_smoothed_time_resolved_regression_pipeline(
    subjects: List[str],
    event: str,
    predictor: str,
    base_dir: str,
    load_dir: str,
    rois: List[str],
    formula: Optional[str] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    win_ms: int = 200,
    downsample_factor: int = 5,
    n_permutations: int = 100,
    covariates: Optional[List[str]] = None,
    min_trials: int = 10,
    seed: int = 42,
    n_jobs_outer: int = 1,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified Subject-backed workflow for single-electrode smoothed timewise regression.

    Produces a dataframe compatible with legacy notebook outputs (`predictor`, `z_beta`,
    `raw_p`, `ts`, `subj`, `unique_label`, `region`, `hemi`).
    """
    covariates = covariates or ['zpm']
    if formula is None:
        rhs = " + ".join([predictor] + covariates)
        formula = f"tfr ~ 1 + {rhs}"

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    completed_subjects = []
    if checkpoint_dir is not None and not overwrite_checkpoints:
        for subject in subjects:
            path = os.path.join(checkpoint_dir, f"{subject}_results.csv")
            if os.path.exists(path):
                completed_subjects.append(subject)

    subjects_to_run = [s for s in subjects if s not in completed_subjects]
    if verbose:
        print(
            f"Running smoothed regression for {len(subjects_to_run)} subjects "
            f"(skipping {len(completed_subjects)} checkpointed)."
        )

    def _run_one(subject: str) -> pd.DataFrame:
        return process_subject_smoothed_regression(
            subject=subject,
            event=event,
            predictor=predictor,
            formula=formula,
            base_dir=base_dir,
            load_dir=load_dir,
            rois=rois,
            freq_range=freq_range,
            win_ms=win_ms,
            downsample_factor=downsample_factor,
            n_permutations=n_permutations,
            covariates=covariates,
            min_trials=min_trials,
            seed=seed,
            regressor_source=regressor_source,
            trial_var_override=trial_var_override,
        )

    if len(subjects_to_run) > 0:
        if n_jobs_outer == 1:
            out_frames = [_run_one(s) for s in subjects_to_run]
        else:
            out_frames = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
                delayed(_run_one)(s) for s in subjects_to_run
            )
        if checkpoint_dir is not None:
            for subject, df_subj in zip(subjects_to_run, out_frames):
                if len(df_subj) == 0:
                    continue
                df_subj.to_csv(os.path.join(checkpoint_dir, f"{subject}_results.csv"), index=False)

    all_frames = []
    if checkpoint_dir is not None:
        for subject in subjects:
            path = os.path.join(checkpoint_dir, f"{subject}_results.csv")
            if os.path.exists(path):
                all_frames.append(pd.read_csv(path))
    else:
        if len(subjects_to_run) > 0:
            all_frames = [df for df in out_frames if len(df) > 0]

    if len(all_frames) == 0:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _default_filtered_subjects(base_dir: str) -> List[str]:
    """
    Return the default 19-subject set used in existing MemoryBandit notebooks.
    """
    demo_file = os.path.join(
        base_dir, 'projects', 'guLab', 'Salman', 'EMU', 'MemoryBanditsDatabase.xlsx'
    )
    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"Demographics file not found: {demo_file}")

    demo_data = pd.read_excel(demo_file, engine='openpyxl')
    subj_df = demo_data[(demo_data.LongVersion == 1) & (demo_data.Minerva == 1)].copy()
    subj_df = subj_df[
        ~subj_df['MSSMCode'].isin(['MS015', 'MS024', 'MS034', 'MS038'])
    ]
    return sorted(subj_df['MSSMCode'].dropna().astype(str).unique().tolist())


def _save_trialwise_2d_cluster_plot(
    result: Dict[str, Any],
    x_values: Optional[np.ndarray],
    y_values: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    title_prefix: str,
    out_file: str,
) -> str:
    """
    Save a generic 3-panel cluster GLM figure for trialwise 2D maps.

    Parameters
    ----------
    result : dict
        Output from TFRClusterPermutationGLM.run().
    x_values : array or None
        Axis values for map dim-1 (columns). If None, uses index coordinates.
    y_values : array or None
        Axis values for map dim-0 (rows). If None, uses index coordinates.
    x_label : str
        Label for horizontal axis.
    y_label : str
        Label for vertical axis.
    title_prefix : str
        Prefix shown in figure suptitle.
    out_file : str
        File path to save figure.
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    def _sym_norm(arr: np.ndarray) -> Optional[TwoSlopeNorm]:
        vmax = float(np.nanmax(np.abs(arr)))
        if not np.isfinite(vmax) or vmax <= 0:
            return None
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    beta_map = np.asarray(result['beta_map'], dtype=np.float64)
    t_map = np.asarray(result['t_map'], dtype=np.float64)
    sig_clusters = list(result.get('significant_clusters', []))

    sig_mask = np.zeros_like(t_map, dtype=bool)
    for c in sig_clusters:
        f0, f1 = c.freq_idx
        t0, t1 = c.time_idx
        sig_mask[f0:f1 + 1, t0:t1 + 1] = True

    n_rows, n_cols = t_map.shape
    xv = np.asarray(x_values, dtype=np.float64) if x_values is not None else np.arange(n_cols, dtype=np.float64)
    yv = np.asarray(y_values, dtype=np.float64) if y_values is not None else np.arange(n_rows, dtype=np.float64)
    if xv.shape[0] != n_cols:
        xv = np.arange(n_cols, dtype=np.float64)
    if yv.shape[0] != n_rows:
        yv = np.arange(n_rows, dtype=np.float64)

    extent = [
        float(np.min(xv)),
        float(np.max(xv)),
        float(np.min(yv)),
        float(np.max(yv)),
    ]
    dx = float(np.median(np.diff(xv))) if len(xv) > 1 else 1.0
    dy = float(np.median(np.diff(yv))) if len(yv) > 1 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        beta_map,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='RdBu_r',
        norm=_sym_norm(beta_map),
    )
    axes[0].set_title('Beta Map')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9)
    cb0.set_label('Beta')

    im1 = axes[1].imshow(
        t_map,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='RdBu_r',
        norm=_sym_norm(t_map),
    )
    axes[1].set_title('T Map')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cb1.set_label('t')

    im2 = axes[2].imshow(
        t_map,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='Greys',
        alpha=0.5,
    )
    overlay = np.ma.masked_where(~sig_mask, sig_mask.astype(float))
    axes[2].imshow(
        overlay,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='viridis',
        alpha=0.7,
    )
    for c in sig_clusters:
        f0, f1 = c.freq_idx
        t0, t1 = c.time_idx
        x0 = float(xv[t0])
        y0 = float(yv[f0])
        width = float(xv[t1] - xv[t0] + dx)
        height = float(yv[f1] - yv[f0] + dy)
        axes[2].add_patch(
            Rectangle((x0, y0), width, height, fill=False, edgecolor='yellow', linewidth=1.5)
        )
    if len(sig_clusters) == 0:
        axes[2].text(
            0.5, 0.5, 'No significant clusters',
            transform=axes[2].transAxes,
            ha='center', va='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        )
    axes[2].set_title('Significant Cluster Mask')
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel(y_label)
    cb2 = fig.colorbar(im2, ax=axes[2], shrink=0.9)
    cb2.set_label('t (background)')

    fig.suptitle(title_prefix, fontsize=11)
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_file


def _save_tfr_glm_cluster_plot(
    result: Dict[str, Any],
    times_ms: np.ndarray,
    freqs: np.ndarray,
    subject: str,
    electrode: str,
    region: str,
    event: str,
    predictor: str,
    n_permutations: int,
    out_file: str,
) -> str:
    """
    Backward-compatible TFR-specific wrapper around generic 2D cluster plotting.
    """
    title_prefix = (
        f'TFR GLM | subj={subject} | ch={electrode} ({region}) | '
        f'event={event} | predictor={predictor} | perms={n_permutations}'
    )
    return _save_trialwise_2d_cluster_plot(
        result=result,
        x_values=np.asarray(times_ms, dtype=np.float64),
        y_values=np.asarray(freqs, dtype=np.float64),
        x_label='Time (ms)',
        y_label='Frequency (Hz)',
        title_prefix=title_prefix,
        out_file=out_file,
    )


def _orient_trialwise_3d_map(
    arr: np.ndarray,
    n_trial_hint: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Coerce a 3D map to (n_trials, dim0, dim1).
    """
    x = np.asarray(arr)
    if x.ndim != 3:
        return None
    if n_trial_hint is not None and n_trial_hint > 0:
        if x.shape[0] == n_trial_hint:
            return x
        if x.shape[2] == n_trial_hint:
            return np.transpose(x, (2, 0, 1))
        if x.shape[1] == n_trial_hint:
            return np.transpose(x, (1, 0, 2))
    # Fallback: assume largest axis is trials (typical PAC export shape)
    trial_axis = int(np.argmax(np.asarray(x.shape)))
    if trial_axis == 0:
        return x
    if trial_axis == 1:
        return np.transpose(x, (1, 0, 2))
    return np.transpose(x, (2, 0, 1))


def process_subject_tfr_cluster_glm(
    subject: str,
    base_dir: str,
    load_dir: str,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    subject_plot_dir: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    max_electrodes_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run TFRClusterPermutationGLM for all valid electrodes in one subject.

    Returns
    -------
    sig_df : DataFrame
        One row per electrode with >=1 significant cluster.
    audit_df : DataFrame
        One row per attempted electrode with status/diagnostics.
    """
    import gc
    import re
    import time

    covariates = list(covariates or ['zpm'])
    covariate_str = ",".join(covariates)

    if checkpoint_file is not None and overwrite_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if checkpoint_file is not None:
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    if subject_plot_dir is not None:
        os.makedirs(subject_plot_dir, exist_ok=True)

    audit_columns = [
        'subject', 'electrode', 'region', 'event', 'predictor', 'covariates',
        'status', 'reason', 'n_aligned_trials', 'n_clusters', 'n_sig_clusters',
        'min_sig_p', 'max_sig_abs_mass', 'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
    ]
    audit_rows: List[Dict[str, Any]] = []
    completed_electrodes: set[str] = set()

    if checkpoint_file is not None and (not overwrite_checkpoint) and os.path.exists(checkpoint_file):
        try:
            existing = pd.read_csv(checkpoint_file)
            if 'electrode' in existing.columns:
                existing['electrode'] = existing['electrode'].astype(str)
                completed_electrodes = set(existing['electrode'].tolist())
            audit_rows = existing.to_dict(orient='records')
        except Exception:
            audit_rows = []
            completed_electrodes = set()

    subject_obj = Subject(
        subject=subject,
        base_dir=base_dir,
        load_dir=load_dir,
        verbose=False,
        cache_mode='reuse',
    )

    reg_df, _ = _build_subject_regressors(
        subject_obj=subject_obj,
        event=event,
        predictor=predictor,
        covariates=covariates,
        regressor_source='learn',
    )
    keep_cols = ['trial', predictor] + [c for c in covariates if c in reg_df.columns]
    reg_df = reg_df[keep_cols].copy().sort_values('trial').reset_index(drop=True)

    trial_ix = reg_df['trial'].to_numpy(dtype=int) - 1
    tfr_obj = subject_obj._ensure_tfr_loaded(event)
    tfr = tfr_obj[0] if isinstance(tfr_obj, list) else tfr_obj
    tfr_labels = [str(ch) for ch in tfr.ch_names]
    ch_index = {ch: i for i, ch in enumerate(tfr_labels)}

    in_bounds = (trial_ix >= 0) & (trial_ix < int(tfr.data.shape[0]))
    reg_in_bounds = reg_df.loc[in_bounds].reset_index(drop=True)
    trial_ix_in_bounds = trial_ix[in_bounds]

    predictor_df_base = reg_in_bounds[[predictor] + [c for c in covariates if c in reg_in_bounds.columns]].copy()
    finite_rows = np.isfinite(predictor_df_base.to_numpy(dtype=np.float64)).all(axis=1)
    predictor_df_base = predictor_df_base.loc[finite_rows].reset_index(drop=True)
    trial_ix_base = trial_ix_in_bounds[finite_rows]
    n_aligned_base = int(len(predictor_df_base))

    elec_df = subject_obj.elec_df.copy()
    elec_df['label'] = elec_df['label'].astype(str)
    region_map = (
        elec_df.groupby('label')['salman_region']
        .first()
        .to_dict()
        if 'salman_region' in elec_df.columns
        else {}
    )
    elec_set = set(elec_df['label'].tolist())
    electrodes = [ch for ch in tfr_labels if ch in elec_set]
    if max_electrodes_per_subject is not None:
        electrodes = electrodes[:max(0, int(max_electrodes_per_subject))]
    electrodes = [e for e in electrodes if e not in completed_electrodes]

    for electrode in electrodes:
        t0 = time.perf_counter()
        region = str(region_map.get(electrode, 'Unknown'))
        row: Dict[str, Any] = {
            'subject': subject,
            'electrode': str(electrode),
            'region': region,
            'event': event,
            'predictor': predictor,
            'covariates': covariate_str,
            'status': 'skipped',
            'reason': '',
            'n_aligned_trials': n_aligned_base,
            'n_clusters': 0,
            'n_sig_clusters': 0,
            'min_sig_p': np.nan,
            'max_sig_abs_mass': np.nan,
            'df_resid': np.nan,
            'tcrit': np.nan,
            'plot_path': '',
            'elapsed_sec': np.nan,
        }

        tfr_data = None
        tfr_data_aligned = None
        glm = None
        result = None

        try:
            if n_aligned_base < int(min_aligned_trials):
                row['status'] = 'skipped'
                row['reason'] = f'aligned_trials<{int(min_aligned_trials)}'
            else:
                idx = ch_index[electrode]
                tfr_data = np.asarray(tfr.data[:, idx, :, :], dtype=np.float64)
                tfr_data_aligned = tfr_data[trial_ix_base, :, :]
                if not np.isfinite(tfr_data_aligned).all():
                    row['status'] = 'skipped'
                    row['reason'] = 'tfr_contains_nan'
                else:
                    glm = TFRClusterPermutationGLM(
                        tfr_data=tfr_data_aligned,
                        predictor_data=predictor_df_base,
                        target_var=predictor,
                        ch_name=str(electrode),
                        alternative=alternative,
                        alpha_cluster=alpha_cluster,
                        alpha_test=alpha_test,
                        standardize_X=True,
                        random_state=random_state,
                    )
                    result = glm.run(
                        n_permutations=int(n_permutations),
                        n_jobs=1,
                        verbose=0,
                    )
                    clusters = result.get('clusters', [])
                    sig_clusters = result.get('significant_clusters', [])
                    row['status'] = 'ok'
                    row['reason'] = ''
                    row['n_clusters'] = int(len(clusters))
                    row['n_sig_clusters'] = int(len(sig_clusters))
                    row['df_resid'] = float(result.get('df_resid', np.nan))
                    row['tcrit'] = float(result.get('tcrit', np.nan))
                    if len(sig_clusters) > 0:
                        row['min_sig_p'] = float(min(c.pvalue for c in sig_clusters))
                        row['max_sig_abs_mass'] = float(max(c.abs_mass for c in sig_clusters))

                    should_plot = bool(save_non_sig_plots or row['n_sig_clusters'] >= 1)
                    if should_plot and subject_plot_dir is not None:
                        token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(electrode)).strip('_') or 'electrode'
                        out_file = os.path.join(
                            subject_plot_dir,
                            f"tfr_glm_sig_{subject}_{token}_{event}.png",
                        )
                        times_ms = np.asarray(tfr.times, dtype=np.float64) * 1000.0
                        freqs = np.asarray(tfr.freqs, dtype=np.float64)
                        _save_tfr_glm_cluster_plot(
                            result=result,
                            times_ms=times_ms,
                            freqs=freqs,
                            subject=subject,
                            electrode=str(electrode),
                            region=region,
                            event=event,
                            predictor=predictor,
                            n_permutations=int(n_permutations),
                            out_file=out_file,
                        )
                        row['plot_path'] = out_file
        except Exception as exc:
            row['status'] = 'error'
            row['reason'] = f'{type(exc).__name__}: {exc}'
        finally:
            row['elapsed_sec'] = float(time.perf_counter() - t0)
            audit_rows.append(row)
            audit_df_tmp = pd.DataFrame(audit_rows)
            if len(audit_df_tmp) > 0 and 'electrode' in audit_df_tmp.columns:
                audit_df_tmp['electrode'] = audit_df_tmp['electrode'].astype(str)
                audit_df_tmp = audit_df_tmp.drop_duplicates(
                    subset=['subject', 'electrode'],
                    keep='last',
                )
            for col in audit_columns:
                if col not in audit_df_tmp.columns:
                    audit_df_tmp[col] = np.nan
            audit_df_tmp = audit_df_tmp[audit_columns]
            audit_rows = audit_df_tmp.to_dict(orient='records')
            if checkpoint_file is not None:
                audit_df_tmp.to_csv(checkpoint_file, index=False)
            del tfr_data, tfr_data_aligned, glm, result
            gc.collect()

    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df) == 0:
        audit_df = pd.DataFrame(columns=audit_columns)
    else:
        audit_df['electrode'] = audit_df['electrode'].astype(str)
        audit_df = audit_df.drop_duplicates(subset=['subject', 'electrode'], keep='last')
        for col in audit_columns:
            if col not in audit_df.columns:
                audit_df[col] = np.nan
        audit_df = audit_df[audit_columns].reset_index(drop=True)

    sig_df = audit_df[
        (audit_df['status'] == 'ok') &
        (pd.to_numeric(audit_df['n_sig_clusters'], errors='coerce').fillna(0).astype(int) >= 1)
    ].copy()

    if checkpoint_file is not None:
        audit_df.to_csv(checkpoint_file, index=False)
    return sig_df, audit_df


def process_subject_crossregion_pac_cluster_glm(
    subject: str,
    base_dir: str,
    load_dir: str,
    pac_dir: str,
    trial_df: Optional[Union[pd.DataFrame, str]] = None,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    subject_plot_dir: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    max_pairs_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cluster GLM for all available cross-region PAC channel pairs in one subject.

    Uses precomputed PAC z-maps saved by MemoryBandit-CrossRegionPAC notebook.
    """
    import gc
    import re
    import time

    covariates = list(covariates or ['zpm'])
    covariate_str = ",".join(covariates)

    default_trial_csv = os.path.join(pac_dir, 'all_subjects_crossregion_pac_combined.csv')
    if trial_df is None:
        trial_df_src = default_trial_csv
    else:
        trial_df_src = trial_df

    if isinstance(trial_df_src, pd.DataFrame):
        all_trial_df = trial_df_src.copy()
    else:
        if not os.path.exists(str(trial_df_src)):
            raise FileNotFoundError(f"PAC trial dataframe not found: {trial_df_src}")
        all_trial_df = pd.read_csv(str(trial_df_src))

    pair_cols = [
        'participant', 'event', 'phase_roi', 'amp_roi',
        'phase_channel', 'amp_channel', 'hemi',
    ]
    required_cols = pair_cols + ['trials', predictor] + [c for c in covariates]
    missing_cols = [c for c in required_cols if c not in all_trial_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required PAC trial columns: {missing_cols}")

    subj_df = all_trial_df[
        (all_trial_df['participant'].astype(str) == str(subject)) &
        (all_trial_df['event'].astype(str) == str(event))
    ].copy()
    if len(subj_df) == 0:
        raise ValueError(f'No PAC trial rows for subject={subject}, event={event}')

    if checkpoint_file is not None and overwrite_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if checkpoint_file is not None:
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    if subject_plot_dir is not None:
        os.makedirs(subject_plot_dir, exist_ok=True)

    audit_columns = [
        'subject', 'event', 'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel', 'hemi',
        'pair_key', 'pac_file', 'predictor', 'covariates',
        'status', 'reason', 'n_aligned_trials',
        'n_clusters', 'n_sig_clusters', 'min_sig_p', 'max_sig_abs_mass',
        'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
    ]
    audit_rows: List[Dict[str, Any]] = []
    completed_pair_keys: set[str] = set()

    if checkpoint_file is not None and (not overwrite_checkpoint) and os.path.exists(checkpoint_file):
        try:
            existing = pd.read_csv(checkpoint_file)
            if 'pair_key' in existing.columns:
                existing['pair_key'] = existing['pair_key'].astype(str)
                completed_pair_keys = set(existing['pair_key'].tolist())
            audit_rows = existing.to_dict(orient='records')
        except Exception:
            audit_rows = []
            completed_pair_keys = set()

    grouped = list(subj_df.groupby(pair_cols, dropna=False))
    if max_pairs_per_subject is not None:
        grouped = grouped[:max(0, int(max_pairs_per_subject))]

    for key_vals, gdf in grouped:
        t0 = time.perf_counter()
        participant, ev, phase_roi, amp_roi, phase_channel, amp_channel, hemi = key_vals
        participant = str(participant)
        ev = str(ev)
        phase_roi = str(phase_roi)
        amp_roi = str(amp_roi)
        phase_channel = str(phase_channel)
        amp_channel = str(amp_channel)
        hemi = str(hemi)
        pair_key = (
            f'{participant}|{ev}|{phase_roi}|{amp_roi}|'
            f'{phase_channel}|{amp_channel}|{hemi}'
        )
        if pair_key in completed_pair_keys:
            continue

        pac_file = os.path.join(
            pac_dir,
            f'{participant}_{phase_roi}-{amp_roi}_{phase_channel}-{amp_channel}_pac.npy'
        )

        row: Dict[str, Any] = {
            'subject': participant,
            'event': ev,
            'phase_roi': phase_roi,
            'amp_roi': amp_roi,
            'phase_channel': phase_channel,
            'amp_channel': amp_channel,
            'hemi': hemi,
            'pair_key': pair_key,
            'pac_file': pac_file,
            'predictor': predictor,
            'covariates': covariate_str,
            'status': 'skipped',
            'reason': '',
            'n_aligned_trials': 0,
            'n_clusters': 0,
            'n_sig_clusters': 0,
            'min_sig_p': np.nan,
            'max_sig_abs_mass': np.nan,
            'df_resid': np.nan,
            'tcrit': np.nan,
            'plot_path': '',
            'elapsed_sec': np.nan,
        }

        pac_map = None
        pac_trials_first = None
        map_aligned = None
        glm = None
        result = None
        reg_df = None

        try:
            if not os.path.exists(pac_file):
                row['status'] = 'skipped'
                row['reason'] = 'missing_pac_file'
            else:
                pac_map = np.load(pac_file, mmap_mode='r')
                n_trial_hint = int(pd.to_numeric(gdf['trials'], errors='coerce').dropna().max()) if len(gdf) > 0 else None
                pac_trials_first = _orient_trialwise_3d_map(pac_map, n_trial_hint=n_trial_hint)
                if pac_trials_first is None or pac_trials_first.ndim != 3:
                    row['status'] = 'skipped'
                    row['reason'] = 'invalid_pac_shape'
                else:
                    reg_df = gdf[['trials', predictor] + [c for c in covariates if c in gdf.columns]].copy()
                    reg_df['trials'] = pd.to_numeric(reg_df['trials'], errors='coerce')
                    reg_df = reg_df.dropna(subset=['trials']).sort_values('trials')
                    reg_df = reg_df.drop_duplicates(subset=['trials'], keep='last').reset_index(drop=True)

                    trial_ix = reg_df['trials'].to_numpy(dtype=int) - 1
                    in_bounds = (trial_ix >= 0) & (trial_ix < int(pac_trials_first.shape[0]))
                    reg_valid = reg_df.loc[in_bounds].reset_index(drop=True)
                    trial_ix_valid = trial_ix[in_bounds]

                    predictor_df = reg_valid[[predictor] + [c for c in covariates if c in reg_valid.columns]].copy()
                    finite_mask = np.isfinite(predictor_df.to_numpy(dtype=np.float64)).all(axis=1)
                    predictor_df = predictor_df.loc[finite_mask].reset_index(drop=True)
                    trial_ix_final = trial_ix_valid[finite_mask]
                    map_aligned = np.asarray(pac_trials_first[trial_ix_final, :, :], dtype=np.float64)

                    row['n_aligned_trials'] = int(len(predictor_df))
                    if row['n_aligned_trials'] < int(min_aligned_trials):
                        row['status'] = 'skipped'
                        row['reason'] = f'aligned_trials<{int(min_aligned_trials)}'
                    elif not np.isfinite(map_aligned).all():
                        row['status'] = 'skipped'
                        row['reason'] = 'pac_contains_nan'
                    else:
                        glm = TFRClusterPermutationGLM(
                            tfr_data=map_aligned,
                            predictor_data=predictor_df,
                            target_var=predictor,
                            ch_name=f'{phase_channel}->{amp_channel}',
                            alternative=alternative,
                            alpha_cluster=alpha_cluster,
                            alpha_test=alpha_test,
                            standardize_X=True,
                            random_state=random_state,
                        )
                        result = glm.run(
                            n_permutations=int(n_permutations),
                            n_jobs=1,
                            verbose=0,
                        )
                        clusters = result.get('clusters', [])
                        sig_clusters = result.get('significant_clusters', [])
                        row['status'] = 'ok'
                        row['reason'] = ''
                        row['n_clusters'] = int(len(clusters))
                        row['n_sig_clusters'] = int(len(sig_clusters))
                        row['df_resid'] = float(result.get('df_resid', np.nan))
                        row['tcrit'] = float(result.get('tcrit', np.nan))
                        if len(sig_clusters) > 0:
                            row['min_sig_p'] = float(min(c.pvalue for c in sig_clusters))
                            row['max_sig_abs_mass'] = float(max(c.abs_mass for c in sig_clusters))

                        should_plot = bool(save_non_sig_plots or row['n_sig_clusters'] >= 1)
                        if should_plot and subject_plot_dir is not None:
                            token = (
                                re.sub(r'[^A-Za-z0-9._-]+', '_', f'{phase_channel}__{amp_channel}')
                                .strip('_')
                                or 'pair'
                            )
                            out_file = os.path.join(
                                subject_plot_dir,
                                f'pac_glm_sig_{participant}_{phase_roi}-{amp_roi}_{token}_{ev}.png'
                            )
                            x_vals = np.arange(map_aligned.shape[2], dtype=np.float64)
                            y_vals = np.arange(map_aligned.shape[1], dtype=np.float64)
                            title_prefix = (
                                f'PAC GLM | subj={participant} | event={ev} | '
                                f'{phase_roi}->{amp_roi} | {phase_channel}->{amp_channel} | '
                                f'predictor={predictor} | perms={int(n_permutations)}'
                            )
                            _save_trialwise_2d_cluster_plot(
                                result=result,
                                x_values=x_vals,
                                y_values=y_vals,
                                x_label='Amplitude Bin',
                                y_label='Phase Bin',
                                title_prefix=title_prefix,
                                out_file=out_file,
                            )
                            row['plot_path'] = out_file
        except Exception as exc:
            row['status'] = 'error'
            row['reason'] = f'{type(exc).__name__}: {exc}'
        finally:
            row['elapsed_sec'] = float(time.perf_counter() - t0)
            audit_rows.append(row)
            audit_df_tmp = pd.DataFrame(audit_rows)
            if len(audit_df_tmp) > 0 and 'pair_key' in audit_df_tmp.columns:
                audit_df_tmp['pair_key'] = audit_df_tmp['pair_key'].astype(str)
                audit_df_tmp = audit_df_tmp.drop_duplicates(subset=['pair_key'], keep='last')
            for col in audit_columns:
                if col not in audit_df_tmp.columns:
                    audit_df_tmp[col] = np.nan
            audit_df_tmp = audit_df_tmp[audit_columns]
            audit_rows = audit_df_tmp.to_dict(orient='records')
            if checkpoint_file is not None:
                audit_df_tmp.to_csv(checkpoint_file, index=False)
            del pac_map, pac_trials_first, map_aligned, glm, result, reg_df
            gc.collect()

    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df) == 0:
        audit_df = pd.DataFrame(columns=audit_columns)
    else:
        if 'pair_key' in audit_df.columns:
            audit_df['pair_key'] = audit_df['pair_key'].astype(str)
            audit_df = audit_df.drop_duplicates(subset=['pair_key'], keep='last')
        for col in audit_columns:
            if col not in audit_df.columns:
                audit_df[col] = np.nan
        audit_df = audit_df[audit_columns].reset_index(drop=True)

    sig_df = audit_df[
        (audit_df['status'] == 'ok') &
        (pd.to_numeric(audit_df['n_sig_clusters'], errors='coerce').fillna(0).astype(int) >= 1)
    ].copy()

    if checkpoint_file is not None:
        audit_df.to_csv(checkpoint_file, index=False)
    return sig_df, audit_df


def run_batch_crossregion_pac_cluster_glm_pipeline(
    subjects: Optional[List[str]],
    base_dir: str,
    load_dir: str,
    pac_dir: str,
    run_dir: str,
    trial_df: Optional[Union[pd.DataFrame, str]] = None,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    overwrite: bool = False,
    resume: bool = True,
    max_subjects: Optional[int] = None,
    max_pairs_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Sequential, checkpointed all-subject/all-pair PAC cluster GLM pipeline.
    """
    from datetime import datetime, timezone

    covariates = list(covariates or ['zpm'])
    if subjects is None:
        subjects = _default_filtered_subjects(base_dir)
    subjects = [str(s) for s in subjects]
    if max_subjects is not None:
        subjects = subjects[:max(0, int(max_subjects))]

    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if trial_df is None:
        trial_df_src: Union[pd.DataFrame, str] = os.path.join(
            pac_dir, 'all_subjects_crossregion_pac_combined.csv'
        )
    else:
        trial_df_src = trial_df

    for subject in subjects:
        subject_plot_dir = os.path.join(plots_dir, subject)
        subject_audit_file = os.path.join(checkpoints_dir, f'{subject}_audit_pairs.csv')
        subject_sig_file = os.path.join(checkpoints_dir, f'{subject}_sig_pairs.csv')
        effective_overwrite = bool(overwrite or (not resume))

        if effective_overwrite:
            if os.path.exists(subject_audit_file):
                os.remove(subject_audit_file)
            if os.path.exists(subject_sig_file):
                os.remove(subject_sig_file)

        try:
            sig_df, audit_df = process_subject_crossregion_pac_cluster_glm(
                subject=subject,
                base_dir=base_dir,
                load_dir=load_dir,
                pac_dir=pac_dir,
                trial_df=trial_df_src,
                event=event,
                predictor=predictor,
                covariates=covariates,
                n_permutations=n_permutations,
                alpha_cluster=alpha_cluster,
                alpha_test=alpha_test,
                alternative=alternative,
                min_aligned_trials=min_aligned_trials,
                random_state=random_state,
                subject_plot_dir=subject_plot_dir,
                checkpoint_file=subject_audit_file,
                overwrite_checkpoint=effective_overwrite,
                max_pairs_per_subject=max_pairs_per_subject,
                save_non_sig_plots=save_non_sig_plots,
            )
        except Exception as exc:
            audit_df = pd.DataFrame([{
                'subject': subject,
                'event': event,
                'phase_roi': np.nan,
                'amp_roi': np.nan,
                'phase_channel': np.nan,
                'amp_channel': np.nan,
                'hemi': np.nan,
                'pair_key': np.nan,
                'pac_file': np.nan,
                'predictor': predictor,
                'covariates': ",".join(covariates),
                'status': 'error',
                'reason': f'{type(exc).__name__}: {exc}',
                'n_aligned_trials': np.nan,
                'n_clusters': 0,
                'n_sig_clusters': 0,
                'min_sig_p': np.nan,
                'max_sig_abs_mass': np.nan,
                'df_resid': np.nan,
                'tcrit': np.nan,
                'plot_path': '',
                'elapsed_sec': np.nan,
            }])
            sig_df = pd.DataFrame()

        audit_df.to_csv(subject_audit_file, index=False)
        sig_df.to_csv(subject_sig_file, index=False)

    sig_frames: List[pd.DataFrame] = []
    audit_frames: List[pd.DataFrame] = []
    for subject in subjects:
        sig_file = os.path.join(checkpoints_dir, f'{subject}_sig_pairs.csv')
        audit_file = os.path.join(checkpoints_dir, f'{subject}_audit_pairs.csv')
        if os.path.exists(sig_file):
            try:
                sig_frames.append(pd.read_csv(sig_file))
            except Exception:
                pass
        if os.path.exists(audit_file):
            try:
                audit_frames.append(pd.read_csv(audit_file))
            except Exception:
                pass

    if len(sig_frames) > 0:
        sig_pairs = pd.concat(sig_frames, ignore_index=True)
        dedupe_cols = [
            'subject', 'event', 'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel', 'hemi'
        ]
        if all(c in sig_pairs.columns for c in dedupe_cols):
            sig_pairs = sig_pairs.drop_duplicates(subset=dedupe_cols, keep='last').reset_index(drop=True)
    else:
        sig_pairs = pd.DataFrame(columns=[
            'subject', 'event', 'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel', 'hemi',
            'predictor', 'covariates', 'n_aligned_trials', 'n_clusters', 'n_sig_clusters',
            'min_sig_p', 'max_sig_abs_mass', 'df_resid', 'tcrit', 'plot_path',
        ])

    if len(audit_frames) > 0:
        audit_pairs = pd.concat(audit_frames, ignore_index=True)
        if 'pair_key' in audit_pairs.columns:
            audit_pairs['pair_key'] = audit_pairs['pair_key'].astype(str)
            audit_pairs = audit_pairs.drop_duplicates(subset=['pair_key'], keep='last').reset_index(drop=True)
    else:
        audit_pairs = pd.DataFrame(columns=[
            'subject', 'event', 'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel', 'hemi',
            'pair_key', 'pac_file', 'predictor', 'covariates', 'status', 'reason',
            'n_aligned_trials', 'n_clusters', 'n_sig_clusters', 'min_sig_p', 'max_sig_abs_mass',
            'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
        ])

    if len(sig_pairs) > 0:
        phase_rows = sig_pairs[[
            'subject', 'event', 'phase_roi', 'phase_channel'
        ]].rename(columns={'phase_roi': 'region', 'phase_channel': 'electrode'})
        amp_rows = sig_pairs[[
            'subject', 'event', 'amp_roi', 'amp_channel'
        ]].rename(columns={'amp_roi': 'region', 'amp_channel': 'electrode'})
        sig_electrodes = pd.concat([phase_rows, amp_rows], ignore_index=True)
        sig_electrodes['electrode'] = sig_electrodes['electrode'].astype(str)
        sig_electrodes = sig_electrodes.drop_duplicates(
            subset=['subject', 'electrode', 'region'],
            keep='last',
        ).reset_index(drop=True)
        region_counts = (
            sig_electrodes
            .groupby('region', dropna=False)
            .agg(
                n_sig_electrodes=('electrode', 'count'),
                n_subjects_with_sig=('subject', 'nunique'),
            )
            .reset_index()
            .sort_values('n_sig_electrodes', ascending=False)
            .reset_index(drop=True)
        )
    else:
        sig_electrodes = pd.DataFrame(columns=['subject', 'event', 'region', 'electrode'])
        region_counts = pd.DataFrame(columns=['region', 'n_sig_electrodes', 'n_subjects_with_sig'])

    sig_pairs_out = os.path.join(run_dir, 'sig_pac_pairs.csv')
    audit_out = os.path.join(run_dir, 'all_pac_pair_audit.csv')
    sig_electrodes_out = os.path.join(run_dir, 'sig_electrodes.csv')
    region_out = os.path.join(run_dir, 'region_counts.csv')
    meta_out = os.path.join(run_dir, 'run_metadata.json')

    sig_pairs.to_csv(sig_pairs_out, index=False)
    audit_pairs.to_csv(audit_out, index=False)
    sig_electrodes.to_csv(sig_electrodes_out, index=False)
    region_counts.to_csv(region_out, index=False)

    failures = int((audit_pairs.get('status', pd.Series(dtype=str)) == 'error').sum()) if len(audit_pairs) > 0 else 0
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'base_dir': base_dir,
        'load_dir': load_dir,
        'pac_dir': pac_dir,
        'run_dir': run_dir,
        'subjects': subjects,
        'event': event,
        'predictor': predictor,
        'covariates': covariates,
        'n_permutations': int(n_permutations),
        'alpha_cluster': float(alpha_cluster),
        'alpha_test': float(alpha_test),
        'alternative': alternative,
        'min_aligned_trials': int(min_aligned_trials),
        'random_state': int(random_state),
        'overwrite': bool(overwrite),
        'resume': bool(resume),
        'max_subjects': None if max_subjects is None else int(max_subjects),
        'max_pairs_per_subject': (
            None if max_pairs_per_subject is None else int(max_pairs_per_subject)
        ),
        'save_non_sig_plots': bool(save_non_sig_plots),
        'n_subjects': int(len(subjects)),
        'n_attempted_pairs': int(len(audit_pairs)),
        'n_sig_pairs': int(len(sig_pairs)),
        'n_sig_electrodes': int(len(sig_electrodes)),
        'n_failures': failures,
        'sig_pac_pairs_csv': sig_pairs_out,
        'audit_pairs_csv': audit_out,
        'sig_electrodes_csv': sig_electrodes_out,
        'region_counts_csv': region_out,
    }
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'sig_pairs': sig_pairs,
        'sig_electrodes': sig_electrodes,
        'region_counts': region_counts,
        'audit_pairs': audit_pairs,
    }


def process_subject_withinregion_pac_cluster_glm(
    subject: str,
    base_dir: str,
    load_dir: str,
    pac_dir: str,
    trial_df: Optional[Union[pd.DataFrame, str]] = None,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    subject_plot_dir: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    max_electrodes_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cluster GLM for within-region/within-electrode PAC maps in one subject.

    Uses precomputed PAC z-maps saved by MemoryBandit-WithinRegionPAC notebook.
    """
    import gc
    import re
    import time

    covariates = list(covariates or ['zpm'])
    covariate_str = ",".join(covariates)

    default_trial_csv = os.path.join(pac_dir, 'all_subjects_withinelectrode_pac_combined.csv')
    trial_df_src = default_trial_csv if trial_df is None else trial_df

    if isinstance(trial_df_src, pd.DataFrame):
        all_trial_df = trial_df_src.copy()
    else:
        if not os.path.exists(str(trial_df_src)):
            raise FileNotFoundError(f"Within PAC trial dataframe not found: {trial_df_src}")
        all_trial_df = pd.read_csv(str(trial_df_src))

    unit_cols = [
        'participant', 'event', 'phase_roi', 'amp_roi',
        'phase_channel', 'amp_channel', 'hemi',
    ]
    required_cols = unit_cols + ['trials', predictor] + [c for c in covariates]
    missing_cols = [c for c in required_cols if c not in all_trial_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required within-PAC trial columns: {missing_cols}")

    subj_df = all_trial_df[
        (all_trial_df['participant'].astype(str) == str(subject)) &
        (all_trial_df['event'].astype(str) == str(event))
    ].copy()
    if len(subj_df) == 0:
        raise ValueError(f'No within-PAC trial rows for subject={subject}, event={event}')

    if checkpoint_file is not None and overwrite_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if checkpoint_file is not None:
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    if subject_plot_dir is not None:
        os.makedirs(subject_plot_dir, exist_ok=True)

    audit_columns = [
        'subject', 'event', 'region', 'electrode', 'hemi',
        'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel',
        'unit_key', 'pac_file', 'predictor', 'covariates',
        'status', 'reason', 'n_aligned_trials',
        'n_clusters', 'n_sig_clusters', 'min_sig_p', 'max_sig_abs_mass',
        'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
    ]
    audit_rows: List[Dict[str, Any]] = []
    completed_unit_keys: set[str] = set()

    if checkpoint_file is not None and (not overwrite_checkpoint) and os.path.exists(checkpoint_file):
        try:
            existing = pd.read_csv(checkpoint_file)
            if 'unit_key' in existing.columns:
                existing['unit_key'] = existing['unit_key'].astype(str)
                completed_unit_keys = set(existing['unit_key'].tolist())
            audit_rows = existing.to_dict(orient='records')
        except Exception:
            audit_rows = []
            completed_unit_keys = set()

    grouped = list(subj_df.groupby(unit_cols, dropna=False))
    if max_electrodes_per_subject is not None:
        grouped = grouped[:max(0, int(max_electrodes_per_subject))]

    for key_vals, gdf in grouped:
        t0 = time.perf_counter()
        participant, ev, phase_roi, amp_roi, phase_channel, amp_channel, hemi = key_vals
        participant = str(participant)
        ev = str(ev)
        phase_roi = str(phase_roi)
        amp_roi = str(amp_roi)
        phase_channel = str(phase_channel)
        amp_channel = str(amp_channel)
        hemi = str(hemi)
        region = phase_roi
        electrode = phase_channel

        unit_key = f'{participant}|{ev}|{phase_roi}|{phase_channel}|{hemi}'
        if unit_key in completed_unit_keys:
            continue

        pac_file = os.path.join(
            pac_dir,
            f'{participant}_{region}_WITHIN_{electrode}_pac.npy'
        )

        row: Dict[str, Any] = {
            'subject': participant,
            'event': ev,
            'region': region,
            'electrode': electrode,
            'hemi': hemi,
            'phase_roi': phase_roi,
            'amp_roi': amp_roi,
            'phase_channel': phase_channel,
            'amp_channel': amp_channel,
            'unit_key': unit_key,
            'pac_file': pac_file,
            'predictor': predictor,
            'covariates': covariate_str,
            'status': 'skipped',
            'reason': '',
            'n_aligned_trials': 0,
            'n_clusters': 0,
            'n_sig_clusters': 0,
            'min_sig_p': np.nan,
            'max_sig_abs_mass': np.nan,
            'df_resid': np.nan,
            'tcrit': np.nan,
            'plot_path': '',
            'elapsed_sec': np.nan,
        }

        pac_map = None
        pac_trials_first = None
        map_aligned = None
        glm = None
        result = None
        reg_df = None

        try:
            if (phase_roi != amp_roi) or (phase_channel != amp_channel):
                row['status'] = 'skipped'
                row['reason'] = 'not_within_identity'
            elif not os.path.exists(pac_file):
                row['status'] = 'skipped'
                row['reason'] = 'missing_pac_file'
            else:
                pac_map = np.load(pac_file, mmap_mode='r')
                n_trial_hint = int(pd.to_numeric(gdf['trials'], errors='coerce').dropna().max()) if len(gdf) > 0 else None
                pac_trials_first = _orient_trialwise_3d_map(pac_map, n_trial_hint=n_trial_hint)
                if pac_trials_first is None or pac_trials_first.ndim != 3:
                    row['status'] = 'skipped'
                    row['reason'] = 'invalid_pac_shape'
                else:
                    reg_df = gdf[['trials', predictor] + [c for c in covariates if c in gdf.columns]].copy()
                    reg_df['trials'] = pd.to_numeric(reg_df['trials'], errors='coerce')
                    reg_df = reg_df.dropna(subset=['trials']).sort_values('trials')
                    reg_df = reg_df.drop_duplicates(subset=['trials'], keep='last').reset_index(drop=True)

                    trial_ix = reg_df['trials'].to_numpy(dtype=int) - 1
                    in_bounds = (trial_ix >= 0) & (trial_ix < int(pac_trials_first.shape[0]))
                    reg_valid = reg_df.loc[in_bounds].reset_index(drop=True)
                    trial_ix_valid = trial_ix[in_bounds]

                    predictor_df = reg_valid[[predictor] + [c for c in covariates if c in reg_valid.columns]].copy()
                    finite_mask = np.isfinite(predictor_df.to_numpy(dtype=np.float64)).all(axis=1)
                    predictor_df = predictor_df.loc[finite_mask].reset_index(drop=True)
                    trial_ix_final = trial_ix_valid[finite_mask]
                    map_aligned = np.asarray(pac_trials_first[trial_ix_final, :, :], dtype=np.float64)

                    row['n_aligned_trials'] = int(len(predictor_df))
                    if row['n_aligned_trials'] < int(min_aligned_trials):
                        row['status'] = 'skipped'
                        row['reason'] = f'aligned_trials<{int(min_aligned_trials)}'
                    elif not np.isfinite(map_aligned).all():
                        row['status'] = 'skipped'
                        row['reason'] = 'pac_contains_nan'
                    else:
                        glm = TFRClusterPermutationGLM(
                            tfr_data=map_aligned,
                            predictor_data=predictor_df,
                            target_var=predictor,
                            ch_name=f'{region}:{electrode}',
                            alternative=alternative,
                            alpha_cluster=alpha_cluster,
                            alpha_test=alpha_test,
                            standardize_X=True,
                            random_state=random_state,
                        )
                        result = glm.run(
                            n_permutations=int(n_permutations),
                            n_jobs=1,
                            verbose=0,
                        )
                        clusters = result.get('clusters', [])
                        sig_clusters = result.get('significant_clusters', [])
                        row['status'] = 'ok'
                        row['reason'] = ''
                        row['n_clusters'] = int(len(clusters))
                        row['n_sig_clusters'] = int(len(sig_clusters))
                        row['df_resid'] = float(result.get('df_resid', np.nan))
                        row['tcrit'] = float(result.get('tcrit', np.nan))
                        if len(sig_clusters) > 0:
                            row['min_sig_p'] = float(min(c.pvalue for c in sig_clusters))
                            row['max_sig_abs_mass'] = float(max(c.abs_mass for c in sig_clusters))

                        should_plot = bool(save_non_sig_plots or row['n_sig_clusters'] >= 1)
                        if should_plot and subject_plot_dir is not None:
                            token = re.sub(r'[^A-Za-z0-9._-]+', '_', f'{region}__{electrode}').strip('_') or 'within'
                            out_file = os.path.join(
                                subject_plot_dir,
                                f'within_pac_glm_sig_{participant}_{token}_{ev}.png'
                            )
                            x_vals = np.arange(map_aligned.shape[2], dtype=np.float64)
                            y_vals = np.arange(map_aligned.shape[1], dtype=np.float64)
                            title_prefix = (
                                f'Within-PAC GLM | subj={participant} | event={ev} | '
                                f'{region} | {electrode} | predictor={predictor} | perms={int(n_permutations)}'
                            )
                            _save_trialwise_2d_cluster_plot(
                                result=result,
                                x_values=x_vals,
                                y_values=y_vals,
                                x_label='Amplitude Bin',
                                y_label='Phase Bin',
                                title_prefix=title_prefix,
                                out_file=out_file,
                            )
                            row['plot_path'] = out_file
        except Exception as exc:
            row['status'] = 'error'
            row['reason'] = f'{type(exc).__name__}: {exc}'
        finally:
            row['elapsed_sec'] = float(time.perf_counter() - t0)
            audit_rows.append(row)
            audit_df_tmp = pd.DataFrame(audit_rows)
            if len(audit_df_tmp) > 0 and 'unit_key' in audit_df_tmp.columns:
                audit_df_tmp['unit_key'] = audit_df_tmp['unit_key'].astype(str)
                audit_df_tmp = audit_df_tmp.drop_duplicates(subset=['unit_key'], keep='last')
            for col in audit_columns:
                if col not in audit_df_tmp.columns:
                    audit_df_tmp[col] = np.nan
            audit_df_tmp = audit_df_tmp[audit_columns]
            audit_rows = audit_df_tmp.to_dict(orient='records')
            if checkpoint_file is not None:
                audit_df_tmp.to_csv(checkpoint_file, index=False)
            del pac_map, pac_trials_first, map_aligned, glm, result, reg_df
            gc.collect()

    audit_df = pd.DataFrame(audit_rows)
    if len(audit_df) == 0:
        audit_df = pd.DataFrame(columns=audit_columns)
    else:
        if 'unit_key' in audit_df.columns:
            audit_df['unit_key'] = audit_df['unit_key'].astype(str)
            audit_df = audit_df.drop_duplicates(subset=['unit_key'], keep='last')
        for col in audit_columns:
            if col not in audit_df.columns:
                audit_df[col] = np.nan
        audit_df = audit_df[audit_columns].reset_index(drop=True)

    sig_df = audit_df[
        (audit_df['status'] == 'ok') &
        (pd.to_numeric(audit_df['n_sig_clusters'], errors='coerce').fillna(0).astype(int) >= 1)
    ].copy()

    if checkpoint_file is not None:
        audit_df.to_csv(checkpoint_file, index=False)
    return sig_df, audit_df


def run_batch_withinregion_pac_cluster_glm_pipeline(
    subjects: Optional[List[str]],
    base_dir: str,
    load_dir: str,
    pac_dir: str,
    run_dir: str,
    trial_df: Optional[Union[pd.DataFrame, str]] = None,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    overwrite: bool = False,
    resume: bool = True,
    max_subjects: Optional[int] = None,
    max_electrodes_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Sequential, checkpointed all-subject/all-within-electrode PAC cluster GLM pipeline.
    """
    from datetime import datetime, timezone

    covariates = list(covariates or ['zpm'])
    if subjects is None:
        subjects = _default_filtered_subjects(base_dir)
    subjects = [str(s) for s in subjects]
    if max_subjects is not None:
        subjects = subjects[:max(0, int(max_subjects))]

    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if trial_df is None:
        trial_df_src: Union[pd.DataFrame, str] = os.path.join(
            pac_dir, 'all_subjects_withinelectrode_pac_combined.csv'
        )
    else:
        trial_df_src = trial_df

    for subject in subjects:
        subject_plot_dir = os.path.join(plots_dir, subject)
        subject_audit_file = os.path.join(checkpoints_dir, f'{subject}_within_audit.csv')
        subject_sig_file = os.path.join(checkpoints_dir, f'{subject}_within_sig.csv')
        effective_overwrite = bool(overwrite or (not resume))

        if effective_overwrite:
            if os.path.exists(subject_audit_file):
                os.remove(subject_audit_file)
            if os.path.exists(subject_sig_file):
                os.remove(subject_sig_file)

        try:
            sig_df, audit_df = process_subject_withinregion_pac_cluster_glm(
                subject=subject,
                base_dir=base_dir,
                load_dir=load_dir,
                pac_dir=pac_dir,
                trial_df=trial_df_src,
                event=event,
                predictor=predictor,
                covariates=covariates,
                n_permutations=n_permutations,
                alpha_cluster=alpha_cluster,
                alpha_test=alpha_test,
                alternative=alternative,
                min_aligned_trials=min_aligned_trials,
                random_state=random_state,
                subject_plot_dir=subject_plot_dir,
                checkpoint_file=subject_audit_file,
                overwrite_checkpoint=effective_overwrite,
                max_electrodes_per_subject=max_electrodes_per_subject,
                save_non_sig_plots=save_non_sig_plots,
            )
        except Exception as exc:
            audit_df = pd.DataFrame([{
                'subject': subject,
                'event': event,
                'region': np.nan,
                'electrode': np.nan,
                'hemi': np.nan,
                'phase_roi': np.nan,
                'amp_roi': np.nan,
                'phase_channel': np.nan,
                'amp_channel': np.nan,
                'unit_key': np.nan,
                'pac_file': np.nan,
                'predictor': predictor,
                'covariates': ",".join(covariates),
                'status': 'error',
                'reason': f'{type(exc).__name__}: {exc}',
                'n_aligned_trials': np.nan,
                'n_clusters': 0,
                'n_sig_clusters': 0,
                'min_sig_p': np.nan,
                'max_sig_abs_mass': np.nan,
                'df_resid': np.nan,
                'tcrit': np.nan,
                'plot_path': '',
                'elapsed_sec': np.nan,
            }])
            sig_df = pd.DataFrame()

        audit_df.to_csv(subject_audit_file, index=False)
        sig_df.to_csv(subject_sig_file, index=False)

    sig_frames: List[pd.DataFrame] = []
    audit_frames: List[pd.DataFrame] = []
    for subject in subjects:
        sig_file = os.path.join(checkpoints_dir, f'{subject}_within_sig.csv')
        audit_file = os.path.join(checkpoints_dir, f'{subject}_within_audit.csv')
        if os.path.exists(sig_file):
            try:
                sig_frames.append(pd.read_csv(sig_file))
            except Exception:
                pass
        if os.path.exists(audit_file):
            try:
                audit_frames.append(pd.read_csv(audit_file))
            except Exception:
                pass

    if len(sig_frames) > 0:
        sig_electrodes = pd.concat(sig_frames, ignore_index=True)
        dedupe_cols = ['subject', 'event', 'region', 'electrode', 'hemi']
        if all(c in sig_electrodes.columns for c in dedupe_cols):
            sig_electrodes['electrode'] = sig_electrodes['electrode'].astype(str)
            sig_electrodes = sig_electrodes.drop_duplicates(
                subset=dedupe_cols,
                keep='last',
            ).reset_index(drop=True)
    else:
        sig_electrodes = pd.DataFrame(columns=[
            'subject', 'event', 'region', 'electrode', 'hemi',
            'predictor', 'covariates', 'n_aligned_trials', 'n_clusters', 'n_sig_clusters',
            'min_sig_p', 'max_sig_abs_mass', 'df_resid', 'tcrit', 'plot_path',
        ])

    if len(audit_frames) > 0:
        audit_df_all = pd.concat(audit_frames, ignore_index=True)
        if 'unit_key' in audit_df_all.columns:
            audit_df_all['unit_key'] = audit_df_all['unit_key'].astype(str)
            audit_df_all = audit_df_all.drop_duplicates(subset=['unit_key'], keep='last').reset_index(drop=True)
    else:
        audit_df_all = pd.DataFrame(columns=[
            'subject', 'event', 'region', 'electrode', 'hemi',
            'phase_roi', 'amp_roi', 'phase_channel', 'amp_channel',
            'unit_key', 'pac_file', 'predictor', 'covariates', 'status', 'reason',
            'n_aligned_trials', 'n_clusters', 'n_sig_clusters', 'min_sig_p', 'max_sig_abs_mass',
            'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
        ])

    if len(sig_electrodes) > 0:
        sig_units = sig_electrodes[['subject', 'region', 'electrode']].copy()
        sig_units['electrode'] = sig_units['electrode'].astype(str)
        sig_units = sig_units.drop_duplicates(subset=['subject', 'region', 'electrode'], keep='last')
        region_counts = (
            sig_units
            .groupby('region', dropna=False)
            .agg(
                n_sig_electrodes=('electrode', 'count'),
                n_subjects_with_sig=('subject', 'nunique'),
            )
            .reset_index()
            .sort_values('n_sig_electrodes', ascending=False)
            .reset_index(drop=True)
        )
    else:
        sig_units = pd.DataFrame(columns=['subject', 'region', 'electrode'])
        region_counts = pd.DataFrame(columns=['region', 'n_sig_electrodes', 'n_subjects_with_sig'])

    sig_out = os.path.join(run_dir, 'sig_within_pac_electrodes.csv')
    audit_out = os.path.join(run_dir, 'all_within_pac_audit.csv')
    region_out = os.path.join(run_dir, 'region_counts_within.csv')
    meta_out = os.path.join(run_dir, 'run_metadata_within.json')

    sig_electrodes.to_csv(sig_out, index=False)
    audit_df_all.to_csv(audit_out, index=False)
    region_counts.to_csv(region_out, index=False)

    failures = int((audit_df_all.get('status', pd.Series(dtype=str)) == 'error').sum()) if len(audit_df_all) > 0 else 0
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'base_dir': base_dir,
        'load_dir': load_dir,
        'pac_dir': pac_dir,
        'run_dir': run_dir,
        'subjects': subjects,
        'event': event,
        'predictor': predictor,
        'covariates': covariates,
        'n_permutations': int(n_permutations),
        'alpha_cluster': float(alpha_cluster),
        'alpha_test': float(alpha_test),
        'alternative': alternative,
        'min_aligned_trials': int(min_aligned_trials),
        'random_state': int(random_state),
        'overwrite': bool(overwrite),
        'resume': bool(resume),
        'max_subjects': None if max_subjects is None else int(max_subjects),
        'max_electrodes_per_subject': (
            None if max_electrodes_per_subject is None else int(max_electrodes_per_subject)
        ),
        'save_non_sig_plots': bool(save_non_sig_plots),
        'n_subjects': int(len(subjects)),
        'n_attempted_units': int(len(audit_df_all)),
        'n_sig_electrodes': int(len(sig_electrodes)),
        'n_failures': failures,
        'sig_within_pac_electrodes_csv': sig_out,
        'all_within_pac_audit_csv': audit_out,
        'region_counts_within_csv': region_out,
    }
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'sig_electrodes': sig_electrodes,
        'region_counts': region_counts,
        'audit': audit_df_all,
        'sig_units': sig_electrodes,
    }


def run_batch_tfr_cluster_glm_pipeline(
    subjects: Optional[List[str]],
    base_dir: str,
    load_dir: str,
    run_dir: str,
    event: str = 'feedback_start',
    predictor: str = 'zrpe',
    covariates: Optional[List[str]] = None,
    n_permutations: int = 200,
    alpha_cluster: float = 0.05,
    alpha_test: float = 0.05,
    alternative: str = 'two-sided',
    min_aligned_trials: int = 20,
    random_state: int = 42,
    overwrite: bool = False,
    resume: bool = True,
    max_subjects: Optional[int] = None,
    max_electrodes_per_subject: Optional[int] = None,
    save_non_sig_plots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Sequential, checkpointed all-subject/all-electrode TFR cluster GLM pipeline.
    """
    from datetime import datetime, timezone

    covariates = list(covariates or ['zpm'])
    if subjects is None:
        subjects = _default_filtered_subjects(base_dir)
    subjects = [str(s) for s in subjects]
    if max_subjects is not None:
        subjects = subjects[:max(0, int(max_subjects))]

    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for subject in subjects:
        subject_plot_dir = os.path.join(plots_dir, subject)
        subject_audit_file = os.path.join(checkpoints_dir, f'{subject}_audit.csv')
        subject_sig_file = os.path.join(checkpoints_dir, f'{subject}_sig.csv')
        effective_overwrite = bool(overwrite or (not resume))

        if effective_overwrite:
            if os.path.exists(subject_audit_file):
                os.remove(subject_audit_file)
            if os.path.exists(subject_sig_file):
                os.remove(subject_sig_file)

        sig_df, audit_df = process_subject_tfr_cluster_glm(
            subject=subject,
            base_dir=base_dir,
            load_dir=load_dir,
            event=event,
            predictor=predictor,
            covariates=covariates,
            n_permutations=n_permutations,
            alpha_cluster=alpha_cluster,
            alpha_test=alpha_test,
            alternative=alternative,
            min_aligned_trials=min_aligned_trials,
            random_state=random_state,
            subject_plot_dir=subject_plot_dir,
            checkpoint_file=subject_audit_file,
            overwrite_checkpoint=effective_overwrite,
            max_electrodes_per_subject=max_electrodes_per_subject,
            save_non_sig_plots=save_non_sig_plots,
        )
        audit_df.to_csv(subject_audit_file, index=False)
        sig_df.to_csv(subject_sig_file, index=False)

    sig_frames = []
    audit_frames = []
    for subject in subjects:
        sig_file = os.path.join(checkpoints_dir, f'{subject}_sig.csv')
        audit_file = os.path.join(checkpoints_dir, f'{subject}_audit.csv')
        if os.path.exists(sig_file):
            try:
                sig_frames.append(pd.read_csv(sig_file))
            except Exception:
                pass
        if os.path.exists(audit_file):
            try:
                audit_frames.append(pd.read_csv(audit_file))
            except Exception:
                pass

    if len(sig_frames) > 0:
        sig_electrodes = pd.concat(sig_frames, ignore_index=True)
        if len(sig_electrodes) > 0:
            sig_electrodes['electrode'] = sig_electrodes['electrode'].astype(str)
            sig_electrodes = sig_electrodes.drop_duplicates(
                subset=['subject', 'electrode'],
                keep='last',
            ).reset_index(drop=True)
    else:
        sig_electrodes = pd.DataFrame(columns=[
            'subject', 'electrode', 'region', 'event', 'predictor', 'covariates',
            'n_aligned_trials', 'n_clusters', 'n_sig_clusters', 'min_sig_p',
            'max_sig_abs_mass', 'df_resid', 'tcrit', 'plot_path',
        ])

    if len(audit_frames) > 0:
        audit_df_all = pd.concat(audit_frames, ignore_index=True)
        if len(audit_df_all) > 0:
            audit_df_all['electrode'] = audit_df_all['electrode'].astype(str)
            audit_df_all = audit_df_all.drop_duplicates(
                subset=['subject', 'electrode'],
                keep='last',
            ).reset_index(drop=True)
    else:
        audit_df_all = pd.DataFrame(columns=[
            'subject', 'electrode', 'region', 'event', 'predictor', 'covariates',
            'status', 'reason', 'n_aligned_trials', 'n_clusters', 'n_sig_clusters',
            'min_sig_p', 'max_sig_abs_mass', 'df_resid', 'tcrit', 'plot_path', 'elapsed_sec',
        ])

    if len(sig_electrodes) > 0:
        unique_sig = sig_electrodes[['subject', 'electrode', 'region']].drop_duplicates()
        region_counts = (
            unique_sig
            .groupby('region', dropna=False)
            .agg(
                n_sig_electrodes=('electrode', 'count'),
                n_subjects_with_sig=('subject', 'nunique'),
            )
            .reset_index()
            .sort_values('n_sig_electrodes', ascending=False)
            .reset_index(drop=True)
        )
    else:
        region_counts = pd.DataFrame(columns=['region', 'n_sig_electrodes', 'n_subjects_with_sig'])

    sig_out = os.path.join(run_dir, 'sig_electrodes.csv')
    region_out = os.path.join(run_dir, 'region_counts.csv')
    audit_out = os.path.join(run_dir, 'all_audit.csv')
    meta_out = os.path.join(run_dir, 'run_metadata.json')

    sig_electrodes.to_csv(sig_out, index=False)
    region_counts.to_csv(region_out, index=False)
    audit_df_all.to_csv(audit_out, index=False)

    failures = int((audit_df_all.get('status', pd.Series(dtype=str)) == 'error').sum()) if len(audit_df_all) > 0 else 0
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'base_dir': base_dir,
        'load_dir': load_dir,
        'run_dir': run_dir,
        'subjects': subjects,
        'event': event,
        'predictor': predictor,
        'covariates': covariates,
        'n_permutations': int(n_permutations),
        'alpha_cluster': float(alpha_cluster),
        'alpha_test': float(alpha_test),
        'alternative': alternative,
        'min_aligned_trials': int(min_aligned_trials),
        'random_state': int(random_state),
        'overwrite': bool(overwrite),
        'resume': bool(resume),
        'max_subjects': None if max_subjects is None else int(max_subjects),
        'max_electrodes_per_subject': (
            None if max_electrodes_per_subject is None else int(max_electrodes_per_subject)
        ),
        'save_non_sig_plots': bool(save_non_sig_plots),
        'n_subjects': int(len(subjects)),
        'n_attempted_electrodes': int(len(audit_df_all)),
        'n_sig_electrodes': int(len(sig_electrodes)),
        'n_failures': failures,
        'sig_electrodes_csv': sig_out,
        'region_counts_csv': region_out,
        'audit_csv': audit_out,
    }
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'sig_electrodes': sig_electrodes,
        'region_counts': region_counts,
        'audit': audit_df_all,
    }


def fit_residualizer(
    C_train: Optional[np.ndarray],
    y_train: np.ndarray
) -> np.ndarray:
    """
    Fit OLS residualizer coefficients for covariate-adjusted predictor residuals.

    Returns beta coefficients for:
        y_train ~= C_train_design @ beta
    where C_train_design contains an intercept column if not already present.
    """
    y_arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if C_train is None:
        return np.array([], dtype=np.float64)

    C_arr = np.asarray(C_train, dtype=np.float64)
    if C_arr.ndim == 1:
        C_arr = C_arr[:, np.newaxis]
    if C_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"C_train rows ({C_arr.shape[0]}) must match y_train rows ({y_arr.shape[0]})"
        )
    if C_arr.size == 0:
        return np.array([], dtype=np.float64)

    valid = np.isfinite(y_arr) & np.all(np.isfinite(C_arr), axis=1)
    if not np.any(valid):
        return np.array([], dtype=np.float64)

    C_valid = C_arr[valid]
    y_valid = y_arr[valid]

    has_intercept = np.any(np.all(np.isclose(C_valid, 1.0, atol=1e-8), axis=0))
    if not has_intercept:
        C_valid = np.column_stack([np.ones(C_valid.shape[0], dtype=np.float64), C_valid])

    beta_cov = np.linalg.pinv(C_valid) @ y_valid
    return np.asarray(beta_cov, dtype=np.float64)


def apply_residualizer(
    C: Optional[np.ndarray],
    y: np.ndarray,
    beta_cov: np.ndarray
) -> np.ndarray:
    """
    Apply pre-fit covariate residualizer to y and return aligned residuals.

    Invalid rows (non-finite y/C) are returned as NaN.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    y_resid = np.full(y_arr.shape, np.nan, dtype=np.float64)
    beta = np.asarray(beta_cov, dtype=np.float64).reshape(-1)

    if C is None or beta.size == 0:
        valid = np.isfinite(y_arr)
        y_resid[valid] = y_arr[valid]
        return y_resid

    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.ndim == 1:
        C_arr = C_arr[:, np.newaxis]
    if C_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"C rows ({C_arr.shape[0]}) must match y rows ({y_arr.shape[0]})"
        )

    valid = np.isfinite(y_arr) & np.all(np.isfinite(C_arr), axis=1)
    if not np.any(valid):
        return y_resid

    C_valid = C_arr[valid]
    y_valid = y_arr[valid]

    if beta.size == C_valid.shape[1] + 1:
        C_design = np.column_stack([np.ones(C_valid.shape[0], dtype=np.float64), C_valid])
    elif beta.size == C_valid.shape[1]:
        C_design = C_valid
    else:
        raise ValueError(
            f"beta_cov length ({beta.size}) incompatible with covariate columns ({C_valid.shape[1]})"
        )

    y_fit = C_design @ beta
    y_resid[valid] = y_valid - y_fit
    return y_resid


def residualize_predictor(
    y: np.ndarray,
    C: Optional[np.ndarray]
) -> np.ndarray:
    """
    Residualize predictor against covariates via OLS (with intercept handling).
    """
    beta_cov = fit_residualizer(C, y)
    return apply_residualizer(C, y, beta_cov)


def _as_2d_float64(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Convert array-like to float64 2D matrix; return matrix and vector flag."""
    arr_np = np.asarray(arr, dtype=np.float64)
    was_vector = (arr_np.ndim == 1)
    if was_vector:
        arr_np = arr_np[:, np.newaxis]
    if arr_np.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got shape {arr_np.shape}")
    return arr_np, was_vector


def _fit_ols_reduced_components(
    y: np.ndarray,
    X_reduced: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit reduced OLS model and return fitted values and residuals.

    `X_reduced` is treated as the complete reduced design matrix (no implicit
    intercept is added). If it has zero columns, fitted values are zeros.
    """
    y_mat, y_was_vector = _as_2d_float64(y)
    n_rows = y_mat.shape[0]

    if X_reduced is None:
        X_red = np.empty((n_rows, 0), dtype=np.float64)
    else:
        X_red = np.asarray(X_reduced, dtype=np.float64)
        if X_red.ndim == 1:
            X_red = X_red[:, np.newaxis]
        if X_red.ndim != 2:
            raise ValueError(f"Reduced design must be 2D, got shape {X_red.shape}")
        if X_red.shape[0] != n_rows:
            raise ValueError(
                f"Reduced design rows ({X_red.shape[0]}) must match y rows ({n_rows})"
            )

    if X_red.shape[1] == 0:
        y_hat = np.zeros_like(y_mat)
    else:
        y_hat = X_red @ (np.linalg.pinv(X_red) @ y_mat)
    resid = y_mat - y_hat

    if y_was_vector:
        return y_hat[:, 0], resid[:, 0]
    return y_hat, resid


def _prepare_freedman_lane_state(
    y: np.ndarray,
    nuisance: Optional[np.ndarray],
    include_intercept: bool = True
) -> Dict[str, np.ndarray]:
    """
    Prepare reduced-model components for Freedman-Lane permutations.

    Reduced model:
        y ~ [intercept] + nuisance
    """
    y_mat, _ = _as_2d_float64(y)
    n_rows = y_mat.shape[0]

    if nuisance is None:
        Z = np.empty((n_rows, 0), dtype=np.float64)
    else:
        Z = np.asarray(nuisance, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, np.newaxis]
        if Z.ndim != 2:
            raise ValueError(f"Nuisance design must be 2D, got shape {Z.shape}")
        if Z.shape[0] != n_rows:
            raise ValueError(
                f"Nuisance rows ({Z.shape[0]}) must match y rows ({n_rows})"
            )

    if include_intercept:
        has_intercept = (
            Z.shape[1] > 0 and np.any(np.all(np.isclose(Z, 1.0, atol=1e-8), axis=0))
        )
        if not has_intercept:
            Z = np.column_stack([np.ones((n_rows, 1), dtype=np.float64), Z])

    y_hat0, resid0 = _fit_ols_reduced_components(y_mat, Z)
    pinv_Z = np.linalg.pinv(Z) if Z.shape[1] > 0 else np.empty((0, n_rows), dtype=np.float64)
    return {
        'y_hat0': np.asarray(y_hat0, dtype=np.float64),
        'resid0': np.asarray(resid0, dtype=np.float64),
        'Z': np.asarray(Z, dtype=np.float64),
        'pinv_Z': pinv_Z,
    }


def _freedman_lane_pseudo_response(
    y_hat0: np.ndarray,
    resid0: np.ndarray,
    perm_ix: np.ndarray
) -> np.ndarray:
    """Construct Freedman-Lane pseudo-response y* = y_hat0 + P e."""
    y_hat0_mat, yhat_was_vector = _as_2d_float64(y_hat0)
    resid0_mat, resid_was_vector = _as_2d_float64(resid0)
    if y_hat0_mat.shape != resid0_mat.shape:
        raise ValueError(
            f"y_hat0 shape {y_hat0_mat.shape} must match resid0 shape {resid0_mat.shape}"
        )
    perm_ix_arr = np.asarray(perm_ix, dtype=np.int64).reshape(-1)
    if perm_ix_arr.size != y_hat0_mat.shape[0]:
        raise ValueError(
            f"Permutation index length ({perm_ix_arr.size}) must match row count ({y_hat0_mat.shape[0]})"
        )
    y_star = y_hat0_mat + resid0_mat[perm_ix_arr]
    if yhat_was_vector and resid_was_vector:
        return y_star[:, 0]
    return y_star


def _residualize_with_precomputed_design(
    y: np.ndarray,
    Z: np.ndarray,
    pinv_Z: Optional[np.ndarray] = None
) -> np.ndarray:
    """Residualize y against a fixed design matrix with optional precomputed pseudo-inverse."""
    y_mat, y_was_vector = _as_2d_float64(y)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr[:, np.newaxis]
    if Z_arr.shape[0] != y_mat.shape[0]:
        raise ValueError(
            f"Design rows ({Z_arr.shape[0]}) must match y rows ({y_mat.shape[0]})"
        )
    if Z_arr.shape[1] == 0:
        y_resid = y_mat.copy()
    else:
        pinv = np.asarray(pinv_Z, dtype=np.float64) if pinv_Z is not None else np.linalg.pinv(Z_arr)
        y_resid = y_mat - (Z_arr @ (pinv @ y_mat))
    if y_was_vector:
        return y_resid[:, 0]
    return y_resid


def _resolve_ridge_alpha_for_key(
    ridge_alpha: float,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]],
    subject: str,
    roi: str,
    predictor_key: str,
) -> float:
    """Resolve per-subject ROI predictor alpha with scalar fallback."""
    alpha_val = ridge_alpha
    if ridge_alpha_map is not None:
        key = (str(subject), str(roi), str(predictor_key))
        if key in ridge_alpha_map:
            alpha_val = ridge_alpha_map[key]
    try:
        alpha_val = float(alpha_val)
    except (TypeError, ValueError):
        alpha_val = float(ridge_alpha)
    if (not np.isfinite(alpha_val)) or (alpha_val <= 0):
        alpha_val = float(ridge_alpha)
    return alpha_val


def _compute_timepoint_holdout_r(
    X_train_3d: np.ndarray,
    y_train: np.ndarray,
    X_test_3d: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute mean Pearson r across timepoints for holdout evaluation.
    """
    X_tr = np.asarray(X_train_3d, dtype=np.float64)
    X_te = np.asarray(X_test_3d, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_te = np.asarray(y_test, dtype=np.float64).reshape(-1)

    if X_tr.ndim != 3 or X_te.ndim != 3:
        return np.nan
    if X_tr.shape[2] != X_te.shape[2]:
        return np.nan

    n_t = X_tr.shape[2]
    per_t_scores: List[float] = []
    for t_idx in range(n_t):
        Xt_tr = X_tr[:, :, t_idx]
        Xt_te = X_te[:, :, t_idx]

        valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(Xt_tr), axis=1)
        valid_te = np.isfinite(y_te) & np.all(np.isfinite(Xt_te), axis=1)

        if np.sum(valid_tr) < 3 or np.sum(valid_te) < 3:
            continue
        if np.std(y_tr[valid_tr]) < 1e-12:
            continue

        model = Ridge(alpha=alpha)
        model.fit(Xt_tr[valid_tr], y_tr[valid_tr])
        y_hat = model.predict(Xt_te[valid_te])
        y_true = y_te[valid_te]

        if np.std(y_hat) < 1e-12 or np.std(y_true) < 1e-12:
            continue
        per_t_scores.append(float(np.corrcoef(y_true, y_hat)[0, 1]))

    if len(per_t_scores) == 0:
        return np.nan
    return float(np.mean(per_t_scores))


def _compute_cv_pearson_r(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    splits: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Compute out-of-fold Pearson correlation for ridge regression.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        return np.nan
    if X_arr.shape[0] < 3:
        return np.nan

    y_true_parts = []
    y_pred_parts = []
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        try:
            model = Ridge(alpha=alpha)
            model.fit(X_arr[train_idx], y_arr[train_idx])
            y_pred = model.predict(X_arr[test_idx])
        except Exception:
            return np.nan
        y_true_parts.append(y_arr[test_idx])
        y_pred_parts.append(y_pred)

    if len(y_true_parts) == 0:
        return np.nan
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    if y_true.size < 3:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _compute_insample_r2(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute in-sample Ridge R^2 on finite rows.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        return np.nan
    if X_arr.shape[0] != y_arr.shape[0]:
        return np.nan
    valid = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
    if np.sum(valid) < 3:
        return np.nan
    y_valid = y_arr[valid]
    if np.std(y_valid) < 1e-12:
        return np.nan
    try:
        model = Ridge(alpha=alpha)
        model.fit(X_arr[valid], y_valid)
        y_pred = model.predict(X_arr[valid])
        return float(r2_score(y_valid, y_pred))
    except Exception:
        return np.nan


def _encode_multiclass_target(
    y_raw: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode arbitrary categorical labels to integer class ids.

    Returns
    -------
    y_encoded : np.ndarray, shape (n_trials,)
        Encoded class ids, with -1 for invalid/NaN labels.
    class_labels : List[str]
        Class label names ordered by assigned integer id.
    """
    y_obj = np.asarray(y_raw, dtype=object).reshape(-1)
    valid = ~pd.isna(y_obj)
    labels_valid = [str(v).strip() for v in y_obj[valid]]
    labels_valid = [v for v in labels_valid if len(v) > 0 and v.lower() != 'nan']
    class_labels = list(pd.unique(np.asarray(labels_valid, dtype=object)))
    mapping = {lbl: idx for idx, lbl in enumerate(class_labels)}

    y_encoded = np.full(y_obj.shape[0], -1, dtype=np.int64)
    for i, v in enumerate(y_obj):
        if pd.isna(v):
            continue
        key = str(v).strip()
        if len(key) == 0 or key.lower() == 'nan':
            continue
        if key in mapping:
            y_encoded[i] = mapping[key]
    return y_encoded, [str(x) for x in class_labels]


def _residualize_X_by_covariates_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    C_train: Optional[np.ndarray],
    C_test: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Residualize neural features with respect to nuisance covariates using
    train-only fit (applied to both train and test).
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    if C_train is None or C_test is None:
        return X_tr, X_te

    Ctr = np.asarray(C_train, dtype=np.float64)
    Cte = np.asarray(C_test, dtype=np.float64)
    if Ctr.ndim == 1:
        Ctr = Ctr[:, np.newaxis]
    if Cte.ndim == 1:
        Cte = Cte[:, np.newaxis]

    Ztr = np.column_stack([np.ones(Ctr.shape[0], dtype=np.float64), Ctr])
    Zte = np.column_stack([np.ones(Cte.shape[0], dtype=np.float64), Cte])
    pinv_Ztr = np.linalg.pinv(Ztr)
    beta = pinv_Ztr @ X_tr
    X_tr_resid = X_tr - (Ztr @ beta)
    X_te_resid = X_te - (Zte @ beta)
    return X_tr_resid, X_te_resid


def _compute_cv_balanced_accuracy_with_cov_resid(
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[np.ndarray],
    alpha: float,
    seed: int,
    n_folds: int,
    classifier_model: str = 'ridge_classifier',
) -> float:
    """
    Compute stratified CV balanced accuracy with foldwise train-only covariate
    residualization of X.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    C_arr = np.asarray(C, dtype=np.float64) if C is not None else None

    if X_arr.ndim != 2 or y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        return np.nan

    valid = (y_arr >= 0) & np.all(np.isfinite(X_arr), axis=1)
    if C_arr is not None:
        if C_arr.ndim == 1:
            C_arr = C_arr[:, np.newaxis]
        valid = valid & np.all(np.isfinite(C_arr), axis=1)

    Xv = X_arr[valid]
    yv = y_arr[valid]
    Cv = C_arr[valid] if C_arr is not None else None
    if Xv.shape[0] < 4:
        return np.nan

    classes, counts = np.unique(yv, return_counts=True)
    if classes.size < 2:
        return np.nan
    max_splits = int(np.min(counts))
    n_splits = min(int(max(2, n_folds)), max_splits)
    if n_splits < 2:
        return np.nan

    if classifier_model != 'ridge_classifier':
        raise ValueError(f"Unsupported classifier_model: {classifier_model}")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    scores: List[float] = []
    for tr_idx, te_idx in splitter.split(Xv, yv):
        Xtr, Xte = Xv[tr_idx], Xv[te_idx]
        ytr, yte = yv[tr_idx], yv[te_idx]
        Ctr = Cv[tr_idx] if Cv is not None else None
        Cte = Cv[te_idx] if Cv is not None else None
        Xtr_res, Xte_res = _residualize_X_by_covariates_train_test(Xtr, Xte, Ctr, Cte)
        clf = RidgeClassifier(alpha=float(alpha))
        clf.fit(Xtr_res, ytr)
        yhat = clf.predict(Xte_res)
        scores.append(float(balanced_accuracy_score(yte, yhat)))

    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))


def tune_alpha_pooled_subject_roi_predictor(
    subject_obj: Subject,
    roi: str,
    predictor: str,
    event: str = 'feedback_start',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    alpha_grid: Optional[np.ndarray] = None,
    seed: int = 42,
    min_trials: int = 10,
    default_ridge_alpha: float = 1.0,
    run_outer_eval: bool = True,
    outer_eval_folds: int = 3,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tune one alpha per subject/ROI/predictor pooled across encoding timepoints.

    Selection uses RidgeCV over pooled trial x timepoint samples.
    Optional outer evaluation computes holdout performance as mean timepoint-wise
    Pearson r with train-only residualization.
    """
    covariates = covariates or ['zpm']
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 13)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).reshape(-1)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid) & (alpha_grid > 0)]
    if alpha_grid.size == 0:
        alpha_grid = np.array([default_ridge_alpha], dtype=np.float64)
    alpha_grid = np.unique(alpha_grid)

    result: Dict[str, Any] = {
        'subj': subject_obj.subject,
        'region': roi,
        'predictor': predictor,
        'event': event,
        'selected_alpha': float(default_ridge_alpha),
        'default_alpha_used': True,
        'status': 'ok',
        'n_trials': 0,
        'n_electrodes': 0,
        'outer_score_mean': np.nan,
        'outer_score_std': np.nan,
        'alpha_grid_json': json.dumps(alpha_grid.astype(float).tolist()),
        'seed': int(seed),
        'selector': 'ridgecv_gcv',
        'run_outer_eval': bool(run_outer_eval),
        'outer_eval_folds': int(outer_eval_folds),
    }

    try:
        signal, _, _ = _get_smoothed_event_signal(
            subject_obj=subject_obj,
            roi=roi,
            event=event,
            freq_range=freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms
        )
    except Exception:
        result['status'] = 'no_roi_electrodes'
        return result

    try:
        regressors, _ = _build_subject_regressors(
            subject_obj=subject_obj,
            event=event,
            predictor=predictor,
            covariates=covariates,
            regressor_source=regressor_source,
            trial_var_override=trial_var_override,
        )
    except Exception:
        result['status'] = 'missing_regressors'
        return result

    trial_ix = regressors['trial'].to_numpy(dtype=np.int64) - 1
    in_range = (trial_ix >= 0) & (trial_ix < signal.shape[0])
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]

    if trial_ix.size == 0:
        result['status'] = 'insufficient_trials'
        return result

    X_all = np.asarray(signal[trial_ix], dtype=np.float64)  # trials x elec x t
    pred_series = regressors[predictor]
    if not pd.api.types.is_numeric_dtype(pred_series):
        result['status'] = 'unsupported_target_type_for_alpha_tuning'
        return result
    y_raw = pred_series.to_numpy(dtype=np.float64)
    cov_cols = [c for c in covariates if c in regressors.columns and c != predictor]
    C_all = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None
    y_resid = residualize_predictor(y_raw, C_all)

    result['n_trials'] = int(X_all.shape[0])
    result['n_electrodes'] = int(X_all.shape[1])

    valid_trials = np.isfinite(y_resid)
    if np.sum(valid_trials) < max(min_trials, 3):
        result['status'] = 'insufficient_trials'
        return result
    if np.std(y_resid[valid_trials]) < 1e-12:
        result['status'] = 'constant_predictor'
        return result

    X_sel = X_all[valid_trials]  # n_tv x elec x t
    y_sel = y_resid[valid_trials]
    n_t = X_sel.shape[2]

    X_pooled = np.transpose(X_sel, (0, 2, 1)).reshape(-1, X_sel.shape[1])
    y_pooled = np.repeat(y_sel, n_t)
    pooled_valid = np.isfinite(y_pooled) & np.all(np.isfinite(X_pooled), axis=1)
    if np.sum(pooled_valid) < max(min_trials, 3):
        result['status'] = 'insufficient_trials'
        return result
    X_pooled = X_pooled[pooled_valid]
    y_pooled = y_pooled[pooled_valid]

    try:
        selector = RidgeCV(alphas=alpha_grid)
        selector.fit(X_pooled, y_pooled)
        selected_alpha = float(selector.alpha_)
        if (not np.isfinite(selected_alpha)) or (selected_alpha <= 0):
            raise ValueError("Invalid alpha from RidgeCV")
    except Exception:
        result['status'] = 'selector_failed'
        result['selected_alpha'] = float(default_ridge_alpha)
        result['default_alpha_used'] = True
        return result

    result['selected_alpha'] = selected_alpha
    result['default_alpha_used'] = bool(np.isclose(selected_alpha, float(default_ridge_alpha)))

    if run_outer_eval:
        n_trials_valid = X_sel.shape[0]
        n_splits = min(max(2, outer_eval_folds), n_trials_valid)
        if n_splits >= 2:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = []
            y_raw_valid = y_raw[valid_trials]
            C_valid = C_all[valid_trials] if C_all is not None else None
            for tr_idx, te_idx in splitter.split(np.arange(n_trials_valid)):
                if len(tr_idx) < 3 or len(te_idx) < 3:
                    continue
                beta_cov = fit_residualizer(
                    C_valid[tr_idx] if C_valid is not None else None,
                    y_raw_valid[tr_idx]
                )
                y_tr_res = apply_residualizer(
                    C_valid[tr_idx] if C_valid is not None else None,
                    y_raw_valid[tr_idx],
                    beta_cov
                )
                y_te_res = apply_residualizer(
                    C_valid[te_idx] if C_valid is not None else None,
                    y_raw_valid[te_idx],
                    beta_cov
                )
                score = _compute_timepoint_holdout_r(
                    X_sel[tr_idx], y_tr_res, X_sel[te_idx], y_te_res, selected_alpha
                )
                if np.isfinite(score):
                    scores.append(float(score))
            if len(scores) > 0:
                result['outer_score_mean'] = float(np.mean(scores))
                result['outer_score_std'] = float(np.std(scores, ddof=0))
    return result


def run_pooled_alpha_tuning_pipeline(
    subjects: List[str],
    rois: List[str],
    predictors: List[str],
    base_dir: str,
    load_dir: str,
    event: str = 'feedback_start',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    alpha_grid: Optional[np.ndarray] = None,
    seed: int = 42,
    min_trials: int = 10,
    default_ridge_alpha: float = 1.0,
    run_outer_eval: bool = True,
    outer_eval_folds: int = 3,
    n_jobs_outer: int = 1,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch pooled alpha tuning for all subject x ROI x predictor tuples.
    """
    covariates = covariates or ['zpm']
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 13)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
    tasks = [(roi, predictor) for roi in rois for predictor in predictors]

    def _run_subject(subject: str) -> List[Dict[str, Any]]:
        rows = []
        try:
            subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
        except Exception:
            for roi, predictor in tasks:
                rows.append({
                    'subj': subject,
                    'region': roi,
                    'predictor': predictor,
                    'event': event,
                    'selected_alpha': float(default_ridge_alpha),
                    'default_alpha_used': True,
                    'status': 'subject_init_failed',
                    'n_trials': 0,
                    'n_electrodes': 0,
                    'outer_score_mean': np.nan,
                    'outer_score_std': np.nan,
                    'alpha_grid_json': json.dumps(alpha_grid.astype(float).tolist()),
                    'seed': int(seed),
                    'selector': 'ridgecv_gcv',
                    'run_outer_eval': bool(run_outer_eval),
                    'outer_eval_folds': int(outer_eval_folds),
                })
            return rows

        for roi, predictor in tasks:
            rows.append(
                tune_alpha_pooled_subject_roi_predictor(
                    subject_obj=subject_obj,
                    roi=roi,
                    predictor=predictor,
                    event=event,
                    covariates=covariates,
                    freq_range=freq_range,
                    downsample_factor=downsample_factor,
                    win_ms=win_ms,
                    alpha_grid=alpha_grid,
                    seed=seed,
                    min_trials=min_trials,
                    default_ridge_alpha=default_ridge_alpha,
                    run_outer_eval=run_outer_eval,
                    outer_eval_folds=outer_eval_folds,
                    regressor_source=regressor_source,
                    trial_var_override=trial_var_override,
                )
            )
        return rows

    if n_jobs_outer == 1:
        all_rows = [_run_subject(s) for s in subjects]
    else:
        all_rows = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
            delayed(_run_subject)(s) for s in subjects
        )

    flat_rows = [r for chunk in all_rows for r in chunk]
    df = pd.DataFrame(flat_rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=[
            'subj', 'region', 'predictor', 'event', 'selected_alpha', 'default_alpha_used',
            'status', 'n_trials', 'n_electrodes', 'outer_score_mean', 'outer_score_std',
            'alpha_grid_json', 'seed', 'selector', 'run_outer_eval', 'outer_eval_folds'
        ])
    return df


def tune_alpha_nested_cv_subject_roi(
    subject_obj: Subject,
    roi: str,
    predictor: str,
    event: str,
    covariates: Optional[List[str]],
    freq_range: Tuple[float, float],
    downsample_factor: int,
    win_ms: int,
    alpha_grid: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    outer_repeats: int,
    seed: int,
    min_trials: int,
    default_ridge_alpha: float = 1.0,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Legacy fallback: tune alpha for one subject/ROI/predictor using nested CV.
    """
    covariates = covariates or ['zpm']
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64).reshape(-1)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid) & (alpha_grid > 0)]
    if alpha_grid.size == 0:
        alpha_grid = np.array([default_ridge_alpha], dtype=np.float64)
    alpha_grid = np.unique(alpha_grid)

    result: Dict[str, Any] = {
        'subj': subject_obj.subject,
        'region': roi,
        'predictor': predictor,
        'event': event,
        'selected_alpha': float(default_ridge_alpha),
        'default_alpha_used': True,
        'status': 'ok',
        'n_trials': 0,
        'n_electrodes': 0,
        'outer_score_mean': np.nan,
        'outer_score_std': np.nan,
        'alpha_log10_iqr': np.nan,
        'alpha_best_per_outer_json': json.dumps([]),
        'alpha_grid_json': json.dumps(alpha_grid.astype(float).tolist()),
        'outer_folds': int(outer_folds),
        'inner_folds': int(inner_folds),
        'outer_repeats': int(outer_repeats),
        'seed': int(seed),
    }

    try:
        signal, _, _ = _get_smoothed_event_signal(
            subject_obj=subject_obj,
            roi=roi,
            event=event,
            freq_range=freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms
        )
    except Exception:
        result['status'] = 'no_roi_electrodes'
        return result

    try:
        regressors, _ = _build_subject_regressors(
            subject_obj=subject_obj,
            event=event,
            predictor=predictor,
            covariates=covariates,
            regressor_source=regressor_source,
            trial_var_override=trial_var_override,
        )
    except Exception:
        result['status'] = 'missing_regressors'
        return result

    trial_ix = regressors['trial'].to_numpy(dtype=np.int64) - 1
    in_range = (trial_ix >= 0) & (trial_ix < signal.shape[0])
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]
    if trial_ix.size == 0:
        result['status'] = 'insufficient_trials_for_nested_cv'
        return result

    X_all = np.asarray(signal[trial_ix], dtype=np.float64)
    y_raw = regressors[predictor].to_numpy(dtype=np.float64)
    cov_cols = [c for c in covariates if c in regressors.columns and c != predictor]
    C_all = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None

    n_trials = X_all.shape[0]
    n_electrodes = X_all.shape[1]
    result['n_trials'] = int(n_trials)
    result['n_electrodes'] = int(n_electrodes)

    y_resid_global = residualize_predictor(y_raw, C_all)
    valid_global = np.isfinite(y_resid_global)
    if np.sum(valid_global) < max(min_trials, 3):
        result['status'] = 'insufficient_trials_for_nested_cv'
        return result
    if np.std(y_resid_global[valid_global]) < 1e-12:
        result['status'] = 'constant_predictor'
        return result

    if n_trials < max(min_trials, outer_folds, 3):
        result['status'] = 'insufficient_trials_for_nested_cv'
        return result

    outer_best_alphas: List[float] = []
    outer_scores: List[float] = []

    for rep in range(max(1, outer_repeats)):
        n_outer = min(outer_folds, n_trials)
        if n_outer < 2:
            continue
        outer_splitter = KFold(
            n_splits=n_outer,
            shuffle=True,
            random_state=seed + rep
        )
        for train_idx, test_idx in outer_splitter.split(np.arange(n_trials)):
            if len(train_idx) < max(min_trials, 3) or len(test_idx) < 3:
                continue

            X_outer_train = X_all[train_idx]
            X_outer_test = X_all[test_idx]
            y_outer_train = y_raw[train_idx]
            y_outer_test = y_raw[test_idx]
            C_outer_train = C_all[train_idx] if C_all is not None else None
            C_outer_test = C_all[test_idx] if C_all is not None else None

            n_inner = min(inner_folds, len(train_idx))
            if n_inner < 2:
                continue

            alpha_scores: List[Tuple[float, float]] = []
            inner_splitter = KFold(
                n_splits=n_inner,
                shuffle=True,
                random_state=seed + 1000 + rep
            )
            for alpha in alpha_grid:
                fold_scores: List[float] = []
                for i_train_rel, i_val_rel in inner_splitter.split(np.arange(len(train_idx))):
                    X_in_train = X_outer_train[i_train_rel]
                    X_in_val = X_outer_train[i_val_rel]
                    y_in_train = y_outer_train[i_train_rel]
                    y_in_val = y_outer_train[i_val_rel]
                    C_in_train = C_outer_train[i_train_rel] if C_outer_train is not None else None
                    C_in_val = C_outer_train[i_val_rel] if C_outer_train is not None else None

                    beta_cov = fit_residualizer(C_in_train, y_in_train)
                    y_train_resid = apply_residualizer(C_in_train, y_in_train, beta_cov)
                    y_val_resid = apply_residualizer(C_in_val, y_in_val, beta_cov)

                    score = _compute_timepoint_holdout_r(
                        X_in_train, y_train_resid, X_in_val, y_val_resid, float(alpha)
                    )
                    if np.isfinite(score):
                        fold_scores.append(float(score))
                if len(fold_scores) > 0:
                    alpha_scores.append((float(alpha), float(np.mean(fold_scores))))

            if len(alpha_scores) == 0:
                continue

            score_values = np.array([s for _, s in alpha_scores], dtype=np.float64)
            max_score = np.nanmax(score_values)
            best_candidates = [a for a, s in alpha_scores if np.isfinite(s) and np.isclose(s, max_score)]
            if len(best_candidates) == 0:
                continue
            best_alpha_outer = float(np.max(best_candidates))

            beta_cov_outer = fit_residualizer(C_outer_train, y_outer_train)
            y_outer_train_resid = apply_residualizer(C_outer_train, y_outer_train, beta_cov_outer)
            y_outer_test_resid = apply_residualizer(C_outer_test, y_outer_test, beta_cov_outer)
            outer_score = _compute_timepoint_holdout_r(
                X_outer_train,
                y_outer_train_resid,
                X_outer_test,
                y_outer_test_resid,
                best_alpha_outer,
            )
            if np.isfinite(outer_score):
                outer_best_alphas.append(best_alpha_outer)
                outer_scores.append(float(outer_score))

    if len(outer_best_alphas) == 0:
        result['status'] = 'insufficient_trials_for_nested_cv'
        result['selected_alpha'] = float(default_ridge_alpha)
        result['default_alpha_used'] = True
        return result

    alpha_logs = np.log10(np.asarray(outer_best_alphas, dtype=np.float64))
    median_log = float(np.median(alpha_logs))
    snap_idx = int(np.argmin(np.abs(np.log10(alpha_grid) - median_log)))
    selected_alpha = float(alpha_grid[snap_idx])

    log_iqr = float(np.percentile(alpha_logs, 75) - np.percentile(alpha_logs, 25))
    unstable = (float(np.max(alpha_logs) - np.min(alpha_logs)) > 2.0)

    result['selected_alpha'] = selected_alpha
    result['default_alpha_used'] = bool(np.isclose(selected_alpha, float(default_ridge_alpha)))
    result['outer_score_mean'] = float(np.mean(outer_scores)) if len(outer_scores) > 0 else np.nan
    result['outer_score_std'] = float(np.std(outer_scores, ddof=0)) if len(outer_scores) > 0 else np.nan
    result['alpha_log10_iqr'] = log_iqr
    result['alpha_best_per_outer_json'] = json.dumps([float(x) for x in outer_best_alphas])
    if unstable:
        result['status'] = 'unstable_alpha'
    return result


def run_nested_alpha_tuning_pipeline(
    subjects: List[str],
    rois: List[str],
    predictors: List[str],
    base_dir: str,
    load_dir: str,
    event: str = 'feedback_start',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    alpha_grid: Optional[np.ndarray] = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    outer_repeats: int = 2,
    seed: int = 42,
    min_trials: int = 10,
    default_ridge_alpha: float = 1.0,
    n_jobs_outer: int = 1,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Legacy fallback: batch nested-CV alpha tuning.
    """
    covariates = covariates or ['zpm']
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 13)
    alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

    tasks = [(roi, predictor) for roi in rois for predictor in predictors]

    def _run_subject(subject: str) -> List[Dict[str, Any]]:
        subject_rows: List[Dict[str, Any]] = []
        try:
            subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
        except Exception:
            for roi, predictor in tasks:
                subject_rows.append({
                    'subj': subject,
                    'region': roi,
                    'predictor': predictor,
                    'event': event,
                    'selected_alpha': float(default_ridge_alpha),
                    'default_alpha_used': True,
                    'status': 'subject_init_failed',
                    'n_trials': 0,
                    'n_electrodes': 0,
                    'outer_score_mean': np.nan,
                    'outer_score_std': np.nan,
                    'alpha_log10_iqr': np.nan,
                    'alpha_best_per_outer_json': json.dumps([]),
                    'alpha_grid_json': json.dumps(alpha_grid.astype(float).tolist()),
                    'outer_folds': int(outer_folds),
                    'inner_folds': int(inner_folds),
                    'outer_repeats': int(outer_repeats),
                    'seed': int(seed),
                })
            return subject_rows

        for roi, predictor in tasks:
            row = tune_alpha_nested_cv_subject_roi(
                subject_obj=subject_obj,
                roi=roi,
                predictor=predictor,
                event=event,
                covariates=covariates,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
                alpha_grid=alpha_grid,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                outer_repeats=outer_repeats,
                seed=seed,
                min_trials=min_trials,
                default_ridge_alpha=default_ridge_alpha,
                regressor_source=regressor_source,
                trial_var_override=trial_var_override,
            )
            subject_rows.append(row)
        return subject_rows

    if n_jobs_outer == 1:
        all_rows = [_run_subject(subject) for subject in subjects]
    else:
        all_rows = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
            delayed(_run_subject)(subject) for subject in subjects
        )

    flat_rows: List[Dict[str, Any]] = [row for chunk in all_rows for row in chunk]
    df = pd.DataFrame(flat_rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=[
            'subj', 'region', 'predictor', 'event', 'selected_alpha', 'default_alpha_used',
            'status', 'n_trials', 'n_electrodes', 'outer_score_mean', 'outer_score_std',
            'alpha_log10_iqr', 'alpha_best_per_outer_json', 'alpha_grid_json',
            'outer_folds', 'inner_folds', 'outer_repeats', 'seed'
        ])
    return df


def process_subject_smoothed_multivariate_regression(
    subject: str,
    event: str,
    predictor: str,
    base_dir: str,
    load_dir: str,
    rois: List[str],
    freq_range: Tuple[float, float] = (70., 201.),
    win_ms: int = 200,
    downsample_factor: int = 5,
    n_permutations: int = 100,
    covariates: Optional[List[str]] = None,
    min_trials: int = 10,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    predictor_key: Optional[str] = None,
    cv_folds: int = 5,
    encoding_metric: str = 'insample_r2',
    null_tail: str = 'greater',
    store_metric_name: bool = True,
    use_surrogates: bool = True,
    surrogate_null: str = 'freedman_lane_residual_permutation',
    target_mode: str = 'auto',
    classifier_alpha: float = 1.0,
    classifier_model: str = 'ridge_classifier',
    verbose: bool = False,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Subject-level multivariate, time-resolved encoding regression.

    For continuous targets, fits ridge:
        y_resid ~ X_electrodes
    where y_resid is predictor residualized against nuisance covariates.

    For categorical targets (e.g., rpe_cat), computes time-resolved
    cross-validated balanced accuracy from neural features with foldwise
    train-only covariate residualization of X.

    Returns one row per ROI x timepoint with permutation z-score and p-value.
    """
    covariates = covariates or ['zpm']
    metric_name = str(encoding_metric).strip().lower()
    null_tail_name = str(null_tail).strip().lower()
    target_mode_name = str(target_mode).strip().lower()
    if metric_name not in {'insample_r2', 'cv_pearson_r', 'cv_balanced_accuracy'}:
        raise ValueError(f"Unsupported encoding_metric: {encoding_metric}")
    if null_tail_name not in {'greater', 'less', 'two_sided_abs'}:
        raise ValueError(f"Unsupported null_tail: {null_tail}")
    if target_mode_name not in {'auto', 'continuous', 'categorical'}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
    rows: List[Dict[str, Any]] = []

    for roi in rois:
        try:
            signal, times_sec, _ = _get_smoothed_event_signal(
                subject_obj=subject_obj,
                roi=roi,
                event=event,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms
            )
            regressors, _ = _build_subject_regressors(
                subject_obj=subject_obj,
                event=event,
                predictor=predictor,
                covariates=covariates,
                regressor_source=regressor_source,
                trial_var_override=trial_var_override,
            )
        except Exception as exc:
            if verbose:
                print(f"Skipping {subject}, ROI={roi}: {exc}")
            continue

        trial_ix = regressors['trial'].to_numpy(dtype=np.int64) - 1
        in_range = (trial_ix >= 0) & (trial_ix < signal.shape[0])
        dropped_alignment = int((~in_range).sum())
        regressors = regressors.loc[in_range].reset_index(drop=True)
        trial_ix = trial_ix[in_range]

        if trial_ix.size == 0:
            if verbose:
                print(f"Skipping {subject}, ROI={roi}: no aligned trials")
            continue

        pred_series = regressors[predictor]
        cov_cols = [c for c in covariates if c in regressors.columns and c != predictor]
        C = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None
        if target_mode_name == 'auto':
            is_categorical = (predictor == 'rpe_cat') or (not pd.api.types.is_numeric_dtype(pred_series))
            target_kind = 'categorical' if is_categorical else 'continuous'
        else:
            target_kind = target_mode_name

        class_labels: List[str] = []
        n_classes = np.nan
        y_obs_resid_core = None
        y_hat0_core = None
        resid0_core = None
        fl_state = None
        y_encoded = None

        if target_kind == 'continuous':
            y_raw = pred_series.to_numpy(dtype=np.float64)
            if C is None:
                core_valid = np.isfinite(y_raw)
            else:
                core_valid = np.isfinite(y_raw) & np.all(np.isfinite(C), axis=1)
            if np.sum(core_valid) < max(3, min_trials):
                if verbose:
                    print(f"Skipping {subject}, ROI={roi}: insufficient finite rows for nuisance model")
                continue
            C_core = C[core_valid] if C is not None else None
            fl_state = _prepare_freedman_lane_state(
                y=y_raw[core_valid],
                nuisance=C_core,
                include_intercept=True,
            )
            y_hat0_core = np.asarray(fl_state['y_hat0'], dtype=np.float64).reshape(-1)
            resid0_core = np.asarray(fl_state['resid0'], dtype=np.float64).reshape(-1)
            y_obs_resid_core = resid0_core.copy()
            if metric_name not in {'insample_r2', 'cv_pearson_r'}:
                raise ValueError(
                    f"encoding_metric='{metric_name}' is invalid for continuous target mode."
                )
            metric_name_effective = metric_name
            surrogate_null_name = 'freedman_lane_residual_permutation'
            alpha_for_roi = _resolve_ridge_alpha_for_key(
                ridge_alpha=ridge_alpha,
                ridge_alpha_map=ridge_alpha_map,
                subject=subject,
                roi=roi,
                predictor_key=(predictor_key or predictor),
            )
        else:
            y_encoded, class_labels = _encode_multiclass_target(pred_series.to_numpy(dtype=object))
            n_classes = int(len(class_labels))
            if C is None:
                core_valid = (y_encoded >= 0)
            else:
                core_valid = (y_encoded >= 0) & np.all(np.isfinite(C), axis=1)
            if np.sum(core_valid) < max(3, min_trials):
                if verbose:
                    print(f"Skipping {subject}, ROI={roi}: insufficient finite rows for categorical model")
                continue
            if n_classes < 2:
                if verbose:
                    print(f"Skipping {subject}, ROI={roi}: insufficient classes")
                continue
            metric_name_effective = 'cv_balanced_accuracy'
            surrogate_null_name = 'label_permutation'
            alpha_for_roi = float(classifier_alpha)
            if (not np.isfinite(alpha_for_roi)) or alpha_for_roi <= 0:
                alpha_for_roi = 1.0

        for t_idx, ts_sec in enumerate(times_sec):
            X_t = np.asarray(signal[trial_ix, :, t_idx], dtype=np.float64)
            valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
            n_valid = int(valid_t.sum())

            metric = np.nan
            z_metric = np.nan
            raw_p = np.nan
            status = 'ok'

            if n_valid < max(3, min_trials):
                status = 'insufficient_trials'
            else:
                try:
                    X_tv = X_t[valid_t]
                    if target_kind == 'continuous':
                        valid_t_core = valid_t[core_valid]
                        y_t = y_obs_resid_core[valid_t_core]
                        if np.std(y_t) < 1e-12:
                            status = 'constant_predictor'
                        else:
                            splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
                            if metric_name_effective == 'cv_pearson_r':
                                n_splits = min(cv_folds, n_valid)
                                if n_splits < 2:
                                    status = 'insufficient_folds'
                                else:
                                    splits = list(
                                        KFold(
                                            n_splits=n_splits,
                                            shuffle=True,
                                            random_state=seed
                                        ).split(np.arange(n_valid))
                                    )
                            if status == 'ok':
                                if metric_name_effective == 'insample_r2':
                                    metric = _compute_insample_r2(X_tv, y_t, alpha_for_roi)
                                else:
                                    metric = _compute_cv_pearson_r(X_tv, y_t, alpha_for_roi, splits or [])

                            if (status != 'ok') or (not np.isfinite(metric)):
                                status = 'failed' if status == 'ok' else status
                            elif not use_surrogates:
                                z_metric = np.nan
                                raw_p = np.nan
                            else:
                                rng = np.random.default_rng(seed + t_idx)
                                perm_metrics = np.full(n_permutations, np.nan, dtype=np.float64)
                                for p_idx in range(n_permutations):
                                    perm_ix = rng.permutation(resid0_core.shape[0])
                                    y_star_core = _freedman_lane_pseudo_response(
                                        y_hat0=y_hat0_core,
                                        resid0=resid0_core,
                                        perm_ix=perm_ix,
                                    )
                                    y_perm_core = _residualize_with_precomputed_design(
                                        y=y_star_core,
                                        Z=fl_state['Z'],
                                        pinv_Z=fl_state['pinv_Z'],
                                    )
                                    y_perm_t = np.asarray(y_perm_core, dtype=np.float64).reshape(-1)[valid_t_core]
                                    if metric_name_effective == 'insample_r2':
                                        perm_metrics[p_idx] = _compute_insample_r2(
                                            X_tv, y_perm_t, alpha_for_roi
                                        )
                                    else:
                                        perm_metrics[p_idx] = _compute_cv_pearson_r(
                                            X_tv, y_perm_t, alpha_for_roi, splits or []
                                        )
                    else:
                        y_t = np.asarray(y_encoded[valid_t], dtype=np.int64)
                        C_t = C[valid_t] if C is not None else None
                        classes_t, counts_t = np.unique(y_t, return_counts=True)
                        if classes_t.size < 2:
                            status = 'insufficient_classes'
                        else:
                            n_splits = min(int(cv_folds), int(np.min(counts_t)))
                            if n_splits < 2:
                                status = 'insufficient_folds'
                            else:
                                metric = _compute_cv_balanced_accuracy_with_cov_resid(
                                    X=X_tv,
                                    y=y_t,
                                    C=C_t,
                                    alpha=alpha_for_roi,
                                    seed=seed + t_idx,
                                    n_folds=n_splits,
                                    classifier_model=classifier_model,
                                )
                                if not np.isfinite(metric):
                                    status = 'failed'
                                elif not use_surrogates:
                                    z_metric = np.nan
                                    raw_p = np.nan
                                else:
                                    rng = np.random.default_rng(seed + t_idx)
                                    perm_metrics = np.full(n_permutations, np.nan, dtype=np.float64)
                                    for p_idx in range(n_permutations):
                                        y_perm = y_t[rng.permutation(y_t.shape[0])]
                                        perm_metrics[p_idx] = _compute_cv_balanced_accuracy_with_cov_resid(
                                            X=X_tv,
                                            y=y_perm,
                                            C=C_t,
                                            alpha=alpha_for_roi,
                                            seed=seed + t_idx,
                                            n_folds=n_splits,
                                            classifier_model=classifier_model,
                                        )

                    if status == 'ok' and use_surrogates:
                        perm_metrics = perm_metrics[np.isfinite(perm_metrics)]
                        if perm_metrics.size == 0:
                            status = 'failed'
                        else:
                            perm_mean = float(np.mean(perm_metrics))
                            perm_std = float(np.std(perm_metrics, ddof=0))
                            perm_std = max(perm_std, 1e-8)
                            z_metric = (float(metric) - perm_mean) / perm_std
                            if null_tail_name == 'greater':
                                raw_p = (
                                    np.sum(perm_metrics >= float(metric)) + 1.0
                                ) / (perm_metrics.size + 1.0)
                            elif null_tail_name == 'less':
                                raw_p = (
                                    np.sum(perm_metrics <= float(metric)) + 1.0
                                ) / (perm_metrics.size + 1.0)
                            else:
                                raw_p = (
                                    np.sum(np.abs(perm_metrics) >= np.abs(float(metric))) + 1.0
                                ) / (perm_metrics.size + 1.0)
                except Exception:
                    status = 'failed'

            rows.append({
                'predictor': predictor,
                'metric': metric,
                'z_metric': z_metric,
                'raw_p': raw_p,
                'ts': float(ts_sec * 1000.0),
                'subj': subject,
                'region': roi,
                'n_trials': n_valid,
                'n_electrodes': int(X_t.shape[1]),
                'ridge_alpha': float(alpha_for_roi),
                'cv_folds': int(min(cv_folds, max(n_valid, 1))),
                'event': event,
                'covariates': ",".join(cov_cols),
                'metric_name': metric_name_effective if store_metric_name else np.nan,
                'null_tail': null_tail_name if store_metric_name else np.nan,
                'surrogate_null': (
                    surrogate_null_name if (store_metric_name and use_surrogates) else (
                        'none' if store_metric_name else np.nan
                    )
                ),
                'use_surrogates': bool(use_surrogates),
                'target_kind': target_kind,
                'n_classes': int(n_classes) if target_kind == 'categorical' else np.nan,
                'class_labels_json': (
                    json.dumps([str(x) for x in class_labels])
                    if target_kind == 'categorical' else np.nan
                ),
                'status': status,
                'dropped_alignment': dropped_alignment,
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            'predictor', 'metric', 'z_metric', 'raw_p', 'ts', 'subj', 'region',
            'n_trials', 'n_electrodes', 'ridge_alpha', 'cv_folds', 'event',
            'covariates', 'metric_name', 'null_tail', 'surrogate_null',
            'use_surrogates', 'target_kind', 'n_classes', 'class_labels_json',
            'status', 'dropped_alignment'
        ])
    return pd.DataFrame(rows)


def run_smoothed_time_resolved_multivariate_regression_pipeline(
    subjects: List[str],
    event: str,
    predictor: str,
    base_dir: str,
    load_dir: str,
    rois: List[str],
    freq_range: Tuple[float, float] = (70., 201.),
    win_ms: int = 200,
    downsample_factor: int = 5,
    n_permutations: int = 100,
    covariates: Optional[List[str]] = None,
    min_trials: int = 10,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    predictor_key: Optional[str] = None,
    cv_folds: int = 5,
    encoding_metric: str = 'insample_r2',
    null_tail: str = 'greater',
    store_metric_name: bool = True,
    use_surrogates: bool = True,
    surrogate_null: str = 'freedman_lane_residual_permutation',
    target_mode: str = 'auto',
    classifier_alpha: float = 1.0,
    classifier_model: str = 'ridge_classifier',
    n_jobs_outer: int = 1,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Subject-first checkpointed multivariate time-resolved encoding pipeline.
    """
    covariates = covariates or ['zpm']
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    completed_subjects: List[str] = []
    if checkpoint_dir is not None and not overwrite_checkpoints:
        for subject in subjects:
            checkpoint_file = os.path.join(checkpoint_dir, f"{subject}_results.csv")
            if os.path.exists(checkpoint_file):
                completed_subjects.append(subject)

    subjects_to_run = [s for s in subjects if s not in completed_subjects]
    if verbose:
        print(
            f"Running multivariate smoothed regression for {len(subjects_to_run)} subjects "
            f"(skipping {len(completed_subjects)} checkpointed)."
        )

    def _run_one(subject: str) -> pd.DataFrame:
        return process_subject_smoothed_multivariate_regression(
            subject=subject,
            event=event,
            predictor=predictor,
            base_dir=base_dir,
            load_dir=load_dir,
            rois=rois,
            freq_range=freq_range,
            win_ms=win_ms,
            downsample_factor=downsample_factor,
            n_permutations=n_permutations,
            covariates=covariates,
            min_trials=min_trials,
            seed=seed,
            ridge_alpha=ridge_alpha,
            ridge_alpha_map=ridge_alpha_map,
            predictor_key=(predictor_key or predictor),
            cv_folds=cv_folds,
            encoding_metric=encoding_metric,
            null_tail=null_tail,
            store_metric_name=store_metric_name,
            use_surrogates=use_surrogates,
            surrogate_null=surrogate_null,
            target_mode=target_mode,
            classifier_alpha=classifier_alpha,
            classifier_model=classifier_model,
            verbose=False,
            regressor_source=regressor_source,
            trial_var_override=trial_var_override,
        )

    out_frames: List[pd.DataFrame] = []
    if len(subjects_to_run) > 0:
        if n_jobs_outer == 1:
            out_frames = [_run_one(subject) for subject in subjects_to_run]
        else:
            out_frames = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
                delayed(_run_one)(subject) for subject in subjects_to_run
            )

        if checkpoint_dir is not None:
            for subject, df_subj in zip(subjects_to_run, out_frames):
                if len(df_subj) == 0:
                    continue
                df_subj.to_csv(
                    os.path.join(checkpoint_dir, f"{subject}_results.csv"),
                    index=False
                )

    all_frames: List[pd.DataFrame] = []
    if checkpoint_dir is not None:
        for subject in subjects:
            checkpoint_file = os.path.join(checkpoint_dir, f"{subject}_results.csv")
            if os.path.exists(checkpoint_file):
                all_frames.append(pd.read_csv(checkpoint_file))
    else:
        all_frames = [df for df in out_frames if len(df) > 0]

    if len(all_frames) == 0:
        return pd.DataFrame(columns=[
            'predictor', 'metric', 'z_metric', 'raw_p', 'ts', 'subj', 'region',
            'n_trials', 'n_electrodes', 'ridge_alpha', 'cv_folds', 'event',
            'covariates', 'metric_name', 'null_tail', 'surrogate_null',
            'use_surrogates', 'target_kind', 'n_classes', 'class_labels_json',
            'status', 'dropped_alignment'
        ])
    return pd.concat(all_frames, ignore_index=True)


def compute_subject_multivariate_reactivation(
    subject_obj: Subject,
    roi: str,
    predictor: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    predictor_key: Optional[str] = None,
    trial_save_dtype: type = np.float32,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multivariate encoding + projection for one subject/ROI/predictor.

    Encoding: ridge over electrode vectors at each encoding timepoint.
    Projection: cue electrode vectors projected through encoding weights,
    yielding (n_feedback_t, n_cue_t, n_trials) reactivation maps.

    Surrogates use Freedman-Lane-style pseudo-responses generated from
    reduced-model residual permutations of y ~ 1 + covariates.
    """
    covariates = covariates or ['zpm']
    fb_hfa, fb_times, electrode_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_encoding,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms
    )
    cue_hfa, cue_times, _ = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_projection,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms
    )
    regressors, _ = _build_subject_regressors(
        subject_obj=subject_obj,
        event=event_encoding,
        predictor=predictor,
        covariates=covariates,
        regressor_source=regressor_source,
        trial_var_override=trial_var_override,
    )

    trial_ix = regressors['trial'].to_numpy(dtype=np.int64) - 1
    in_range = (
        (trial_ix >= 0) &
        (trial_ix < fb_hfa.shape[0]) &
        (trial_ix < cue_hfa.shape[0])
    )
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]
    if trial_ix.size < subject_obj.min_trials:
        raise ValueError(
            f"Insufficient aligned trials for {subject_obj.subject}, ROI={roi}: "
            f"{trial_ix.size} < {subject_obj.min_trials}"
        )

    y_raw = regressors[predictor].to_numpy(dtype=np.float64)
    cov_cols = [c for c in covariates if c in regressors.columns and c != predictor]
    C = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None
    if C is None:
        core_valid = np.isfinite(y_raw)
    else:
        core_valid = np.isfinite(y_raw) & np.all(np.isfinite(C), axis=1)
    C_core = C[core_valid] if C is not None else None
    fl_state = _prepare_freedman_lane_state(
        y=y_raw[core_valid],
        nuisance=C_core,
        include_intercept=True,
    )
    y_hat0_core = np.asarray(fl_state['y_hat0'], dtype=np.float64).reshape(-1)
    resid0_core = np.asarray(fl_state['resid0'], dtype=np.float64).reshape(-1)
    y_resid = np.full(y_raw.shape, np.nan, dtype=np.float64)
    y_resid[core_valid] = resid0_core
    alpha_to_use = _resolve_ridge_alpha_for_key(
        ridge_alpha=ridge_alpha,
        ridge_alpha_map=ridge_alpha_map,
        subject=subject_obj.subject,
        roi=roi,
        predictor_key=(predictor_key or predictor),
    )

    if np.sum(core_valid) < subject_obj.min_trials:
        raise ValueError(
            f"Insufficient valid residualized trials for {subject_obj.subject}, ROI={roi}: "
            f"{int(np.sum(core_valid))} < {subject_obj.min_trials}"
        )
    if np.std(y_resid[core_valid]) < 1e-12:
        raise ValueError(
            f"Constant residualized predictor for {subject_obj.subject}, ROI={roi}, predictor={predictor}"
        )

    fb_aligned = np.asarray(fb_hfa[trial_ix], dtype=np.float32)   # (n_trials, n_elec, n_fb)
    cue_aligned = np.asarray(cue_hfa[trial_ix], dtype=np.float32)  # (n_trials, n_elec, n_cue)
    n_trials, n_electrodes, n_fb = fb_aligned.shape
    n_cue = cue_aligned.shape[-1]

    weights = np.zeros((n_fb, n_electrodes), dtype=np.float32)
    activation_patterns = np.zeros((n_fb, n_electrodes), dtype=np.float32)
    intercepts = np.zeros((n_fb,), dtype=np.float32)

    X_all = fb_aligned.transpose(0, 2, 1).reshape(-1, fb_aligned.shape[1])
    Sigma = np.cov(X_all, rowvar=False)  # (n_electrodes, n_electrodes)

    for t_idx in range(n_fb):
        X_t = np.asarray(fb_aligned[:, :, t_idx], dtype=np.float64)
        valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
        if np.sum(valid_t) < max(3, subject_obj.min_trials):
            continue
        y_t = y_resid[valid_t]
        if np.std(y_t) < 1e-12:
            continue
        model = Ridge(alpha=alpha_to_use)
        model.fit(X_t[valid_t], y_t)
        weights[t_idx] = model.coef_.astype(np.float32, copy=False)
        # sigma_x_t = np.cov(X_t[valid_t], rowvar=False)
        # activation_patterns[t_idx] = (sigma_x_t @ model.coef_).astype(np.float32)
        activation_patterns[t_idx] = (Sigma @ model.coef_).astype(np.float32)
        intercepts[t_idx] = np.float32(model.intercept_)

    lrt_axis = _build_lrt_trial_axis(
        subject_obj=subject_obj,
        event_projection=event_projection,
        cue_n_trials=cue_hfa.shape[0],
    )
    trial_ix_lrt = np.asarray(lrt_axis['trial_index_lrt'], dtype=np.int64)
    memory_outcomes_lrt = np.asarray(lrt_axis['memory_outcomes_lrt'], dtype=object)
    cue_lrt_aligned = np.asarray(cue_hfa[trial_ix_lrt], dtype=np.float32)  # (n_trials_lrt, n_elec, n_cue)

    projection_real = np.einsum('nec,fe->fcn', cue_aligned, weights) + intercepts[:, np.newaxis, np.newaxis]
    projection_real_lrt = np.einsum('nec,fe->fcn', cue_lrt_aligned, weights) + intercepts[:, np.newaxis, np.newaxis]

    if (not use_surrogates) or n_surrogates <= 0:
        z_reactivation = projection_real.astype(trial_save_dtype, copy=False)
        z_reactivation_lrt = projection_real_lrt.astype(trial_save_dtype, copy=False)
    else:
        rng = np.random.default_rng(seed)
        running_mean = np.zeros_like(projection_real, dtype=np.float32)
        running_M2 = np.zeros_like(projection_real, dtype=np.float32)
        running_mean_lrt = np.zeros_like(projection_real_lrt, dtype=np.float32)
        running_M2_lrt = np.zeros_like(projection_real_lrt, dtype=np.float32)
        n_total = 0

        for _ in range(n_surrogates):
            perm_ix = rng.permutation(resid0_core.shape[0])
            y_star_core = _freedman_lane_pseudo_response(
                y_hat0=y_hat0_core,
                resid0=resid0_core,
                perm_ix=perm_ix,
            )
            y_perm_core = _residualize_with_precomputed_design(
                y=y_star_core,
                Z=fl_state['Z'],
                pinv_Z=fl_state['pinv_Z'],
            )
            y_perm = np.full(y_raw.shape, np.nan, dtype=np.float64)
            y_perm[core_valid] = np.asarray(y_perm_core, dtype=np.float64).reshape(-1)

            w_perm = np.zeros((n_fb, n_electrodes), dtype=np.float32)
            b_perm = np.zeros((n_fb,), dtype=np.float32)
            for t_idx in range(n_fb):
                X_t = np.asarray(fb_aligned[:, :, t_idx], dtype=np.float64)
                valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
                if np.sum(valid_t) < max(3, subject_obj.min_trials):
                    continue
                y_t = y_perm[valid_t]
                if np.std(y_t) < 1e-12:
                    continue
                model = Ridge(alpha=alpha_to_use)
                model.fit(X_t[valid_t], y_t)
                w_perm[t_idx] = model.coef_.astype(np.float32, copy=False)
                b_perm[t_idx] = np.float32(model.intercept_)

            proj_s = np.einsum('nec,fe->fcn', cue_aligned, w_perm) + b_perm[:, np.newaxis, np.newaxis]
            proj_s_lrt = np.einsum('nec,fe->fcn', cue_lrt_aligned, w_perm) + b_perm[:, np.newaxis, np.newaxis]
            n_total += 1
            delta = proj_s - running_mean
            running_mean += delta / np.float32(n_total)
            delta2 = proj_s - running_mean
            running_M2 += delta * delta2
            delta_lrt = proj_s_lrt - running_mean_lrt
            running_mean_lrt += delta_lrt / np.float32(n_total)
            delta2_lrt = proj_s_lrt - running_mean_lrt
            running_M2_lrt += delta_lrt * delta2_lrt

        surrogate_std = np.sqrt(running_M2 / np.float32(max(n_total, 1)))
        surrogate_std = np.maximum(surrogate_std, 1e-8)
        surrogate_std_lrt = np.sqrt(running_M2_lrt / np.float32(max(n_total, 1)))
        surrogate_std_lrt = np.maximum(surrogate_std_lrt, 1e-8)
        z_reactivation = ((projection_real - running_mean) / surrogate_std).astype(
            trial_save_dtype, copy=False
        )
        z_reactivation_lrt = ((projection_real_lrt - running_mean_lrt) / surrogate_std_lrt).astype(
            trial_save_dtype, copy=False
        )

    mem_map = {int(t): m for t, m in zip(trial_ix_lrt.tolist(), memory_outcomes_lrt.tolist())}
    memory_outcomes = np.array([mem_map.get(int(ti), np.nan) for ti in trial_ix], dtype=object)

    return {
        'z_reactivation': z_reactivation.astype(trial_save_dtype, copy=False),
        'z_reactivation_lrt': z_reactivation_lrt.astype(trial_save_dtype, copy=False),
        'projection_real': projection_real.astype(np.float32, copy=False),
        'projection_real_lrt': projection_real_lrt.astype(np.float32, copy=False),
        'feedback_times': fb_times.astype(np.float32, copy=False),
        'cue_times': cue_times.astype(np.float32, copy=False),
        'weights': weights.astype(np.float32, copy=False),
        'activation_patterns': activation_patterns.astype(np.float32, copy=False),
        'intercepts': intercepts.astype(np.float32, copy=False),
        'electrode_labels': np.asarray(electrode_labels, dtype=str),
        'trial_index': trial_ix.astype(np.int32),
        'trial_index_lrt': trial_ix_lrt.astype(np.int32, copy=False),
        'memory_outcomes': memory_outcomes,
        'memory_outcomes_lrt': memory_outcomes_lrt,
        'predictor': predictor,
        'roi': roi,
        'subject': subject_obj.subject,
        'event_encoding': event_encoding,
        'event_projection': event_projection,
        'n_surrogates': int(n_surrogates),
        'ridge_alpha': float(alpha_to_use),
        'covariates': np.asarray(cov_cols, dtype=str),
        'residualized_predictor': y_resid.astype(np.float32, copy=False),
        'n_trials_train': int(n_trials),
        'n_trials_lrt': int(trial_ix_lrt.size),
        'lrt_axis_mode': 'all_day1_cue_trials',
    }


def run_multivariate_reactivation_pipeline(
    subjects: List[str],
    rois: List[str],
    predictors: List[str],
    base_dir: str,
    load_dir: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    n_jobs_outer: int = 1,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Subject-first multivariate reactivation pipeline.
    """
    covariates = covariates or ['zpm']
    tasks = [(roi, predictor) for roi in rois for predictor in predictors]

    def _run_subject(subject: str) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
        subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
        out: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
        for roi, predictor in tasks:
            try:
                result = compute_subject_multivariate_reactivation(
                    subject_obj=subject_obj,
                    roi=roi,
                    predictor=predictor,
                    event_encoding=event_encoding,
                    event_projection=event_projection,
                    covariates=covariates,
                    freq_range=freq_range,
                    downsample_factor=downsample_factor,
                    win_ms=win_ms,
                    n_surrogates=n_surrogates,
                    use_surrogates=use_surrogates,
                    seed=seed,
                    ridge_alpha=ridge_alpha,
                    ridge_alpha_map=ridge_alpha_map,
                    predictor_key=predictor,
                    trial_save_dtype=trial_save_dtype,
                    regressor_source=regressor_source,
                    trial_var_override=trial_var_override,
                )

                if save_trial_level and trial_save_dir is not None:
                    roi_dir = os.path.join(trial_save_dir, roi)
                    os.makedirs(roi_dir, exist_ok=True)
                    out_file = os.path.join(roi_dir, f"{subject}_{predictor}_zreactivation.npz")
                    if overwrite_trial_level or (not os.path.exists(out_file)):
                        np.savez_compressed(
                            out_file,
                            z_reactivation=result['z_reactivation'].astype(trial_save_dtype, copy=False),
                            z_reactivation_lrt=result['z_reactivation_lrt'].astype(trial_save_dtype, copy=False),
                            feedback_times=result['feedback_times'],
                            cue_times=result['cue_times'],
                            memory_outcomes=result['memory_outcomes'],
                            memory_outcomes_lrt=result['memory_outcomes_lrt'],
                            trial_index=result['trial_index'],
                            trial_index_lrt=result['trial_index_lrt'].astype(np.int32, copy=False),
                            electrode_labels=result['electrode_labels'],
                            subject=subject,
                            roi=roi,
                            predictor=predictor,
                            event_encoding=event_encoding,
                            event_projection=event_projection,
                            n_surrogates=int(n_surrogates),
                            ridge_alpha=float(result['ridge_alpha']),
                            covariates=result['covariates'],
                            residualized_predictor=result['residualized_predictor'],
                            projection_real=result['projection_real'],
                            projection_real_lrt=result['projection_real_lrt'],
                            weights=result['weights'],
                            activation_patterns=result['activation_patterns'],
                            intercepts=result['intercepts'],
                            n_trials_train=np.int32(result['n_trials_train']),
                            n_trials_lrt=np.int32(result['n_trials_lrt']),
                            lrt_axis_mode=np.asarray(result['lrt_axis_mode']),
                            use_surrogates=bool(use_surrogates),
                            seed=int(seed),
                        )

                out.append((roi, predictor, result))
            except Exception as exc:
                if verbose:
                    print(f"Multivariate processing failed: {subject}, ROI={roi}, predictor={predictor}: {exc}")
                out.append((roi, predictor, None))
        return out

    if n_jobs_outer == 1:
        batch_results = [_run_subject(subject) for subject in subjects]
    else:
        batch_results = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
            delayed(_run_subject)(subject) for subject in subjects
        )

    results = {
        roi: {
            predictor: {
                'subject_list': [],
                'feedback_times_list': [],
                'cue_times_list': [],
                'weights_list': [],
                'activation_patterns_list': [],
                'memory_outcomes': [],
            }
            for predictor in predictors
        }
        for roi in rois
    }

    for subject_batch in batch_results:
        for roi, predictor, result in subject_batch:
            if result is None:
                continue
            results[roi][predictor]['subject_list'].append(result['subject'])
            results[roi][predictor]['feedback_times_list'].append(result['feedback_times'])
            results[roi][predictor]['cue_times_list'].append(result['cue_times'])
            results[roi][predictor]['weights_list'].append(result['weights'])
            results[roi][predictor]['activation_patterns_list'].append(result['activation_patterns'])
            results[roi][predictor]['memory_outcomes'].append(result['memory_outcomes'])
    return results


def build_multivariate_csv_path(
    load_dir: str,
    band: str,
    event: str,
    predictor: str,
    suffix: str = 'smoothed_final'
) -> str:
    """
    Build default path for multivariate time-resolved encoding CSV output.
    """
    return os.path.join(
        load_dir, f"multivar_time_resolved_perm_{band}_{event}_{predictor}_{suffix}.csv"
    )


PAIRWISE_RPE_CONTRASTS: Dict[str, Tuple[str, str]] = {
    'pos_vs_neutral': ('pos', 'neutral'),
    'pos_vs_negative': ('pos', 'neg'),
}


def _canonicalize_rpe_cat_label(label: Any) -> Optional[str]:
    """Normalize mixed rpe_cat labels to {'neg', 'neutral', 'pos'}."""
    if pd.isna(label):
        return None
    key = str(label).strip().lower()
    if len(key) == 0 or key == 'nan':
        return None
    if key in {'pos', 'positive'}:
        return 'pos'
    if key in {'neg', 'negative'}:
        return 'neg'
    if key == 'neutral':
        return 'neutral'
    return key


def build_rpe_pair_contrast_regressors(
    subject_obj: Subject,
    contrast_name: str,
    event: str = 'feedback_start',
    covariates: Optional[List[str]] = None,
    regressor_source: Optional[str] = 'learn',
    trial_var_override: Optional[str] = None,
    positive_value: float = 0.5,
    negative_value: float = -0.5,
) -> Tuple[pd.DataFrame, str]:
    """
    Build subject-level numeric pairwise contrast regressors from `rpe_cat`.

    Returns a dataframe aligned to the requested event with columns:
    `trial`, `rpe_cat`, `rpe_pair_contrast`, plus requested covariates.
    """
    covariates = covariates or ['zpm']
    if contrast_name not in PAIRWISE_RPE_CONTRASTS:
        raise ValueError(
            f"Unsupported contrast_name='{contrast_name}'. "
            f"Expected one of: {sorted(PAIRWISE_RPE_CONTRASTS)}"
        )

    reg_df, trial_var = _build_subject_regressors(
        subject_obj=subject_obj,
        event=event,
        predictor='rpe_cat',
        covariates=covariates,
        include_columns=[],
        regressor_source=regressor_source,
        trial_var_override=trial_var_override,
    )
    reg_df = reg_df.copy()
    reg_df['rpe_cat'] = reg_df['rpe_cat'].map(_canonicalize_rpe_cat_label)

    pos_label, ref_label = PAIRWISE_RPE_CONTRASTS[contrast_name]
    reg_df = reg_df[reg_df['rpe_cat'].isin([pos_label, ref_label])].copy()
    if len(reg_df) == 0:
        raise ValueError(
            f"No rows available for subject={subject_obj.subject}, contrast={contrast_name}"
        )

    reg_df['rpe_pair_contrast'] = np.where(
        reg_df['rpe_cat'].astype(str) == pos_label,
        float(positive_value),
        float(negative_value),
    ).astype(np.float32)
    reg_df['contrast_name'] = str(contrast_name)

    counts = reg_df['rpe_cat'].value_counts(dropna=False)
    if int(counts.get(pos_label, 0)) < 3 or int(counts.get(ref_label, 0)) < 3:
        raise ValueError(
            f"Insufficient per-class trials for subject={subject_obj.subject}, "
            f"contrast={contrast_name}: {counts.to_dict()}"
        )
    return reg_df, trial_var


def compute_subject_pairwise_multivariate_reactivation(
    subject_obj: Subject,
    roi: str,
    contrast_name: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    predictor_key: Optional[str] = None,
    trial_save_dtype: type = np.float32,
    regressor_source: Optional[str] = 'learn',
    trial_var_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Multivariate ACC reactivation from explicit pairwise RPE-category contrasts.

    Training uses only the requested pairwise subset; projection is performed on
    all available Day1 cue trials using the all-trial LRT axis.
    """
    covariates = covariates or ['zpm']
    fb_hfa, fb_times, electrode_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_encoding,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
    )
    cue_hfa, cue_times, _ = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_projection,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
    )

    regressors, _ = build_rpe_pair_contrast_regressors(
        subject_obj=subject_obj,
        contrast_name=contrast_name,
        event=event_encoding,
        covariates=covariates,
        regressor_source=regressor_source,
        trial_var_override=trial_var_override,
    )

    trial_ix = regressors['trial'].to_numpy(dtype=np.int64) - 1
    in_range = (
        (trial_ix >= 0) &
        (trial_ix < fb_hfa.shape[0]) &
        (trial_ix < cue_hfa.shape[0])
    )
    regressors = regressors.loc[in_range].reset_index(drop=True)
    trial_ix = trial_ix[in_range]
    if trial_ix.size < subject_obj.min_trials:
        raise ValueError(
            f"Insufficient aligned trials for {subject_obj.subject}, ROI={roi}, "
            f"contrast={contrast_name}: {trial_ix.size} < {subject_obj.min_trials}"
        )

    y_raw = regressors['rpe_pair_contrast'].to_numpy(dtype=np.float64)
    cov_cols = [c for c in covariates if c in regressors.columns and c != 'rpe_pair_contrast']
    C = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None
    if C is None:
        core_valid = np.isfinite(y_raw)
    else:
        core_valid = np.isfinite(y_raw) & np.all(np.isfinite(C), axis=1)
    C_core = C[core_valid] if C is not None else None
    fl_state = _prepare_freedman_lane_state(
        y=y_raw[core_valid],
        nuisance=C_core,
        include_intercept=True,
    )
    y_hat0_core = np.asarray(fl_state['y_hat0'], dtype=np.float64).reshape(-1)
    resid0_core = np.asarray(fl_state['resid0'], dtype=np.float64).reshape(-1)
    y_resid = np.full(y_raw.shape, np.nan, dtype=np.float64)
    y_resid[core_valid] = resid0_core

    if np.sum(core_valid) < subject_obj.min_trials:
        raise ValueError(
            f"Insufficient valid residualized trials for {subject_obj.subject}, ROI={roi}, "
            f"contrast={contrast_name}: {int(np.sum(core_valid))} < {subject_obj.min_trials}"
        )
    if np.std(y_resid[core_valid]) < 1e-12:
        raise ValueError(
            f"Constant residualized predictor for {subject_obj.subject}, ROI={roi}, "
            f"contrast={contrast_name}"
        )

    alpha_to_use = _resolve_ridge_alpha_for_key(
        ridge_alpha=ridge_alpha,
        ridge_alpha_map=ridge_alpha_map,
        subject=subject_obj.subject,
        roi=roi,
        predictor_key=(predictor_key or contrast_name),
    )

    fb_aligned = np.asarray(fb_hfa[trial_ix], dtype=np.float32)
    cue_aligned = np.asarray(cue_hfa[trial_ix], dtype=np.float32)
    n_trials, n_electrodes, n_fb = fb_aligned.shape
    n_cue = cue_aligned.shape[-1]

    weights = np.zeros((n_fb, n_electrodes), dtype=np.float32)
    activation_patterns = np.zeros((n_fb, n_electrodes), dtype=np.float32)
    intercepts = np.zeros((n_fb,), dtype=np.float32)

    X_all = fb_aligned.transpose(0, 2, 1).reshape(-1, fb_aligned.shape[1])
    Sigma = np.cov(X_all, rowvar=False)

    for t_idx in range(n_fb):
        X_t = np.asarray(fb_aligned[:, :, t_idx], dtype=np.float64)
        valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
        if np.sum(valid_t) < max(3, subject_obj.min_trials):
            continue
        y_t = y_resid[valid_t]
        if np.std(y_t) < 1e-12:
            continue
        model = Ridge(alpha=alpha_to_use)
        model.fit(X_t[valid_t], y_t)
        weights[t_idx] = model.coef_.astype(np.float32, copy=False)
        activation_patterns[t_idx] = (Sigma @ model.coef_).astype(np.float32)
        intercepts[t_idx] = np.float32(model.intercept_)

    lrt_axis = _build_lrt_trial_axis(
        subject_obj=subject_obj,
        event_projection=event_projection,
        cue_n_trials=cue_hfa.shape[0],
    )
    trial_ix_lrt = np.asarray(lrt_axis['trial_index_lrt'], dtype=np.int64)
    memory_outcomes_lrt = np.asarray(lrt_axis['memory_outcomes_lrt'], dtype=object)
    cue_lrt_aligned = np.asarray(cue_hfa[trial_ix_lrt], dtype=np.float32)

    projection_real = np.einsum('nec,fe->fcn', cue_aligned, weights) + intercepts[:, np.newaxis, np.newaxis]
    projection_real_lrt = np.einsum('nec,fe->fcn', cue_lrt_aligned, weights) + intercepts[:, np.newaxis, np.newaxis]

    if (not use_surrogates) or n_surrogates <= 0:
        z_reactivation = projection_real.astype(trial_save_dtype, copy=False)
        z_reactivation_lrt = projection_real_lrt.astype(trial_save_dtype, copy=False)
    else:
        rng = np.random.default_rng(seed)
        running_mean = np.zeros_like(projection_real, dtype=np.float32)
        running_M2 = np.zeros_like(projection_real, dtype=np.float32)
        running_mean_lrt = np.zeros_like(projection_real_lrt, dtype=np.float32)
        running_M2_lrt = np.zeros_like(projection_real_lrt, dtype=np.float32)
        n_total = 0

        for _ in range(n_surrogates):
            perm_ix = rng.permutation(resid0_core.shape[0])
            y_star_core = _freedman_lane_pseudo_response(
                y_hat0=y_hat0_core,
                resid0=resid0_core,
                perm_ix=perm_ix,
            )
            y_perm_core = _residualize_with_precomputed_design(
                y=y_star_core,
                Z=fl_state['Z'],
                pinv_Z=fl_state['pinv_Z'],
            )
            y_perm = np.full(y_raw.shape, np.nan, dtype=np.float64)
            y_perm[core_valid] = np.asarray(y_perm_core, dtype=np.float64).reshape(-1)

            w_perm = np.zeros((n_fb, n_electrodes), dtype=np.float32)
            b_perm = np.zeros((n_fb,), dtype=np.float32)
            for t_idx in range(n_fb):
                X_t = np.asarray(fb_aligned[:, :, t_idx], dtype=np.float64)
                valid_t = core_valid & np.all(np.isfinite(X_t), axis=1)
                if np.sum(valid_t) < max(3, subject_obj.min_trials):
                    continue
                y_t = y_perm[valid_t]
                if np.std(y_t) < 1e-12:
                    continue
                model = Ridge(alpha=alpha_to_use)
                model.fit(X_t[valid_t], y_t)
                w_perm[t_idx] = model.coef_.astype(np.float32, copy=False)
                b_perm[t_idx] = np.float32(model.intercept_)

            proj_s = np.einsum('nec,fe->fcn', cue_aligned, w_perm) + b_perm[:, np.newaxis, np.newaxis]
            proj_s_lrt = np.einsum('nec,fe->fcn', cue_lrt_aligned, w_perm) + b_perm[:, np.newaxis, np.newaxis]
            n_total += 1

            delta = proj_s - running_mean
            running_mean += delta / np.float32(n_total)
            running_M2 += delta * (proj_s - running_mean)

            delta_lrt = proj_s_lrt - running_mean_lrt
            running_mean_lrt += delta_lrt / np.float32(n_total)
            running_M2_lrt += delta_lrt * (proj_s_lrt - running_mean_lrt)

        surrogate_std = np.sqrt(running_M2 / np.float32(max(n_total, 1)))
        surrogate_std = np.maximum(surrogate_std, 1e-8)
        surrogate_std_lrt = np.sqrt(running_M2_lrt / np.float32(max(n_total, 1)))
        surrogate_std_lrt = np.maximum(surrogate_std_lrt, 1e-8)
        z_reactivation = ((projection_real - running_mean) / surrogate_std).astype(
            trial_save_dtype, copy=False
        )
        z_reactivation_lrt = ((projection_real_lrt - running_mean_lrt) / surrogate_std_lrt).astype(
            trial_save_dtype, copy=False
        )

    train_counts = regressors['rpe_cat'].value_counts(dropna=False).to_dict()
    pos_label, ref_label = PAIRWISE_RPE_CONTRASTS[contrast_name]
    return {
        'z_reactivation': z_reactivation.astype(trial_save_dtype, copy=False),
        'z_reactivation_lrt': z_reactivation_lrt.astype(trial_save_dtype, copy=False),
        'projection_real': projection_real.astype(np.float32, copy=False),
        'projection_real_lrt': projection_real_lrt.astype(np.float32, copy=False),
        'feedback_times': fb_times.astype(np.float32, copy=False),
        'cue_times': cue_times.astype(np.float32, copy=False),
        'weights': weights.astype(np.float32, copy=False),
        'activation_patterns': activation_patterns.astype(np.float32, copy=False),
        'intercepts': intercepts.astype(np.float32, copy=False),
        'electrode_labels': np.asarray(electrode_labels, dtype=str),
        'trial_index': trial_ix.astype(np.int32, copy=False),
        'trial_index_lrt': trial_ix_lrt.astype(np.int32, copy=False),
        'memory_outcomes_lrt': memory_outcomes_lrt,
        'contrast_name': str(contrast_name),
        'contrast_positive_label': np.asarray(pos_label),
        'contrast_reference_label': np.asarray(ref_label),
        'contrast_training_counts_json': json.dumps({str(k): int(v) for k, v in train_counts.items()}),
        'roi': roi,
        'subject': subject_obj.subject,
        'event_encoding': event_encoding,
        'event_projection': event_projection,
        'n_surrogates': int(n_surrogates),
        'ridge_alpha': float(alpha_to_use),
        'covariates': np.asarray(cov_cols, dtype=str),
        'residualized_predictor': y_resid.astype(np.float32, copy=False),
        'n_trials_train': int(n_trials),
        'n_trials_lrt': int(trial_ix_lrt.size),
        'lrt_axis_mode': 'all_day1_cue_trials',
    }


def run_pairwise_multivariate_reactivation_pipeline(
    subjects: List[str],
    rois: List[str],
    contrast_names: List[str],
    base_dir: str,
    load_dir: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    ridge_alpha: float = 1.0,
    ridge_alpha_map: Optional[Dict[Tuple[str, str, str], float]] = None,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    n_jobs_outer: int = 1,
    verbose: bool = True,
    regressor_source: Optional[str] = 'learn',
    trial_var_override: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Subject-first multivariate reactivation pipeline for explicit RPE pairwise contrasts.
    """
    covariates = covariates or ['zpm']
    tasks = [(roi, contrast_name) for roi in rois for contrast_name in contrast_names]

    def _run_subject(subject: str) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
        subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
        out: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
        for roi, contrast_name in tasks:
            try:
                result = compute_subject_pairwise_multivariate_reactivation(
                    subject_obj=subject_obj,
                    roi=roi,
                    contrast_name=contrast_name,
                    event_encoding=event_encoding,
                    event_projection=event_projection,
                    covariates=covariates,
                    freq_range=freq_range,
                    downsample_factor=downsample_factor,
                    win_ms=win_ms,
                    n_surrogates=n_surrogates,
                    use_surrogates=use_surrogates,
                    seed=seed,
                    ridge_alpha=ridge_alpha,
                    ridge_alpha_map=ridge_alpha_map,
                    predictor_key=contrast_name,
                    trial_save_dtype=trial_save_dtype,
                    regressor_source=regressor_source,
                    trial_var_override=trial_var_override,
                )

                if save_trial_level and trial_save_dir is not None:
                    roi_dir = os.path.join(trial_save_dir, roi)
                    os.makedirs(roi_dir, exist_ok=True)
                    out_file = os.path.join(roi_dir, f"{subject}_{contrast_name}_zreactivation.npz")
                    if overwrite_trial_level or (not os.path.exists(out_file)):
                        np.savez_compressed(
                            out_file,
                            z_reactivation=result['z_reactivation'].astype(trial_save_dtype, copy=False),
                            z_reactivation_lrt=result['z_reactivation_lrt'].astype(trial_save_dtype, copy=False),
                            feedback_times=result['feedback_times'],
                            cue_times=result['cue_times'],
                            trial_index=result['trial_index'],
                            trial_index_lrt=result['trial_index_lrt'],
                            memory_outcomes_lrt=result['memory_outcomes_lrt'],
                            electrode_labels=result['electrode_labels'],
                            subject=subject,
                            roi=roi,
                            contrast_name=contrast_name,
                            event_encoding=event_encoding,
                            event_projection=event_projection,
                            n_surrogates=int(n_surrogates),
                            ridge_alpha=float(result['ridge_alpha']),
                            covariates=result['covariates'],
                            residualized_predictor=result['residualized_predictor'],
                            projection_real=result['projection_real'],
                            projection_real_lrt=result['projection_real_lrt'],
                            weights=result['weights'],
                            activation_patterns=result['activation_patterns'],
                            intercepts=result['intercepts'],
                            n_trials_train=np.int32(result['n_trials_train']),
                            n_trials_lrt=np.int32(result['n_trials_lrt']),
                            lrt_axis_mode=np.asarray(result['lrt_axis_mode']),
                            use_surrogates=bool(use_surrogates),
                            seed=int(seed),
                            contrast_positive_label=result['contrast_positive_label'],
                            contrast_reference_label=result['contrast_reference_label'],
                            contrast_training_counts_json=np.asarray(result['contrast_training_counts_json']),
                        )
                out.append((roi, contrast_name, result))
            except Exception as exc:
                if verbose:
                    print(
                        f"Pairwise multivariate processing failed: {subject}, ROI={roi}, "
                        f"contrast={contrast_name}: {exc}"
                    )
                out.append((roi, contrast_name, None))
        return out

    if n_jobs_outer == 1:
        batch_results = [_run_subject(subject) for subject in subjects]
    else:
        batch_results = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
            delayed(_run_subject)(subject) for subject in subjects
        )

    results = {
        roi: {
            contrast_name: {
                'subject_list': [],
                'feedback_times_list': [],
                'cue_times_list': [],
                'weights_list': [],
                'activation_patterns_list': [],
                'memory_outcomes_lrt': [],
            }
            for contrast_name in contrast_names
        }
        for roi in rois
    }
    for subject_batch in batch_results:
        for roi, contrast_name, result in subject_batch:
            if result is None:
                continue
            results[roi][contrast_name]['subject_list'].append(result['subject'])
            results[roi][contrast_name]['feedback_times_list'].append(result['feedback_times'])
            results[roi][contrast_name]['cue_times_list'].append(result['cue_times'])
            results[roi][contrast_name]['weights_list'].append(result['weights'])
            results[roi][contrast_name]['activation_patterns_list'].append(result['activation_patterns'])
            results[roi][contrast_name]['memory_outcomes_lrt'].append(result['memory_outcomes_lrt'])
    return results


def collapse_reactivation_feedback_windows(
    reactivation: np.ndarray,
    feedback_times: np.ndarray,
    windows: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Collapse a (feedback_time, cue_time, trial) tensor into cue_time-by-trial
    matrices by averaging feedback rows within fixed windows.
    """
    R = np.asarray(reactivation, dtype=np.float32)
    fb_times = np.asarray(feedback_times, dtype=np.float32).reshape(-1)
    if R.ndim != 3:
        raise ValueError(f"Expected 3D reactivation array, got shape {R.shape}")
    if R.shape[0] != fb_times.shape[0]:
        raise ValueError(
            f"feedback_times length mismatch: len={fb_times.shape[0]}, reactivation={R.shape}"
        )

    out: Dict[str, np.ndarray] = {}
    for window_name, (tmin, tmax) in windows.items():
        mask = (fb_times >= float(tmin)) & (fb_times <= float(tmax))
        if not np.any(mask):
            raise ValueError(
                f"No feedback rows found for window '{window_name}' [{tmin}, {tmax}]"
            )
        out[str(window_name)] = np.nanmean(R[mask], axis=0).astype(np.float32, copy=False)
    return out


def _as_single_tfr(tfr_obj: Any) -> Any:
    """Normalize MNE read_tfrs output across versions."""
    if isinstance(tfr_obj, (list, tuple)):
        if len(tfr_obj) == 0:
            raise ValueError("TFR loader returned an empty list")
        return tfr_obj[0]
    return tfr_obj


def _interp_last_axis_to_target(
    data: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Interpolate data shaped (..., source_time) onto target_times.

    Targets outside the source time range are set to NaN instead of endpoint-clamped.
    """
    arr = np.asarray(data, dtype=np.float64)
    source = np.asarray(source_times, dtype=np.float64).reshape(-1)
    target = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if arr.shape[-1] != source.size:
        raise ValueError(f"Last data axis ({arr.shape[-1]}) does not match source_times ({source.size})")
    if source.size == target.size and np.allclose(source, target, atol=1e-6, rtol=0):
        return arr.astype(np.float32, copy=False)

    flat = arr.reshape(-1, source.size)
    out = np.full((flat.shape[0], target.size), np.nan, dtype=np.float64)
    outside = (target < np.nanmin(source)) | (target > np.nanmax(source))
    for row_idx, row in enumerate(flat):
        valid = np.isfinite(row) & np.isfinite(source)
        if np.sum(valid) == 0:
            continue
        if np.sum(valid) == 1:
            out[row_idx, ~outside] = row[valid][0]
        else:
            out[row_idx] = np.interp(target, source[valid], row[valid])
            out[row_idx, outside] = np.nan
    return out.reshape(arr.shape[:-1] + (target.size,)).astype(np.float32, copy=False)


def extract_theta_power_from_cue_tfr(
    subject_obj: Subject,
    roi: str = 'HPC',
    event: str = 'cue_on',
    freq_band: Tuple[float, float] = (4., 8.),
    target_times: Optional[np.ndarray] = None,
    trial_index: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract cue-locked theta power from baseline-z-scored TFRs.

    The saved TFR values are assumed to already be baseline-normalized/z-scored.
    This helper averages those values across theta frequencies and aligns them to
    the requested cue-time grid without re-baselining, smoothing, or Hilbert power.
    """
    tfr = _as_single_tfr(subject_obj._ensure_tfr_loaded(event)).copy()
    roi_labels = subject_obj.elec_df[
        subject_obj.elec_df.salman_region.astype(str) == str(roi)
    ]['label'].astype(str).tolist()
    keep_labels = [ch for ch in roi_labels if ch in tfr.ch_names]
    if len(keep_labels) == 0:
        raise ValueError(f"No {roi} electrodes found in TFR for {subject_obj.subject}")

    tfr = tfr.pick(keep_labels)
    tfr_data = np.asarray(tfr.data, dtype=np.float32)
    if tfr_data.ndim != 4:
        raise ValueError(
            f"Expected trial-level TFR data with shape (trials, channels, freqs, times), "
            f"got {tfr_data.shape}"
        )

    if trial_index is not None:
        trial_ix = np.asarray(trial_index, dtype=np.int64).reshape(-1)
        in_range = (trial_ix >= 0) & (trial_ix < tfr_data.shape[0])
        if not np.all(in_range):
            bad = int(np.sum(~in_range))
            raise ValueError(
                f"{bad} trial_index entries are outside TFR trial bounds "
                f"[0, {tfr_data.shape[0] - 1}]"
            )
        tfr_data = tfr_data[trial_ix]
        used_trial_index = trial_ix.astype(np.int64, copy=False)
    else:
        used_trial_index = np.arange(tfr_data.shape[0], dtype=np.int64)

    freqs = np.asarray(tfr.freqs, dtype=np.float64)
    freq_mask = (freqs >= float(freq_band[0])) & (freqs <= float(freq_band[1]))
    if not np.any(freq_mask):
        raise ValueError(
            f"No TFR frequencies found in band {freq_band}; available range is "
            f"{float(np.nanmin(freqs)):.3f}-{float(np.nanmax(freqs)):.3f} Hz"
        )

    theta_power = np.nanmean(tfr_data[:, :, freq_mask, :], axis=2)
    tfr_times = np.asarray(tfr.times, dtype=np.float32)
    if target_times is None:
        target_times_arr = tfr_times.copy()
    else:
        target_times_arr = np.asarray(target_times, dtype=np.float32).reshape(-1)

    theta_power = _interp_last_axis_to_target(theta_power, tfr_times, target_times_arr)
    theta_power = np.transpose(theta_power, (1, 2, 0)).astype(np.float32, copy=False)

    return {
        'power': theta_power,
        'target_times': target_times_arr.astype(np.float32, copy=False),
        'tfr_times': tfr_times.astype(np.float32, copy=False),
        'freqs': freqs.astype(np.float32, copy=False),
        'freq_mask': freq_mask,
        'freq_band': np.asarray(freq_band, dtype=np.float32),
        'electrode_labels': np.asarray(tfr.ch_names, dtype=str),
        'trial_index': used_trial_index,
        'theta_power_source': 'baseline_zscored_tfr',
    }


def extract_theta_phase_from_cue_epochs(
    subject_obj: Subject,
    roi: str = 'HPC',
    event: str = 'cue_on',
    freq_band: Tuple[float, float] = (4., 8.),
    target_times: Optional[np.ndarray] = None,
    trial_index: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract cue-locked theta phase per electrode/trial on a target time grid.

    Phase is sampled from the Hilbert transform of 4-8 Hz filtered raw epochs at
    each requested cue-time center.
    """
    epochs_file = os.path.join(
        subject_obj.base_dir,
        'projects',
        'guLab',
        'Salman',
        'EphysAnalyses',
        subject_obj.subject,
        'neural',
        'Day1',
        f'{event}-epo.fif',
    )
    if not os.path.exists(epochs_file):
        raise FileNotFoundError(f"Epochs file not found: {epochs_file}")

    epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)
    roi_labels = subject_obj.elec_df[
        subject_obj.elec_df.salman_region.astype(str) == str(roi)
    ]['label'].astype(str).tolist()
    keep_labels = [ch for ch in roi_labels if ch in epochs.ch_names]
    if len(keep_labels) == 0:
        raise ValueError(f"No {roi} electrodes found in epochs for {subject_obj.subject}")

    epochs = epochs.copy().pick(keep_labels)
    data = np.asarray(epochs.get_data(), dtype=np.float64)
    if trial_index is not None:
        trial_ix = np.asarray(trial_index, dtype=np.int64).reshape(-1)
        in_range = (trial_ix >= 0) & (trial_ix < data.shape[0])
        if not np.all(in_range):
            bad = int(np.sum(~in_range))
            raise ValueError(
                f"{bad} trial_index entries are outside epoch trial bounds "
                f"[0, {data.shape[0] - 1}]"
            )
        data = data[trial_ix]
        used_trial_index = trial_ix.astype(np.int64, copy=False)
    else:
        used_trial_index = np.arange(data.shape[0], dtype=np.int64)

    sfreq = float(epochs.info['sfreq'])
    n_trials, n_electrodes, n_samples = data.shape
    reshaped = data.reshape(-1, n_samples)
    theta = mne.filter.filter_data(
        reshaped.astype(np.float64, copy=True),
        sfreq=sfreq,
        l_freq=float(freq_band[0]),
        h_freq=float(freq_band[1]),
        verbose=False,
    )
    theta = theta.reshape(n_trials, n_electrodes, n_samples)
    analytic = hilbert(theta, axis=-1)
    inst_phase = np.angle(analytic).astype(np.float32, copy=False)

    epoch_times = np.asarray(epochs.times, dtype=np.float32)
    if target_times is None:
        target_times = epoch_times.copy()
    target_times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    center_indices = np.array(
        [int(np.argmin(np.abs(epoch_times - t))) for t in target_times],
        dtype=np.int64,
    )

    phase = np.full((n_electrodes, len(target_times), n_trials), np.nan, dtype=np.float32)

    for t_idx, center_idx in enumerate(center_indices):
        if center_idx < 0 or center_idx >= n_samples:
            continue
        phase[:, t_idx, :] = inst_phase[:, :, center_idx].T.astype(np.float32, copy=False)

    return {
        'phase': phase,
        'target_times': target_times.astype(np.float32, copy=False),
        'epoch_times': epoch_times.astype(np.float32, copy=False),
        'center_indices': center_indices.astype(np.int32, copy=False),
        'sfreq': float(sfreq),
        'freq_band': np.asarray(freq_band, dtype=np.float32),
        'electrode_labels': np.asarray(keep_labels, dtype=str),
        'trial_index': used_trial_index,
        'theta_phase_source': 'hilbert_filtered_epochs',
    }


def extract_theta_features_from_cue_epochs(
    subject_obj: Subject,
    roi: str = 'HPC',
    event: str = 'cue_on',
    freq_band: Tuple[float, float] = (4., 8.),
    window_ms: int = 200,
    target_times: Optional[np.ndarray] = None,
    trial_index: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible theta feature extractor.

    Power is now loaded from baseline-z-scored TFRs. Phase is still computed from
    filtered cue epochs. ``window_ms`` is retained only for old call sites and is
    not applied to TFR power.
    """
    power = extract_theta_power_from_cue_tfr(
        subject_obj=subject_obj,
        roi=roi,
        event=event,
        freq_band=freq_band,
        target_times=target_times,
        trial_index=trial_index,
    )
    phase = extract_theta_phase_from_cue_epochs(
        subject_obj=subject_obj,
        roi=roi,
        event=event,
        freq_band=freq_band,
        target_times=power['target_times'],
        trial_index=trial_index,
    )

    power_labels = np.asarray(power['electrode_labels']).astype(str)
    phase_labels = np.asarray(phase['electrode_labels']).astype(str)
    common_labels = [label for label in power_labels if label in set(phase_labels)]
    if len(common_labels) == 0:
        raise ValueError(f"No common {roi} electrodes found between TFR power and epoch phase")

    power_ix = np.array([int(np.where(power_labels == label)[0][0]) for label in common_labels], dtype=np.int64)
    phase_ix = np.array([int(np.where(phase_labels == label)[0][0]) for label in common_labels], dtype=np.int64)

    return {
        'power': np.asarray(power['power'][power_ix], dtype=np.float32),
        'phase': np.asarray(phase['phase'][phase_ix], dtype=np.float32),
        'target_times': np.asarray(power['target_times'], dtype=np.float32),
        'tfr_times': np.asarray(power['tfr_times'], dtype=np.float32),
        'epoch_times': np.asarray(phase['epoch_times'], dtype=np.float32),
        'center_indices': np.asarray(phase['center_indices'], dtype=np.int32),
        'window_ms': int(window_ms),
        'sfreq': float(phase['sfreq']),
        'freq_band': np.asarray(freq_band, dtype=np.float32),
        'electrode_labels': np.asarray(common_labels, dtype=str),
        'trial_index': np.asarray(power['trial_index'], dtype=np.int64),
        'theta_power_source': power['theta_power_source'],
        'theta_phase_source': phase['theta_phase_source'],
    }


def compute_time_resolved_theta_power_reactivation_beta(
    theta_power: np.ndarray,
    reactivation: np.ndarray,
    trial_mask: Optional[np.ndarray] = None,
    min_trials: int = 5,
    n_permutations: int = 0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Regress trial-wise reactivation on trial-wise theta power at each cue timepoint.

    Optionally computes a shuffle-based null distribution at each cue timepoint
    by permuting theta-power trial order (within timepoint), then z-scores the
    observed beta against that null.
    """
    x = np.asarray(theta_power, dtype=np.float64)
    y = np.asarray(reactivation, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: theta_power={x.shape}, reactivation={y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D arrays (time, trial), got {x.ndim}D")

    n_times = x.shape[0]
    betas = np.full(n_times, np.nan, dtype=np.float32)
    intercepts = np.full(n_times, np.nan, dtype=np.float32)
    z_betas = np.full(n_times, np.nan, dtype=np.float32)
    p_betas = np.full(n_times, np.nan, dtype=np.float32)
    null_mean = np.full(n_times, np.nan, dtype=np.float32)
    null_std = np.full(n_times, np.nan, dtype=np.float32)
    n_valid_trials = np.zeros(n_times, dtype=np.int32)

    base_mask = None
    if trial_mask is not None:
        base_mask = np.asarray(trial_mask, dtype=bool).reshape(-1)
        if base_mask.shape[0] != x.shape[1]:
            raise ValueError(
                f"trial_mask length mismatch: len={base_mask.shape[0]}, trials={x.shape[1]}"
            )

    rng = np.random.default_rng(seed)
    for t_idx in range(n_times):
        mask = np.isfinite(x[t_idx]) & np.isfinite(y[t_idx])
        if base_mask is not None:
            mask &= base_mask
        n_valid = int(mask.sum())
        n_valid_trials[t_idx] = np.int32(n_valid)
        if n_valid < int(min_trials):
            continue

        x_t = x[t_idx, mask]
        y_t = y[t_idx, mask]
        x_centered = x_t - np.nanmean(x_t)
        y_centered = y_t - np.nanmean(y_t)
        denom = float(np.dot(x_centered, x_centered))
        if (not np.isfinite(denom)) or denom <= 1e-12:
            continue

        beta = float(np.dot(x_centered, y_centered) / denom)
        intercept = float(np.nanmean(y_t) - beta * np.nanmean(x_t))
        betas[t_idx] = np.float32(beta)
        intercepts[t_idx] = np.float32(intercept)

        if int(n_permutations) > 0:
            perm_betas = np.full(int(n_permutations), np.nan, dtype=np.float64)
            for p_idx in range(int(n_permutations)):
                x_perm = x_t[rng.permutation(x_t.shape[0])]
                x_perm_centered = x_perm - np.nanmean(x_perm)
                denom_perm = float(np.dot(x_perm_centered, x_perm_centered))
                if (not np.isfinite(denom_perm)) or denom_perm <= 1e-12:
                    continue
                perm_betas[p_idx] = float(np.dot(x_perm_centered, y_centered) / denom_perm)

            perm_betas = perm_betas[np.isfinite(perm_betas)]
            if perm_betas.size > 0:
                mu = float(np.nanmean(perm_betas))
                sigma = float(np.nanstd(perm_betas))
                if (not np.isfinite(sigma)) or sigma < 1e-8:
                    sigma = 1e-8
                null_mean[t_idx] = np.float32(mu)
                null_std[t_idx] = np.float32(sigma)
                z_betas[t_idx] = np.float32((beta - mu) / sigma)
                p_betas[t_idx] = np.float32(
                    (np.sum(np.abs(perm_betas) >= np.abs(beta)) + 1.0) / (perm_betas.size + 1.0)
                )
        else:
            z_betas[t_idx] = np.float32(beta)

    return {
        'beta': betas,
        'z_beta': z_betas,
        'p_beta': p_betas,
        'null_mean': null_mean,
        'null_std': null_std,
        'intercept': intercepts,
        'n_valid_trials': n_valid_trials,
    }


def _compute_phase_bin_profile(
    phase_vals: np.ndarray,
    value_vals: np.ndarray,
    phase_bins: np.ndarray,
) -> np.ndarray:
    """Mean values per phase bin with NaNs for empty bins."""
    phase_vals = np.asarray(phase_vals, dtype=np.float64).reshape(-1)
    value_vals = np.asarray(value_vals, dtype=np.float64).reshape(-1)
    if phase_vals.shape != value_vals.shape:
        raise ValueError("phase_vals and value_vals must have matching shape")
    bin_idx = np.digitize(phase_vals, phase_bins, right=False) - 1
    n_bins = len(phase_bins) - 1
    profile = np.full(n_bins, np.nan, dtype=np.float64)
    for b_idx in range(n_bins):
        mask = bin_idx == b_idx
        if np.any(mask):
            profile[b_idx] = np.nanmean(value_vals[mask])
    return profile


def _modulation_index_from_profile(profile: np.ndarray) -> float:
    """Compute Tort-style modulation index after positive shifting."""
    prof = np.asarray(profile, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prof)
    if int(valid.sum()) == 0:
        return np.nan
    fill_value = float(np.nanmean(prof[valid]))
    prof = np.where(valid, prof, fill_value)
    prof = prof - np.nanmin(prof)
    prof = prof + 1e-6
    total = float(np.sum(prof))
    if (not np.isfinite(total)) or total <= 0:
        return np.nan
    probs = prof / total
    n_bins = probs.size
    return float(np.sum(probs * np.log((probs + 1e-12) * n_bins)) / np.log(n_bins))


def compute_time_resolved_theta_phase_reactivation_mi(
    theta_phase: np.ndarray,
    reactivation: np.ndarray,
    trial_mask: Optional[np.ndarray] = None,
    n_bins: int = 18,
    n_permutations: int = 200,
    seed: int = 42,
    min_trials: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Compute time-resolved PAC-style MI between theta phase and trial-wise reactivation.
    """
    phase_arr = np.asarray(theta_phase, dtype=np.float64)
    reac_arr = np.asarray(reactivation, dtype=np.float64)
    if phase_arr.shape != reac_arr.shape:
        raise ValueError(f"Shape mismatch: theta_phase={phase_arr.shape}, reactivation={reac_arr.shape}")
    if phase_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays (time, trial), got {phase_arr.ndim}D")

    phase_bins = np.linspace(-np.pi, np.pi, int(n_bins) + 1, dtype=np.float64)
    phase_centers = ((phase_bins[:-1] + phase_bins[1:]) / 2.0).astype(np.float32)
    n_times = phase_arr.shape[0]
    mi = np.full(n_times, np.nan, dtype=np.float32)
    z_mi = np.full(n_times, np.nan, dtype=np.float32)
    pref_phase = np.full(n_times, np.nan, dtype=np.float32)
    n_valid_trials = np.zeros(n_times, dtype=np.int32)
    profiles = np.full((n_times, len(phase_centers)), np.nan, dtype=np.float32)
    null_mean = np.full(n_times, np.nan, dtype=np.float32)
    null_std = np.full(n_times, np.nan, dtype=np.float32)

    base_mask = None
    if trial_mask is not None:
        base_mask = np.asarray(trial_mask, dtype=bool).reshape(-1)
        if base_mask.shape[0] != phase_arr.shape[1]:
            raise ValueError(
                f"trial_mask length mismatch: len={base_mask.shape[0]}, trials={phase_arr.shape[1]}"
            )

    rng = np.random.default_rng(seed)
    for t_idx in range(n_times):
        mask = np.isfinite(phase_arr[t_idx]) & np.isfinite(reac_arr[t_idx])
        if base_mask is not None:
            mask &= base_mask
        n_valid = int(mask.sum())
        n_valid_trials[t_idx] = np.int32(n_valid)
        if n_valid < int(min_trials):
            continue

        phase_t = phase_arr[t_idx, mask]
        reac_t = reac_arr[t_idx, mask]
        profile = _compute_phase_bin_profile(phase_t, reac_t, phase_bins)
        profiles[t_idx] = profile.astype(np.float32, copy=False)
        obs_mi = _modulation_index_from_profile(profile)
        if not np.isfinite(obs_mi):
            continue
        mi[t_idx] = np.float32(obs_mi)
        finite_profile = np.where(np.isfinite(profile), profile, -np.inf)
        pref_idx = int(np.argmax(finite_profile))
        pref_phase[t_idx] = np.float32(phase_centers[pref_idx])

        if n_permutations <= 0:
            z_mi[t_idx] = np.float32(obs_mi)
            continue

        perm_vals = np.full(int(n_permutations), np.nan, dtype=np.float64)
        for p_idx in range(int(n_permutations)):
            shuffled = reac_t[rng.permutation(reac_t.shape[0])]
            perm_profile = _compute_phase_bin_profile(phase_t, shuffled, phase_bins)
            perm_vals[p_idx] = _modulation_index_from_profile(perm_profile)

        mu = float(np.nanmean(perm_vals))
        sigma = float(np.nanstd(perm_vals))
        if (not np.isfinite(sigma)) or sigma < 1e-8:
            sigma = 1e-8
        null_mean[t_idx] = np.float32(mu)
        null_std[t_idx] = np.float32(sigma)
        z_mi[t_idx] = np.float32((obs_mi - mu) / sigma)

    return {
        'mi': mi,
        'z_mi': z_mi,
        'preferred_phase': pref_phase,
        'phase_profile': profiles,
        'phase_centers': phase_centers,
        'n_valid_trials': n_valid_trials,
        'null_mean': null_mean,
        'null_std': null_std,
    }


def compute_subject_univariate_reactivation(
    subject_obj: Subject,
    roi: str,
    predictor: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    include_intercept: bool = False,
    trial_save_dtype: type = np.float32,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Univariate encoding + projection for one subject/ROI.

    Encoding is single-electrode timewise OLS:
        neural_tfr ~ 1 + predictor + covariates
    Projection is electrode-wise:
        weight(feedback_time, electrode) * cue_activity(electrode, cue_time)

    Surrogates use Freedman-Lane-style pseudo-responses generated from
    reduced-model residual permutations of Y ~ 1 + covariates.
    """
    covariates = covariates or ['zpm']
    fb_hfa, fb_times, electrode_labels = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_encoding,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms
    )
    cue_hfa, cue_times, _ = _get_smoothed_event_signal(
        subject_obj=subject_obj,
        roi=roi,
        event=event_projection,
        freq_range=freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms
    )
    regressors, _ = _build_subject_regressors(
        subject_obj=subject_obj,
        event=event_encoding,
        predictor=predictor,
        covariates=covariates,
        regressor_source=regressor_source,
        trial_var_override=trial_var_override,
    )

    trial_ix = regressors['trial'].values - 1
    valid = (trial_ix >= 0) & (trial_ix < fb_hfa.shape[0]) & (trial_ix < cue_hfa.shape[0])
    regressors = regressors.loc[valid].reset_index(drop=True)
    trial_ix = trial_ix[valid]
    if len(trial_ix) < subject_obj.min_trials:
        raise ValueError(
            f"Insufficient aligned trials for {subject_obj.subject}, ROI={roi}: "
            f"{len(trial_ix)} < {subject_obj.min_trials}"
        )

    Y = fb_hfa[trial_ix]  # (n_trials, n_electrodes, n_feedback_t)
    cue_aligned = cue_hfa[trial_ix]  # (n_trials, n_electrodes, n_cue_t)
    n_trials, n_electrodes, n_fb = Y.shape
    n_cue = cue_aligned.shape[-1]

    cov_cols = [c for c in covariates if c in regressors.columns and c != predictor]
    x_cols = [predictor] + cov_cols
    X = regressors[x_cols].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones((X.shape[0], 1), dtype=np.float64), X])

    pinv_X = np.linalg.pinv(X)
    Y2 = Y.reshape(n_trials, -1).astype(np.float64)
    B = pinv_X @ Y2  # (n_regressors+1, n_electrodes*n_fb)

    predictor_idx = 1
    intercepts = B[0].reshape(n_electrodes, n_fb).T.astype(np.float32)
    weights = B[predictor_idx].reshape(n_electrodes, n_fb).T.astype(np.float32)

    projection_real = np.einsum('nec,fe->fcen', cue_aligned.astype(np.float32), weights)
    if include_intercept:
        projection_real = projection_real + intercepts[:, np.newaxis, :, np.newaxis]

    if not use_surrogates or n_surrogates <= 0:
        z_projection = projection_real.astype(trial_save_dtype, copy=False)
        null_weights = None
    else:
        rng = np.random.default_rng(seed)
        running_mean = np.zeros_like(projection_real, dtype=np.float32)
        running_M2 = np.zeros_like(projection_real, dtype=np.float32)
        null_weights_list = []
        n_total = 0
        C = regressors[cov_cols].to_numpy(dtype=np.float64) if len(cov_cols) > 0 else None
        fl_state = _prepare_freedman_lane_state(
            y=Y2,
            nuisance=C,
            include_intercept=True,
        )

        for _ in range(n_surrogates):
            perm_ix = rng.permutation(n_trials)
            Y_star = _freedman_lane_pseudo_response(
                y_hat0=fl_state['y_hat0'],
                resid0=fl_state['resid0'],
                perm_ix=perm_ix,
            )
            Bs = pinv_X @ np.asarray(Y_star, dtype=np.float64)
            ws = Bs[predictor_idx].reshape(n_electrodes, n_fb).T.astype(np.float32)
            null_weights_list.append(ws)

            proj_s = np.einsum('nec,fe->fcen', cue_aligned.astype(np.float32), ws)
            if include_intercept:
                ints = Bs[0].reshape(n_electrodes, n_fb).T.astype(np.float32)
                proj_s = proj_s + ints[:, np.newaxis, :, np.newaxis]

            n_total += 1
            delta = proj_s - running_mean
            running_mean += delta / np.float32(n_total)
            delta2 = proj_s - running_mean
            running_M2 += delta * delta2

        surrogate_std = np.sqrt(running_M2 / np.float32(max(n_total, 1)))
        surrogate_std = np.maximum(surrogate_std, 1e-8)
        z_projection = ((projection_real - running_mean) / surrogate_std).astype(trial_save_dtype, copy=False)
        null_weights = np.stack(null_weights_list, axis=0).astype(np.float32)

    memory_outcomes = np.array([])
    memory_df = subject_obj.combined_df[
        (subject_obj.combined_df.participant == subject_obj.subject) &
        (subject_obj.combined_df.condition == 'Day1')
    ].copy()
    if len(memory_df) > 0:
        if 'memory_outcome' not in memory_df.columns:
            memory_df['memory_outcome'] = memory_df[
                ['hits', 'misses', 'correct_rejections', 'false_alarms']
            ].idxmax(axis=1)
        trial_col = _resolve_trial_var_for_event(memory_df, event_projection)
        memory_df = memory_df.dropna(subset=[trial_col]).sort_values(by=[trial_col])
        mem_trials = memory_df[trial_col].astype(int).values - 1
        mem_map = dict(zip(mem_trials, memory_df['memory_outcome'].values))
        memory_outcomes = np.array([mem_map.get(int(ti), np.nan) for ti in trial_ix], dtype=object)

    return {
        'subject': subject_obj.subject,
        'roi': roi,
        'predictor': predictor,
        'event_encoding': event_encoding,
        'event_projection': event_projection,
        'weights': weights,
        'intercepts': intercepts,
        'projection_real': projection_real.astype(np.float32, copy=False),
        'z_projection': z_projection,
        'feedback_times': fb_times.astype(np.float32, copy=False),
        'cue_times': cue_times.astype(np.float32, copy=False),
        'electrode_labels': electrode_labels.astype(str),
        'trial_index': trial_ix.astype(np.int32),
        'memory_outcomes': memory_outcomes,
        'null_weights': null_weights,
    }


def run_univariate_reactivation_pipeline(
    subjects: List[str],
    rois: List[str],
    predictors: List[str],
    base_dir: str,
    load_dir: str,
    event_encoding: str = 'feedback_start',
    event_projection: str = 'cue_on',
    covariates: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (70., 201.),
    downsample_factor: int = 5,
    win_ms: int = 200,
    min_trials: int = 10,
    n_surrogates: int = 250,
    use_surrogates: bool = True,
    seed: int = 42,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    n_jobs_outer: int = 1,
    verbose: bool = True,
    regressor_source: Optional[str] = None,
    trial_var_override: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Subject-first univariate encoding/projection pipeline.
    """
    covariates = covariates or ['zpm']
    tasks = [(roi, predictor) for roi in rois for predictor in predictors]

    def _run_subject(subject: str) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
        subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
        out = []
        for roi, predictor in tasks:
            try:
                result = compute_subject_univariate_reactivation(
                    subject_obj=subject_obj,
                    roi=roi,
                    predictor=predictor,
                    event_encoding=event_encoding,
                    event_projection=event_projection,
                    covariates=covariates,
                    freq_range=freq_range,
                    downsample_factor=downsample_factor,
                    win_ms=win_ms,
                    n_surrogates=n_surrogates,
                    use_surrogates=use_surrogates,
                    seed=seed,
                    trial_save_dtype=trial_save_dtype,
                    regressor_source=regressor_source,
                    trial_var_override=trial_var_override,
                )
                if save_trial_level and trial_save_dir is not None:
                    roi_dir = os.path.join(trial_save_dir, "univariate", roi)
                    os.makedirs(roi_dir, exist_ok=True)
                    out_file = os.path.join(
                        roi_dir, f"{subject}_{predictor}_zprojection.npz"
                    )
                    if overwrite_trial_level or (not os.path.exists(out_file)):
                        np.savez_compressed(
                            out_file,
                            z_projection=result['z_projection'].astype(trial_save_dtype, copy=False),
                            projection_real=result['projection_real'].astype(np.float32, copy=False),
                            feedback_times=result['feedback_times'],
                            cue_times=result['cue_times'],
                            weights=result['weights'],
                            intercepts=result['intercepts'],
                            electrode_labels=result['electrode_labels'].astype(str),
                            trial_index=result['trial_index'].astype(np.int32),
                            memory_outcomes=result['memory_outcomes'],
                            predictor=predictor,
                            roi=roi,
                            subject=subject,
                            event_encoding=event_encoding,
                            event_projection=event_projection,
                            n_surrogates=int(n_surrogates),
                            seed=int(seed),
                            use_surrogates=bool(use_surrogates),
                        )
                out.append((roi, predictor, result))
            except Exception as e:
                if verbose:
                    print(f"Univariate processing failed: {subject}, {roi}, {predictor}: {e}")
                out.append((roi, predictor, None))
        return out

    if n_jobs_outer == 1:
        batch_results = [_run_subject(s) for s in subjects]
    else:
        batch_results = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
            delayed(_run_subject)(s) for s in subjects
        )

    results = {
        roi: {
            predictor: {
                'subject_list': [],
                'feedback_times_list': [],
                'cue_times_list': [],
                'weights_list': [],
                'memory_outcomes': [],
            }
            for predictor in predictors
        }
        for roi in rois
    }
    for subject_batch in batch_results:
        for roi, predictor, result in subject_batch:
            if result is None:
                continue
            results[roi][predictor]['subject_list'].append(result['subject'])
            results[roi][predictor]['feedback_times_list'].append(result['feedback_times'])
            results[roi][predictor]['cue_times_list'].append(result['cue_times'])
            results[roi][predictor]['weights_list'].append(result['weights'])
            results[roi][predictor]['memory_outcomes'].append(result['memory_outcomes'])
    return results


def build_legacy_single_elec_csv_path(
    load_dir: str,
    band: str,
    event: str,
    predictor: str,
    suffix: str = 'smoothed_final'
) -> str:
    """Build legacy-compatible single-electrode regression CSV path."""
    return os.path.join(
        load_dir, f"single_elec_time_resolved_perm_{band}_{event}_{predictor}_{suffix}.csv"
    )


def load_univariate_trial_level(
    roi: str,
    subject: str,
    trial_save_dir: str,
    predictor: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load univariate trial-level projection file.

    Returns
    -------
    z_projection : array, shape (n_feedback_t, n_cue_t, n_electrodes, n_trials)
    feedback_times : array
    cue_times : array
    meta : dict
    """
    path = os.path.join(trial_save_dir, "univariate", roi, f"{subject}_{predictor}_zprojection.npz")
    npz = np.load(path, allow_pickle=True)
    z_projection = np.asarray(npz['z_projection'])
    feedback_times = np.asarray(npz['feedback_times'])
    cue_times = np.asarray(npz['cue_times'])
    meta = {k: npz[k] for k in npz.files if k not in {'z_projection', 'feedback_times', 'cue_times'}}
    return z_projection, feedback_times, cue_times, meta


def summarize_univariate_hits_misses(
    z_projection: np.ndarray,
    memory_outcomes: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Summarize hits-vs-misses contrasts from univariate projection.

    Parameters
    ----------
    z_projection : array
        Expected shape (n_fb, n_cue, n_electrodes, n_trials).
    memory_outcomes : array
        Trial labels including 'hits' and 'misses'.

    Returns
    -------
    dict
        Keys: hits_mask, misses_mask, hits_mean_per_electrode, misses_mean_per_electrode,
        diff_per_electrode, hits_mean_agg, misses_mean_agg, diff_agg.
    """
    z_projection = np.asarray(z_projection)
    memory_outcomes = np.asarray(memory_outcomes)
    if z_projection.ndim != 4:
        raise ValueError(f"Expected 4D z_projection, got {z_projection.ndim}D")
    if z_projection.shape[-1] != len(memory_outcomes):
        raise ValueError(
            f"Trial mismatch: z_projection has {z_projection.shape[-1]} trials, "
            f"memory_outcomes has {len(memory_outcomes)}"
        )

    hits_mask = memory_outcomes == 'hits'
    misses_mask = memory_outcomes == 'misses'
    if np.sum(hits_mask) == 0 or np.sum(misses_mask) == 0:
        raise ValueError("Need at least one hit and one miss trial.")

    hits_mean_per_electrode = np.mean(z_projection[:, :, :, hits_mask], axis=3)
    misses_mean_per_electrode = np.mean(z_projection[:, :, :, misses_mask], axis=3)
    diff_per_electrode = hits_mean_per_electrode - misses_mean_per_electrode

    hits_mean_agg = np.mean(hits_mean_per_electrode, axis=2)
    misses_mean_agg = np.mean(misses_mean_per_electrode, axis=2)
    diff_agg = hits_mean_agg - misses_mean_agg

    return {
        'hits_mask': hits_mask,
        'misses_mask': misses_mask,
        'hits_mean_per_electrode': hits_mean_per_electrode,
        'misses_mean_per_electrode': misses_mean_per_electrode,
        'diff_per_electrode': diff_per_electrode,
        'hits_mean_agg': hits_mean_agg,
        'misses_mean_agg': misses_mean_agg,
        'diff_agg': diff_agg,
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_cluster_test_per_roi(
    diff_arrays: List[np.ndarray],
    feedback_times_list: List[np.ndarray],
    cue_times_list: List[np.ndarray],
    n_permutations: int = 1024,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], Optional[List], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Aggregate difference arrays across subjects and run 1-sample permutation cluster test.
    
    Parameters:
    -----------
    diff_arrays : list of arrays
        Each array is (n_feedback_t, n_cue_t) difference for one subject
    feedback_times_list : list of arrays
        Time vectors for feedback period (one per subject)
    cue_times_list : list of arrays
        Time vectors for cue period (one per subject)
    n_permutations : int, default=1024
        Number of permutations for cluster test
    seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    t_obs : array or None, shape (n_fb_common, n_cue_common)
        T-statistic map
    clusters : list or None
        Cluster objects from MNE
    cluster_pvals : array or None
        P-values for each cluster
    h0 : array or None
        Null distribution
    feedback_times_common : array or None
        Common feedback time vector
    cue_times_common : array or None
        Common cue time vector
    D : array or None, shape (n_subjects, n_fb_common, n_cue_common)
        Aggregated difference array
    """
    if len(diff_arrays) == 0:
        if verbose:
            print("No valid subjects for this ROI")
        return None, None, None, None, None, None, None
    
    # Stack into (n_subjects, n_fb, n_cue)
    D = np.stack(diff_arrays, axis=0)
    n_subjects = D.shape[0]
    
    if verbose:
        print(f"Aggregated {n_subjects} subjects into array of shape {D.shape}")
    
    # Use first subject's time vectors (assuming all have same dimensions)
    feedback_times_common = feedback_times_list[0]
    cue_times_common = cue_times_list[0]
    
    if verbose:
        print(f"Running 1-sample permutation cluster test ({n_permutations} permutations)...")
    
    # Run 1-sample cluster test
    t_obs, clusters, cluster_pvals, h0 = permutation_cluster_1samp_test(
        D,
        n_permutations=n_permutations,
        tail=0,  # One-tailed test
        n_jobs=-1,
        seed=seed,
        verbose=verbose
    )
    
    # Reshape t_obs back to 2D if needed
    if t_obs.ndim > 2:
        t_obs = t_obs.reshape(D.shape[1], D.shape[2])
    
    return t_obs, clusters, cluster_pvals, h0, feedback_times_common, cue_times_common, D


def run_2sample_cluster_test_per_roi(
    roi: str,
    trial_save_dir: str,
    dependent_var: Optional[str] = None,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run 2-sample permutation cluster test (hits vs misses) for an ROI.
    
    Aggregates all trials across subjects and tests if hits > misses.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    trial_save_dir : str
        Directory containing saved trial-level data
    n_permutations : int, default=1024
        Number of permutations
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'roi': str
        - 't_obs': array
        - 'clusters': list
        - 'cluster_pvals': array
        - 'cue_times': array
        - 'feedback_times': array
        - 'subjects': list
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"2-SAMPLE CLUSTER TEST (HITS vs MISSES) FOR ROI: {roi}")
        print(f"{'='*70}")
    
    # Load all trial-level data
    hits_data, misses_data, fb_times_list, cue_times_list, subjects = load_all_trial_level_per_roi(
        roi=roi,
        trial_save_dir=trial_save_dir,
        dependent_var=dependent_var,
        aggregate_electrodes=True,
        verbose=verbose
    )

    if len(hits_data) == 0 or len(misses_data) == 0:
        if verbose:
            print(f"No valid hits/misses data found for ROI {roi}")
        return {
            'roi': roi,
            't_obs': None,
            'clusters': [],
            'cluster_pvals': np.array([]),
            'cue_times': None,
            'feedback_times': None,
            'subjects': []
        }
    
    # Get common time arrays (use median subject)
    fb_dims = [len(ft) for ft in fb_times_list]
    cue_dims = [len(ct) for ct in cue_times_list]
    median_idx = np.argsort(fb_dims)[len(fb_dims)//2]
    feedback_times_common = fb_times_list[median_idx]
    cue_times_common = cue_times_list[median_idx]
    
    # Run 2-sample test: hits vs misses
    rearanged_hits = np.transpose(np.concatenate(hits_data, axis=2), (2, 0, 1))
    rearanged_misses = np.transpose(np.concatenate(misses_data, axis=2), (2, 0, 1))
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [rearanged_hits, rearanged_misses],
        n_permutations=n_permutations,
        tail=0,  # One-tailed: hits > misses
        seed=cluster_seed,
        verbose=False
    )
    
    if verbose:
        n_sig = np.sum(cluster_p_values <= 0.05)
        print(f"ROI {roi}: {n_sig} significant clusters (p <= 0.05)")
        if len(cluster_p_values) > 0:
            print(f"  Cluster p-values: {cluster_p_values}")
    
    return {
        'roi': roi,
        't_obs': T_obs,
        'clusters': clusters,
        'cluster_pvals': cluster_p_values,
        'cue_times': cue_times_common,
        'feedback_times': feedback_times_common,
        'subjects': subjects
    }


def run_subject_level_1sample_test(
    roi: str,
    subject: str,
    trial_save_dir: str,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    demean_subject: bool = False,
    verbose: bool = True,
    aggregate_electrodes: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run 1-sample permutation cluster test for a single subject.
    
    Tests if mean z-reactivation differs from zero across all trials.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    subject : str
        Subject identifier
    trial_save_dir : str
        Directory containing saved trial-level data
    n_permutations : int, default=1024
        Number of permutations
    cluster_seed : int, default=42
        Random seed for permutations
    demean_subject : bool, default=False
        If True, subtract subject mean before testing
    verbose : bool, default=True
        Print progress
    aggregate_electrodes : bool, default=True
        If True and data is electrode-level (4D), aggregate across electrodes (mean).
        If False, process each electrode separately (not implemented yet).
        
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'subject': str
        - 'roi': str
        - 't_obs': array
        - 'clusters': list
        - 'cluster_pvals': array
        - 'feedback_times': array
        - 'cue_times': array
        - 'n_trials': int
    """
    try:
        z_reactivation, fb_times, cue_times, meta = load_trial_level(
            roi, subject, trial_save_dir
        )
        
        # Handle both 3D (vector-level) and 4D (electrode-level) arrays
        if z_reactivation.ndim == 4:
            # Electrode-level: (n_fb, n_cue, n_electrodes, n_trials)
            if aggregate_electrodes:
                # Aggregate across electrodes (mean)
                z_reactivation = np.mean(z_reactivation, axis=2)  # (n_fb, n_cue, n_trials)
            else:
                # For now, raise error - per-electrode processing not implemented
                raise NotImplementedError("Per-electrode processing not yet implemented. Use aggregate_electrodes=True.")
        
        # Reshape: (n_fb, n_cue, n_trials) -> (n_trials, n_fb*n_cue)
        n_fb, n_cue, n_trials = z_reactivation.shape
        
        rearanged_z_reactivation = np.transpose(z_reactivation, (2, 0, 1))
        # Run 1-sample test: test if mean differs from 0
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            rearanged_z_reactivation,
            n_permutations=n_permutations,
            tail=0,  # Two-tailed
            seed=cluster_seed,
            verbose=False
        )
        
        # Reshape T_obs back to 2D
        t_obs = T_obs.reshape(n_fb, n_cue)
        
        return {
            'subject': subject,
            'roi': roi,
            't_obs': t_obs,
            'clusters': clusters,
            'cluster_pvals': cluster_p_values,
            'feedback_times': fb_times,
            'cue_times': cue_times,
            'n_trials': n_trials
        }
        
    except Exception as e:
        if verbose:
            print(f"Error processing {subject}: {e}")
        return None


def run_subject_wise_2sample_cluster_test(
    roi: str,
    subject: str,
    dependent_var: str,
    trial_save_dir: str,
    memory_outcomes: np.ndarray,
    comparison: str,  # 'hits_vs_misses' or 'hits_vs_false_alarms'
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = True,
    aggregate_electrodes: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Run 2-sample permutation cluster test for a single subject.
    
    Tests if z-reactivation differs between two memory outcome groups (hits vs misses
    or hits vs false_alarms) for a single subject.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    subject : str
        Subject identifier
    dependent_var : str
        Name of the dependent variable used for encoding models
    trial_save_dir : str
        Directory containing saved trial-level data
    memory_outcomes : array, shape (n_trials,)
        Memory outcome labels ('hits', 'misses', 'false_alarms', 'correct_rejections')
    comparison : str
        Comparison type: 'hits_vs_misses' or 'hits_vs_false_alarms'
    n_permutations : int, default=1024
        Number of permutations
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
    aggregate_electrodes : bool, default=True
        If True and data is electrode-level (4D), aggregate across electrodes (mean).
        If False, process each electrode separately (not implemented yet).
        
    Returns:
    --------
    dict or None
        Dictionary containing:
        - 'subject': str
        - 'roi': str
        - 'dependent_var': str
        - 'comparison': str
        - 'clusters': list
        - 'cluster_pvals': array
        - 'cluster_statistics': list of floats (sum of t-statistics within each significant cluster)
        - 'n_significant_clusters': int
        - 't_obs': array, shape (n_fb, n_cue)
        - 'feedback_times': array
        - 'cue_times': array
    """
    try:
        z_reactivation, fb_times, cue_times, meta = load_trial_level(
            roi, subject, trial_save_dir, dependent_var=dependent_var
        )
        
        # Handle both 3D (vector-level) and 4D (electrode-level) arrays
        if z_reactivation.ndim == 4:
            # Electrode-level: (n_fb, n_cue, n_electrodes, n_trials)
            if aggregate_electrodes:
                # Aggregate across electrodes (mean)
                z_reactivation = np.mean(z_reactivation, axis=2)  # (n_fb, n_cue, n_trials)
            else:
                # For now, raise error - per-electrode processing not implemented
                raise NotImplementedError("Per-electrode processing not yet implemented. Use aggregate_electrodes=True.")
        
        # Check that memory_outcomes length matches z_reactivation trials
        n_fb, n_cue, n_trials = z_reactivation.shape
        if len(memory_outcomes) != n_trials:
            if verbose:
                print(f"Warning: {subject} has {len(memory_outcomes)} memory outcomes but {n_trials} trials in z_reactivation")
            return None
        
        # Filter trials based on comparison type
        if comparison == 'hits_vs_misses':
            group1_mask = memory_outcomes == 'hits'
            group2_mask = memory_outcomes == 'misses'
        elif comparison == 'hits_vs_false_alarms':
            group1_mask = memory_outcomes == 'hits'
            group2_mask = memory_outcomes == 'false_alarms'
        else:
            raise ValueError(f"Unknown comparison type: {comparison}")
        
        # Extract data for each group
        group1_data = z_reactivation[:, :, group1_mask]  # (n_fb, n_cue, n_group1)
        group2_data = z_reactivation[:, :, group2_mask]  # (n_fb, n_cue, n_group2)
        
        if group1_data.shape[2] < 2 or group2_data.shape[2] < 2:
            if verbose:
                print(f"Warning: {subject} has insufficient trials for {comparison}: {group1_data.shape[2]} vs {group2_data.shape[2]}")
            return None
        
        # Reshape for cluster test: (n_fb, n_cue, n_trials) -> (n_trials, n_fb*n_cue)
        # Flatten spatial dimensions
        group1_reshaped = np.transpose(group1_data, (2, 0, 1))  # (n_group1, n_fb, n_cue)
        group1_reshaped = group1_reshaped.reshape(group1_reshaped.shape[0], -1)  # (n_group1, n_fb*n_cue)
        
        group2_reshaped = np.transpose(group2_data, (2, 0, 1))  # (n_group2, n_fb, n_cue)
        group2_reshaped = group2_reshaped.reshape(group2_reshaped.shape[0], -1)  # (n_group2, n_fb*n_cue)
        
        # Run 2-sample cluster test
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [group1_reshaped, group2_reshaped],
            n_permutations=n_permutations,
            tail=1,  # Two-tailed test
            seed=cluster_seed,
            verbose=False
        )
        
        # Reshape T_obs back to 2D
        t_obs = T_obs.reshape(n_fb, n_cue)
        
        # Extract cluster statistics for significant clusters only
        cluster_statistics = []
        n_significant = 0
        
        if len(clusters) > 0:
            for cluster, pval in zip(clusters, cluster_p_values):
                if pval <= 0.05:
                    n_significant += 1
                    # Get cluster indices (flattened)
                    if isinstance(cluster, (list, tuple, np.ndarray)):
                        cluster_indices = np.array(cluster)
                    elif hasattr(cluster, '__iter__'):
                        cluster_indices = np.array(list(cluster))
                    else:
                        cluster_indices = np.array([cluster])
                    
                    # Extract t-statistics within this cluster and sum them
                    cluster_t_stats = T_obs[cluster_indices]
                    cluster_stat = np.sum(cluster_t_stats)
                    cluster_statistics.append(cluster_stat)
        
        return {
            'subject': subject,
            'roi': roi,
            'dependent_var': dependent_var,
            'comparison': comparison,
            'clusters': clusters,
            'cluster_pvals': cluster_p_values,
            'cluster_statistics': cluster_statistics,
            'n_significant_clusters': n_significant,
            't_obs': t_obs,
            'feedback_times': fb_times,
            'cue_times': cue_times
        }
        
    except Exception as e:
        if verbose:
            print(f"Error processing {subject} ({comparison}): {e}")
        return None


def compute_all_subjects_cluster_statistics(
    roi: str,
    dependent_var: str,
    subjects: List[str],
    memory_outcomes_dict: Dict[str, np.ndarray],  # subject -> memory_outcomes array
    trial_save_dir: str,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute cluster statistics for all subjects and both comparisons.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    dependent_var : str
        Name of the dependent variable used for encoding models
    subjects : list of str
        List of subject identifiers
    memory_outcomes_dict : dict
        Dictionary mapping subject -> memory_outcomes array
    trial_save_dir : str
        Directory containing saved trial-level data
    n_permutations : int, default=1024
        Number of permutations for cluster test
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    dict
        Nested dictionary: {'hits_vs_misses': {subject: [cluster_stats]}, 
                            'hits_vs_false_alarms': {subject: [cluster_stats]}}
        Only includes significant cluster statistics (p <= 0.05)
    """
    results = {
        'hits_vs_misses': {},
        'hits_vs_false_alarms': {}
    }
    
    comparisons = ['hits_vs_misses', 'hits_vs_false_alarms']
    
    for comparison in comparisons:
        if verbose:
            print(f"\nProcessing {comparison} for ROI {roi}, dependent_var {dependent_var}...")
        
        for subject in subjects:
            if subject not in memory_outcomes_dict:
                if verbose:
                    print(f"  Skipping {subject}: no memory outcomes available")
                continue
            
            memory_outcomes = memory_outcomes_dict[subject]
            
            result = run_subject_wise_2sample_cluster_test(
                roi=roi,
                subject=subject,
                dependent_var=dependent_var,
                trial_save_dir=trial_save_dir,
                memory_outcomes=memory_outcomes,
                comparison=comparison,
                n_permutations=n_permutations,
                cluster_seed=cluster_seed,
                verbose=False
            )
            
            if result is not None and len(result['cluster_statistics']) > 0:
                results[comparison][subject] = result['cluster_statistics']
                if verbose:
                    print(f"  {subject}: {result['n_significant_clusters']} significant clusters")
            elif result is not None:
                # Subject processed but no significant clusters
                if verbose:
                    print(f"  {subject}: 0 significant clusters")
            else:
                if verbose:
                    print(f"  {subject}: failed to process")
    
    return results


def run_per_electrode_2sample_cluster_test(
    z_reactivation: np.ndarray,
    memory_outcomes: np.ndarray,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = False
) -> np.ndarray:
    """
    Run 2-sample permutation cluster test (hits vs misses) for each electrode separately.
    
    Tests if z-reactivation differs between hits and misses for each electrode independently.
    Returns the count of significant clusters per electrode.
    
    Parameters:
    -----------
    z_reactivation : array, shape (n_feedback_t, n_cue_t, n_electrodes, n_trials)
        Z-scored reactivation data (must be 4D, electrode-level)
    memory_outcomes : array, shape (n_trials,)
        Memory outcome labels ('hits', 'misses', 'false_alarms', 'correct_rejections')
    n_permutations : int, default=1024
        Number of permutations
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=False
        Print progress
        
    Returns:
    --------
    significant_cluster_counts : array, shape (n_electrodes,)
        Count of significant clusters (p <= 0.05) for each electrode
    """
    if z_reactivation.ndim != 4:
        raise ValueError(f"Expected 4D array (electrode-level), got {z_reactivation.ndim}D")
    
    n_fb, n_cue, n_electrodes, n_trials = z_reactivation.shape
    
    if len(memory_outcomes) != n_trials:
        raise ValueError(f"Memory outcomes length ({len(memory_outcomes)}) doesn't match n_trials ({n_trials})")
    
    # Filter by hits and misses
    hits_mask = memory_outcomes == 'hits'
    misses_mask = memory_outcomes == 'misses'
    
    n_hits = np.sum(hits_mask)
    n_misses = np.sum(misses_mask)
    
    if n_hits < 2 or n_misses < 2:
        if verbose:
            print(f"Warning: Insufficient trials (hits: {n_hits}, misses: {n_misses})")
        return np.zeros(n_electrodes)
    
    # Store significant cluster counts for each electrode
    significant_cluster_counts = np.zeros(n_electrodes, dtype=int)
    
    for electrode_idx in range(n_electrodes):
        # Extract data for this electrode: (n_fb, n_cue, n_trials)
        electrode_data = z_reactivation[:, :, electrode_idx, :]
        
        # Extract hits and misses for this electrode
        hits_data = electrode_data[:, :, hits_mask]  # (n_fb, n_cue, n_hits)
        misses_data = electrode_data[:, :, misses_mask]  # (n_fb, n_cue, n_misses)
        
        # Reshape for cluster test: (n_fb, n_cue, n_trials) -> (n_trials, n_fb*n_cue)
        hits_reshaped = np.transpose(hits_data, (2, 0, 1))  # (n_hits, n_fb, n_cue)
        hits_reshaped = hits_reshaped.reshape(hits_reshaped.shape[0], -1)  # (n_hits, n_fb*n_cue)
        
        misses_reshaped = np.transpose(misses_data, (2, 0, 1))  # (n_misses, n_fb, n_cue)
        misses_reshaped = misses_reshaped.reshape(misses_reshaped.shape[0], -1)  # (n_misses, n_fb*n_cue)
        
        # Run 2-sample cluster test: tail=1 means testing if hits > misses
        try:
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [hits_reshaped, misses_reshaped],
                n_permutations=n_permutations,
                tail=1,  # One-tailed: hits > misses
                seed=cluster_seed,
                verbose=False
            )
            
            # Count significant clusters
            n_sig = np.sum(cluster_p_values <= 0.05) if len(cluster_p_values) > 0 else 0
            significant_cluster_counts[electrode_idx] = n_sig
            
        except Exception as e:
            if verbose:
                print(f"Error processing electrode {electrode_idx}: {e}")
            significant_cluster_counts[electrode_idx] = 0
    
    return significant_cluster_counts


def run_per_electrode_2sample_cluster_test_with_t_obs(
    z_reactivation: np.ndarray,
    memory_outcomes: np.ndarray,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = False
) -> tuple:
    """
    Run 2-sample permutation cluster test (hits vs misses) per electrode and return
    cluster counts and observed t-statistic maps.
    
    Same as run_per_electrode_2sample_cluster_test but also returns t_obs reshaped
    to (n_fb, n_cue) for each electrode.
    
    Parameters:
    -----------
    z_reactivation : array, shape (n_feedback_t, n_cue_t, n_electrodes, n_trials)
        Z-scored reactivation data (must be 4D, electrode-level)
    memory_outcomes : array, shape (n_trials,)
        Memory outcome labels ('hits', 'misses', ...)
    n_permutations : int, default=1024
        Number of permutations
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=False
        Print progress
        
    Returns:
    --------
    significant_cluster_counts : array, shape (n_electrodes,)
        Count of significant clusters (p <= 0.05) per electrode
    t_obs_maps : array, shape (n_electrodes, n_fb, n_cue)
        Observed t-statistic map per electrode (feedback_time x cue_time)
    """
    if z_reactivation.ndim != 4:
        raise ValueError(f"Expected 4D array (electrode-level), got {z_reactivation.ndim}D")
    
    n_fb, n_cue, n_electrodes, n_trials = z_reactivation.shape
    
    if len(memory_outcomes) != n_trials:
        raise ValueError(f"Memory outcomes length ({len(memory_outcomes)}) doesn't match n_trials ({n_trials})")
    
    hits_mask = memory_outcomes == 'hits'
    misses_mask = memory_outcomes == 'misses'
    n_hits = np.sum(hits_mask)
    n_misses = np.sum(misses_mask)
    
    if n_hits < 2 or n_misses < 2:
        if verbose:
            print(f"Warning: Insufficient trials (hits: {n_hits}, misses: {n_misses})")
        return np.zeros(n_electrodes, dtype=int), np.zeros((n_electrodes, n_fb, n_cue))
    
    significant_cluster_counts = np.zeros(n_electrodes, dtype=int)
    t_obs_maps = np.zeros((n_electrodes, n_fb, n_cue))
    
    for electrode_idx in range(n_electrodes):
        electrode_data = z_reactivation[:, :, electrode_idx, :]
        hits_data = electrode_data[:, :, hits_mask]
        misses_data = electrode_data[:, :, misses_mask]
        hits_reshaped = np.transpose(hits_data, (2, 0, 1))
        misses_reshaped = np.transpose(misses_data, (2, 0, 1))
        
        try:
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [hits_reshaped, misses_reshaped],
                n_permutations=n_permutations,
                tail=1,
                seed=cluster_seed,
                verbose=False
            )
            n_sig = np.sum(cluster_p_values <= 0.05) if len(cluster_p_values) > 0 else 0
            significant_cluster_counts[electrode_idx] = n_sig
            t_obs_maps[electrode_idx] = T_obs.reshape(n_fb, n_cue)
        except Exception as e:
            if verbose:
                print(f"Error processing electrode {electrode_idx}: {e}")
            significant_cluster_counts[electrode_idx] = 0
            t_obs_maps[electrode_idx] = 0.0
    
    return significant_cluster_counts, t_obs_maps


def compute_per_electrode_cluster_counts(
    roi: str,
    dependent_var: str,
    subjects: List[str],
    memory_outcomes_dict: Dict[str, np.ndarray],
    trial_save_dir: str,
    n_permutations: int = 1024,
    cluster_seed: int = 42,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Compute per-electrode cluster counts for all subjects in an ROI.
    
    For each subject, runs per-electrode 2-sample cluster tests (hits vs misses)
    and counts how many electrodes have at least one significant cluster.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    dependent_var : str
        Name of the dependent variable used for encoding models
    subjects : list of str
        List of subject identifiers
    memory_outcomes_dict : dict
        Dictionary mapping subject -> memory_outcomes array
    trial_save_dir : str
        Directory containing saved trial-level data
    n_permutations : int, default=1024
        Number of permutations for cluster test
    cluster_seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    dict
        Dictionary mapping subject -> count of electrodes with significant clusters
    """
    results = {}
    
    for subject in subjects:
        if subject not in memory_outcomes_dict:
            if verbose:
                print(f"  Skipping {subject}: no memory outcomes available")
            continue
        
        try:
            # Load trial-level data
            z_reactivation, fb_times, cue_times, meta = load_trial_level(
                roi, subject, trial_save_dir, dependent_var=dependent_var
            )
            
            # Check if electrode-level (4D)
            if z_reactivation.ndim != 4:
                if verbose:
                    print(f"  Skipping {subject}: data is not electrode-level (shape: {z_reactivation.shape})")
                continue
            
            memory_outcomes = memory_outcomes_dict[subject]
            
            # Run per-electrode cluster test
            if verbose:
                print(f"  Processing {subject}...")
            
            significant_cluster_counts = run_per_electrode_2sample_cluster_test(
                z_reactivation,
                memory_outcomes,
                n_permutations=n_permutations,
                cluster_seed=cluster_seed,
                verbose=False
            )
            
            # Count electrodes with >0 significant clusters
            n_electrodes_with_sig = np.sum(significant_cluster_counts > 0)
            results[subject] = n_electrodes_with_sig
            
            if verbose:
                print(f"    {subject}: {n_electrodes_with_sig} electrodes with significant clusters")
                
        except Exception as e:
            if verbose:
                print(f"  Error processing {subject}: {e}")
            continue
    
    return results


def plot_electrode_cluster_counts(
    cluster_counts_dict: Dict[str, Dict[str, Dict[str, int]]],  # roi -> dependent_var -> subject -> count
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot bar chart of mean electrode counts (with significant clusters) across ROIs.
    
    Parameters:
    -----------
    cluster_counts_dict : dict
        Nested dictionary: roi -> dependent_var -> subject -> count
        Structure: {roi: {dependent_var: {subject: count}}}
    figsize : tuple, default=(12, 6)
        Figure size
    """
    # Extract ROIs and dependent_vars
    rois = sorted(cluster_counts_dict.keys())
    dependent_vars = set()
    for roi_data in cluster_counts_dict.values():
        dependent_vars.update(roi_data.keys())
    dependent_vars = sorted(dependent_vars)
    
    # Compute means and standard errors for each ROI/dependent_var combination
    means = {}  # (roi, dependent_var) -> mean
    sems = {}   # (roi, dependent_var) -> sem
    
    for roi in rois:
        for dependent_var in dependent_vars:
            if dependent_var not in cluster_counts_dict[roi]:
                continue
            
            # Get all counts for this combination
            counts = list(cluster_counts_dict[roi][dependent_var].values())
            
            if len(counts) > 0:
                means[(roi, dependent_var)] = np.mean(counts)
                sems[(roi, dependent_var)] = np.std(counts) / np.sqrt(len(counts))
            else:
                means[(roi, dependent_var)] = 0.0
                sems[(roi, dependent_var)] = 0.0
    
    # Create separate figure for each dependent variable
    for dependent_var in dependent_vars:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get values and errors for this dependent variable
        values = []
        errors = []
        roi_labels = []
        
        for roi in rois:
            key = (roi, dependent_var)
            if key in means:
                values.append(means[key])
                errors.append(sems[key])
                roi_labels.append(roi)
        
        if len(values) == 0:
            print(f"No data for dependent variable: {dependent_var}")
            continue
        
        # Create bar chart with error bars
        x_positions = np.arange(len(roi_labels))
        bars = ax.bar(x_positions, values, yerr=errors, capsize=5, 
                     color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Customize plot
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_ylabel('Mean Number of Electrodes\nwith Significant Clusters', fontsize=12)
        ax.set_title(f'Electrodes with Significant Hits > Misses Clusters\n{dependent_var}', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(roi_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n{dependent_var} Summary:")
        print("=" * 70)
        for roi in rois:
            key = (roi, dependent_var)
            if key in means:
                n_subjects = len(cluster_counts_dict[roi][dependent_var])
                print(f"  {roi}: Mean = {means[key]:.2f} ± {sems[key]:.2f} (n={n_subjects} subjects)")


def run_encoding_weight_cluster_test(
    weights_list: List[List[np.ndarray]],
    null_weights_list: List[np.ndarray],
    feedback_times_list: List[np.ndarray],
    n_permutations: int = 1024,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], Optional[List], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run cluster-based permutation test for encoding weight significance.
    
    Tests if encoding weights are significantly different from zero during feedback period.
    Uses null weights from RPE-shuffled surrogates to z-score actual weights,
    then tests z-scored weight magnitudes against 0 using 1-sample cluster test.
    
    Parameters:
    -----------
    weights_list : list of lists
        Each element is a list of n_feedback_t arrays, each of shape (n_electrodes,)
        Actual encoding weights for each subject
    null_weights_list : list of arrays
        Each element is an array of shape (n_surrogates, n_feedback_t, n_electrodes)
        Null encoding weights from surrogate computation
    feedback_times_list : list of arrays
        Time vectors for feedback period (one per subject)
    n_permutations : int, default=1024
        Number of permutations for cluster test
    seed : int, default=42
        Random seed for permutations
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    t_obs : array or None, shape (n_fb_common,)
        T-statistic map (1D: feedback time only)
    clusters : list or None
        Cluster objects from MNE
    cluster_pvals : array or None
        P-values for each cluster
    h0 : array or None
        Null distribution
    feedback_times_common : array or None
        Common feedback time vector
    z_weights : array or None, shape (n_subjects, n_fb_common)
        Z-scored weight magnitudes for each subject and feedback timepoint
    """
    if len(weights_list) == 0:
        if verbose:
            print("No valid subjects for encoding weight cluster test")
        return None, None, None, None, None, None
    
    n_subjects = len(weights_list)
    
    # Convert weights_list to array: (n_subjects, n_feedback_t, n_electrodes)
    # First, find common dimensions
    n_fb_list = [len(w) for w in weights_list]
    n_electrodes_list = [len(w[0]) for w in weights_list if len(w) > 0]
    
    if len(set(n_fb_list)) > 1 or len(set(n_electrodes_list)) > 1:
        if verbose:
            print(f"Warning: Subjects have different dimensions. Using first subject's dimensions.")
    
    n_fb_common = n_fb_list[0] if len(n_fb_list) > 0 else 0
    n_electrodes_common = n_electrodes_list[0] if len(n_electrodes_list) > 0 else 0
    
    # Stack actual weights: (n_subjects, n_feedback_t, n_electrodes)
    weights_array = np.stack(
        [np.stack(w_list, axis=0).astype(np.float32) for w_list in weights_list],
        axis=0
    )

    # Stack null stats to subject-aligned arrays
    # null_weights_list[s] expected shape (n_surrogates, n_feedback_t, n_electrodes)
    null_weights_mean = np.stack(
        [np.asarray(nw, dtype=np.float32).mean(axis=0) for nw in null_weights_list],
        axis=0
    )
    null_weights_std = np.stack(
        [np.asarray(nw, dtype=np.float32).std(axis=0) for nw in null_weights_list],
        axis=0
    )
    null_weights_std = np.maximum(null_weights_std, 1e-8)

    # Subject/electrode/time z-score then aggregate electrode magnitudes
    z_weights_full = (weights_array - null_weights_mean) / null_weights_std
    z_weights = np.linalg.norm(z_weights_full, axis=2)
    
    # Use first subject's time vector (assuming all have same dimensions)
    feedback_times_common = feedback_times_list[0]
    
    if verbose:
        print(f"Running encoding weight cluster test for {n_subjects} subjects...")
        print(f"  Weight shape: {z_weights.shape}")
    
    # Run 1-sample cluster test on z-scored weight magnitudes
    t_obs, clusters, cluster_pvals, h0 = permutation_cluster_1samp_test(
        z_weights,
        threshold=0,
        n_permutations=n_permutations,
        tail=1,
        n_jobs=-1,
        seed=seed,
        verbose=verbose
    )
    
    return t_obs, clusters, cluster_pvals, h0, feedback_times_common, z_weights


# =============================================================================
# MULTINOMIAL REGRESSION
# =============================================================================

def multinomial_regression(
    df: pd.DataFrame,
    response_col: str,
    predictor_col: List[str]
) -> Any:
    """
    Fit multinomial logistic regression model.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    response_col : str
        Name of response variable column (categorical)
    predictor_col : list of str
        List of predictor variable column names
        
    Returns:
    --------
    result
        Statsmodels multinomial regression result object
    """
    # Drop rows with NaN values in the dependent or predictor columns
    rel_cols = [response_col] + predictor_col
    df_clean = df[rel_cols].dropna()

    # Encode the categorical response variable
    # hits=0, misses=1, false_alarms=2, correct_rejections=3
    df_clean['response_encoded'] = df_clean[response_col].apply(
        lambda x: 0 if x=='hits' else 1 if x=='misses' else 2 if x=='false_alarms' else 3
    ).values.ravel()
    
    # Prepare the predictor variable (add constant for intercept)
    X = sm.add_constant(df_clean[predictor_col])
    
    # The dependent variable is the encoded response
    y = df_clean['response_encoded']
        
    # Fit the multinomial logistic regression model
    model = sm.MNLogit(y, X)
    result = model.fit(disp=0)
    
    return result


def _encode_memory_outcome(memory_outcome: np.ndarray) -> np.ndarray:
    """
    Encode memory outcome categories.
    
    Parameters:
    -----------
    memory_outcome : array
        Array of memory outcome strings
        
    Returns:
    --------
    encoded : array
        Encoded array: hits=0, misses=1, false_alarms=2, correct_rejections=3
    """
    encoding_map = {'hits': 0, 'misses': 1, 'false_alarms': 2, 'correct_rejections': 3}
    return np.array([encoding_map.get(x, np.nan) for x in memory_outcome])


def _fit_pixel_regression(
    y_pixel: np.ndarray,
    y_encoded: np.ndarray,
    valid_mask: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Fit multinomial regression for a single pixel using pre-processed data.
    
    This is an optimized function that works with numpy arrays and avoids
    repeated pandas operations. Designed for parallel processing.
    
    Parameters:
    -----------
    y_pixel : array, shape (n_trials,)
        Predictor values for this pixel
    y_encoded : array, shape (n_trials,)
        Pre-encoded response variable
    valid_mask : array, shape (n_trials,), dtype=bool
        Mask indicating valid (non-NaN) rows
        
    Returns:
    --------
    misses_coef : float
        Coefficient for misses category
    correct_rejections_coef : float
        Coefficient for correct_rejections category
    false_alarms_coef : float
        Coefficient for false_alarms category
    lrt : float
        Likelihood ratio test statistic
    aic : float
        Akaike Information Criterion
    """
    # Apply valid mask to both predictor and response
    y_pixel_valid = y_pixel[valid_mask]
    y_encoded_valid = y_encoded[valid_mask]
    
    # Check for NaN in predictor and create final mask
    final_mask = ~np.isnan(y_pixel_valid)
    if np.sum(final_mask) < 10:  # Minimum trials required
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    
    y_final = y_encoded_valid[final_mask]
    X_final = y_pixel_valid[final_mask]
    
    # Add constant term
    X = sm.add_constant(X_final)
    
    # Fit multinomial regression
    # Create DataFrame with named column to match original function structure
    X_df = pd.DataFrame(X, columns=['const', 'z_reactivation'])
    
    try:
        model = sm.MNLogit(y_final, X_df)
        result = model.fit(disp=0)
        
        # Extract coefficients: params.loc['z_reactivation'] gives coefficients for each outcome
        # Columns: misses (0), correct_rejections (1), false_alarms (2)
        # (hits is the reference category)
        predictor_coefs = result.params.loc['z_reactivation'].values
        
        lrt = 2 * (result.llf - result.llnull)
        aic = result.aic
        
        # Return coefficients: misses, correct_rejections, false_alarms, lrt, aic
        return (predictor_coefs[0], predictor_coefs[1], predictor_coefs[2], lrt, aic)
    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


def _process_pixel(
    ij_tuple: Tuple[int, int],
    rearanged_z_reactivation: np.ndarray,
    y_encoded: np.ndarray,
    valid_mask: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Process a single pixel (i, j) combination for parallel regression.
    
    Parameters:
    -----------
    ij_tuple : tuple
        (i, j) indices for the pixel
    rearanged_z_reactivation : array, shape (n_trials, n_i, n_j)
        Rearranged z-reactivation data
    y_encoded : array, shape (n_trials,)
        Pre-encoded response variable
    valid_mask : array, shape (n_trials,), dtype=bool
        Mask indicating valid rows
        
    Returns:
    --------
    tuple
        (misses_coef, correct_rejections_coef, false_alarms_coef, lrt, aic)
    """
    i, j = ij_tuple
    y_pixel = rearanged_z_reactivation[:, i, j]
    return _fit_pixel_regression(y_pixel, y_encoded, valid_mask)


def compute_subject_lrt_map(
    z_reactivation: np.ndarray,
    memory_outcomes: np.ndarray,
    n_jobs: int = -1,
    verbose: bool = False,
    aggregate_electrodes: bool = True
) -> np.ndarray:
    """
    Compute LRT map for a single subject.
    
    Computes likelihood ratio test statistics for multinomial regression
    at each (feedback_t, cue_t) pixel for a single subject.
    
    Parameters:
    -----------
    z_reactivation : array
        If vector-level: shape (n_feedback_t, n_cue_t, n_trials)
        If electrode-level: shape (n_feedback_t, n_cue_t, n_electrodes, n_trials)
        Z-scored reactivation data for one subject
    memory_outcomes : array, shape (n_trials,)
        Memory outcome labels ('hits', 'misses', 'false_alarms', 'correct_rejections')
    n_jobs : int, default=-1
        Number of parallel jobs for pixel processing
    verbose : bool, default=False
        Print progress messages
    aggregate_electrodes : bool, default=True
        If True and data is electrode-level (4D), aggregate across electrodes (mean).
        If False, process each electrode separately (not implemented yet).
        
    Returns:
    --------
    lrt_map : array, shape (n_feedback_t, n_cue_t)
        LRT statistic for each pixel
    """
    # Handle both 3D (vector-level) and 4D (electrode-level) arrays
    if z_reactivation.ndim == 4:
        # Electrode-level: (n_fb, n_cue, n_electrodes, n_trials)
        if aggregate_electrodes:
            # Aggregate across electrodes (mean)
            z_reactivation = np.mean(z_reactivation, axis=2)  # (n_fb, n_cue, n_trials)
        else:
            # For now, raise error - per-electrode processing not implemented
            raise NotImplementedError("Per-electrode processing not yet implemented. Use aggregate_electrodes=True.")
    
    # Rearrange to (n_trials, n_feedback_t, n_cue_t)
    rearanged_z_reactivation = np.transpose(z_reactivation, (2, 0, 1))
    
    # Pre-process: encode memory outcome and create valid mask
    y_encoded = _encode_memory_outcome(memory_outcomes)
    valid_mask = ~np.isnan(y_encoded)
    
    # Get dimensions
    n_i, n_j = rearanged_z_reactivation.shape[1], rearanged_z_reactivation.shape[2]
    pixel_combinations = [(i, j) for i in range(n_i) for j in range(n_j)]
    
    # Initialize result array (only LRT, not coefficients)
    lrt_map = np.zeros((n_i, n_j))
    
    # Process all pixels in parallel
    if verbose:
        print(f"Processing {len(pixel_combinations)} pixels in parallel...")
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_pixel)(ij, rearanged_z_reactivation, y_encoded, valid_mask)
        for ij in pixel_combinations
    )
    
    # Extract LRT values (4th element of each result tuple)
    for idx, (i, j) in enumerate(pixel_combinations):
        _, _, _, lrt, _ = results[idx]
        lrt_map[i, j] = lrt
    
    if verbose:
        print("LRT map computation complete!")
    
    return lrt_map


def compute_all_subjects_lrt_maps(
    roi: str,
    subjects: List[str],
    combined_df: pd.DataFrame,
    trial_save_dir: str,
    n_jobs: int = -1,
    verbose: bool = True,
    dependent_var: str = 'rpe'
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Compute LRT maps for all subjects.
    
    Loads trial-level data for each subject and computes LRT map separately,
    resulting in a 3D array suitable for group-level permutation testing.
    
    Parameters:
    -----------
    roi : str
        Region of interest
    subjects : list of str
        List of subject identifiers
    combined_df : DataFrame
        DataFrame containing memory outcomes for all subjects
    trial_save_dir : str
        Directory containing saved trial-level data
    n_jobs : int, default=-1
        Number of parallel jobs for pixel processing within each subject
    verbose : bool, default=True
        Print progress messages
    dependent_var : str, default='rpe'
        Name of the dependent variable used for encoding models
        
    Returns:
    --------
    lrt_maps : array, shape (n_subj, n_feedback_t, n_cue_t)
        3D array of LRT maps, one per subject
    valid_subjects : list of str
        List of subjects with valid data
    feedback_times_common : array
        Common feedback time vector (using median dimensions)
    cue_times_common : array
        Common cue time vector (using median dimensions)
    """
    lrt_maps_list = []
    valid_subjects = []
    feedback_times_list = []
    cue_times_list = []

    combined_df['memory_outcome'] = combined_df[['hits', 'misses', 'correct_rejections', 
                                                'false_alarms']].idxmax(1).to_frame('memory_outcome')  
    
    for subj in subjects:
        try:
            # Load trial-level data
            z_reactivation, fb_times, cue_times, meta = load_trial_level(
                roi, subj, trial_save_dir, dependent_var=dependent_var
            )
            
            # Get memory outcomes for this subject
            subj_df = combined_df[
                (combined_df.participant == subj) & 
                (combined_df.condition == 'Day1')
            ]
            
            # Handle both 3D and 4D arrays for trial count check
            if z_reactivation.ndim == 4:
                n_trials_reactivation = z_reactivation.shape[3]
            else:
                n_trials_reactivation = z_reactivation.shape[2]
            
            if len(subj_df) != n_trials_reactivation:
                if verbose:
                    print(f"WARNING: {subj} has {len(subj_df)} trials, but {n_trials_reactivation} z-reactivation values")
                continue
            
            memory_outcomes = subj_df['memory_outcome'].values
            
            # Compute LRT map for this subject
            if verbose:
                print(f"Computing LRT map for {subj}...")
            
            lrt_map = compute_subject_lrt_map(
                z_reactivation,
                memory_outcomes,
                n_jobs=n_jobs,
                verbose=False
            )
            
            lrt_maps_list.append(lrt_map)
            valid_subjects.append(subj)
            feedback_times_list.append(fb_times)
            cue_times_list.append(cue_times)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {subj}: {e}")
            continue
    
    if len(lrt_maps_list) == 0:
        if verbose:
            print("No valid subjects found!")
        return None, [], None, None
    
    # Find common dimensions (use median)
    fb_dims = [len(ft) for ft in feedback_times_list]
    cue_dims = [len(ct) for ct in cue_times_list]
    median_fb_idx = np.argsort(fb_dims)[len(fb_dims)//2]
    median_cue_idx = np.argsort(cue_dims)[len(cue_dims)//2]
    
    feedback_times_common = feedback_times_list[median_fb_idx]
    cue_times_common = cue_times_list[median_cue_idx]
    n_fb_common = len(feedback_times_common)
    n_cue_common = len(cue_times_common)
    
    # Interpolate all LRT maps to common dimensions if needed
    lrt_maps_interp = []
    for lrt_map, fb_t, cue_t in zip(lrt_maps_list, feedback_times_list, cue_times_list):
        if lrt_map.shape[0] == n_fb_common and lrt_map.shape[1] == n_cue_common:
            lrt_maps_interp.append(lrt_map)
        else:
            # Interpolate to common grid
            interp_func = RectBivariateSpline(fb_t, cue_t, lrt_map, kx=1, ky=1)
            lrt_map_interp = interp_func(feedback_times_common, cue_times_common, grid=True)
            lrt_maps_interp.append(lrt_map_interp)
    
    # Stack into 3D array: (n_subj, n_feedback_t, n_cue_t)
    lrt_maps = np.stack(lrt_maps_interp, axis=0)
    
    if verbose:
        print(f"\nComputed LRT maps for {len(valid_subjects)} subjects")
        print(f"LRT maps shape: {lrt_maps.shape}")
    
    return lrt_maps, valid_subjects, feedback_times_common, cue_times_common


def compute_all_subjects_lrt_maps_from_npz(
    roi: str,
    subjects: List[str],
    trial_save_dir: str,
    n_jobs: int = -1,
    verbose: bool = True,
    dependent_var: str = 'rpe',
    aggregate_electrodes: bool = True,
    min_valid_trials: int = 10,
) -> Tuple[Optional[np.ndarray], List[str], Optional[np.ndarray], Optional[np.ndarray], Dict[str, str]]:
    """
    Compute subject-level LRT maps directly from NPZ trial-level outputs.

    Prioritizes all-trial LRT keys (`z_reactivation_lrt`, `memory_outcomes_lrt`) when
    present, otherwise falls back to legacy (`z_reactivation`, `memory_outcomes`).
    """
    lrt_maps_list: List[np.ndarray] = []
    valid_subjects: List[str] = []
    feedback_times_list: List[np.ndarray] = []
    cue_times_list: List[np.ndarray] = []
    skipped_reasons: Dict[str, str] = {}
    valid_labels = np.array(['hits', 'misses', 'false_alarms', 'correct_rejections'], dtype=object)

    for subj in subjects:
        try:
            z_reactivation, fb_times, cue_times, meta = load_trial_level(
                roi, subj, trial_save_dir, dependent_var=dependent_var
            )
        except Exception:
            skipped_reasons[subj] = 'missing_or_unreadable_npz'
            continue

        if ('z_reactivation_lrt' in meta) and ('memory_outcomes_lrt' in meta):
            z_src = np.asarray(meta['z_reactivation_lrt'])
            mem = np.asarray(meta['memory_outcomes_lrt'], dtype=object)
        else:
            z_src = np.asarray(z_reactivation)
            mem = np.asarray(meta.get('memory_outcomes', np.array([], dtype=object)), dtype=object)

        if z_src.ndim == 4:
            if aggregate_electrodes:
                z_src = np.nanmean(z_src, axis=2)
            else:
                skipped_reasons[subj] = 'electrode_level_not_supported_without_aggregation'
                continue
        if z_src.ndim != 3:
            skipped_reasons[subj] = f'unexpected_ndim_{z_src.ndim}'
            continue
        if len(mem) != z_src.shape[-1]:
            skipped_reasons[subj] = 'trial_mismatch'
            continue

        class_mask = np.isin(mem, valid_labels)
        if int(np.sum(class_mask)) < int(min_valid_trials):
            skipped_reasons[subj] = 'insufficient_valid_trials'
            continue

        mem_use = mem[class_mask]
        if len(np.unique(mem_use)) < 2:
            skipped_reasons[subj] = 'missing_memory_class'
            continue

        z_use = z_src[:, :, class_mask]
        try:
            lrt_map = compute_subject_lrt_map(
                z_use,
                mem_use,
                n_jobs=n_jobs,
                verbose=False,
                aggregate_electrodes=aggregate_electrodes,
            )
        except Exception:
            skipped_reasons[subj] = 'lrt_compute_failed'
            continue

        if np.all(~np.isfinite(lrt_map)):
            skipped_reasons[subj] = 'all_nan_lrt_map'
            continue

        lrt_maps_list.append(np.asarray(lrt_map, dtype=np.float64))
        valid_subjects.append(subj)
        feedback_times_list.append(np.asarray(fb_times, dtype=np.float64))
        cue_times_list.append(np.asarray(cue_times, dtype=np.float64))

    if len(lrt_maps_list) == 0:
        if verbose:
            print("No valid subjects found for NPZ-based LRT maps.")
        return None, [], None, None, skipped_reasons

    fb_dims = [len(ft) for ft in feedback_times_list]
    cue_dims = [len(ct) for ct in cue_times_list]
    median_fb_idx = np.argsort(fb_dims)[len(fb_dims) // 2]
    median_cue_idx = np.argsort(cue_dims)[len(cue_dims) // 2]

    feedback_times_common = feedback_times_list[median_fb_idx]
    cue_times_common = cue_times_list[median_cue_idx]
    n_fb_common = len(feedback_times_common)
    n_cue_common = len(cue_times_common)

    lrt_maps_interp: List[np.ndarray] = []
    for lrt_map, fb_t, cue_t in zip(lrt_maps_list, feedback_times_list, cue_times_list):
        if lrt_map.shape[0] == n_fb_common and lrt_map.shape[1] == n_cue_common:
            lrt_maps_interp.append(lrt_map)
        else:
            interp_func = RectBivariateSpline(fb_t, cue_t, lrt_map, kx=1, ky=1)
            lrt_maps_interp.append(interp_func(feedback_times_common, cue_times_common, grid=True))

    lrt_maps = np.stack(lrt_maps_interp, axis=0)
    if verbose:
        print(f"Computed NPZ-based LRT maps for {len(valid_subjects)} subjects")
        print(f"LRT maps shape: {lrt_maps.shape}")
    return lrt_maps, valid_subjects, feedback_times_common, cue_times_common, skipped_reasons



# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_encoding_weight_clusters(
    roi: str,
    encoding_cluster_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot encoding weight significance results for a single ROI.
    
    Parameters:
    -----------
    roi : str
        ROI name
    encoding_cluster_results : dict
        Results from run_encoding_weight_cluster_test
    figsize : tuple, default=(12, 5)
        Figure size
    """
    if roi not in encoding_cluster_results:
        print(f"No encoding cluster results for ROI {roi}")
        return
    
    result = encoding_cluster_results[roi]
    t_obs = result['t_obs']
    clusters = result['clusters']
    cluster_pvals = result['cluster_pvals']
    fb_times = result['feedback_times']
    z_weights = result['z_weights']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Mean z-scored weight magnitude across subjects
    mean_z_magnitude = np.mean(z_weights, axis=0)
    ste_z_magnitude = np.std(z_weights, axis=0)/np.sqrt(z_weights.shape[0])
    
    axes[0].plot(fb_times, mean_z_magnitude, 'b-', linewidth=2, label='Mean')
    axes[0].fill_between(fb_times, 
                         mean_z_magnitude - ste_z_magnitude,
                         mean_z_magnitude + ste_z_magnitude,
                         alpha=0.3, color='blue', label='±1 SD')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Feedback Time (s)')
    axes[0].set_ylabel('Z-scored Weight ')
    axes[0].set_title(f'{roi}: Encoding Weight Magnitude\n(Mean across subjects)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: T-statistic map
    axes[1].plot(fb_times, t_obs, 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Feedback Time (s)')
    axes[1].set_ylabel('T-statistic')
    axes[1].set_title(f'{roi}: T-statistic Map\n(Encoding Weight Significance)')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Significant clusters masked
    t_obs_masked = t_obs.copy()
    if len(clusters) > 0:
        # Create mask for significant clusters
        sig_mask = np.zeros_like(t_obs, dtype=bool)
        for cluster_idx, (cluster, pval) in enumerate(zip(clusters, cluster_pvals)):
            if pval <= 0.05:
                # Get cluster indices
                if isinstance(cluster, (list, tuple, np.ndarray)):
                    cluster_indices = np.array(cluster)
                elif hasattr(cluster, '__iter__'):
                    cluster_indices = np.array(list(cluster))
                else:
                    cluster_indices = np.array([cluster])
                
                sig_mask[cluster_indices] = True
        
        # Mask non-significant areas
        t_obs_masked[~sig_mask] = np.nan
    
    axes[2].plot(fb_times, t_obs_masked, 'r-', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Feedback Time (s)')
    axes[2].set_ylabel('T-statistic')
    axes[2].set_title(f'{roi}: Significant Clusters\n(p <= 0.05)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    n_sig = np.sum(cluster_pvals <= 0.05) if len(cluster_pvals) > 0 else 0
    print(f"\n{roi}: {len(result['subjects'])} subjects, {n_sig} significant encoding clusters (p ≤ 0.05)")


def plot_cluster_statistics_bar_chart(
    cluster_stats_dict: Dict[str, Dict[str, Dict[str, List[float]]]],  # roi -> dependent_var -> comparison -> [stats]
    figsize: Tuple[int, int] = (15, 8)
) -> None:
    """
    Plot bar chart of mean cluster statistics across subjects.
    
    Parameters:
    -----------
    cluster_stats_dict : dict
        Nested dictionary: roi -> dependent_var -> comparison -> list of cluster statistics
        Structure: {roi: {dependent_var: {'hits_vs_misses': [stats], 'hits_vs_false_alarms': [stats]}}}
    figsize : tuple, default=(15, 8)
        Figure size
    """
    # Extract ROIs and dependent_vars
    rois = sorted(cluster_stats_dict.keys())
    dependent_vars = set()
    for roi_data in cluster_stats_dict.values():
        dependent_vars.update(roi_data.keys())
    dependent_vars = sorted(dependent_vars)
    
    comparisons = ['hits_vs_misses', 'hits_vs_false_alarms']
    comparison_labels = ['Hits vs Misses', 'Hits vs False Alarms']
    
    # Compute means for each ROI/dependent_var/comparison combination
    means = {}  # (roi, dependent_var, comparison) -> mean
    for roi in rois:
        for dependent_var in dependent_vars:
            if dependent_var not in cluster_stats_dict[roi]:
                continue
            for comparison in comparisons:
                if comparison not in cluster_stats_dict[roi][dependent_var]:
                    continue
                
                # Get all cluster statistics for this combination
                all_stats = []
                for subject_stats in cluster_stats_dict[roi][dependent_var][comparison].values():
                    all_stats.extend(subject_stats)
                
                if len(all_stats) > 0:
                    means[(roi, dependent_var, comparison)] = np.mean(all_stats)
                else:
                    means[(roi, dependent_var, comparison)] = 0.0
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    n_rois = len(rois)
    n_dependent_vars = len(dependent_vars)
    n_comparisons = len(comparisons)
    
    # Width of each bar group
    group_width = 0.8
    bar_width = group_width / (n_dependent_vars * n_comparisons)
    
    # X positions for each ROI
    x_positions = np.arange(n_rois)
    
    # Colors for dependent_vars
    var_colors = plt.cm.Set2(np.linspace(0, 1, n_dependent_vars))
    
    # Plot bars
    bars = []
    bar_labels = []
    
    for var_idx, dependent_var in enumerate(dependent_vars):
        for comp_idx, comparison in enumerate(comparisons):
            # Calculate offset for this bar within the group
            offset = (var_idx * n_comparisons + comp_idx) * bar_width - group_width/2 + bar_width/2
            
            # Get values for this combination
            values = []
            for roi in rois:
                key = (roi, dependent_var, comparison)
                values.append(means.get(key, 0.0))
            
            # Create bar
            bar = ax.bar(x_positions + offset, values, bar_width,
                        label=f"{dependent_var} - {comparison_labels[comp_idx]}",
                        color=var_colors[var_idx], alpha=0.7 if comp_idx == 0 else 0.5)
            bars.append(bar)
    
    # Customize plot
    ax.set_xlabel('ROI', fontsize=12)
    ax.set_ylabel('Mean Cluster Statistic', fontsize=12)
    ax.set_title('Mean Cluster Statistics: Hits vs Misses and Hits vs False Alarms', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(rois, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 70)
    for roi in rois:
        print(f"\n{roi}:")
        for dependent_var in dependent_vars:
            if dependent_var not in cluster_stats_dict[roi]:
                continue
            print(f"  {dependent_var}:")
            for comparison, label in zip(comparisons, comparison_labels):
                if comparison not in cluster_stats_dict[roi][dependent_var]:
                    continue
                all_stats = []
                for subject_stats in cluster_stats_dict[roi][dependent_var][comparison].values():
                    all_stats.extend(subject_stats)
                if len(all_stats) > 0:
                    mean_stat = np.mean(all_stats)
                    n_subjects = len(cluster_stats_dict[roi][dependent_var][comparison])
                    print(f"    {label}: Mean = {mean_stat:.3f} (n={n_subjects} subjects)")


# =============================================================================
# PROCESSING HELPERS
# =============================================================================

def _process_one_subject_batch(
    subject: str,
    tasks: List[Tuple[str, str]],
    base_dir: str,
    load_dir: str,
    downsample_factor: int,
    win_ms: int,
    hfa_freq_range: Tuple[float, float],
    min_trials: int,
    n_surrogates: int,
    use_surrogates: bool,
    ridge_alpha: float,
    seed: int,
    n_jobs: int,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    electrode_level: bool = False,
    surrogate_chunk_size: Optional[int] = None,
) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """
    Process one subject for multiple (roi, dependent_var) tasks. Creates Subject once
    and reuses it for all tasks. Designed for parallel execution (one job per subject).

    Returns:
        List of (roi, dependent_var, result_dict or None) for each task.
    """
    subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
    results = []
    for (roi, dependent_var) in tasks:
        data = subject_obj.get_data_for_roi_and_dependent_var(
            roi=roi,
            dependent_var=dependent_var,
            freq_range=hfa_freq_range,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
            verbose=False,
        )
        if data is None:
            results.append((roi, dependent_var, None))
            continue
        result = compute_subject_reactivation(
            data,
            n_surrogates=n_surrogates,
            use_surrogates=use_surrogates,
            ridge_alpha=ridge_alpha,
            seed=seed,
            verbose=False,
            n_jobs=n_jobs,
            trial_save_dtype=trial_save_dtype,
            electrode_level=electrode_level,
            surrogate_chunk_size=surrogate_chunk_size,
        )
        reactivation_real, z_reactivation, fb_times, cue_times, weights, activation_patterns, null_weights = result
        if save_trial_level and trial_save_dir is not None:
            roi_dir = os.path.join(trial_save_dir, roi)
            os.makedirs(roi_dir, exist_ok=True)
            out_file = os.path.join(roi_dir, f"{subject}_{dependent_var}_zreactivation.npz")
            if overwrite_trial_level or not os.path.exists(out_file):
                np.savez_compressed(
                    out_file,
                    z_reactivation=z_reactivation.astype(trial_save_dtype, copy=False),
                    feedback_times=fb_times,
                    cue_times=cue_times,
                    memory_outcomes=np.asarray(data['memory_outcomes']),
                    trial_index=np.arange(z_reactivation.shape[-1], dtype=np.int32),
                    electrode_labels=np.asarray(data.get('electrode_labels', []), dtype=str),
                    subject=subject,
                    roi=roi,
                    n_surrogates=int(n_surrogates),
                    ridge_alpha=float(ridge_alpha),
                    downsample_factor=int(downsample_factor),
                    win_ms=int(win_ms),
                    hfa_freq_range=np.asarray(hfa_freq_range),
                    seed=int(seed),
                    electrode_level=bool(electrode_level),
                    activation_patterns=np.stack(activation_patterns, axis=0).astype(np.float32),
                )
        results.append((roi, dependent_var, {
            'subject': subject,
            'feedback_times': fb_times,
            'cue_times': cue_times,
            'memory_outcomes': data['memory_outcomes'],
            'weights': weights,
            'activation_patterns': activation_patterns,
            'null_weights': null_weights,
        }))
    return results


def _process_single_subject(
    subject: str,
    roi: str,
    base_dir: str,
    load_dir: str,
    downsample_factor: int,
    win_ms: int,
    hfa_freq_range: Tuple[float, float],
    min_trials: int,
    n_surrogates: int,
    use_surrogates: bool,
    ridge_alpha: float,
    seed: int,
    n_jobs: int,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    dependent_var: str = 'rpe',
    electrode_level: bool = False,
    subject_obj: Optional[Subject] = None
) -> Optional[Dict[str, Any]]:
    """
    Helper function to process a single subject. Designed for parallel execution.
    
    Parameters:
    -----------
    subject : str
        Subject identifier
    roi : str
        Region of interest
    base_dir : str
        Base directory
    load_dir : str
        Load directory
    downsample_factor : int
        Downsampling factor
    win_ms : int
        Smoothing window size (ms)
    hfa_freq_range : tuple
        HFA frequency range
    min_trials : int
        Minimum trials required
    n_surrogates : int
        Number of surrogates
    use_surrogates : bool
        Whether to use surrogates
    ridge_alpha : float
        Ridge regression alpha
    seed : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    save_trial_level : bool, default=True
        Whether to save trial-level data
    trial_save_dir : str or None, default=None
        Directory to save trial-level data
    overwrite_trial_level : bool, default=True
        Whether to overwrite existing files
    trial_save_dtype : type, default=np.float32
        Data type for saving
    dependent_var : str, default='rpe'
        Name of the dependent variable column in learn_df to use for encoding models
    electrode_level : bool, default=False
        If True, compute reactivation at electrode level.
        If False, compute vector-level reactivation.
    subject_obj : Subject or None, default=None
        Optional Subject object with pre-loaded data. If provided, uses this instead
        of calling load_subject_data, enabling efficient processing of multiple
        dependent variables.
        
    Returns:
    --------
    dict or None
        Dictionary containing subject results
    """
    # Load subject data
    # Use pre-loaded Subject object for efficient processing
    data = subject_obj.get_data_for_roi_and_dependent_var(
        roi=roi,
        dependent_var=dependent_var,
        freq_range=hfa_freq_range,
        downsample_factor=downsample_factor,
        win_ms=win_ms,
        verbose=False
    )
    
    if data is None:
        return None
    
    # Compute reactivation
    result = compute_subject_reactivation(
        data,
        n_surrogates=n_surrogates,
        use_surrogates=use_surrogates,
        ridge_alpha=ridge_alpha,
        seed=seed,
        verbose=False,
        n_jobs=n_jobs,
        trial_save_dtype=trial_save_dtype,
        electrode_level=electrode_level
    )
    
    reactivation_real, z_reactivation, fb_times, cue_times, weights, activation_patterns, null_weights = result
    
    # Save trial-level z-reactivation
    if save_trial_level and trial_save_dir is not None:
        roi_dir = os.path.join(trial_save_dir, roi)
        os.makedirs(roi_dir, exist_ok=True)
        out_file = os.path.join(roi_dir, f"{subject}_{dependent_var}_zreactivation.npz")

        if (not overwrite_trial_level) and os.path.exists(out_file):
            pass
        else:
            np.savez_compressed(
                out_file,
                z_reactivation=z_reactivation.astype(trial_save_dtype, copy=False),
                feedback_times=fb_times,
                cue_times=cue_times,
                memory_outcomes=np.asarray(data['memory_outcomes']),
                trial_index=np.arange(z_reactivation.shape[-1], dtype=np.int32),
                electrode_labels=np.asarray(data.get('electrode_labels', []), dtype=str),
                subject=subject,
                roi=roi,
                n_surrogates=int(n_surrogates),
                ridge_alpha=float(ridge_alpha),
                downsample_factor=int(downsample_factor),
                win_ms=int(win_ms),
                hfa_freq_range=np.asarray(hfa_freq_range),
                seed=int(seed),
                electrode_level=bool(electrode_level),
                activation_patterns=np.stack(activation_patterns, axis=0).astype(np.float32),
            )
    
    return {
        'subject': subject,
        'reactivation_real': reactivation_real,
        'feedback_times': fb_times,
        'cue_times': cue_times,
        'memory_outcomes': data['memory_outcomes'],
        'weights': weights,
        'activation_patterns': activation_patterns,
        'null_weights': null_weights
    }


def run_reactivation_pipeline(
    subjects: List[str],
    rois: List[str],
    dependent_vars: List[str],
    base_dir: str,
    load_dir: str,
    downsample_factor: int = 5,
    win_ms: int = 200,
    hfa_freq_range: Tuple[float, float] = (70.0, 201.0),
    min_trials: int = 10,
    n_surrogates: int = 500,
    use_surrogates: bool = True,
    ridge_alpha: float = 1.0,
    seed: int = 42,
    n_jobs: int = -1,
    save_trial_level: bool = True,
    trial_save_dir: Optional[str] = None,
    overwrite_trial_level: bool = True,
    trial_save_dtype: type = np.float32,
    electrode_level: bool = False,
    surrogate_chunk_size: Optional[int] = None,
    n_jobs_outer: int = 1,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run reactivation computation for all subjects, ROIs, and dependent variables
    using subject-first parallelism. Each worker loads one subject's TFR once and
    processes all (roi, dependent_var) pairs for that subject.

    When n_jobs_outer > 1, each worker holds one subject's TFR in memory; total
    memory is roughly n_jobs_outer times one-subject memory.

    Returns:
        results_per_roi_per_var[roi][dependent_var] with keys:
        feedback_times_list, cue_times_list, subject_list, weights_list,
        activation_patterns_list, null_weights_list, memory_outcomes (list of arrays).
    """
    tasks = [(roi, var) for roi in rois for var in dependent_vars]
    jobs = [(subject, tasks) for subject in subjects]

    results_per_roi_per_var = {}
    for roi in rois:
        results_per_roi_per_var[roi] = {}
        for var in dependent_vars:
            results_per_roi_per_var[roi][var] = {
                'feedback_times_list': [],
                'cue_times_list': [],
                'subject_list': [],
                'weights_list': [],
                'activation_patterns_list': [],
                'null_weights_list': [],
                'memory_outcomes': [],
            }

    def _run_batch(args):
        subj, task_list = args
        return _process_one_subject_batch(
            subj,
            task_list,
            base_dir=base_dir,
            load_dir=load_dir,
            downsample_factor=downsample_factor,
            win_ms=win_ms,
            hfa_freq_range=hfa_freq_range,
            min_trials=min_trials,
            n_surrogates=n_surrogates,
            use_surrogates=use_surrogates,
            ridge_alpha=ridge_alpha,
            seed=seed,
            n_jobs=n_jobs,
            save_trial_level=save_trial_level,
            trial_save_dir=trial_save_dir,
            overwrite_trial_level=overwrite_trial_level,
            trial_save_dtype=trial_save_dtype,
            electrode_level=electrode_level,
            surrogate_chunk_size=surrogate_chunk_size,
        )

    batch_results = Parallel(n_jobs=n_jobs_outer, verbose=10 if verbose else 0)(
        delayed(_run_batch)((subject, tasks)) for subject, tasks in jobs
    )

    for batch in batch_results:
        for (roi, dependent_var, result) in batch:
            if result is None:
                continue
            results_per_roi_per_var[roi][dependent_var]['feedback_times_list'].append(result['feedback_times'])
            results_per_roi_per_var[roi][dependent_var]['cue_times_list'].append(result['cue_times'])
            results_per_roi_per_var[roi][dependent_var]['subject_list'].append(result['subject'])
            results_per_roi_per_var[roi][dependent_var]['weights_list'].append(result['weights'])
            results_per_roi_per_var[roi][dependent_var]['activation_patterns_list'].append(result['activation_patterns'])
            results_per_roi_per_var[roi][dependent_var]['null_weights_list'].append(result['null_weights'])
            results_per_roi_per_var[roi][dependent_var]['memory_outcomes'].append(result['memory_outcomes'])

    return results_per_roi_per_var


# =============================================================================
# ROI OSCILLATION / FOOOF ANALYSIS
# =============================================================================

ROI_COLOR_MAP: Dict[str, np.ndarray] = {
    'HPC': np.array([0.44314, 0.2902, 1.0], dtype=np.float32),
    'AMY': np.array([0.49412, 0.0, 0.90196], dtype=np.float32),
    'OFC': np.array([1.0, 0.75686, 0.41569], dtype=np.float32),
    'ACC': np.array([0.7098, 0.27059, 0.68235], dtype=np.float32),
    'dmPFC': np.array([0.40784, 0.01961, 0.0], dtype=np.float32),
    'dlPFC': np.array([1.0, 0.06275, 0.06275], dtype=np.float32),
}

OSCILLATION_EVENT_WINDOWS: Dict[str, Tuple[float, float]] = {
    'feedback_start': (0.0, 1.5),
    'cue_on': (0.0, 1.0),
}

OSCILLATION_FOOOF_DEFAULTS: Dict[str, Any] = {
    'peak_width_limits': [1, 8],
    'min_peak_height': 0.1,
    'peak_threshold': 1.0,
    'max_n_peaks': 3,
    'freq_range': [2, 40],
}

OSCILLATION_PROPERTY_LABELS: Dict[str, str] = {
    'peak_freq': 'Peak frequency (Hz)',
    'peak_height': 'Peak height',
    'peak_bandwidth': 'Peak bandwidth (Hz)',
    'n_peaks': 'Detected peaks',
    'aperiodic_exponent': 'Aperiodic exponent',
    'aperiodic_offset': 'Aperiodic offset',
}


def _oscillation_trial_column(event: str) -> str:
    """Resolve the required Day1 trial index for oscillation analyses."""
    if event == 'cue_on':
        return 'trials_mem'
    if event == 'feedback_start':
        return 'trials_gamble'
    raise ValueError(f"Unsupported event '{event}' for oscillation analysis")


def _nan_zscore_1d(values: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Z-score a 1D vector while preserving NaNs."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr)
    if int(np.sum(valid)) < 2:
        return out
    mean_val = float(np.nanmean(arr[valid]))
    std_val = float(np.nanstd(arr[valid]))
    if not np.isfinite(std_val) or std_val <= 0:
        out[valid] = 0.0
        return out
    out[valid] = (arr[valid] - mean_val) / std_val
    return out


def _get_roi_electrode_metadata(
    subject_obj: Subject,
    roi: str,
    picked_labels: List[str],
) -> pd.DataFrame:
    """Resolve electrode metadata for picked ROI channels."""
    elec_sub = subject_obj.elec_df[
        subject_obj.elec_df.label.astype(str).isin([str(x) for x in picked_labels])
    ][['label', 'salman_region', 'hemisphere']].copy()
    elec_sub['label'] = elec_sub['label'].astype(str)
    elec_sub = elec_sub.drop_duplicates(subset=['label']).set_index('label')

    rows = []
    for ch in picked_labels:
        ch_key = str(ch)
        if ch_key in elec_sub.index:
            row = elec_sub.loc[ch_key]
            region = str(row.get('salman_region', roi))
            hemi = row.get('hemisphere', np.nan)
            hemi = str(hemi).lower() if pd.notna(hemi) else np.nan
        else:
            region = roi
            hemi = np.nan
        rows.append({
            'channel': ch_key,
            'region': region,
            'hemisphere': hemi,
        })
    return pd.DataFrame(rows)


def get_subject_roi_event_spectra(
    subject_obj: Subject,
    roi: str,
    event: str,
    event_window: Optional[Tuple[float, float]] = None,
    freq_range: Tuple[float, float] = (2.0, 40.0),
    condition: str = 'Day1',
    min_trials: int = 10,
) -> Dict[str, Any]:
    """
    Load and align subject-level ROI spectra for one event.

    Returns trial-aligned mean power spectra with shape:
    (n_trials, n_electrodes, n_freqs)
    """
    if event_window is None:
        event_window = OSCILLATION_EVENT_WINDOWS[event]
    if float(event_window[1]) <= float(event_window[0]):
        raise ValueError(f"Invalid event_window={event_window}")

    roi_electrodes = subject_obj.elec_df[
        subject_obj.elec_df.salman_region == roi
    ]['label'].astype(str).tolist()
    n_roi_electrodes_metadata = int(len(roi_electrodes))
    if len(roi_electrodes) == 0:
        raise ValueError(f"No electrodes found in ROI {roi} for subject {subject_obj.subject}")

    epochs_all = subject_obj._ensure_epochs_loaded(event)
    valid_roi_electrodes = [ch for ch in roi_electrodes if ch in epochs_all.ch_names]
    n_roi_electrodes_in_epochs = int(len(valid_roi_electrodes))
    if len(valid_roi_electrodes) == 0:
        raise ValueError(f"No ROI channels are present in epochs for ROI {roi} in {subject_obj.subject}")

    epochs = epochs_all.copy().pick(valid_roi_electrodes)
    picked_labels = [str(ch) for ch in epochs.ch_names]
    if len(picked_labels) == 0:
        raise ValueError(f"No picked channels remain for ROI {roi} in {subject_obj.subject}")

    subj_combined_df = subject_obj.combined_df[
        (subject_obj.combined_df.participant == subject_obj.subject) &
        (subject_obj.combined_df.condition == condition)
    ].copy()
    if len(subj_combined_df) == 0:
        raise ValueError(f"no_{condition.lower()}_trials_for_{subject_obj.subject}")

    trial_col = _oscillation_trial_column(event)
    if trial_col not in subj_combined_df.columns:
        raise ValueError(f"Missing trial column '{trial_col}' for {subject_obj.subject}")

    subj_combined_df = subj_combined_df.dropna(subset=[trial_col]).copy()
    if len(subj_combined_df) == 0:
        raise ValueError(f"no_valid_{trial_col}_rows_for_{subject_obj.subject}")

    trial_vals = pd.to_numeric(subj_combined_df[trial_col], errors='coerce').to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(trial_vals) &
        ((trial_vals - 1) >= 0) &
        ((trial_vals - 1) < len(epochs))
    )
    if int(np.sum(valid)) < int(min_trials):
        raise ValueError(
            f"Insufficient aligned trials for {subject_obj.subject}, ROI={roi}, "
            f"event={event}: {int(np.sum(valid))} < {int(min_trials)}"
        )

    trial_ix = trial_vals[valid].astype(np.int64) - 1
    epochs_aligned = epochs[trial_ix]
    psd = epochs_aligned.compute_psd(
        method='multitaper',
        fmin=float(freq_range[0]),
        fmax=float(freq_range[1]),
        tmin=float(event_window[0]),
        tmax=float(event_window[1]),
        verbose=False,
    )
    spectral_power = psd.get_data() if hasattr(psd, 'get_data') else psd.data
    spectral_power = np.asarray(spectral_power, dtype=np.float32)
    if spectral_power.ndim == 2:
        spectral_power = spectral_power[np.newaxis, :, :]
    if spectral_power.ndim != 3:
        raise ValueError(
            f"Unexpected PSD shape for {subject_obj.subject}, ROI={roi}, event={event}: "
            f"{spectral_power.shape}"
        )
    freqs = np.asarray(psd.freqs, dtype=np.float64)
    if int(len(freqs)) == 0:
        raise ValueError(f"No PSD frequencies retained for range {freq_range}")

    psd_ch_names = [str(ch) for ch in getattr(psd, 'ch_names', picked_labels)]
    n_psd_spectra = int(spectral_power.shape[1])
    n_psd_channel_names = int(len(psd_ch_names))
    psd_channel_name_mismatch = bool(n_psd_channel_names != n_psd_spectra)
    if psd_channel_name_mismatch:
        n_keep = int(min(n_psd_channel_names, n_psd_spectra))
        if n_keep == 0:
            raise ValueError(
                f"No usable PSD channels for {subject_obj.subject}, ROI={roi}, event={event}: "
                f"{n_psd_channel_names} names vs {n_psd_spectra} spectra"
            )
        psd_ch_names = psd_ch_names[:n_keep]
        spectral_power = spectral_power[:, :n_keep, :]
    n_roi_electrodes_in_psd = int(len(psd_ch_names))
    dropped_by_psd_labels = [ch for ch in picked_labels if ch not in psd_ch_names]

    electrode_meta = _get_roi_electrode_metadata(subject_obj, roi, psd_ch_names)
    electrode_labels = electrode_meta['channel'].to_numpy(dtype=object)
    electrode_region = electrode_meta['region'].to_numpy(dtype=object)
    electrode_hemi = electrode_meta['hemisphere'].to_numpy(dtype=object)

    return {
        'subject': subject_obj.subject,
        'roi': roi,
        'event': event,
        'condition': condition,
        'trial_column': trial_col,
        'trial_indices': trial_ix.astype(np.int64, copy=False),
        'n_trials': int(len(trial_ix)),
        'n_dropped_alignment': int((~valid).sum()),
        'event_window': tuple(float(x) for x in event_window),
        'freq_range': tuple(float(x) for x in freq_range),
        'freqs': freqs.astype(np.float32),
        'times': np.asarray(epochs_aligned.times, dtype=np.float32),
        'n_freq_bins': int(len(freqs)),
        'low_n_freq_bins': bool(len(freqs) < 20),
        'psd_method': 'multitaper',
        'n_roi_electrodes_metadata': n_roi_electrodes_metadata,
        'n_roi_electrodes_in_epochs': n_roi_electrodes_in_epochs,
        'n_roi_electrodes_in_psd': n_roi_electrodes_in_psd,
        'n_dropped_by_psd': int(len(dropped_by_psd_labels)),
        'dropped_by_psd_labels': ','.join(dropped_by_psd_labels),
        'n_psd_channel_names': n_psd_channel_names,
        'n_psd_spectra': n_psd_spectra,
        'psd_channel_name_mismatch': psd_channel_name_mismatch,
        'spectral_power': spectral_power,
        'electrode_labels': electrode_labels,
        'electrode_region': electrode_region,
        'electrode_hemi': electrode_hemi,
    }


def _fit_fooof_spectrum(
    freqs: np.ndarray,
    power_spectrum: np.ndarray,
    fooof_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit FOOOF to one electrode spectrum and extract the dominant peak."""
    from fooof import FOOOF

    settings = dict(OSCILLATION_FOOOF_DEFAULTS)
    if fooof_kwargs is not None:
        settings.update(fooof_kwargs)

    fit_range = settings.pop('freq_range', [float(np.nanmin(freqs)), float(np.nanmax(freqs))])
    settings.setdefault('verbose', False)

    freqs_arr = np.asarray(freqs, dtype=np.float64)
    power_arr = np.asarray(power_spectrum, dtype=np.float64)
    valid = np.isfinite(freqs_arr) & np.isfinite(power_arr) & (power_arr > 0)
    valid &= (freqs_arr >= float(fit_range[0])) & (freqs_arr <= float(fit_range[1]))
    if int(np.sum(valid)) < 3:
        return {
            'raw_psd': power_arr,
            'corrected_psd': np.full(power_arr.shape, np.nan, dtype=np.float64),
            'peak_freq': np.nan,
            'peak_height': np.nan,
            'peak_bandwidth': np.nan,
            'n_peaks': 0,
            'aperiodic_offset': np.nan,
            'aperiodic_exponent': np.nan,
            'r_squared': np.nan,
            'fit_error': np.nan,
            'fooof_model': None,
        }

    fit_freqs = freqs_arr[valid]
    fit_power = power_arr[valid]
    model = FOOOF(**settings)

    try:
        model.fit(fit_freqs, fit_power, fit_range)
    except Exception:
        return {
            'raw_psd': power_arr,
            'corrected_psd': np.full(power_arr.shape, np.nan, dtype=np.float64),
            'peak_freq': np.nan,
            'peak_height': np.nan,
            'peak_bandwidth': np.nan,
            'n_peaks': 0,
            'aperiodic_offset': np.nan,
            'aperiodic_exponent': np.nan,
            'r_squared': np.nan,
            'fit_error': np.nan,
            'fooof_model': None,
        }

    corrected_fit = getattr(model, '_spectrum_flat', None)
    if corrected_fit is None:
        ap_fit = getattr(model, '_ap_fit', None)
        if ap_fit is None:
            corrected_fit = np.full(fit_power.shape, np.nan, dtype=np.float64)
        else:
            corrected_fit = np.asarray(model.power_spectrum, dtype=np.float64) - np.asarray(ap_fit, dtype=np.float64)
    corrected_fit = np.asarray(corrected_fit, dtype=np.float64)

    corrected_full = np.full(power_arr.shape, np.nan, dtype=np.float64)
    corrected_full[valid] = corrected_fit

    peak_params = np.asarray(getattr(model, 'peak_params_', np.empty((0, 3))), dtype=np.float64)
    if peak_params.size == 0:
        peak_params = np.empty((0, 3), dtype=np.float64)
    elif peak_params.ndim == 1:
        peak_params = peak_params.reshape(1, -1)

    if peak_params.shape[0] > 0:
        peak_idx = int(np.nanargmax(peak_params[:, 1]))
        peak_freq = float(peak_params[peak_idx, 0])
        peak_height = float(peak_params[peak_idx, 1])
        peak_bandwidth = float(peak_params[peak_idx, 2]) if peak_params.shape[1] > 2 else np.nan
    else:
        peak_freq = np.nan
        peak_height = np.nan
        peak_bandwidth = np.nan

    aperiodic_params = np.asarray(getattr(model, 'aperiodic_params_', []), dtype=np.float64)
    ap_offset = float(aperiodic_params[0]) if aperiodic_params.size >= 1 else np.nan
    ap_exponent = float(aperiodic_params[-1]) if aperiodic_params.size >= 2 else np.nan

    return {
        'raw_psd': power_arr,
        'corrected_psd': corrected_full,
        'peak_freq': peak_freq,
        'peak_height': peak_height,
        'peak_bandwidth': peak_bandwidth,
        'n_peaks': int(peak_params.shape[0]),
        'aperiodic_offset': ap_offset,
        'aperiodic_exponent': ap_exponent,
        'r_squared': float(getattr(model, 'r_squared_', np.nan)),
        'fit_error': float(getattr(model, 'error_', np.nan)),
        'fooof_model': model,
    }


def compute_subject_roi_oscillation_features(
    subject_obj: Subject,
    roi: str,
    condition: str = 'Day1',
    freq_range: Tuple[float, float] = (2.0, 40.0),
    event_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    fooof_kwargs: Optional[Dict[str, Any]] = None,
    min_trials: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Compute electrode-level PSD and FOOOF features for one subject/ROI."""
    event_windows = dict(OSCILLATION_EVENT_WINDOWS if event_windows is None else event_windows)
    electrode_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for event in ('feedback_start', 'cue_on'):
        aligned = get_subject_roi_event_spectra(
            subject_obj=subject_obj,
            roi=roi,
            event=event,
            event_window=event_windows[event],
            freq_range=freq_range,
            condition=condition,
            min_trials=min_trials,
        )

        freqs = np.asarray(aligned['freqs'], dtype=np.float64)
        spectra = np.asarray(aligned['spectral_power'], dtype=np.float64)
        trial_mean_spectra = np.nanmean(spectra, axis=0)
        if trial_mean_spectra.ndim != 2:
            raise ValueError(
                f"Unexpected trial-mean PSD shape for {subject_obj.subject}, ROI={roi}, event={event}: "
                f"{trial_mean_spectra.shape}"
            )
        if trial_mean_spectra.shape[0] != len(aligned['electrode_labels']):
            raise ValueError(
                f"PSD channel count mismatch for {subject_obj.subject}, ROI={roi}, event={event}: "
                f"{trial_mean_spectra.shape[0]} spectra vs {len(aligned['electrode_labels'])} electrodes"
            )

        audit_rows.append({
            'participant': subject_obj.subject,
            'roi': roi,
            'event': event,
            'condition': condition,
            'trial_column': aligned['trial_column'],
            'n_trials': int(aligned['n_trials']),
            'n_dropped_alignment': int(aligned['n_dropped_alignment']),
            'n_freq_bins': int(aligned['n_freq_bins']),
            'low_n_freq_bins': bool(aligned['low_n_freq_bins']),
            'psd_method': aligned['psd_method'],
            'n_roi_electrodes_metadata': int(aligned['n_roi_electrodes_metadata']),
            'n_roi_electrodes_in_epochs': int(aligned['n_roi_electrodes_in_epochs']),
            'n_roi_electrodes_in_psd': int(aligned['n_roi_electrodes_in_psd']),
            'n_dropped_by_psd': int(aligned['n_dropped_by_psd']),
            'dropped_by_psd_labels': aligned['dropped_by_psd_labels'],
            'n_psd_channel_names': int(aligned['n_psd_channel_names']),
            'n_psd_spectra': int(aligned['n_psd_spectra']),
            'psd_channel_name_mismatch': bool(aligned['psd_channel_name_mismatch']),
            'event_tmin': float(aligned['event_window'][0]),
            'event_tmax': float(aligned['event_window'][1]),
        })

        for chan_idx, channel in enumerate(aligned['electrode_labels']):
            raw_psd = np.asarray(trial_mean_spectra[chan_idx], dtype=np.float64)
            raw_psd_z = _nan_zscore_1d(raw_psd)
            fooof_res = _fit_fooof_spectrum(
                freqs=freqs,
                power_spectrum=raw_psd,
                fooof_kwargs=fooof_kwargs,
            )
            corrected_psd = np.asarray(fooof_res['corrected_psd'], dtype=np.float64)
            corrected_psd_z = _nan_zscore_1d(corrected_psd)

            hemi = aligned['electrode_hemi'][chan_idx]
            region = aligned['electrode_region'][chan_idx]
            color = ROI_COLOR_MAP.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32))
            electrode_key = f"{subject_obj.subject}_{channel}"

            for freq_idx, freq in enumerate(freqs):
                electrode_rows.append({
                    'participant': subject_obj.subject,
                    'roi': roi,
                    'event': event,
                    'condition': condition,
                    'channel': str(channel),
                    'electrode_key': electrode_key,
                    'hemisphere': hemi,
                    'region': region,
                    'frequency': float(freq),
                    'psd_raw': float(raw_psd[freq_idx]) if np.isfinite(raw_psd[freq_idx]) else np.nan,
                    'psd_raw_z': float(raw_psd_z[freq_idx]) if np.isfinite(raw_psd_z[freq_idx]) else np.nan,
                    'psd_corrected': float(corrected_psd[freq_idx]) if np.isfinite(corrected_psd[freq_idx]) else np.nan,
                    'psd_corrected_z': float(corrected_psd_z[freq_idx]) if np.isfinite(corrected_psd_z[freq_idx]) else np.nan,
                    'peak_freq': float(fooof_res['peak_freq']) if np.isfinite(fooof_res['peak_freq']) else np.nan,
                    'peak_height': float(fooof_res['peak_height']) if np.isfinite(fooof_res['peak_height']) else np.nan,
                    'peak_bandwidth': float(fooof_res['peak_bandwidth']) if np.isfinite(fooof_res['peak_bandwidth']) else np.nan,
                    'n_peaks': int(fooof_res['n_peaks']),
                    'aperiodic_offset': float(fooof_res['aperiodic_offset']) if np.isfinite(fooof_res['aperiodic_offset']) else np.nan,
                    'aperiodic_exponent': float(fooof_res['aperiodic_exponent']) if np.isfinite(fooof_res['aperiodic_exponent']) else np.nan,
                    'r_squared': float(fooof_res['r_squared']) if np.isfinite(fooof_res['r_squared']) else np.nan,
                    'fit_error': float(fooof_res['fit_error']) if np.isfinite(fooof_res['fit_error']) else np.nan,
                    'n_trials': int(aligned['n_trials']),
                    'n_freq_bins': int(aligned['n_freq_bins']),
                    'low_n_freq_bins': bool(aligned['low_n_freq_bins']),
                    'psd_method': aligned['psd_method'],
                    'n_roi_electrodes_metadata': int(aligned['n_roi_electrodes_metadata']),
                    'n_roi_electrodes_in_epochs': int(aligned['n_roi_electrodes_in_epochs']),
                    'n_roi_electrodes_in_psd': int(aligned['n_roi_electrodes_in_psd']),
                    'n_dropped_by_psd': int(aligned['n_dropped_by_psd']),
                    'n_psd_channel_names': int(aligned['n_psd_channel_names']),
                    'n_psd_spectra': int(aligned['n_psd_spectra']),
                    'psd_channel_name_mismatch': bool(aligned['psd_channel_name_mismatch']),
                    'color_r': float(color[0]),
                    'color_g': float(color[1]),
                    'color_b': float(color[2]),
                })

    return {
        'electrode_psd_df': pd.DataFrame(electrode_rows),
        'audit_df': pd.DataFrame(audit_rows),
    }


def _summarize_roi_oscillation_psd(electrode_psd_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrected/z-scored PSD by ROI, event, and frequency."""
    if electrode_psd_df.empty:
        return pd.DataFrame()

    grouped = (
        electrode_psd_df
        .groupby(['roi', 'event', 'frequency'], as_index=False)
        .agg(
            psd_corrected_z_mean=('psd_corrected_z', 'mean'),
            psd_corrected_z_se=('psd_corrected_z', lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x)))) if np.sum(np.isfinite(x)) > 1 else np.nan),
            psd_raw_z_mean=('psd_raw_z', 'mean'),
            psd_raw_z_se=('psd_raw_z', lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x)))) if np.sum(np.isfinite(x)) > 1 else np.nan),
            psd_corrected_mean=('psd_corrected', 'mean'),
            psd_raw_mean=('psd_raw', 'mean'),
            n_electrodes=('electrode_key', 'nunique'),
            n_subjects=('participant', 'nunique'),
            n_freq_bins=('n_freq_bins', 'max'),
            low_n_freq_bins=('low_n_freq_bins', 'max'),
        )
    )
    return grouped


def summarize_oscillation_electrodes(electrode_psd_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse frequency-level output to one row per participant/ROI/channel/event."""
    if electrode_psd_df.empty:
        return pd.DataFrame()

    cols = [
        'participant', 'roi', 'event', 'condition', 'channel', 'electrode_key',
        'hemisphere', 'region', 'peak_freq', 'peak_height', 'peak_bandwidth',
        'n_peaks', 'aperiodic_offset', 'aperiodic_exponent', 'r_squared',
        'fit_error', 'n_trials', 'n_freq_bins', 'low_n_freq_bins', 'psd_method',
        'n_roi_electrodes_metadata', 'n_roi_electrodes_in_epochs',
        'n_roi_electrodes_in_psd', 'n_dropped_by_psd',
        'n_psd_channel_names', 'n_psd_spectra', 'psd_channel_name_mismatch',
        'color_r', 'color_g', 'color_b',
    ]
    available = [col for col in cols if col in electrode_psd_df.columns]
    summary_df = (
        electrode_psd_df[available]
        .drop_duplicates(subset=['participant', 'roi', 'channel', 'event'])
        .sort_values(['roi', 'participant', 'channel', 'event'])
        .reset_index(drop=True)
    )
    return summary_df


def build_oscillation_matched_pairs(
    electrode_summary_df: pd.DataFrame,
    properties: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build matched feedback/cue electrode pairs for requested properties."""
    if electrode_summary_df.empty:
        return pd.DataFrame()

    properties = list(OSCILLATION_PROPERTY_LABELS.keys()) if properties is None else list(properties)
    base_cols = ['participant', 'roi', 'channel', 'electrode_key']
    fb = electrode_summary_df[electrode_summary_df.event == 'feedback_start'][base_cols + properties].copy()
    cue = electrode_summary_df[electrode_summary_df.event == 'cue_on'][base_cols + properties].copy()
    fb = fb.rename(columns={prop: f'{prop}_feedback_start' for prop in properties})
    cue = cue.rename(columns={prop: f'{prop}_cue_on' for prop in properties})
    matched = fb.merge(cue, on=base_cols, how='inner')

    rows: List[Dict[str, Any]] = []
    for _, row in matched.iterrows():
        base = {col: row[col] for col in base_cols}
        for prop in properties:
            feedback_val = row.get(f'{prop}_feedback_start', np.nan)
            cue_val = row.get(f'{prop}_cue_on', np.nan)
            if not (np.isfinite(feedback_val) and np.isfinite(cue_val)):
                continue
            rows.append({
                **base,
                'property': prop,
                'label': OSCILLATION_PROPERTY_LABELS.get(prop, prop),
                'feedback_start': float(feedback_val),
                'cue_on': float(cue_val),
                'delta_feedback_minus_cue': float(feedback_val - cue_val),
            })
    return pd.DataFrame(rows)


def summarize_oscillation_property_stats(
    electrode_summary_df: pd.DataFrame,
    matched_pairs_df: Optional[pd.DataFrame] = None,
    properties: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize event-wise and paired property statistics by ROI."""
    if electrode_summary_df.empty:
        return pd.DataFrame()

    properties = list(OSCILLATION_PROPERTY_LABELS.keys()) if properties is None else list(properties)
    if matched_pairs_df is None:
        matched_pairs_df = build_oscillation_matched_pairs(electrode_summary_df, properties=properties)

    rows: List[Dict[str, Any]] = []
    for roi in sorted(electrode_summary_df['roi'].dropna().unique().tolist()):
        roi_df = electrode_summary_df[electrode_summary_df.roi == roi].copy()
        for prop in properties:
            fb_vals = roi_df.loc[roi_df.event == 'feedback_start', prop].to_numpy(dtype=np.float64)
            cue_vals = roi_df.loc[roi_df.event == 'cue_on', prop].to_numpy(dtype=np.float64)
            fb_vals = fb_vals[np.isfinite(fb_vals)]
            cue_vals = cue_vals[np.isfinite(cue_vals)]

            pair_prop = matched_pairs_df[
                (matched_pairs_df.roi == roi) &
                (matched_pairs_df.property == prop)
            ].copy()
            n_pairs = int(len(pair_prop))
            if n_pairs >= 2:
                test_res = ttest_rel(
                    pair_prop['feedback_start'].to_numpy(dtype=np.float64),
                    pair_prop['cue_on'].to_numpy(dtype=np.float64),
                    nan_policy='omit',
                )
                t_stat = float(test_res.statistic)
                p_value = float(test_res.pvalue)
                status = 'ok'
            else:
                t_stat = np.nan
                p_value = np.nan
                status = 'insufficient_pairs'

            rows.append({
                'roi': roi,
                'property': prop,
                'label': OSCILLATION_PROPERTY_LABELS.get(prop, prop),
                'n_feedback': int(len(fb_vals)),
                'n_cue': int(len(cue_vals)),
                'n_pairs': n_pairs,
                'mean_feedback': float(np.nanmean(fb_vals)) if len(fb_vals) > 0 else np.nan,
                'mean_cue': float(np.nanmean(cue_vals)) if len(cue_vals) > 0 else np.nan,
                'mean_difference': float(np.nanmean(pair_prop['delta_feedback_minus_cue'])) if n_pairs > 0 else np.nan,
                't_stat': t_stat,
                'p_value': p_value,
                'p_fdr': np.nan,
                'significant_fdr': False,
                'status': status,
            })

    stats_df = pd.DataFrame(rows)
    valid = (stats_df.status == 'ok') & np.isfinite(stats_df.p_value.to_numpy(dtype=np.float64))
    if int(np.sum(valid)) > 0:
        _, p_fdr, _, _ = multipletests(
            stats_df.loc[valid, 'p_value'].to_numpy(dtype=np.float64),
            method='fdr_bh',
        )
        stats_df.loc[valid, 'p_fdr'] = p_fdr
        stats_df.loc[valid, 'significant_fdr'] = p_fdr < 0.05
    return stats_df.sort_values(['property', 'roi']).reset_index(drop=True)


def _process_subject_oscillation_batch(
    subject: str,
    rois: List[str],
    base_dir: str,
    load_dir: str,
    condition: str,
    freq_range: Tuple[float, float],
    event_windows: Dict[str, Tuple[float, float]],
    fooof_kwargs: Optional[Dict[str, Any]],
    min_trials: int,
    cache_mode: str,
    verbose: bool,
) -> Dict[str, pd.DataFrame]:
    """Worker for subject-level oscillation feature extraction."""
    subject_obj = Subject(
        subject=subject,
        base_dir=base_dir,
        load_dir=load_dir,
        min_trials=min_trials,
        verbose=verbose,
        cache_mode=cache_mode,
    )
    electrode_dfs: List[pd.DataFrame] = []
    audit_dfs: List[pd.DataFrame] = []

    for roi in rois:
        try:
            result = compute_subject_roi_oscillation_features(
                subject_obj=subject_obj,
                roi=roi,
                condition=condition,
                freq_range=freq_range,
                event_windows=event_windows,
                fooof_kwargs=fooof_kwargs,
                min_trials=min_trials,
            )
        except (ValueError, FileNotFoundError) as exc:
            audit_dfs.append(pd.DataFrame([{
                'participant': subject,
                'roi': roi,
                'event': 'all',
                'condition': condition,
                'trial_column': np.nan,
                'n_trials': 0,
                'n_dropped_alignment': np.nan,
                'n_freq_bins': np.nan,
                'low_n_freq_bins': np.nan,
                'psd_method': 'multitaper',
                'n_roi_electrodes_metadata': np.nan,
                'n_roi_electrodes_in_epochs': np.nan,
                'n_roi_electrodes_in_psd': np.nan,
                'n_dropped_by_psd': np.nan,
                'dropped_by_psd_labels': np.nan,
                'n_psd_channel_names': np.nan,
                'n_psd_spectra': np.nan,
                'psd_channel_name_mismatch': np.nan,
                'event_tmin': np.nan,
                'event_tmax': np.nan,
                'status': str(exc),
            }]))
            continue

        if not result['electrode_psd_df'].empty:
            electrode_dfs.append(result['electrode_psd_df'])
        if not result['audit_df'].empty:
            audit_df = result['audit_df'].copy()
            audit_df['status'] = 'ok'
            audit_dfs.append(audit_df)

    return {
        'electrode_psd_df': pd.concat(electrode_dfs, ignore_index=True) if len(electrode_dfs) > 0 else pd.DataFrame(),
        'audit_df': pd.concat(audit_dfs, ignore_index=True) if len(audit_dfs) > 0 else pd.DataFrame(),
    }


def run_roi_oscillation_comparison(
    subjects: List[str],
    rois: Optional[List[str]],
    base_dir: str,
    load_dir: str,
    condition: str = 'Day1',
    freq_range: Tuple[float, float] = (2.0, 40.0),
    event_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    fooof_kwargs: Optional[Dict[str, Any]] = None,
    min_trials: int = 10,
    n_jobs: int = 1,
    cache_mode: str = 'reuse',
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Batch ROI oscillation comparison between feedback_start and cue_on.

    Returns:
        electrode_psd_df: long-form frequency-level electrode spectra and FOOOF features
        roi_psd_df: ROI x event x frequency summary
        electrode_summary_df: one row per participant x ROI x channel x event
        matched_pairs_df: matched feedback/cue electrode pairs by property
        property_stats_df: ROI-level paired statistics for each property
        peak_stats_df: ROI-level paired peak-frequency statistics
        audit_df: subject/ROI audit table
    """
    rois = list(ROI_COLOR_MAP.keys()) if rois is None else list(rois)
    event_windows = dict(OSCILLATION_EVENT_WINDOWS if event_windows is None else event_windows)
    fooof_settings = dict(OSCILLATION_FOOOF_DEFAULTS)
    if fooof_kwargs is not None:
        fooof_settings.update(fooof_kwargs)

    batch_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_process_subject_oscillation_batch)(
            subject=subject,
            rois=rois,
            base_dir=base_dir,
            load_dir=load_dir,
            condition=condition,
            freq_range=freq_range,
            event_windows=event_windows,
            fooof_kwargs=fooof_settings,
            min_trials=min_trials,
            cache_mode=cache_mode,
            verbose=False,
        )
        for subject in subjects
    )

    electrode_frames = [res['electrode_psd_df'] for res in batch_results if not res['electrode_psd_df'].empty]
    audit_frames = [res['audit_df'] for res in batch_results if not res['audit_df'].empty]

    electrode_psd_df = pd.concat(electrode_frames, ignore_index=True) if len(electrode_frames) > 0 else pd.DataFrame()
    audit_df = pd.concat(audit_frames, ignore_index=True) if len(audit_frames) > 0 else pd.DataFrame()
    roi_psd_df = _summarize_roi_oscillation_psd(electrode_psd_df)
    electrode_summary_df = summarize_oscillation_electrodes(electrode_psd_df)
    matched_pairs_df = build_oscillation_matched_pairs(electrode_summary_df)
    property_stats_df = summarize_oscillation_property_stats(
        electrode_summary_df=electrode_summary_df,
        matched_pairs_df=matched_pairs_df,
    )
    peak_stats_df = property_stats_df[property_stats_df.property == 'peak_freq'].reset_index(drop=True)

    return {
        'electrode_psd_df': electrode_psd_df,
        'electrode_summary_df': electrode_summary_df,
        'matched_pairs_df': matched_pairs_df,
        'roi_psd_df': roi_psd_df,
        'property_stats_df': property_stats_df,
        'peak_stats_df': peak_stats_df,
        'audit_df': audit_df,
    }


def plot_roi_oscillation_psd_summary(
    roi_psd_df: pd.DataFrame,
    rois: Optional[List[str]] = None,
    colors: Optional[Dict[str, np.ndarray]] = None,
    spectrum: Literal['corrected_z', 'raw_z'] = 'corrected_z',
    xscale: Literal['linear', 'log'] = 'linear',
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ROI-wise PSD curves for both events.

    spectrum
        ``corrected_z``: FOOOF aperiodic-removed spectrum, z-scored along frequency per electrode.
        ``raw_z``: raw trial-mean spectrum, z-scored along frequency per electrode (uncorrected).
    xscale
        ``linear`` (default) or ``log`` for frequency on the x-axis (log uses base-10).
        Log mode uses major ticks at 1, 4, 8, 13, and 30 Hz.
    """
    if roi_psd_df.empty:
        raise ValueError("roi_psd_df is empty")

    if spectrum == 'corrected_z':
        mean_col, se_col = 'psd_corrected_z_mean', 'psd_corrected_z_se'
        ylabel = 'Corrected PSD (z)'
    else:
        mean_col, se_col = 'psd_raw_z_mean', 'psd_raw_z_se'
        ylabel = 'Uncorrected PSD (z)'

    # Log x: default LogFormatter uses scientific notation and can hide most labels; we use plain Hz labels.
    _log_xticks_hz = np.array([1.0, 4.0, 8.0, 13.0, 30.0], dtype=float)

    def _log_freq_tick_fmt(x: float, pos: Optional[int]) -> str:
        for xi, s in zip(
            _log_xticks_hz, ('1', '4', '8', '13', '30')
        ):
            if np.isclose(x, xi, rtol=1e-12, atol=1e-6):
                return s
        return ''

    colors = ROI_COLOR_MAP if colors is None else colors
    rois = sorted(roi_psd_df['roi'].unique().tolist()) if rois is None else list(rois)
    ncols = min(3, max(1, len(rois)))
    nrows = int(np.ceil(len(rois) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
        dpi=150,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, roi in zip(axes.flatten(), rois):
        roi_df = roi_psd_df[roi_psd_df.roi == roi].copy()
        base_color = colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        n_electrodes = int(roi_df['n_electrodes'].max()) if 'n_electrodes' in roi_df.columns and len(roi_df) > 0 else 0
        for event, style in [('feedback_start', '-'), ('cue_on', '--')]:
            event_df = roi_df[roi_df.event == event].sort_values('frequency')
            if event_df.empty:
                continue
            line_color = np.clip(base_color * (0.85 if event == 'feedback_start' else 1.0), 0.0, 1.0)
            ax.plot(
                event_df['frequency'],
                event_df[mean_col],
                linestyle=style,
                linewidth=2,
                color=line_color,
                label=event,
            )
            if se_col in event_df.columns:
                lower = event_df[mean_col] - event_df[se_col]
                upper = event_df[mean_col] + event_df[se_col]
                ax.fill_between(event_df['frequency'], lower, upper, color=line_color, alpha=0.15)
        ax.set_title(f'{roi} (n={n_electrodes})')
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color='k', linewidth=0.75, alpha=0.4)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)
        if xscale == 'log':
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(_log_xticks_hz))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(_log_freq_tick_fmt))
            ax.set_xticks(_log_xticks_hz)
        else:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            f_min = float(np.nanmin(roi_df['frequency'])) if len(roi_df) else 0.0
            f_max = float(np.nanmax(roi_df['frequency'])) if len(roi_df) else 0.0
            if np.isfinite(f_min) and np.isfinite(f_max) and f_max >= f_min:
                start = int(np.floor(f_min / 5.0) * 5)
                stop = int(np.ceil(f_max / 5.0) * 5) + 1
                ax.set_xticks(np.arange(start, stop, 5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(
            axis='both',
            which='major',
            direction='out',
            length=7,
            width=1.2,
            colors='black',
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelbottom=True,
            labelleft=True,
        )
        ax.tick_params(
            axis='both',
            which='minor',
            direction='out',
            length=4,
            width=0.9,
            colors='black',
            bottom=True,
            left=True,
            top=False,
            right=False,
        )
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel('Frequency (Hz)')
    for ax in axes.flatten()[len(rois):]:
        ax.set_visible(False)
    if xscale == 'log':
        f_data_min = float(roi_psd_df['frequency'].min())
        f_data_max = float(roi_psd_df['frequency'].max())
        x_left = max(0.5, min(1.0, f_data_min * 0.95))
        x_right = max(30.0, f_data_max * 1.02)
        for ax in axes.flatten():
            if ax.get_visible():
                ax.set_xlim(x_left, x_right)
    fig.tight_layout()
    return fig, axes


def plot_roi_peak_frequency_histograms(
    electrode_psd_df: pd.DataFrame,
    rois: Optional[List[str]] = None,
    colors: Optional[Dict[str, np.ndarray]] = None,
    bins: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ROI-wise histograms of dominant FOOOF peak frequency."""
    if electrode_psd_df.empty:
        raise ValueError("electrode_psd_df is empty")

    colors = ROI_COLOR_MAP if colors is None else colors
    rois = sorted(electrode_psd_df['roi'].unique().tolist()) if rois is None else list(rois)
    bins = np.arange(2.0, 41.0, 2.0) if bins is None else bins
    peak_df = electrode_psd_df[
        ['participant', 'roi', 'channel', 'electrode_key', 'event', 'peak_freq']
    ].drop_duplicates()
    peak_df = peak_df[np.isfinite(peak_df['peak_freq'])].copy()

    fig, axes = plt.subplots(len(rois), 1, figsize=(8, max(3, 2.4 * len(rois))), sharex=True, dpi=150)
    axes = np.atleast_1d(axes)
    for ax, roi in zip(axes, rois):
        roi_df = peak_df[peak_df.roi == roi]
        base_color = colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        title_bits = []
        for event, alpha in [('feedback_start', 0.45), ('cue_on', 0.75)]:
            event_df = roi_df[roi_df.event == event]
            if event_df.empty:
                title_bits.append(f"{event}=0")
                continue
            ax.hist(
                event_df['peak_freq'].to_numpy(dtype=np.float64),
                bins=bins,
                color=base_color,
                alpha=alpha,
                label=event,
            )
            title_bits.append(f"{event}={event_df['electrode_key'].nunique()}")
        ax.set_title(f"{roi} ({', '.join(title_bits)})" if len(title_bits) > 0 else roi)
        ax.set_ylabel('Count')
        ax.legend(frameon=False, fontsize=8, loc='upper right')

    axes[-1].set_xlabel('Peak frequency (Hz)')
    fig.tight_layout()
    return fig, axes


def plot_roi_property_summaries(
    electrode_summary_df: pd.DataFrame,
    properties: Optional[List[str]] = None,
    rois: Optional[List[str]] = None,
    colors: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ROI-level summaries for periodic and aperiodic FOOOF properties."""
    if electrode_summary_df.empty:
        raise ValueError("electrode_summary_df is empty")

    colors = ROI_COLOR_MAP if colors is None else colors
    properties = list(OSCILLATION_PROPERTY_LABELS.keys()) if properties is None else list(properties)
    rois = sorted(electrode_summary_df['roi'].unique().tolist()) if rois is None else list(rois)

    n_props = len(properties)
    ncols = 2
    nrows = int(np.ceil(n_props / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 4.2 * nrows)), dpi=150)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, prop in zip(axes.flatten(), properties):
        prop_df = electrode_summary_df[
            ['roi', 'event', 'electrode_key', prop]
        ].dropna(subset=[prop]).copy()
        if prop_df.empty:
            ax.set_visible(False)
            continue

        event_order = ['feedback_start', 'cue_on']
        summary_df = (
            prop_df.groupby(['roi', 'event'])[prop]
            .agg(['mean', 'sem'])
            .reset_index()
            .rename(columns={'mean': 'value_mean', 'sem': 'value_sem'})
        )
        x_positions = np.arange(len(rois), dtype=float)
        bar_width = 0.36
        event_offsets = {
            'feedback_start': -bar_width / 2.0,
            'cue_on': bar_width / 2.0,
        }
        event_alphas = {
            'feedback_start': 0.85,
            'cue_on': 0.55,
        }
        for event in event_order:
            event_df = (
                summary_df[summary_df.event == event]
                .set_index('roi')
                .reindex(rois)
                .reset_index()
            )
            means = event_df['value_mean'].to_numpy(dtype=float)
            sems = event_df['value_sem'].fillna(0.0).to_numpy(dtype=float)
            bar_x = x_positions + event_offsets[event]
            bar_colors = [colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32)) for roi in rois]
            ax.bar(
                bar_x,
                means,
                width=bar_width,
                yerr=sems,
                color=bar_colors,
                alpha=event_alphas[event],
                edgecolor='black',
                linewidth=0.6,
                capsize=2.5,
                label=event,
                zorder=2,
            )
        rng = np.random.default_rng(0)
        for roi_idx, roi in enumerate(rois):
            roi_color = colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32))
            for event in event_order:
                event_vals = prop_df.loc[
                    (prop_df['roi'] == roi) & (prop_df['event'] == event),
                    prop
                ].to_numpy(dtype=float)
                if event_vals.size == 0:
                    continue
                x_center = roi_idx + event_offsets[event]
                x_jitter = rng.uniform(-0.045, 0.045, size=event_vals.size)
                ax.scatter(
                    np.full(event_vals.size, x_center, dtype=float) + x_jitter,
                    event_vals,
                    s=12,
                    color=roi_color,
                    alpha=0.45 if event == 'cue_on' else 0.6,
                    edgecolors='none',
                    zorder=3,
                )

        # Clean duplicate legends from layered seaborn calls.
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) >= 2:
            ax.legend(handles[:2], labels[:2], frameon=False, fontsize=8, loc='best')
        else:
            ax.legend([], [], frameon=False)

        sample_counts = (
            prop_df.groupby(['roi', 'event'])['electrode_key']
            .nunique()
            .reset_index(name='n')
        )
        y_min, y_max = ax.get_ylim()
        y_text = y_max - 0.04 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else y_max
        for roi_idx, roi in enumerate(rois):
            fb_n = sample_counts.loc[
                (sample_counts.roi == roi) & (sample_counts.event == 'feedback_start'),
                'n'
            ]
            cue_n = sample_counts.loc[
                (sample_counts.roi == roi) & (sample_counts.event == 'cue_on'),
                'n'
            ]
            ax.text(
                roi_idx,
                y_text,
                f"n={int(fb_n.iloc[0]) if len(fb_n) else 0}/{int(cue_n.iloc[0]) if len(cue_n) else 0}",
                ha='center',
                va='top',
                fontsize=7,
                color='dimgray',
            )

        ax.set_title(OSCILLATION_PROPERTY_LABELS.get(prop, prop))
        ax.set_xlabel('ROI')
        ax.set_ylabel(OSCILLATION_PROPERTY_LABELS.get(prop, prop))
        ax.set_xticks(np.arange(len(rois)))
        ax.set_xticklabels(rois, rotation=30)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='y', which='major', length=4)
        ax.tick_params(axis='y', which='minor', length=2)
        ax.tick_params(axis='x', rotation=30)

    for ax in axes.flatten()[n_props:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_roi_property_deltas(
    matched_pairs_df: pd.DataFrame,
    properties: Optional[List[str]] = None,
    rois: Optional[List[str]] = None,
    colors: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot paired event deltas for matched electrodes across ROIs."""
    if matched_pairs_df.empty:
        raise ValueError("matched_pairs_df is empty")

    colors = ROI_COLOR_MAP if colors is None else colors
    properties = ['peak_freq', 'peak_height', 'peak_bandwidth', 'aperiodic_exponent', 'aperiodic_offset'] if properties is None else list(properties)
    rois = sorted(matched_pairs_df['roi'].unique().tolist()) if rois is None else list(rois)

    n_props = len(properties)
    ncols = 2
    nrows = int(np.ceil(n_props / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 4.0 * nrows)), dpi=150)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, prop in zip(axes.flatten(), properties):
        prop_df = matched_pairs_df[matched_pairs_df.property == prop].copy()
        if prop_df.empty:
            ax.set_visible(False)
            continue
        summary_df = (
            prop_df.groupby('roi')['delta_feedback_minus_cue']
            .agg(['mean', 'sem'])
            .reindex(rois)
            .reset_index()
            .rename(columns={'mean': 'value_mean', 'sem': 'value_sem'})
        )
        x_positions = np.arange(len(rois), dtype=float)
        ax.bar(
            x_positions,
            summary_df['value_mean'].to_numpy(dtype=float),
            yerr=summary_df['value_sem'].fillna(0.0).to_numpy(dtype=float),
            width=0.62,
            color=[colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32)) for roi in rois],
            edgecolor='black',
            linewidth=0.6,
            capsize=2.5,
            alpha=0.75,
            zorder=2,
        )
        rng = np.random.default_rng(0)
        for roi_idx, roi in enumerate(rois):
            roi_vals = prop_df.loc[
                prop_df['roi'] == roi, 'delta_feedback_minus_cue'
            ].to_numpy(dtype=float)
            if roi_vals.size == 0:
                continue
            x_jitter = rng.uniform(-0.08, 0.08, size=roi_vals.size)
            ax.scatter(
                np.full(roi_vals.size, roi_idx, dtype=float) + x_jitter,
                roi_vals,
                s=12,
                color=colors.get(roi, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
                alpha=0.55,
                edgecolors='none',
                zorder=3,
            )
        counts = prop_df.groupby('roi')['electrode_key'].nunique()
        y_min, y_max = ax.get_ylim()
        y_text = y_max - 0.04 * (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else y_max
        for roi_idx, roi in enumerate(rois):
            ax.text(
                roi_idx,
                y_text,
                f"n={int(counts.get(roi, 0))}",
                ha='center',
                va='top',
                fontsize=7,
                color='dimgray',
            )
        ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
        ax.set_title(f"{OSCILLATION_PROPERTY_LABELS.get(prop, prop)} delta")
        ax.set_xlabel('ROI')
        ax.set_ylabel('feedback_start - cue_on')
        ax.set_xticks(np.arange(len(rois)))
        ax.set_xticklabels(rois, rotation=30)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis='y', which='major', length=4)
        ax.tick_params(axis='y', which='minor', length=2)
        ax.tick_params(axis='x', rotation=30)

    for ax in axes.flatten()[n_props:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes
