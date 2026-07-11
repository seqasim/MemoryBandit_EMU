from pathlib import Path
import inspect

import nbformat as nbf


NOTEBOOK_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-ACC_HPC_RPE_Memory_Overlap_PairMI_Regression.ipynb"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(inspect.cleandoc(text) + "\n")


cells = [
    code(
        """
        %matplotlib inline
        %reload_ext autoreload
        %autoreload 2
        """
    ),
    md(
        """
        # ACC-HPC Pair-Level MI Regression In The ACC RPE-Memory Overlap Window

        This notebook mirrors the ACC overlap-window regression workflow, but replaces
        electrode-level HFA with pair-level ACC-HPC local Gaussian mutual information.

        The core dependent variable is trial-level local MI for each ACC-HPC pair in:

        `RPE_MEMORY_OVERLAP_WINDOW_ACC = (0.37, 0.60)`

        The pair-level regression is:

        `mi_local ~ zrpe * phit`
        """
    ),
    code(
        """
        import re
        import warnings
        from itertools import product
        from pathlib import Path
        from typing import Any, Dict, List, Tuple

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import patsy
        import seaborn as sns
        from IPython.display import display
        from scipy import stats
        from statsmodels.regression.linear_model import OLS
        from tqdm.auto import tqdm

        from plot_config import set_plot_style
        from mv_regression_utils import Subject
        from mv_granger_utils import (
            load_behavior_tables,
            preprocess_combined_df,
            preprocess_learn_df,
        )

        warnings.filterwarnings("ignore")
        set_plot_style()
        np.random.seed(42)
        """
    ),
    code(
        """
        # ---- Config ----
        base_dir = Path('/Volumes/T7 Shield')
        load_dir = base_dir / 'work/qasims01/MemoryBanditData/EMU'

        analysis_name = 'acc_hpc_pair_mi_overlap_regression'
        analysis_root = Path(
            '/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/analysis_outputs'
        ) / analysis_name
        figure_root = analysis_root / 'figures'
        analysis_root.mkdir(parents=True, exist_ok=True)
        figure_root.mkdir(parents=True, exist_ok=True)

        acc_reinstatement_csv = (
            '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/crossdecoding/rpe_category/'
            'ACC_rpe_category_HFA_subject_reinstatement.csv'
        )
        acc_hit_decoding_csv = (
            '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/decoding_cue/hit_miss/'
            'ACC_memory_outcome_hit_miss_hfa_subject_reinstatement.csv'
        )
        acc_reinstatement_stats_txt = (
            '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/crossdecoding/rpe_category/'
            'ACC_rpe_category_HFA_crossdecoding_stats.txt'
        )
        acc_hit_decoding_stats_txt = (
            '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/decoding_cue/hit_miss/'
            'ACC_memory_outcome_hit_miss_hfa_200_stats.txt'
        )

        EVENT = 'cue_on'
        CONDITION = 'Day1'
        ACC_ROI = 'ACC'
        HPC_ROI = 'HPC'

        RPE_REINSTATEMENT_CUE_WINDOW_ACC = (0.31, 0.72)
        HIT_DECODING_CUE_WINDOW_ACC = (0.37, 0.60)
        RPE_MEMORY_OVERLAP_WINDOW_ACC = (0.37, 0.60)

        FREQ_RANGE = (70.0, 201.0)
        DOWNSAMPLE_FACTOR = 5
        WIN_MS = 200
        HFA_SMOOTH_METHOD = 'movmean'
        HFA_ARTIFACT_ABS_THRESH = 10.0

        REG_EPS = 1e-6
        PCA_VAR_KEEP = 0.99
        PCA_MAX_COMPONENTS = 20
        N_PERMUTATIONS = 1000
        RANDOM_STATE = 42
        MIN_TRIALS = 15

        mi_trial_cache_csv = analysis_root / 'acc_hpc_pair_mi_trial_cache.csv'
        pair_results_csv = analysis_root / 'acc_hpc_pair_mi_regression_results.csv'
        pair_audit_csv = analysis_root / 'acc_hpc_pair_subject_audit.csv'
        zscore_figure_pdf = figure_root / 'acc_hpc_pair_mi_overlap_zscore_distribution.pdf'
        zscore_figure_png = figure_root / 'acc_hpc_pair_mi_overlap_zscore_distribution.png'

        print('analysis_root:', analysis_root)
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', RPE_MEMORY_OVERLAP_WINDOW_ACC)
        """
    ),
    code(
        """
        def extract_cluster_time_range(
            stats_path: str,
            cue_label: str = 'Cue time range',
            time_label: str = 'Time range',
        ):
            text = Path(stats_path).read_text()
            for label in [cue_label, time_label]:
                match = re.search(rf'{label}:\\s*([0-9.]+)\\s*[–-]\\s*([0-9.]+)\\s*s', text)
                if match:
                    return (float(match.group(1)), float(match.group(2))), text
            raise ValueError(f'Could not parse cluster time range from {stats_path}')


        rpe_cluster_window, _ = extract_cluster_time_range(
            acc_reinstatement_stats_txt,
            cue_label='Cue time range',
        )
        hit_cluster_window, _ = extract_cluster_time_range(
            acc_hit_decoding_stats_txt,
            cue_label='Cue time range',
            time_label='Time range',
        )

        overlap_window = (
            max(rpe_cluster_window[0], hit_cluster_window[0]),
            min(rpe_cluster_window[1], hit_cluster_window[1]),
        )

        if not np.allclose(overlap_window, RPE_MEMORY_OVERLAP_WINDOW_ACC):
            raise ValueError(
                f'Parsed overlap {overlap_window} does not match expected '
                f'{RPE_MEMORY_OVERLAP_WINDOW_ACC}'
            )

        acc_reinstatement_tbl = pd.read_csv(acc_reinstatement_csv)
        acc_hit_tbl = pd.read_csv(acc_hit_decoding_csv)
        shared_subjects = sorted(
            set(acc_reinstatement_tbl['Subject']).intersection(acc_hit_tbl['Subject'])
        )

        overlap_df = pd.DataFrame(
            [
                {
                    'metric': 'ACC RPE reinstatement',
                    'window_start_s': rpe_cluster_window[0],
                    'window_end_s': rpe_cluster_window[1],
                },
                {
                    'metric': 'ACC hit decoding',
                    'window_start_s': hit_cluster_window[0],
                    'window_end_s': hit_cluster_window[1],
                },
                {
                    'metric': 'ACC overlap',
                    'window_start_s': overlap_window[0],
                    'window_end_s': overlap_window[1],
                },
            ]
        )

        print('Shared ACC subjects:', len(shared_subjects))
        print('ACC RPE reinstatement cue cluster:', rpe_cluster_window)
        print('ACC hit decoding cue cluster:', hit_cluster_window)
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', overlap_window)
        display(overlap_df)
        """
    ),
    md("## Load and preprocess behavioral data"),
    code(
        """
        learn_df_raw, combined_df_raw = load_behavior_tables(str(load_dir))
        learn_df = preprocess_learn_df(learn_df_raw)
        combined_df = preprocess_combined_df(combined_df_raw)

        learn_df = learn_df[learn_df['participant'].isin(shared_subjects)].copy()
        combined_df = combined_df[
            combined_df['participant'].isin(shared_subjects) &
            (combined_df['condition'] == CONDITION)
        ].copy()

        required_combined_cols = [
            'participant',
            'condition',
            'trials_mem',
            'memory_outcome',
            'phit',
            'zrpe',
        ]
        missing_combined_cols = [
            c for c in required_combined_cols if c not in combined_df.columns
        ]
        if missing_combined_cols:
            raise ValueError(
                f'preprocess_combined_df did not provide required columns: {missing_combined_cols}'
            )

        print('Behavior rows after shared-subject Day1 filtering:', len(combined_df))
        display(
            combined_df[
                ['participant', 'condition', 'trials_mem', 'memory_outcome', 'phit', 'zrpe']
            ].head()
        )
        """
    ),
    md("## Helper functions for ACC-HPC pairwise local MI"),
    code(
        """
        def _get_smoothed_event_signal(
            subject_obj: Subject,
            roi: str,
            event: str,
            freq_range: Tuple[float, float],
            downsample_factor: int,
            win_ms: int,
            smooth_method: str = 'movmean',
            artifact_abs_thresh: float = 10.0,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            neural = subject_obj._get_or_build_event_neural_data(
                event=event,
                roi=roi,
                freq_range=freq_range,
                downsample_factor=downsample_factor,
                win_ms=win_ms,
                smooth_method=smooth_method,
                artifact_abs_thresh=artifact_abs_thresh,
            )
            signal = np.asarray(neural['hfa'], dtype=np.float64)
            times = np.asarray(neural['times'], dtype=np.float64)
            labels = np.asarray(neural.get('electrode_labels', []), dtype=str)
            hemis = np.asarray(neural.get('electrode_hemi', []), dtype=str)
            if labels.size == 0:
                raise ValueError(f'Missing electrode_labels for {subject_obj.subject} {event}/{roi}')
            if hemis.size == 0:
                hemis = np.repeat('unknown', labels.size)
            return signal, times, labels, hemis


        def _resolve_trial_var_for_event(df: pd.DataFrame, event: str) -> str:
            if str(event).strip().lower() == 'cue_on':
                for candidate in ('trials_mem', 'trials_gamble', 'trials'):
                    if candidate in df.columns:
                        return candidate
            for candidate in ('trials', 'trials_gamble', 'trials_mem'):
                if candidate in df.columns:
                    return candidate
            raise ValueError(f'Could not determine trial variable for event={event!r}')


        def _build_cue_trial_axis(subject_obj: Subject, n_epochs: int, event: str = 'cue_on') -> Dict[str, Any]:
            cdf = subject_obj.combined_df[
                (subject_obj.combined_df.participant == subject_obj.subject) &
                (subject_obj.combined_df.condition == CONDITION)
            ].copy()
            if len(cdf) == 0:
                raise ValueError('no_day1_trials')

            trial_col = _resolve_trial_var_for_event(cdf, event)
            cdf = cdf.dropna(subset=[trial_col]).copy()
            trial_vals = pd.to_numeric(cdf[trial_col], errors='coerce').to_numpy(dtype=np.float64)
            in_range = np.isfinite(trial_vals) & ((trial_vals - 1) >= 0) & ((trial_vals - 1) < int(n_epochs))
            if int(np.sum(in_range)) == 0:
                raise ValueError('no_in_range_trials')

            trial_ix = trial_vals[in_range].astype(np.int64) - 1
            return {
                'trial_index': trial_ix.astype(np.int64, copy=False),
                'trial_number': trial_vals[in_range].astype(np.int64, copy=False),
                'trial_col': trial_col,
                'n_dropped': int((~in_range).sum()),
            }


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
                return {'ok': False, 'reason': 'bad_input'}

            n = int(X.shape[0])
            px = int(X.shape[1])
            py = int(Y.shape[1])
            if n < 4 or px < 1 or py < 1:
                return {'ok': False, 'reason': 'insufficient_shape'}

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
                return {'ok': False, 'reason': 'no_valid_cc'}

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
                'ok': True,
                'rho': rho,
                'Uscore': Uscore,
                'Vscore': Vscore,
                'global_mi': global_mi,
                'n_cc': int(rho.size),
            }


        def _local_mi_from_canonical(Uscore: np.ndarray, Vscore: np.ndarray, rho: np.ndarray) -> np.ndarray:
            Uscore = np.asarray(Uscore, dtype=np.float64)
            Vscore = np.asarray(Vscore, dtype=np.float64)
            rho = np.asarray(rho, dtype=np.float64)
            one_minus = np.clip(1.0 - rho ** 2, 1e-12, None)

            const = -0.5 * np.log(one_minus)
            quad = -(
                ((rho ** 2) * (Uscore ** 2 + Vscore ** 2) - (2.0 * rho * Uscore * Vscore)) /
                (2.0 * one_minus)
            )
            local_per_cc = const.reshape(1, -1) + quad
            return np.sum(local_per_cc, axis=1)


        def _normalize_memory_outcome(value: Any) -> str:
            value = str(value).strip().lower()
            mapping = {
                'hit': 'hit',
                'hits': 'hit',
                'miss': 'miss',
                'misses': 'miss',
            }
            return mapping.get(value, value)


        def _pair_hemisphere_label(acc_hemi: str, hpc_hemi: str) -> str:
            return f'{str(acc_hemi).lower()}_acc__{str(hpc_hemi).lower()}_hpc'
        """
    ),
    md("## Compute trial-level local MI for each ACC-HPC pair"),
    code(
        """
        def compute_subject_pair_mi_rows(subject_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            subj = Subject(subject_id, str(base_dir), str(load_dir), min_trials=10, verbose=False)
            subj._learn_df = learn_df[learn_df['participant'] == subject_id].copy()
            subj._combined_df = combined_df[combined_df['participant'] == subject_id].copy()

            acc_hfa, cue_times_acc, acc_labels, acc_hemis = _get_smoothed_event_signal(
                subject_obj=subj,
                roi=ACC_ROI,
                event=EVENT,
                freq_range=FREQ_RANGE,
                downsample_factor=DOWNSAMPLE_FACTOR,
                win_ms=WIN_MS,
                smooth_method=HFA_SMOOTH_METHOD,
                artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
            )
            hpc_hfa, cue_times_hpc, hpc_labels, hpc_hemis = _get_smoothed_event_signal(
                subject_obj=subj,
                roi=HPC_ROI,
                event=EVENT,
                freq_range=FREQ_RANGE,
                downsample_factor=DOWNSAMPLE_FACTOR,
                win_ms=WIN_MS,
                smooth_method=HFA_SMOOTH_METHOD,
                artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
            )

            if not np.allclose(np.asarray(cue_times_acc), np.asarray(cue_times_hpc), atol=1e-6, rtol=0):
                raise ValueError(f'ACC/HPC cue time vectors mismatch for {subject_id}')

            cue_times = np.asarray(cue_times_acc, dtype=np.float64)
            time_mask = (
                (cue_times > RPE_MEMORY_OVERLAP_WINDOW_ACC[0]) &
                (cue_times <= RPE_MEMORY_OVERLAP_WINDOW_ACC[1])
            )
            if not np.any(time_mask):
                raise ValueError(f'No cue samples in overlap window for {subject_id}')

            n_epochs = int(min(acc_hfa.shape[0], hpc_hfa.shape[0]))
            axis_pack = _build_cue_trial_axis(subject_obj=subj, n_epochs=n_epochs, event=EVENT)
            trial_index = np.asarray(axis_pack['trial_index'], dtype=np.int64)
            trial_number = np.asarray(axis_pack['trial_number'], dtype=np.int64)

            acc = np.asarray(acc_hfa[trial_index, :, :], dtype=np.float64)
            hpc = np.asarray(hpc_hfa[trial_index, :, :], dtype=np.float64)

            acc_window = np.asarray(acc[:, :, time_mask], dtype=np.float64)
            hpc_window = np.asarray(hpc[:, :, time_mask], dtype=np.float64)
            n_window_timepoints = int(np.sum(time_mask))

            behavior_df = (
                combined_df[
                    (combined_df['participant'] == subject_id) &
                    (combined_df['condition'] == CONDITION)
                ][['participant', 'trials_mem', 'memory_outcome', 'phit', 'zrpe']]
                .dropna(subset=['trials_mem'])
                .copy()
            )
            behavior_df['trial_number_epoch1'] = pd.to_numeric(
                behavior_df['trials_mem'],
                errors='coerce',
            ).astype('Int64')
            behavior_df['memory_outcome'] = behavior_df['memory_outcome'].map(_normalize_memory_outcome)
            behavior_df = behavior_df.dropna(subset=['trial_number_epoch1', 'zrpe', 'memory_outcome']).copy()
            behavior_df['trial_number_epoch1'] = behavior_df['trial_number_epoch1'].astype(int)
            behavior_df = behavior_df[
                behavior_df['memory_outcome'].isin(['hit', 'miss'])
            ].copy()

            if len(behavior_df) == 0:
                return pd.DataFrame(), {
                    'participant': subject_id,
                    'n_acc_electrodes': int(acc_window.shape[1]),
                    'n_hpc_electrodes': int(hpc_window.shape[1]),
                    'n_pairs': 0,
                    'n_trial_rows': 0,
                    'n_window_timepoints': n_window_timepoints,
                    'status': 'no_hit_miss_trials',
                }

            pair_rows: List[Dict[str, Any]] = []
            for acc_idx, hpc_idx in product(range(acc_window.shape[1]), range(hpc_window.shape[1])):
                acc_label = str(acc_labels[acc_idx])
                hpc_label = str(hpc_labels[hpc_idx])
                acc_hemi = str(acc_hemis[acc_idx]) if acc_idx < len(acc_hemis) else 'unknown'
                hpc_hemi = str(hpc_hemis[hpc_idx]) if hpc_idx < len(hpc_hemis) else 'unknown'
                pair_id = f'{subject_id}::{acc_label}__{hpc_label}'
                hemisphere_label = _pair_hemisphere_label(acc_hemi, hpc_hemi)

                pair_df = pd.DataFrame(
                    {
                        'participant': subject_id,
                        'pair_id': pair_id,
                        'acc_channel': acc_label,
                        'hpc_channel': hpc_label,
                        'acc_hemi': acc_hemi,
                        'hpc_hemi': hpc_hemi,
                        'hemisphere_label': hemisphere_label,
                        'trial_index_epoch0': trial_index.astype(int),
                        'trial_number_epoch1': trial_number.astype(int),
                        'acc_hfa_mean': np.nanmean(acc_window[:, acc_idx, :], axis=1),
                        'hpc_hfa_mean': np.nanmean(hpc_window[:, hpc_idx, :], axis=1),
                    }
                )
                pair_df = pair_df.merge(
                    behavior_df[
                        ['trial_number_epoch1', 'memory_outcome', 'phit', 'zrpe']
                    ].drop_duplicates(subset=['trial_number_epoch1']),
                    on='trial_number_epoch1',
                    how='inner',
                )
                pair_df = pair_df.replace([np.inf, -np.inf], np.nan)
                pair_df = pair_df.dropna(subset=['acc_hfa_mean', 'hpc_hfa_mean', 'zrpe', 'phit'])
                if len(pair_df) < MIN_TRIALS:
                    continue

                Xraw = np.asarray(acc_window[:, acc_idx, :], dtype=np.float64)
                Yraw = np.asarray(hpc_window[:, hpc_idx, :], dtype=np.float64)
                valid_trial_mask = np.isfinite(Xraw).all(axis=1) & np.isfinite(Yraw).all(axis=1)
                valid_trial_numbers = set(pair_df['trial_number_epoch1'].tolist())
                valid_pair_mask = np.array(
                    [int(tn) in valid_trial_numbers for tn in trial_number.astype(int)],
                    dtype=bool,
                ) & valid_trial_mask
                if int(np.sum(valid_pair_mask)) < MIN_TRIALS:
                    continue

                ordered_trial_numbers = trial_number.astype(int)[valid_pair_mask]
                Xpair = Xraw[valid_pair_mask, :]
                Ypair = Yraw[valid_pair_mask, :]
                pair_df = pair_df[
                    pair_df['trial_number_epoch1'].isin(ordered_trial_numbers)
                ].copy()
                pair_df['trial_number_epoch1'] = pd.Categorical(
                    pair_df['trial_number_epoch1'],
                    categories=ordered_trial_numbers.tolist(),
                    ordered=True,
                )
                pair_df = pair_df.sort_values('trial_number_epoch1').reset_index(drop=True)
                pair_df['trial_number_epoch1'] = pair_df['trial_number_epoch1'].astype(int)

                Xz, _ = _zscore_cols(Xpair)
                Yz, _ = _zscore_cols(Ypair)
                if Xz.shape[1] < 1 or Yz.shape[1] < 1 or Xz.shape[0] != Yz.shape[0]:
                    continue

                Xr = _pca_reduce(Xz, var_keep=PCA_VAR_KEEP, max_components=PCA_MAX_COMPONENTS)
                Yr = _pca_reduce(Yz, var_keep=PCA_VAR_KEEP, max_components=PCA_MAX_COMPONENTS)
                model = _fit_gaussian_cca_local_model(Xr, Yr, reg_eps=REG_EPS)
                if not model.get('ok', False):
                    continue

                mi_local = _local_mi_from_canonical(
                    model['Uscore'],
                    model['Vscore'],
                    model['rho'],
                )
                if mi_local.shape[0] != len(pair_df):
                    continue

                pair_df = pair_df.reset_index(drop=True).copy()
                pair_df['mi_local'] = mi_local
                pair_df['n_cc'] = int(model['n_cc'])
                pair_df['global_mi'] = float(model['global_mi'])
                pair_df['n_window_timepoints'] = n_window_timepoints
                pair_rows.extend(pair_df.to_dict(orient='records'))

            pair_df_out = pd.DataFrame(pair_rows)
            audit_row = {
                'participant': subject_id,
                'n_acc_electrodes': int(acc_window.shape[1]),
                'n_hpc_electrodes': int(hpc_window.shape[1]),
                'n_pairs': int(pair_df_out['pair_id'].nunique()) if len(pair_df_out) else 0,
                'n_trial_rows': int(len(pair_df_out)),
                'n_window_timepoints': n_window_timepoints,
                'status': 'ok' if len(pair_df_out) else 'no_valid_pairs',
            }
            return pair_df_out, audit_row


        mi_frames = []
        audit_rows = []
        for subject_id in tqdm(shared_subjects, desc='ACC-HPC MI subjects'):
            try:
                subject_df, audit_row = compute_subject_pair_mi_rows(subject_id)
            except Exception as exc:
                subject_df = pd.DataFrame()
                audit_row = {
                    'participant': subject_id,
                    'n_acc_electrodes': np.nan,
                    'n_hpc_electrodes': np.nan,
                    'n_pairs': 0,
                    'n_trial_rows': 0,
                    'status': f'error: {exc}',
                }
            mi_frames.append(subject_df)
            audit_rows.append(audit_row)

        pair_mi_df = pd.concat(mi_frames, ignore_index=True) if mi_frames else pd.DataFrame()
        pair_audit_df = pd.DataFrame(audit_rows).sort_values('participant').reset_index(drop=True)

        if len(pair_mi_df) == 0:
            raise ValueError('No pair-level MI rows were generated.')

        required_pair_cols = [
            'participant',
            'pair_id',
            'acc_channel',
            'hpc_channel',
            'hemisphere_label',
            'trial_number_epoch1',
            'mi_local',
            'zrpe',
            'memory_outcome',
            'phit',
        ]
        missing_pair_cols = [c for c in required_pair_cols if c not in pair_mi_df.columns]
        if missing_pair_cols:
            raise ValueError(f'Missing required pair-level columns: {missing_pair_cols}')

        pair_mi_df = pair_mi_df.sort_values(
            ['participant', 'pair_id', 'trial_number_epoch1']
        ).reset_index(drop=True)
        pair_audit_df.to_csv(pair_audit_csv, index=False)

        print('Pair-level MI trial rows:', len(pair_mi_df))
        print('Unique ACC-HPC pairs:', pair_mi_df['pair_id'].nunique())
        display(pair_audit_df)
        display(
            pair_mi_df[
                [
                    'participant',
                    'pair_id',
                    'acc_channel',
                    'hpc_channel',
                    'trial_number_epoch1',
                    'mi_local',
                    'zrpe',
                    'memory_outcome',
                    'phit',
                ]
            ].head()
        )
        """
    ),
    md("## Cache trial-level pair MI"),
    code(
        """
        pair_mi_df.to_csv(mi_trial_cache_csv, index=False)
        print('Saved mi_trial_cache_csv:', mi_trial_cache_csv)
        """
    ),
    md("## Pair-level permutation regression"),
    code(
        """
        def permutation_interaction_regression(
            data: pd.DataFrame,
            formula: str = 'mi_local ~ 1 + zrpe + phit + zrpe:phit',
            interaction_term: str = 'zrpe:phit',
            n_permutations: int = 1000,
            random_state: int = 42,
        ):
            y, X = patsy.dmatrices(formula, data, return_type='dataframe')
            y_values = y.values.ravel()
            model = OLS(y, X).fit()

            if interaction_term not in X.columns:
                raise ValueError(f'Interaction term {interaction_term} not found in design matrix.')

            actual_beta = float(model.params[interaction_term])
            rng = np.random.default_rng(random_state)
            null_betas = np.empty(n_permutations, dtype=float)

            for perm_idx in range(n_permutations):
                y_perm = rng.permutation(y_values)
                perm_model = OLS(y_perm, X).fit()
                null_betas[perm_idx] = float(perm_model.params[interaction_term])

            null_mean = float(np.mean(null_betas))
            null_std = float(np.std(null_betas, ddof=1))
            z_beta = float((actual_beta - null_mean) / null_std) if null_std > 0 else np.nan
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_beta)))) if np.isfinite(z_beta) else np.nan

            return {
                'actual_beta': actual_beta,
                'null_mean': null_mean,
                'null_std': null_std,
                'z_beta': z_beta,
                'p_value': p_value,
                'n_trials': int(len(data)),
                'raw_model_p': float(model.pvalues.get(interaction_term, np.nan)),
            }


        regression_rows = []
        sanity_result = None
        for pair_id, pair_df in tqdm(
            pair_mi_df.groupby('pair_id'),
            total=pair_mi_df['pair_id'].nunique(),
            desc='ACC-HPC pairs',
        ):
            pair_df = pair_df.copy()
            pair_df = pair_df.replace([np.inf, -np.inf], np.nan)
            pair_df = pair_df.dropna(subset=['mi_local', 'zrpe', 'phit'])

            if len(pair_df) < MIN_TRIALS:
                continue
            if pair_df['phit'].nunique() < 2:
                continue
            if pair_df['zrpe'].nunique() < 2:
                continue

            result = permutation_interaction_regression(
                pair_df,
                n_permutations=N_PERMUTATIONS,
                random_state=RANDOM_STATE,
            )

            row = {
                'participant': str(pair_df['participant'].iloc[0]),
                'pair_id': str(pair_id),
                'acc_channel': str(pair_df['acc_channel'].iloc[0]),
                'hpc_channel': str(pair_df['hpc_channel'].iloc[0]),
                'hemisphere_label': str(pair_df['hemisphere_label'].iloc[0]),
                'raw_weight': result['actual_beta'],
                'z_scored_weight': result['z_beta'],
                'p_value': result['p_value'],
                'raw_model_p': result['raw_model_p'],
                'n_trials': result['n_trials'],
                'null_mean': result['null_mean'],
                'null_std': result['null_std'],
            }
            regression_rows.append(row)

            if sanity_result is None:
                sanity_result = row.copy()

        pair_results_df = pd.DataFrame(regression_rows).sort_values(
            ['participant', 'pair_id']
        ).reset_index(drop=True)
        if len(pair_results_df) == 0:
            raise ValueError('No pair-level regression results were produced.')

        display(pair_results_df.head())
        print('Analyzed ACC-HPC pairs:', len(pair_results_df))
        """
    ),
    md("## Sanity check"),
    code(
        """
        if sanity_result is None:
            raise ValueError('Sanity result was not captured.')

        sanity_df = pd.DataFrame([sanity_result])[
            [
                'participant',
                'pair_id',
                'acc_channel',
                'hpc_channel',
                'n_trials',
                'raw_weight',
                'null_mean',
                'null_std',
                'z_scored_weight',
                'p_value',
            ]
        ]
        print('Sanity-check pair regression:')
        display(sanity_df)
        """
    ),
    md("## Save results and plot pair-level z-scored interaction weights"),
    code(
        """
        if pair_results_df.duplicated(['participant', 'pair_id']).any():
            raise ValueError('Duplicate (participant, pair_id) rows found in pair_results_df.')

        pair_results_df.to_csv(pair_results_csv, index=False)
        print('Saved pair_results_csv:', pair_results_csv)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        sns.histplot(
            data=pair_results_df,
            x='z_scored_weight',
            bins=18,
            color='black',
            alpha=0.65,
            edgecolor='white',
            ax=ax,
        )
        ax.axvline(0, color='black', linestyle='-', linewidth=1.0)
        ax.axvline(1.96, color='red', linestyle='--', linewidth=1.0)
        ax.axvline(-1.96, color='red', linestyle='--', linewidth=1.0)
        ax.set(
            title='ACC-HPC pair distribution: z-scored zrpe:memory_outcome MI weights',
            xlabel='Interaction weight z-score (zrpe:phit)',
            ylabel='Pair count',
        )
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(zscore_figure_pdf, bbox_inches='tight')
        fig.savefig(zscore_figure_png, bbox_inches='tight')
        display(fig)
        plt.close(fig)

        print('Saved zscore_figure_pdf:', zscore_figure_pdf)
        print('Saved zscore_figure_png:', zscore_figure_png)
        """
    ),
    md("## Final checks"),
    code(
        """
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', RPE_MEMORY_OVERLAP_WINDOW_ACC)
        print('Pair-level MI rows:', len(pair_mi_df))
        print('Unique participants:', pair_results_df['participant'].nunique())
        print('Unique ACC-HPC pairs:', pair_results_df['pair_id'].nunique())
        print('Any duplicate participant/pair rows:', pair_results_df.duplicated(['participant', 'pair_id']).any())
        print('Cross-hemisphere labels found:', sorted(pair_mi_df['hemisphere_label'].dropna().unique().tolist())[:12])
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "lfpanalysis",
        "language": "python",
        "name": "lfpanalysis",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote {NOTEBOOK_PATH}")
