import nbformat as nbf
from textwrap import dedent
from pathlib import Path


OUT_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-ACC_HPC_MixedReinstatement_ShuffleNull.ipynb"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.x"},
}

cells = []

cells.append(
    md(
        """
        # ACC Encoding to HPC Retrieval Mixed Reinstatement With Within-Subject Shuffle Nulls

        This notebook implements a trialwise ACC-encoding to HPC-retrieval reinstatement
        analysis with mixed-effects regression and a within-subject surrogate null.

        Primary question:

        Does trialwise reinstatement predict later memory, and is that relationship modulated by RPE?

        Main outputs:

        - empirical mixed-model beta maps for `memory`, `rpe`, and `memory:rpe`
        - within-subject shuffled null mean and SD maps
        - z-scored beta maps relative to the surrogate null
        - permutation p-value maps and FDR-corrected masks

        Secondary follow-up:

        - pairwise cross-phase similarity mixed model on flattened `S[i, j]`
        """
    )
)

cells.append(
    code(
        """
        %matplotlib inline
        %reload_ext autoreload
        %autoreload 2

        import os
        import json
        import warnings
        from dataclasses import dataclass
        from typing import Any, Dict, List, Optional, Tuple

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from tqdm.auto import tqdm
        from joblib import Parallel, delayed
        from IPython.display import display
        from matplotlib.backends.backend_pdf import PdfPages
        from statsmodels.formula.api import mixedlm
        from statsmodels.stats.multitest import multipletests

        from plot_config import set_plot_style
        from mv_regression_utils import (
            Subject,
            load_subject_crossregion_hfa,
            _padded_row_zscore_cosine_similarity,
            _canonicalize_rpe_cat_label,
        )

        warnings.filterwarnings("ignore")
        set_plot_style()
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Paths + subject discovery
        # =============================
        base_dir = '/Volumes/T7 Shield'
        load_dir = f'{base_dir}/work/qasims01/MemoryBanditData/EMU'

        demo_file = f'{base_dir}/projects/guLab/Salman/EMU/MemoryBanditsDatabase.xlsx'
        demo_data = pd.read_excel(demo_file, engine='openpyxl')
        subj_df = demo_data[(demo_data.LongVersion == 1) & (demo_data.Minerva == 1)]
        subj_df = subj_df[~subj_df['MSSMCode'].isin(['MS015', 'MS024', 'MS034', 'MS038'])]
        subjects = subj_df.MSSMCode.astype(str).tolist()

        print('n subjects:', len(subjects))
        print(subjects)
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Configuration
        # =============================
        ENCODING_ROI = 'ACC'
        RETRIEVAL_ROI = 'HPC'

        EVENT_ENCODING = 'feedback_start'
        EVENT_CUE = 'cue_on'
        FREQ_RANGE = (70.0, 201.0)
        WIN_MS = 200
        DOWNSAMPLE_FACTOR = 5
        MIN_TRIALS = 10

        N_SURROGATES = 500
        SEED = 42

        RUN_PRIMARY_TRIALWISE = True
        RUN_PRIMARY_SURROGATES = True
        RUN_SECONDARY_PAIRWISE = False
        RUN_SECONDARY_SURROGATES = False

        TIMEPOINT_STRIDE = 1
        MAX_SUBJECTS = None
        USE_CONTINUOUS_RPE = True
        N_JOBS_SURROGATES = min(8, os.cpu_count() or 1)
        SURROGATE_CHUNK_SIZE = 1
        PARALLEL_BACKEND = 'loky'
        UNCORRECTED_P_ALPHA = 0.05
        FDR_ALPHA = 0.05
        SIGNIFICANCE_MASK_MODE = 'uncorrected'

        # MixedLM can be slow over the full time x time grid.
        # For fast smoke tests, lower N_SURROGATES, increase TIMEPOINT_STRIDE,
        # or set MAX_SUBJECTS to a small integer.
        MIXEDLM_METHOD = 'lbfgs'
        ALLOW_OLS_FALLBACK = False

        OUT_DIR = os.path.join('analysis_outputs', 'acc_hpc_mixed_reinstatement_null')
        os.makedirs(OUT_DIR, exist_ok=True)

        if MAX_SUBJECTS is not None:
            subjects = subjects[:int(MAX_SUBJECTS)]

        CONFIG = {
            'encoding_roi': ENCODING_ROI,
            'retrieval_roi': RETRIEVAL_ROI,
            'event_encoding': EVENT_ENCODING,
            'event_cue': EVENT_CUE,
            'freq_range': FREQ_RANGE,
            'win_ms': WIN_MS,
            'downsample_factor': DOWNSAMPLE_FACTOR,
            'min_trials': MIN_TRIALS,
            'n_surrogates': N_SURROGATES,
            'seed': SEED,
            'run_primary_trialwise': RUN_PRIMARY_TRIALWISE,
            'run_primary_surrogates': RUN_PRIMARY_SURROGATES,
            'run_secondary_pairwise': RUN_SECONDARY_PAIRWISE,
            'run_secondary_surrogates': RUN_SECONDARY_SURROGATES,
            'timepoint_stride': TIMEPOINT_STRIDE,
            'max_subjects': MAX_SUBJECTS,
            'use_continuous_rpe': USE_CONTINUOUS_RPE,
            'n_jobs_surrogates': N_JOBS_SURROGATES,
            'surrogate_chunk_size': SURROGATE_CHUNK_SIZE,
            'parallel_backend': PARALLEL_BACKEND,
            'uncorrected_p_alpha': UNCORRECTED_P_ALPHA,
            'fdr_alpha': FDR_ALPHA,
            'significance_mask_mode': SIGNIFICANCE_MASK_MODE,
            'mixedlm_method': MIXEDLM_METHOD,
            'allow_ols_fallback': ALLOW_OLS_FALLBACK,
        }

        with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
            json.dump(CONFIG, f, indent=2)

        CONFIG
        """
    )
)

cells.append(
    md(
        """
        ## Helpers

        The notebook keeps helpers local on purpose so the workflow is easy to iterate on
        before promoting anything into `mv_regression_utils.py`.
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Notebook-local helpers
        # =============================
        @dataclass
        class SubjectReinstatementData:
            subject: str
            encoding_hfa: np.ndarray
            cue_hfa: np.ndarray
            feedback_times: np.ndarray
            cue_times: np.ndarray
            memory_numeric: np.ndarray
            memory_label: np.ndarray
            rpe_value: np.ndarray
            rpe_category_code: np.ndarray
            n_trials: int
            n_feedback_times: int
            n_cue_times: int


        def aligned_trial_dataframe(subject_obj: Subject) -> pd.DataFrame:
            subj_combined_df = subject_obj.combined_df[
                (subject_obj.combined_df.condition == 'Day1') &
                (subject_obj.combined_df.participant == subject_obj.subject)
            ].copy()
            if len(subj_combined_df) == 0:
                raise ValueError('no_day1_trials')
            needed = ['trials_gamble', 'trials_mem', 'hits']
            missing = [c for c in needed if c not in subj_combined_df.columns]
            if missing:
                raise ValueError(f'missing_trial_columns:{missing}')
            subj_combined_df = subj_combined_df.dropna(subset=['trials_gamble', 'trials_mem']).copy()
            if len(subj_combined_df) == 0:
                raise ValueError('no_day1_trials_after_dropna')
            return subj_combined_df.reset_index(drop=True)


        def _infer_rpe_value_and_category(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            rpe_value = pd.to_numeric(df['rpe'], errors='coerce').to_numpy(dtype=np.float64) if 'rpe' in df.columns else np.full(len(df), np.nan)
            if 'rpe_cat' in df.columns:
                rpe_cat_raw = df['rpe_cat'].to_numpy(dtype=object)
            else:
                rpe_cat_raw = np.full(len(df), np.nan, dtype=object)

            rpe_cat_code = np.full(len(df), np.nan, dtype=np.float64)
            for i, lab in enumerate(rpe_cat_raw):
                key = _canonicalize_rpe_cat_label(lab)
                if key == 'neg':
                    rpe_cat_code[i] = -1.0
                elif key == 'neutral':
                    rpe_cat_code[i] = 0.0
                elif key == 'pos':
                    rpe_cat_code[i] = 1.0
            return rpe_value, rpe_cat_code


        def load_subject_reinstatement_inputs(
            subject: str,
            base_dir: str,
            load_dir: str,
            encoding_roi: str = ENCODING_ROI,
            retrieval_roi: str = RETRIEVAL_ROI,
            min_trials: int = MIN_TRIALS,
        ) -> SubjectReinstatementData:
            subject_obj = Subject(subject, base_dir, load_dir, min_trials=min_trials, verbose=False)
            data = load_subject_crossregion_hfa(
                subject_obj=subject_obj,
                encoding_roi=encoding_roi,
                cue_roi=retrieval_roi,
                event_encoding=EVENT_ENCODING,
                event_cue=EVENT_CUE,
                freq_range=FREQ_RANGE,
                downsample_factor=DOWNSAMPLE_FACTOR,
                win_ms=WIN_MS,
                min_trials=min_trials,
            )

            subj_combined_df = aligned_trial_dataframe(subject_obj)
            trial_ix_encoding = subj_combined_df['trials_gamble'].to_numpy(dtype=np.int64) - 1
            trial_ix_cue = subj_combined_df['trials_mem'].to_numpy(dtype=np.int64) - 1
            aligned_enc = np.asarray(data['trial_index'], dtype=np.int64)
            aligned_cue = np.asarray(data['retrieval_trial_index'], dtype=np.int64)
            aligned_pairs = {(int(e), int(c)) for e, c in zip(aligned_enc, aligned_cue)}
            keep_mask = np.array(
                [(int(e), int(c)) in aligned_pairs for e, c in zip(trial_ix_encoding, trial_ix_cue)],
                dtype=bool,
            )
            combined_sel = subj_combined_df.loc[keep_mask].reset_index(drop=True)
            enc_hfa_all = np.asarray(data['encoding_hfa'], dtype=np.float32)
            cue_hfa_all = np.asarray(data['cue_hfa'], dtype=np.float32)
            if len(combined_sel) != enc_hfa_all.shape[0]:
                pair_df = pd.DataFrame({
                    'trial_index': trial_ix_encoding,
                    'retrieval_trial_index': trial_ix_cue,
                })
                pair_df = pair_df.loc[keep_mask].reset_index(drop=True)
                pair_df['row_idx'] = np.arange(len(pair_df), dtype=np.int64)
                target = pd.DataFrame({
                    'trial_index': aligned_enc,
                    'retrieval_trial_index': aligned_cue,
                    'order_idx': np.arange(len(aligned_enc), dtype=np.int64),
                })
                merged = target.merge(pair_df, on=['trial_index', 'retrieval_trial_index'], how='left', sort=False)
                if merged['row_idx'].isna().any():
                    raise ValueError(f'could_not_reconstruct_aligned_metadata:{subject}')
                combined_sel = combined_sel.iloc[merged['row_idx'].to_numpy(dtype=np.int64)].reset_index(drop=True)
            subj_sel = combined_sel.copy()

            hits_col = pd.to_numeric(subj_sel['hits'], errors='coerce').to_numpy(dtype=np.float64)
            if np.any(~np.isfinite(hits_col)):
                raise ValueError(f'invalid_hits_values:{subject}')
            memory_numeric = hits_col.astype(np.int64)
            memory_label = np.where(memory_numeric.astype(bool), 'remembered', 'forgotten')

            rpe_value, rpe_cat_code = _infer_rpe_value_and_category(subj_sel)
            if USE_CONTINUOUS_RPE and np.sum(np.isfinite(rpe_value)) == len(rpe_value):
                rpe_to_use = rpe_value.astype(np.float64)
            else:
                rpe_to_use = rpe_cat_code.astype(np.float64)

            valid_meta = np.isfinite(memory_numeric.astype(np.float64)) & np.isfinite(rpe_to_use)
            if np.sum(valid_meta) < min_trials:
                raise ValueError(f'insufficient_valid_trials_after_rpe_filter:{subject}:{int(np.sum(valid_meta))}')

            encoding_hfa = np.asarray(enc_hfa_all[valid_meta], dtype=np.float32)
            cue_hfa = np.asarray(cue_hfa_all[valid_meta], dtype=np.float32)
            memory_numeric = np.asarray(memory_numeric[valid_meta], dtype=np.float64)
            memory_label = np.asarray(memory_label[valid_meta], dtype=object)
            rpe_value = np.asarray(rpe_to_use[valid_meta], dtype=np.float64)
            rpe_category_code = np.asarray(rpe_cat_code[valid_meta], dtype=np.float64)

            if encoding_hfa.shape[0] != cue_hfa.shape[0]:
                raise ValueError(f'trial_count_mismatch:{subject}')

            return SubjectReinstatementData(
                subject=subject,
                encoding_hfa=encoding_hfa,
                cue_hfa=cue_hfa,
                feedback_times=np.asarray(data['feedback_times'], dtype=np.float32),
                cue_times=np.asarray(data['cue_times'], dtype=np.float32),
                memory_numeric=memory_numeric,
                memory_label=memory_label,
                rpe_value=rpe_value,
                rpe_category_code=rpe_category_code,
                n_trials=int(encoding_hfa.shape[0]),
                n_feedback_times=int(encoding_hfa.shape[2]),
                n_cue_times=int(cue_hfa.shape[2]),
            )


        def compute_trialwise_reinstatement_at_timepoint(E: np.ndarray, R: np.ndarray) -> Dict[str, np.ndarray]:
            S = _padded_row_zscore_cosine_similarity(E, R)
            diag = np.diag(S).astype(np.float64, copy=False)
            n_trials = S.shape[0]
            if n_trials > 1:
                mismatch = (S.sum(axis=1) - diag) / float(n_trials - 1)
            else:
                mismatch = np.zeros_like(diag)
            ri = diag - mismatch
            return {
                'similarity': S.astype(np.float32),
                'match': diag.astype(np.float32),
                'mismatch': mismatch.astype(np.float32),
                'ri': ri.astype(np.float32),
            }


        def compute_subject_trialwise_ri_tensor(subj_data: SubjectReinstatementData) -> Dict[str, np.ndarray]:
            n_trials = subj_data.n_trials
            n_fb = subj_data.n_feedback_times
            n_cue = subj_data.n_cue_times
            ri = np.full((n_trials, n_fb, n_cue), np.nan, dtype=np.float32)
            match = np.full((n_trials, n_fb, n_cue), np.nan, dtype=np.float32)
            mismatch = np.full((n_trials, n_fb, n_cue), np.nan, dtype=np.float32)
            for t_fb in range(n_fb):
                E = subj_data.encoding_hfa[:, :, t_fb]
                for t_cue in range(n_cue):
                    R = subj_data.cue_hfa[:, :, t_cue]
                    out = compute_trialwise_reinstatement_at_timepoint(E, R)
                    ri[:, t_fb, t_cue] = out['ri']
                    match[:, t_fb, t_cue] = out['match']
                    mismatch[:, t_fb, t_cue] = out['mismatch']
            return {'ri': ri, 'match': match, 'mismatch': mismatch}


        def zscore_within_subject(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float64)
            mu = np.nanmean(arr)
            sd = np.nanstd(arr)
            if (not np.isfinite(sd)) or sd < 1e-8:
                return np.full(arr.shape, 0.0, dtype=np.float64)
            return (arr - mu) / sd


        def build_trialwise_rows_for_subject(
            subj_data: SubjectReinstatementData,
            ri_tensor: np.ndarray,
            t_fb_idx: int,
            t_cue_idx: int,
            use_permuted_labels: Optional[Dict[str, np.ndarray]] = None,
        ) -> pd.DataFrame:
            if use_permuted_labels is None:
                mem = subj_data.memory_numeric
                rpe = subj_data.rpe_value
                rpe_cat = subj_data.rpe_category_code
            else:
                mem = np.asarray(use_permuted_labels['memory'], dtype=np.float64)
                rpe = np.asarray(use_permuted_labels['rpe'], dtype=np.float64)
                rpe_cat = np.asarray(use_permuted_labels['rpe_cat'], dtype=np.float64)

            df = pd.DataFrame({
                'subject': subj_data.subject,
                'trial_id': np.arange(subj_data.n_trials, dtype=np.int64),
                'memory': mem.astype(np.float64),
                'rpe': rpe.astype(np.float64),
                'rpe_category': rpe_cat.astype(np.float64),
                'ri': np.asarray(ri_tensor[:, t_fb_idx, t_cue_idx], dtype=np.float64),
            })
            df = df[np.isfinite(df['ri']) & np.isfinite(df['memory']) & np.isfinite(df['rpe'])].reset_index(drop=True)
            return df


        def fit_trialwise_mixed_model(
            df_timepoint: pd.DataFrame,
            formula: str = 'ri ~ memory * rpe',
            group_col: str = 'subject',
            method: str = MIXEDLM_METHOD,
            allow_ols_fallback: bool = ALLOW_OLS_FALLBACK,
        ) -> Dict[str, Any]:
            if len(df_timepoint) < 8 or df_timepoint[group_col].nunique() < 2:
                return {'status': 'insufficient_rows'}
            needed_terms = ['memory', 'rpe']
            for term in needed_terms:
                if np.nanstd(df_timepoint[term].to_numpy(dtype=np.float64)) < 1e-8:
                    return {'status': f'degenerate_{term}'}
            try:
                model = mixedlm(formula, df_timepoint, groups=df_timepoint[group_col])
                result = model.fit(method=method, reml=False, disp=False)
                params = result.params.to_dict()
                bse = result.bse.to_dict()
                tvalues = result.tvalues.to_dict() if hasattr(result, 'tvalues') else {}
                pvalues = result.pvalues.to_dict() if hasattr(result, 'pvalues') else {}
                return {
                    'status': 'ok',
                    'engine': 'mixedlm',
                    'params': params,
                    'bse': bse,
                    'tvalues': tvalues,
                    'pvalues': pvalues,
                    'converged': bool(getattr(result, 'converged', False)),
                    'n_subjects': int(df_timepoint[group_col].nunique()),
                    'n_rows': int(len(df_timepoint)),
                }
            except Exception as exc:
                if not allow_ols_fallback:
                    return {'status': f'fit_error:{type(exc).__name__}:{exc}'}
                import statsmodels.formula.api as smf
                try:
                    ols = smf.ols(formula + ' + C(subject)', data=df_timepoint).fit()
                    return {
                        'status': 'ok_ols_fallback',
                        'engine': 'ols_subject_fe',
                        'params': ols.params.to_dict(),
                        'bse': ols.bse.to_dict(),
                        'tvalues': ols.tvalues.to_dict(),
                        'pvalues': ols.pvalues.to_dict(),
                        'converged': True,
                        'n_subjects': int(df_timepoint[group_col].nunique()),
                        'n_rows': int(len(df_timepoint)),
                    }
                except Exception as exc2:
                    return {'status': f'fit_error:{type(exc2).__name__}:{exc2}'}


        def permute_trialwise_predictors_within_subject(subj_data: SubjectReinstatementData, rng: np.random.Generator) -> Dict[str, np.ndarray]:
            perm_ix = rng.permutation(subj_data.n_trials)
            return {
                'memory': np.asarray(subj_data.memory_numeric[perm_ix], dtype=np.float64),
                'rpe': np.asarray(subj_data.rpe_value[perm_ix], dtype=np.float64),
                'rpe_cat': np.asarray(subj_data.rpe_category_code[perm_ix], dtype=np.float64),
                'perm_ix': perm_ix.astype(np.int64),
            }


        def update_running_null_stats(
            running_mean: np.ndarray,
            running_M2: np.ndarray,
            x: np.ndarray,
            n_done: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            delta = x - running_mean
            running_mean = running_mean + (delta / float(n_done))
            running_M2 = running_M2 + delta * (x - running_mean)
            return running_mean, running_M2


        def fdr_correct_timegrid(p_map: np.ndarray, alpha: float = 0.05) -> np.ndarray:
            p_arr = np.asarray(p_map, dtype=np.float64)
            valid = np.isfinite(p_arr)
            out = np.zeros_like(p_arr, dtype=bool)
            if np.sum(valid) == 0:
                return out
            rej, _, _, _ = multipletests(p_arr[valid], alpha=alpha, method='fdr_bh')
            out[valid] = rej
            return out


        def uncorrected_p_mask(p_map: np.ndarray, alpha: float = 0.05) -> np.ndarray:
            p_arr = np.asarray(p_map, dtype=np.float64)
            return np.isfinite(p_arr) & (p_arr < float(alpha))


        def init_term_maps(shape: Tuple[int, int], fill_value=np.nan, dtype=np.float32) -> Dict[str, np.ndarray]:
            return {
                'memory': np.full(shape, fill_value, dtype=dtype),
                'rpe': np.full(shape, fill_value, dtype=dtype),
                'memory:rpe': np.full(shape, fill_value, dtype=dtype),
            }


        def _extract_term_dict(res: Dict[str, Any], field: str) -> Dict[str, float]:
            payload = res.get(field, {})
            return {
                'memory': float(payload.get('memory', np.nan)),
                'rpe': float(payload.get('rpe', np.nan)),
                'memory:rpe': float(payload.get('memory:rpe', np.nan)),
            }


        def chunk_sequence(seq, chunk_size: int):
            chunk_size = max(1, int(chunk_size))
            return [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]


        def reduce_surrogate_maps_into_null(
            surrogate_maps: Dict[str, np.ndarray],
            empirical_beta_maps: Dict[str, np.ndarray],
            null_mean: Dict[str, np.ndarray],
            null_M2: Dict[str, np.ndarray],
            null_counts: Dict[str, np.ndarray],
            extreme_counts: Dict[str, np.ndarray],
        ) -> None:
            for term in ('memory', 'rpe', 'memory:rpe'):
                x = np.asarray(surrogate_maps[term], dtype=np.float64)
                valid = np.isfinite(x)
                if np.any(valid):
                    mean_prev = np.asarray(null_mean[term], dtype=np.float64)
                    M2_prev = np.asarray(null_M2[term], dtype=np.float64)
                    counts_prev = np.asarray(null_counts[term], dtype=np.int32)
                    counts_new = counts_prev.astype(np.float64) + valid.astype(np.float64)
                    delta = np.where(valid, x - mean_prev, 0.0)
                    mean_new = np.where(valid, mean_prev + (delta / np.maximum(counts_new, 1.0)), mean_prev)
                    delta2 = np.where(valid, x - mean_new, 0.0)
                    M2_new = np.where(valid, M2_prev + (delta * delta2), M2_prev)
                    null_mean[term] = mean_new.astype(np.float32)
                    null_M2[term] = M2_new.astype(np.float32)
                    null_counts[term] = counts_new.astype(np.int32)

                emp = np.asarray(empirical_beta_maps[term], dtype=np.float64)
                extreme_counts[term] += (valid & np.isfinite(emp) & (np.abs(x) >= np.abs(emp))).astype(np.int32)


        def select_empirical_mask_bundle(empirical_trialwise, mode: str):
            mode_key = str(mode).strip().lower()
            shape = np.asarray(empirical_trialwise['beta_maps']['memory']).shape
            if mode_key == 'fdr':
                return empirical_trialwise['fdr_masks'], f'empirical-fdr<{FDR_ALPHA:g}'
            if mode_key == 'uncorrected':
                return empirical_trialwise['uncorrected_masks'], f'empirical-uncorrected-p<{UNCORRECTED_P_ALPHA:g}'
            if mode_key == 'none':
                empty = {
                    'memory': np.zeros(shape, dtype=bool),
                    'rpe': np.zeros(shape, dtype=bool),
                    'memory:rpe': np.zeros(shape, dtype=bool),
                }
                return empty, 'no-mask'
            raise ValueError(f'Unsupported SIGNIFICANCE_MASK_MODE: {mode}')


        def plot_metric_map(ax, arr, fb_t, cue_t, title, cmap='RdBu_r', center_zero=True, mask=None):
            A = np.asarray(arr, dtype=np.float64)
            vmax = np.nanmax(np.abs(A)) if center_zero else np.nanmax(A)
            if (not np.isfinite(vmax)) or vmax == 0:
                vmax = 1.0
            vmin = -vmax if center_zero else np.nanmin(A)
            im = ax.imshow(
                A,
                origin='lower',
                aspect='auto',
                extent=[float(cue_t.min()), float(cue_t.max()), float(fb_t.min()), float(fb_t.max())],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if mask is not None and np.any(mask):
                ax.contour(cue_t, fb_t, np.asarray(mask, dtype=int), levels=[0.5], colors='k', linewidths=1.2)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.4)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.4)
            ax.set_title(title)
            ax.set_xlabel('Cue time (s)')
            ax.set_ylabel('Feedback time (s)')
            return im
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Subject loading + QC
        # =============================
        subject_data_list = []
        load_errors = []
        for subj in tqdm(subjects, desc='Load subjects'):
            try:
                sd = load_subject_reinstatement_inputs(subj, base_dir=base_dir, load_dir=load_dir)
                subject_data_list.append(sd)
            except Exception as exc:
                load_errors.append((subj, repr(exc)))

        print('loaded subjects:', len(subject_data_list))
        print('load errors:', len(load_errors))
        if load_errors:
            display(pd.DataFrame(load_errors, columns=['subject', 'error']).head(20))

        if len(subject_data_list) == 0:
            raise RuntimeError('No subjects loaded successfully.')

        fb_t = subject_data_list[0].feedback_times[::TIMEPOINT_STRIDE]
        cue_t = subject_data_list[0].cue_times[::TIMEPOINT_STRIDE]
        n_fb = len(fb_t)
        n_cue = len(cue_t)
        print('grid shape:', (n_fb, n_cue))

        qc_rows = []
        for sd in subject_data_list:
            qc_rows.append({
                'subject': sd.subject,
                'n_trials': sd.n_trials,
                'n_hits': int(np.sum(sd.memory_numeric == 1)),
                'n_misses': int(np.sum(sd.memory_numeric == 0)),
                'rpe_mean': float(np.nanmean(sd.rpe_value)),
                'rpe_std': float(np.nanstd(sd.rpe_value)),
                'n_finite_rpe': int(np.sum(np.isfinite(sd.rpe_value))),
            })
        qc_df = pd.DataFrame(qc_rows).sort_values('subject').reset_index(drop=True)
        qc_df.to_csv(os.path.join(OUT_DIR, 'trialwise_counts.csv'), index=False)
        display(qc_df.head())
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Precompute trialwise RI tensors
        # =============================
        ri_cache = {}
        for sd in tqdm(subject_data_list, desc='Compute RI tensors'):
            ri_cache[sd.subject] = compute_subject_trialwise_ri_tensor(sd)

        ex = subject_data_list[0]
        ex_ri = ri_cache[ex.subject]['ri']
        print('example tensor shape:', ex_ri.shape)

        # Smoke-test trialwise RI algebra at one timepoint
        E0 = ex.encoding_hfa[:, :, 0]
        R0 = ex.cue_hfa[:, :, 0]
        chk = compute_trialwise_reinstatement_at_timepoint(E0, R0)
        print('match shape:', chk['match'].shape, 'mismatch shape:', chk['mismatch'].shape, 'ri shape:', chk['ri'].shape)
        print('allclose:', np.allclose(chk['ri'], chk['match'] - chk['mismatch']))
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Primary empirical mixed-model maps
        # =============================
        def run_empirical_trialwise_maps(subject_data_list, ri_cache, stride=1):
            fb_idxs = list(range(0, subject_data_list[0].n_feedback_times, stride))
            cue_idxs = list(range(0, subject_data_list[0].n_cue_times, stride))
            shape = (len(fb_idxs), len(cue_idxs))

            beta_maps = init_term_maps(shape)
            se_maps = init_term_maps(shape)
            t_maps = init_term_maps(shape)
            p_maps = init_term_maps(shape)
            n_rows_map = np.full(shape, np.nan, dtype=np.float32)
            n_subjects_map = np.full(shape, np.nan, dtype=np.float32)
            converged_map = np.zeros(shape, dtype=bool)
            status_rows = []

            for a, t_fb_idx in enumerate(tqdm(fb_idxs, desc='Empirical fb')):
                for b, t_cue_idx in enumerate(cue_idxs):
                    rows = []
                    for sd in subject_data_list:
                        df_sub = build_trialwise_rows_for_subject(sd, ri_cache[sd.subject]['ri'], t_fb_idx, t_cue_idx)
                        if len(df_sub) == 0:
                            continue
                        df_sub = df_sub.copy()
                        df_sub['rpe'] = zscore_within_subject(df_sub['rpe'].to_numpy(dtype=np.float64))
                        rows.append(df_sub)
                    if not rows:
                        status_rows.append({'t_fb_idx': t_fb_idx, 't_cue_idx': t_cue_idx, 'status': 'no_rows'})
                        continue
                    df_tp = pd.concat(rows, axis=0, ignore_index=True)
                    res = fit_trialwise_mixed_model(df_tp)
                    status_rows.append({
                        't_fb_idx': t_fb_idx,
                        't_cue_idx': t_cue_idx,
                        'status': res.get('status', 'unknown'),
                        'engine': res.get('engine', ''),
                        'converged': bool(res.get('converged', False)),
                        'n_rows': int(res.get('n_rows', 0)) if 'n_rows' in res else 0,
                        'n_subjects': int(res.get('n_subjects', 0)) if 'n_subjects' in res else 0,
                    })
                    if not str(res.get('status', '')).startswith('ok'):
                        continue
                    params = _extract_term_dict(res, 'params')
                    bse = _extract_term_dict(res, 'bse')
                    tvals = _extract_term_dict(res, 'tvalues')
                    pvals = _extract_term_dict(res, 'pvalues')
                    for term in ('memory', 'rpe', 'memory:rpe'):
                        beta_maps[term][a, b] = np.float32(params[term])
                        se_maps[term][a, b] = np.float32(bse[term])
                        t_maps[term][a, b] = np.float32(tvals[term])
                        p_maps[term][a, b] = np.float32(pvals[term])
                    n_rows_map[a, b] = np.float32(res.get('n_rows', np.nan))
                    n_subjects_map[a, b] = np.float32(res.get('n_subjects', np.nan))
                    converged_map[a, b] = bool(res.get('converged', False))

            status_df = pd.DataFrame(status_rows)
            fdr_masks = {term: fdr_correct_timegrid(p_maps[term], alpha=FDR_ALPHA) for term in ('memory', 'rpe', 'memory:rpe')}
            uncorrected_masks = {term: uncorrected_p_mask(p_maps[term], alpha=UNCORRECTED_P_ALPHA) for term in ('memory', 'rpe', 'memory:rpe')}
            return {
                'fb_idxs': np.asarray(fb_idxs, dtype=np.int64),
                'cue_idxs': np.asarray(cue_idxs, dtype=np.int64),
                'beta_maps': beta_maps,
                'se_maps': se_maps,
                't_maps': t_maps,
                'p_maps': p_maps,
                'fdr_masks': fdr_masks,
                'uncorrected_masks': uncorrected_masks,
                'n_rows_map': n_rows_map,
                'n_subjects_map': n_subjects_map,
                'converged_map': converged_map,
                'status_df': status_df,
            }


        empirical_trialwise = None
        if RUN_PRIMARY_TRIALWISE:
            empirical_trialwise = run_empirical_trialwise_maps(subject_data_list, ri_cache, stride=TIMEPOINT_STRIDE)
            empirical_trialwise['status_df'].to_csv(os.path.join(OUT_DIR, 'trialwise_empirical_fit_status.csv'), index=False)
            print('empirical statuses:')
            display(empirical_trialwise['status_df']['status'].value_counts(dropna=False).rename_axis('status').reset_index(name='n'))
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Primary within-subject surrogate null maps
        # =============================
        def compute_one_trialwise_surrogate_map(
            surrogate_idx: int,
            surrogate_seed: int,
            subject_data_list,
            ri_cache,
            fb_idxs,
            cue_idxs,
        ):
            rng = np.random.default_rng(int(surrogate_seed))
            shape = (len(fb_idxs), len(cue_idxs))
            subj_perm = {
                sd.subject: permute_trialwise_predictors_within_subject(sd, rng)
                for sd in subject_data_list
            }
            sur_beta = init_term_maps(shape)
            status_counts = {}

            for a, t_fb_idx in enumerate(fb_idxs):
                for b, t_cue_idx in enumerate(cue_idxs):
                    rows = []
                    for sd in subject_data_list:
                        df_sub = build_trialwise_rows_for_subject(
                            sd,
                            ri_cache[sd.subject]['ri'],
                            int(t_fb_idx),
                            int(t_cue_idx),
                            use_permuted_labels=subj_perm[sd.subject],
                        )
                        if len(df_sub) == 0:
                            continue
                        df_sub = df_sub.copy()
                        df_sub['rpe'] = zscore_within_subject(df_sub['rpe'].to_numpy(dtype=np.float64))
                        rows.append(df_sub)
                    if not rows:
                        status_counts['no_rows'] = status_counts.get('no_rows', 0) + 1
                        continue
                    df_tp = pd.concat(rows, axis=0, ignore_index=True)
                    res = fit_trialwise_mixed_model(df_tp)
                    status = str(res.get('status', 'unknown'))
                    if not status.startswith('ok'):
                        status_counts[status] = status_counts.get(status, 0) + 1
                        continue
                    params = _extract_term_dict(res, 'params')
                    for term in ('memory', 'rpe', 'memory:rpe'):
                        sur_beta[term][a, b] = np.float32(params[term])

            return {
                'surrogate_idx': int(surrogate_idx),
                'sur_beta': sur_beta,
                'status_counts': status_counts,
            }


        def compute_trialwise_surrogate_chunk(
            surrogate_indices,
            surrogate_seeds,
            subject_data_list,
            ri_cache,
            fb_idxs,
            cue_idxs,
        ):
            results = []
            for surrogate_idx, surrogate_seed in zip(surrogate_indices, surrogate_seeds):
                results.append(
                    compute_one_trialwise_surrogate_map(
                        surrogate_idx=int(surrogate_idx),
                        surrogate_seed=int(surrogate_seed),
                        subject_data_list=subject_data_list,
                        ri_cache=ri_cache,
                        fb_idxs=fb_idxs,
                        cue_idxs=cue_idxs,
                    )
                )
            return results


        def run_trialwise_surrogate_maps(
            subject_data_list,
            ri_cache,
            empirical_res,
            n_surrogates=500,
            seed=42,
            n_jobs_surrogates=1,
            surrogate_chunk_size=1,
            parallel_backend='loky',
        ):
            fb_idxs = empirical_res['fb_idxs']
            cue_idxs = empirical_res['cue_idxs']
            shape = (len(fb_idxs), len(cue_idxs))

            null_mean = init_term_maps(shape, fill_value=0.0)
            null_M2 = init_term_maps(shape, fill_value=0.0)
            null_counts = {
                'memory': np.zeros(shape, dtype=np.int32),
                'rpe': np.zeros(shape, dtype=np.int32),
                'memory:rpe': np.zeros(shape, dtype=np.int32),
            }
            extreme_counts = {
                'memory': np.zeros(shape, dtype=np.int32),
                'rpe': np.zeros(shape, dtype=np.int32),
                'memory:rpe': np.zeros(shape, dtype=np.int32),
            }
            status_summary_rows = []

            seed_rng = np.random.default_rng(int(seed))
            surrogate_indices = list(range(int(n_surrogates)))
            surrogate_seeds = seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=int(n_surrogates), dtype=np.uint32
            ).astype(np.uint64).tolist()
            index_chunks = chunk_sequence(surrogate_indices, surrogate_chunk_size)
            seed_chunks = chunk_sequence(surrogate_seeds, surrogate_chunk_size)
            chunk_inputs = list(zip(index_chunks, seed_chunks))

            worker_results = Parallel(n_jobs=int(n_jobs_surrogates), backend=parallel_backend)(
                delayed(compute_trialwise_surrogate_chunk)(
                    surrogate_indices=idx_chunk,
                    surrogate_seeds=seed_chunk,
                    subject_data_list=subject_data_list,
                    ri_cache=ri_cache,
                    fb_idxs=fb_idxs,
                    cue_idxs=cue_idxs,
                )
                for idx_chunk, seed_chunk in tqdm(chunk_inputs, desc='Dispatch surrogate chunks')
            )

            completed = 0
            for chunk_result in tqdm(worker_results, desc='Reduce surrogate chunks'):
                for res in chunk_result:
                    completed += 1
                    reduce_surrogate_maps_into_null(
                        surrogate_maps=res['sur_beta'],
                        empirical_beta_maps=empirical_res['beta_maps'],
                        null_mean=null_mean,
                        null_M2=null_M2,
                        null_counts=null_counts,
                        extreme_counts=extreme_counts,
                    )
                    for status, count in sorted(res.get('status_counts', {}).items()):
                        status_summary_rows.append({
                            'surrogate': int(res['surrogate_idx']),
                            'status': str(status),
                            'count': int(count),
                        })

            null_std = {}
            z_beta = {}
            p_beta = {}
            fdr_mask = {}
            for term in ('memory', 'rpe', 'memory:rpe'):
                counts = np.asarray(null_counts[term], dtype=np.int32)
                denom = np.maximum(counts.astype(np.float64), 1.0)
                std = np.sqrt(np.asarray(null_M2[term], dtype=np.float64) / denom)
                std = np.where(counts > 0, np.maximum(std, 1e-8), np.nan)
                emp = np.asarray(empirical_res['beta_maps'][term], dtype=np.float64)
                mu = np.asarray(null_mean[term], dtype=np.float64)
                z = (emp - mu) / std
                p = (extreme_counts[term].astype(np.float64) + 1.0) / (denom + 1.0)
                null_std[term] = std.astype(np.float32)
                z_beta[term] = z.astype(np.float32)
                p_beta[term] = p.astype(np.float32)
                fdr_mask[term] = fdr_correct_timegrid(p)

            return {
                'n_surrogates_done': int(n_surrogates),
                'null_mean': {k: np.asarray(v, dtype=np.float32) for k, v in null_mean.items()},
                'null_std': null_std,
                'null_counts': {k: np.asarray(v, dtype=np.int32) for k, v in null_counts.items()},
                'z_beta': z_beta,
                'p_beta': p_beta,
                'fdr_mask': fdr_mask,
                'surrogate_fail_df': pd.DataFrame(status_summary_rows),
                'n_surrogates_completed': int(completed),
            }


        trialwise_null = None
        if RUN_PRIMARY_TRIALWISE and RUN_PRIMARY_SURROGATES and int(N_SURROGATES) > 0:
            trialwise_null = run_trialwise_surrogate_maps(
                subject_data_list,
                ri_cache,
                empirical_trialwise,
                n_surrogates=N_SURROGATES,
                seed=SEED,
                n_jobs_surrogates=N_JOBS_SURROGATES,
                surrogate_chunk_size=SURROGATE_CHUNK_SIZE,
                parallel_backend=PARALLEL_BACKEND,
            )
            trialwise_null['surrogate_fail_df'].to_csv(os.path.join(OUT_DIR, 'trialwise_surrogate_failures.csv'), index=False)
            print('trialwise surrogate status rows:', len(trialwise_null['surrogate_fail_df']))
            print('trialwise surrogates completed:', int(trialwise_null['n_surrogates_completed']))
        elif RUN_PRIMARY_TRIALWISE and RUN_PRIMARY_SURROGATES and int(N_SURROGATES) <= 0:
            print('Skipping surrogate nulls because N_SURROGATES <= 0; using empirical mixed-model p-values for significance masks.')
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Save primary outputs
        # =============================
        if empirical_trialwise is not None:
            np.savez_compressed(
                os.path.join(OUT_DIR, 'trialwise_empirical_betas.npz'),
                feedback_times=fb_t.astype(np.float32),
                cue_times=cue_t.astype(np.float32),
                fb_idxs=np.asarray(empirical_trialwise['fb_idxs'], dtype=np.int64),
                cue_idxs=np.asarray(empirical_trialwise['cue_idxs'], dtype=np.int64),
                beta_memory=np.asarray(empirical_trialwise['beta_maps']['memory'], dtype=np.float32),
                beta_rpe=np.asarray(empirical_trialwise['beta_maps']['rpe'], dtype=np.float32),
                beta_memory_x_rpe=np.asarray(empirical_trialwise['beta_maps']['memory:rpe'], dtype=np.float32),
                se_memory=np.asarray(empirical_trialwise['se_maps']['memory'], dtype=np.float32),
                se_rpe=np.asarray(empirical_trialwise['se_maps']['rpe'], dtype=np.float32),
                se_memory_x_rpe=np.asarray(empirical_trialwise['se_maps']['memory:rpe'], dtype=np.float32),
                t_memory=np.asarray(empirical_trialwise['t_maps']['memory'], dtype=np.float32),
                t_rpe=np.asarray(empirical_trialwise['t_maps']['rpe'], dtype=np.float32),
                t_memory_x_rpe=np.asarray(empirical_trialwise['t_maps']['memory:rpe'], dtype=np.float32),
                p_memory=np.asarray(empirical_trialwise['p_maps']['memory'], dtype=np.float32),
                p_rpe=np.asarray(empirical_trialwise['p_maps']['rpe'], dtype=np.float32),
                p_memory_x_rpe=np.asarray(empirical_trialwise['p_maps']['memory:rpe'], dtype=np.float32),
                fdr_memory=np.asarray(empirical_trialwise['fdr_masks']['memory'], dtype=np.int8),
                fdr_rpe=np.asarray(empirical_trialwise['fdr_masks']['rpe'], dtype=np.int8),
                fdr_memory_x_rpe=np.asarray(empirical_trialwise['fdr_masks']['memory:rpe'], dtype=np.int8),
                uncorr_memory=np.asarray(empirical_trialwise['uncorrected_masks']['memory'], dtype=np.int8),
                uncorr_rpe=np.asarray(empirical_trialwise['uncorrected_masks']['rpe'], dtype=np.int8),
                uncorr_memory_x_rpe=np.asarray(empirical_trialwise['uncorrected_masks']['memory:rpe'], dtype=np.int8),
                n_rows=np.asarray(empirical_trialwise['n_rows_map'], dtype=np.float32),
                n_subjects=np.asarray(empirical_trialwise['n_subjects_map'], dtype=np.float32),
                converged=np.asarray(empirical_trialwise['converged_map'], dtype=np.int8),
            )

        if trialwise_null is not None:
            np.savez_compressed(
                os.path.join(OUT_DIR, 'trialwise_null_stats.npz'),
                feedback_times=fb_t.astype(np.float32),
                cue_times=cue_t.astype(np.float32),
                n_surrogates=np.int32(trialwise_null['n_surrogates_done']),
                null_mean_memory=np.asarray(trialwise_null['null_mean']['memory'], dtype=np.float32),
                null_mean_rpe=np.asarray(trialwise_null['null_mean']['rpe'], dtype=np.float32),
                null_mean_memory_x_rpe=np.asarray(trialwise_null['null_mean']['memory:rpe'], dtype=np.float32),
                null_std_memory=np.asarray(trialwise_null['null_std']['memory'], dtype=np.float32),
                null_std_rpe=np.asarray(trialwise_null['null_std']['rpe'], dtype=np.float32),
                null_std_memory_x_rpe=np.asarray(trialwise_null['null_std']['memory:rpe'], dtype=np.float32),
                z_memory=np.asarray(trialwise_null['z_beta']['memory'], dtype=np.float32),
                z_rpe=np.asarray(trialwise_null['z_beta']['rpe'], dtype=np.float32),
                z_memory_x_rpe=np.asarray(trialwise_null['z_beta']['memory:rpe'], dtype=np.float32),
                p_memory=np.asarray(trialwise_null['p_beta']['memory'], dtype=np.float32),
                p_rpe=np.asarray(trialwise_null['p_beta']['rpe'], dtype=np.float32),
                p_memory_x_rpe=np.asarray(trialwise_null['p_beta']['memory:rpe'], dtype=np.float32),
                fdr_memory=np.asarray(trialwise_null['fdr_mask']['memory'], dtype=np.int8),
                fdr_rpe=np.asarray(trialwise_null['fdr_mask']['rpe'], dtype=np.int8),
                fdr_memory_x_rpe=np.asarray(trialwise_null['fdr_mask']['memory:rpe'], dtype=np.int8),
            )

            np.savez_compressed(
                os.path.join(OUT_DIR, 'trialwise_zmaps.npz'),
                z_memory=np.asarray(trialwise_null['z_beta']['memory'], dtype=np.float32),
                z_rpe=np.asarray(trialwise_null['z_beta']['rpe'], dtype=np.float32),
                z_memory_x_rpe=np.asarray(trialwise_null['z_beta']['memory:rpe'], dtype=np.float32),
            )

            np.savez_compressed(
                os.path.join(OUT_DIR, 'trialwise_pmaps_fdr.npz'),
                p_memory=np.asarray(trialwise_null['p_beta']['memory'], dtype=np.float32),
                p_rpe=np.asarray(trialwise_null['p_beta']['rpe'], dtype=np.float32),
                p_memory_x_rpe=np.asarray(trialwise_null['p_beta']['memory:rpe'], dtype=np.float32),
                fdr_memory=np.asarray(trialwise_null['fdr_mask']['memory'], dtype=np.int8),
                fdr_rpe=np.asarray(trialwise_null['fdr_mask']['rpe'], dtype=np.int8),
                fdr_memory_x_rpe=np.asarray(trialwise_null['fdr_mask']['memory:rpe'], dtype=np.int8),
            )
        elif empirical_trialwise is not None:
            np.savez_compressed(
                os.path.join(OUT_DIR, 'trialwise_pmaps_fdr.npz'),
                p_memory=np.asarray(empirical_trialwise['p_maps']['memory'], dtype=np.float32),
                p_rpe=np.asarray(empirical_trialwise['p_maps']['rpe'], dtype=np.float32),
                p_memory_x_rpe=np.asarray(empirical_trialwise['p_maps']['memory:rpe'], dtype=np.float32),
                uncorr_memory=np.asarray(empirical_trialwise['uncorrected_masks']['memory'], dtype=np.int8),
                uncorr_rpe=np.asarray(empirical_trialwise['uncorrected_masks']['rpe'], dtype=np.int8),
                uncorr_memory_x_rpe=np.asarray(empirical_trialwise['uncorrected_masks']['memory:rpe'], dtype=np.int8),
                alpha=np.float32(UNCORRECTED_P_ALPHA),
            )
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Plot primary results
        # =============================
        if empirical_trialwise is None:
            print('Primary empirical maps not available.')
        else:
            if trialwise_null is not None:
                sig_label = 'surrogate-null'
                display_maps = {
                    'stat': trialwise_null['z_beta'],
                    'p': trialwise_null['p_beta'],
                    'mask': trialwise_null['fdr_mask'],
                }
                stat_title = 'Null-z beta'
            else:
                empirical_masks, sig_label = select_empirical_mask_bundle(
                    empirical_trialwise,
                    SIGNIFICANCE_MASK_MODE,
                )
                display_maps = {
                    'stat': empirical_trialwise['t_maps'],
                    'p': empirical_trialwise['p_maps'],
                    'mask': empirical_masks,
                }
                stat_title = 'Empirical t/Wald z'

            pdf_path = os.path.join(OUT_DIR, 'acc_hpc_trialwise_mixed_null_results.pdf')
            with PdfPages(pdf_path) as pdf:
                fig, axes = plt.subplots(3, 3, figsize=(15, 13))
                term_order = [('memory', 'Memory'), ('rpe', 'RPE'), ('memory:rpe', 'Memory x RPE')]
                for row, (term_key, term_label) in enumerate(term_order):
                    im0 = plot_metric_map(
                        axes[row, 0],
                        empirical_trialwise['beta_maps'][term_key],
                        fb_t,
                        cue_t,
                        f'Empirical beta: {term_label}',
                    )
                    plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)

                    im1 = plot_metric_map(
                        axes[row, 1],
                        display_maps['stat'][term_key],
                        fb_t,
                        cue_t,
                        f'{stat_title}: {term_label}',
                        mask=display_maps['mask'][term_key],
                    )
                    plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)

                    im2 = plot_metric_map(
                        axes[row, 2],
                        -np.log10(np.maximum(display_maps['p'][term_key], 1e-12)),
                        fb_t,
                        cue_t,
                        f'-log10 p ({sig_label}): {term_label}',
                        cmap='viridis',
                        center_zero=False,
                        mask=display_maps['mask'][term_key],
                    )
                    plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.show()

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, (term_key, term_label) in zip(axes, term_order):
                    mask = np.asarray(display_maps['mask'][term_key], dtype=bool)
                    ax.imshow(mask.astype(int), origin='lower', aspect='auto', cmap='Greys')
                    ax.set_title(f'Significance mask ({sig_label}): {term_label}')
                    ax.set_xlabel('Cue index')
                    ax.set_ylabel('Feedback index')
                plt.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.show()

            print('saved:', pdf_path)
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Positive-RPE follow-up plots
        # =============================
        def _label_from_rpe_cat_code(x: float) -> str:
            if not np.isfinite(x):
                return 'na'
            if float(x) < 0:
                return 'negative'
            if float(x) > 0:
                return 'positive'
            return 'neutral'


        def _choose_followup_mask(empirical_trialwise, trialwise_null):
            if trialwise_null is not None:
                mask = np.asarray(trialwise_null['fdr_mask']['memory:rpe'], dtype=bool)
            else:
                empirical_masks, _ = select_empirical_mask_bundle(
                    empirical_trialwise,
                    SIGNIFICANCE_MASK_MODE,
                )
                mask = np.asarray(empirical_masks['memory:rpe'], dtype=bool)

            if np.any(mask):
                return mask

            beta = np.asarray(empirical_trialwise['beta_maps']['memory:rpe'], dtype=float)
            peak_mask = np.zeros_like(beta, dtype=bool)
            if np.any(np.isfinite(beta)):
                peak_mask[np.unravel_index(int(np.nanargmax(np.abs(beta))), beta.shape)] = True
            return peak_mask


        def build_masked_trialwise_followup_df(subject_data_list, ri_cache, empirical_trialwise, mask):
            rows = []
            fb_idxs = np.asarray(empirical_trialwise['fb_idxs'], dtype=int)
            cue_idxs = np.asarray(empirical_trialwise['cue_idxs'], dtype=int)
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (len(fb_idxs), len(cue_idxs)):
                raise ValueError('follow-up mask shape mismatch')

            for sd in subject_data_list:
                ri_tensor = np.asarray(ri_cache[sd.subject]['ri'], dtype=np.float64)
                ri_sub = ri_tensor[:, fb_idxs][:, :, cue_idxs]
                masked_ri = np.array([np.nanmean(ri_sub[i][mask]) for i in range(sd.n_trials)], dtype=np.float64)
                for i in range(sd.n_trials):
                    rows.append({
                        'subject': sd.subject,
                        'trial_id': int(i),
                        'memory': float(sd.memory_numeric[i]),
                        'memory_label': 'hit' if int(sd.memory_numeric[i]) == 1 else 'miss',
                        'rpe': float(sd.rpe_value[i]),
                        'rpe_category_code': float(sd.rpe_category_code[i]),
                        'rpe_category_label': _label_from_rpe_cat_code(sd.rpe_category_code[i]),
                        'ri_masked': float(masked_ri[i]),
                    })
            df = pd.DataFrame(rows)
            df = df[np.isfinite(df['ri_masked']) & np.isfinite(df['memory']) & np.isfinite(df['rpe'])].reset_index(drop=True)
            return df


        def compute_category_diff_maps(subject_data_list, ri_cache, empirical_trialwise):
            fb_idxs = np.asarray(empirical_trialwise['fb_idxs'], dtype=int)
            cue_idxs = np.asarray(empirical_trialwise['cue_idxs'], dtype=int)
            shape = (len(fb_idxs), len(cue_idxs))
            cat_defs = [('negative', -1.0), ('neutral', 0.0), ('positive', 1.0)]
            out = {}
            for cat_name, cat_code in cat_defs:
                subj_maps = []
                for sd in subject_data_list:
                    m_cat = np.asarray(sd.rpe_category_code == cat_code, dtype=bool)
                    m_hit = np.asarray(sd.memory_numeric == 1, dtype=bool)
                    m_miss = np.asarray(sd.memory_numeric == 0, dtype=bool)
                    keep_hit = m_cat & m_hit
                    keep_miss = m_cat & m_miss
                    if np.sum(keep_hit) < 1 or np.sum(keep_miss) < 1:
                        continue
                    ri_tensor = np.asarray(ri_cache[sd.subject]['ri'], dtype=np.float64)
                    ri_sub = ri_tensor[:, fb_idxs][:, :, cue_idxs]
                    hit_map = np.nanmean(ri_sub[keep_hit], axis=0)
                    miss_map = np.nanmean(ri_sub[keep_miss], axis=0)
                    subj_maps.append(hit_map - miss_map)
                if subj_maps:
                    out[cat_name] = np.stack(subj_maps, axis=0).astype(np.float32)
                else:
                    out[cat_name] = np.empty((0,) + shape, dtype=np.float32)
            return out


        if empirical_trialwise is None:
            print('Skipping positive-RPE follow-up: empirical maps unavailable.')
        else:
            follow_mask = _choose_followup_mask(empirical_trialwise, trialwise_null)
            df_follow = build_masked_trialwise_followup_df(subject_data_list, ri_cache, empirical_trialwise, follow_mask)
            if len(df_follow) == 0:
                print('Skipping positive-RPE follow-up: no valid masked trialwise rows.')
            else:
                follow_dir = os.path.join(OUT_DIR, 'positive_rpe_followup')
                os.makedirs(follow_dir, exist_ok=True)
                df_follow.to_csv(os.path.join(follow_dir, 'masked_trialwise_ri_followup.csv'), index=False)

                if np.sum(follow_mask) == 1:
                    peak_idx = np.argwhere(follow_mask)[0]
                    print(f'Follow-up mask fallback to single peak voxel at fb_idx={int(peak_idx[0])}, cue_idx={int(peak_idx[1])}')
                else:
                    print(f'Follow-up mask voxels: {int(np.sum(follow_mask))}')

                df_subject_cat = (
                    df_follow.groupby(['subject', 'rpe_category_label', 'memory_label'], as_index=False)['ri_masked']
                    .mean()
                )
                piv = df_subject_cat.pivot_table(
                    index=['subject', 'rpe_category_label'],
                    columns='memory_label',
                    values='ri_masked',
                    aggfunc='mean'
                ).reset_index()
                if 'hit' not in piv.columns:
                    piv['hit'] = np.nan
                if 'miss' not in piv.columns:
                    piv['miss'] = np.nan
                piv['hit_minus_miss'] = piv['hit'] - piv['miss']
                piv.to_csv(os.path.join(follow_dir, 'subject_category_hit_minus_miss.csv'), index=False)

                cat_order = ['negative', 'neutral', 'positive']
                color_map = {'negative': '#6b7280', 'neutral': '#9ca3af', 'positive': '#d94841'}

                category_maps = compute_category_diff_maps(subject_data_list, ri_cache, empirical_trialwise)
                pos_map = np.nanmean(category_maps['positive'], axis=0) if category_maps['positive'].shape[0] > 0 else np.full_like(follow_mask, np.nan, dtype=float)
                neu_map = np.nanmean(category_maps['neutral'], axis=0) if category_maps['neutral'].shape[0] > 0 else np.full_like(follow_mask, np.nan, dtype=float)
                neg_map = np.nanmean(category_maps['negative'], axis=0) if category_maps['negative'].shape[0] > 0 else np.full_like(follow_mask, np.nan, dtype=float)
                pos_minus_neu = pos_map - neu_map
                pos_minus_neg = pos_map - neg_map

                np.savez_compressed(
                    os.path.join(follow_dir, 'positive_specific_time_maps.npz'),
                    positive_hit_minus_miss=np.asarray(pos_map, dtype=np.float32),
                    positive_minus_neutral=np.asarray(pos_minus_neu, dtype=np.float32),
                    positive_minus_negative=np.asarray(pos_minus_neg, dtype=np.float32),
                    feedback_times=fb_t.astype(np.float32),
                    cue_times=cue_t.astype(np.float32),
                    follow_mask=np.asarray(follow_mask, dtype=np.int8),
                )

                pdf_path = os.path.join(follow_dir, 'positive_rpe_followup_plots.pdf')
                with PdfPages(pdf_path) as pdf:
                    # 1. Predicted-lines / trend plot
                    fig, ax = plt.subplots(figsize=(7, 5))
                    for mem_val, mem_label in [(0.0, 'miss'), (1.0, 'hit')]:
                        dsub = df_follow[df_follow['memory'] == mem_val].copy()
                        if len(dsub) == 0:
                            continue
                        ax.scatter(
                            dsub['rpe'],
                            dsub['ri_masked'],
                            s=20,
                            alpha=0.18,
                            label=f'{mem_label} trials',
                            color='#1f77b4' if mem_val == 0.0 else '#d94841',
                        )
                        if dsub['rpe'].nunique() >= 2:
                            xx = np.linspace(float(np.nanmin(dsub['rpe'])), float(np.nanmax(dsub['rpe'])), 100)
                            b1, b0 = np.polyfit(dsub['rpe'].to_numpy(dtype=float), dsub['ri_masked'].to_numpy(dtype=float), 1)
                            ax.plot(
                                xx,
                                (b1 * xx) + b0,
                                linewidth=2.5,
                                color='#1f77b4' if mem_val == 0.0 else '#d94841',
                                label=f'{mem_label} fit',
                            )
                    ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
                    ax.axhline(0.0, color='k', linestyle='-', linewidth=0.6, alpha=0.4)
                    ax.set_xlabel('RPE')
                    ax.set_ylabel('Masked trialwise RI')
                    ax.set_title('Masked ACC→HPC RI vs RPE\\nseparate fits for hits and misses')
                    ax.legend(frameon=False, fontsize=9)
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.show()

                    # 2. Positive-vs-rest subject summary
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                    ax0, ax1 = axes
                    piv_plot = piv[piv['rpe_category_label'].isin(cat_order)].copy()
                    subjects_plot = sorted(piv_plot['subject'].unique())
                    x_pos = {cat: idx for idx, cat in enumerate(cat_order)}
                    for subj in subjects_plot:
                        ds = piv_plot[piv_plot['subject'] == subj].copy()
                        ds['x'] = ds['rpe_category_label'].map(x_pos)
                        ds = ds.sort_values('x')
                        ax0.plot(ds['x'], ds['hit_minus_miss'], color='0.75', linewidth=0.8, alpha=0.7)
                        ax0.scatter(
                            ds['x'],
                            ds['hit_minus_miss'],
                            c=[color_map.get(x, '0.5') for x in ds['rpe_category_label']],
                            s=28,
                            alpha=0.9,
                            edgecolors='none',
                        )
                    means = piv_plot.groupby('rpe_category_label')['hit_minus_miss'].mean().reindex(cat_order)
                    sems = piv_plot.groupby('rpe_category_label')['hit_minus_miss'].sem().reindex(cat_order)
                    xx = np.arange(len(cat_order))
                    ax0.errorbar(xx, means.to_numpy(dtype=float), yerr=sems.to_numpy(dtype=float), color='k', linewidth=2.0, capsize=4)
                    ax0.scatter(xx, means.to_numpy(dtype=float), c=[color_map[c] for c in cat_order], s=90, edgecolors='k', linewidths=0.4, zorder=3)
                    ax0.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                    ax0.set_xticks(xx)
                    ax0.set_xticklabels(cat_order)
                    ax0.set_ylabel('Masked RI (hit - miss)')
                    ax0.set_title('Hit-miss difference by RPE category')

                    contrast_rows = []
                    for subj in subjects_plot:
                        ds = piv_plot[piv_plot['subject'] == subj].set_index('rpe_category_label')
                        if {'positive', 'neutral'}.issubset(ds.index):
                            contrast_rows.append({'subject': subj, 'contrast': 'positive - neutral', 'value': float(ds.loc['positive', 'hit_minus_miss'] - ds.loc['neutral', 'hit_minus_miss'])})
                        if {'positive', 'negative'}.issubset(ds.index):
                            contrast_rows.append({'subject': subj, 'contrast': 'positive - negative', 'value': float(ds.loc['positive', 'hit_minus_miss'] - ds.loc['negative', 'hit_minus_miss'])})
                    df_contrast = pd.DataFrame(contrast_rows)
                    if len(df_contrast) > 0:
                        contrast_order = ['positive - neutral', 'positive - negative']
                        for subj in sorted(df_contrast['subject'].unique()):
                            ds = df_contrast[df_contrast['subject'] == subj].copy()
                            ds['x'] = ds['contrast'].map({c: i for i, c in enumerate(contrast_order)})
                            ds = ds.sort_values('x')
                            ax1.plot(ds['x'], ds['value'], color='0.78', linewidth=0.8, alpha=0.7)
                            ax1.scatter(ds['x'], ds['value'], color='#d94841', s=28, alpha=0.9, edgecolors='none')
                        means_c = df_contrast.groupby('contrast')['value'].mean().reindex(contrast_order)
                        sems_c = df_contrast.groupby('contrast')['value'].sem().reindex(contrast_order)
                        xx2 = np.arange(len(contrast_order))
                        ax1.errorbar(xx2, means_c.to_numpy(dtype=float), yerr=sems_c.to_numpy(dtype=float), color='k', linewidth=2.0, capsize=4)
                        ax1.scatter(xx2, means_c.to_numpy(dtype=float), color='#d94841', s=90, edgecolors='k', linewidths=0.4, zorder=3)
                        ax1.set_xticks(xx2)
                        ax1.set_xticklabels(contrast_order, rotation=10)
                    ax1.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
                    ax1.set_ylabel('Positive-category advantage\\nin hit-miss RI')
                    ax1.set_title('Positive-specific subject contrasts')
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.show()

                    # 3. Positive-specific time maps
                    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
                    maps_to_plot = [
                        (pos_map, 'Positive RPE: hit - miss'),
                        (pos_minus_neu, 'Positive minus neutral'),
                        (pos_minus_neg, 'Positive minus negative'),
                    ]
                    for ax, (arr, ttl) in zip(axes, maps_to_plot):
                        im = plot_metric_map(ax, arr, fb_t, cue_t, ttl, mask=follow_mask)
                        plt.colorbar(im, ax=ax, shrink=0.82)
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=300, bbox_inches='tight')
                    plt.show()

                print('Saved positive-RPE follow-up plots to:', pdf_path)
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Secondary pairwise similarity mixed model
        # =============================
        def build_pairwise_similarity_rows_for_subject(
            subj_data: SubjectReinstatementData,
            t_fb_idx: int,
            t_cue_idx: int,
            use_permuted_labels: Optional[Dict[str, np.ndarray]] = None,
        ) -> pd.DataFrame:
            E = subj_data.encoding_hfa[:, :, t_fb_idx]
            R = subj_data.cue_hfa[:, :, t_cue_idx]
            S = _padded_row_zscore_cosine_similarity(E, R)
            n = subj_data.n_trials

            if use_permuted_labels is None:
                mem = subj_data.memory_numeric
                rpe = subj_data.rpe_category_code
            else:
                mem = np.asarray(use_permuted_labels['memory'], dtype=np.float64)
                rpe = np.asarray(use_permuted_labels['rpe_cat'], dtype=np.float64)

            ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            mem_i = mem[ii.ravel()]
            mem_j = mem[jj.ravel()]
            rpe_i = rpe[ii.ravel()]
            rpe_j = rpe[jj.ravel()]
            df = pd.DataFrame({
                'subject': subj_data.subject,
                'enc_trial': ii.ravel().astype(np.int64),
                'ret_trial': jj.ravel().astype(np.int64),
                'similarity': S.reshape(-1).astype(np.float64),
                'item_match': (ii.ravel() == jj.ravel()).astype(np.float64),
                'memory_match': (mem_i == mem_j).astype(np.float64),
                'rpe_match': (rpe_i == rpe_j).astype(np.float64),
            })
            return df[np.isfinite(df['similarity'])].reset_index(drop=True)


        def fit_pairwise_mixed_model(df_timepoint: pd.DataFrame) -> Dict[str, Any]:
            if len(df_timepoint) < 10 or df_timepoint['subject'].nunique() < 2:
                return {'status': 'insufficient_rows'}
            for col in ('item_match', 'memory_match', 'rpe_match'):
                if np.nanstd(df_timepoint[col].to_numpy(dtype=np.float64)) < 1e-8:
                    return {'status': f'degenerate_{col}'}
            try:
                model = mixedlm('similarity ~ item_match + memory_match + rpe_match', df_timepoint, groups=df_timepoint['subject'])
                result = model.fit(method=MIXEDLM_METHOD, reml=False, disp=False)
                return {
                    'status': 'ok',
                    'engine': 'mixedlm',
                    'params': result.params.to_dict(),
                    'bse': result.bse.to_dict(),
                    'tvalues': result.tvalues.to_dict() if hasattr(result, 'tvalues') else {},
                    'converged': bool(getattr(result, 'converged', False)),
                    'n_subjects': int(df_timepoint['subject'].nunique()),
                    'n_rows': int(len(df_timepoint)),
                }
            except Exception as exc:
                return {'status': f'fit_error:{type(exc).__name__}:{exc}'}


        pairwise_results = None
        pairwise_null = None
        if RUN_SECONDARY_PAIRWISE:
            fb_idxs = list(range(0, subject_data_list[0].n_feedback_times, TIMEPOINT_STRIDE))
            cue_idxs = list(range(0, subject_data_list[0].n_cue_times, TIMEPOINT_STRIDE))
            shape = (len(fb_idxs), len(cue_idxs))
            beta_maps = {
                'item_match': np.full(shape, np.nan, dtype=np.float32),
                'memory_match': np.full(shape, np.nan, dtype=np.float32),
                'rpe_match': np.full(shape, np.nan, dtype=np.float32),
            }
            status_rows = []
            for a, t_fb_idx in enumerate(tqdm(fb_idxs, desc='Pairwise fb')):
                for b, t_cue_idx in enumerate(cue_idxs):
                    rows = [build_pairwise_similarity_rows_for_subject(sd, t_fb_idx, t_cue_idx) for sd in subject_data_list]
                    rows = [x for x in rows if len(x) > 0]
                    if not rows:
                        continue
                    df_tp = pd.concat(rows, axis=0, ignore_index=True)
                    res = fit_pairwise_mixed_model(df_tp)
                    status_rows.append({'t_fb_idx': t_fb_idx, 't_cue_idx': t_cue_idx, 'status': res.get('status', 'unknown')})
                    if not str(res.get('status', '')).startswith('ok'):
                        continue
                    beta_maps['item_match'][a, b] = np.float32(res['params'].get('item_match', np.nan))
                    beta_maps['memory_match'][a, b] = np.float32(res['params'].get('memory_match', np.nan))
                    beta_maps['rpe_match'][a, b] = np.float32(res['params'].get('rpe_match', np.nan))
            pairwise_results = {'beta_maps': beta_maps, 'status_df': pd.DataFrame(status_rows)}
            pairwise_results['status_df'].to_csv(os.path.join(OUT_DIR, 'pairwise_empirical_fit_status.csv'), index=False)

            np.savez_compressed(
                os.path.join(OUT_DIR, 'pairwise_empirical_betas.npz'),
                feedback_times=fb_t.astype(np.float32),
                cue_times=cue_t.astype(np.float32),
                beta_item_match=np.asarray(beta_maps['item_match'], dtype=np.float32),
                beta_memory_match=np.asarray(beta_maps['memory_match'], dtype=np.float32),
                beta_rpe_match=np.asarray(beta_maps['rpe_match'], dtype=np.float32),
            )

            if RUN_SECONDARY_SURROGATES:
                null_mean = {
                    'item_match': np.zeros(shape, dtype=np.float32),
                    'memory_match': np.zeros(shape, dtype=np.float32),
                    'rpe_match': np.zeros(shape, dtype=np.float32),
                }
                null_M2 = {
                    'item_match': np.zeros(shape, dtype=np.float32),
                    'memory_match': np.zeros(shape, dtype=np.float32),
                    'rpe_match': np.zeros(shape, dtype=np.float32),
                }
                null_counts = {
                    'item_match': np.zeros(shape, dtype=np.int32),
                    'memory_match': np.zeros(shape, dtype=np.int32),
                    'rpe_match': np.zeros(shape, dtype=np.int32),
                }
                extreme_counts = {
                    'item_match': np.zeros(shape, dtype=np.int32),
                    'memory_match': np.zeros(shape, dtype=np.int32),
                    'rpe_match': np.zeros(shape, dtype=np.int32),
                }
                rng = np.random.default_rng(SEED)
                fail_rows = []

                for s_idx in tqdm(range(int(N_SURROGATES)), desc='Pairwise surrogates'):
                    subj_perm = {
                        sd.subject: permute_trialwise_predictors_within_subject(sd, rng)
                        for sd in subject_data_list
                    }
                    sur_beta = {
                        'item_match': np.full(shape, np.nan, dtype=np.float32),
                        'memory_match': np.full(shape, np.nan, dtype=np.float32),
                        'rpe_match': np.full(shape, np.nan, dtype=np.float32),
                    }
                    for a, t_fb_idx in enumerate(fb_idxs):
                        for b, t_cue_idx in enumerate(cue_idxs):
                            rows = [
                                build_pairwise_similarity_rows_for_subject(
                                    sd,
                                    int(t_fb_idx),
                                    int(t_cue_idx),
                                    use_permuted_labels=subj_perm[sd.subject],
                                )
                                for sd in subject_data_list
                            ]
                            rows = [x for x in rows if len(x) > 0]
                            if not rows:
                                fail_rows.append({'surrogate': s_idx, 't_fb_idx': int(t_fb_idx), 't_cue_idx': int(t_cue_idx), 'status': 'no_rows'})
                                continue
                            df_tp = pd.concat(rows, axis=0, ignore_index=True)
                            res = fit_pairwise_mixed_model(df_tp)
                            if not str(res.get('status', '')).startswith('ok'):
                                fail_rows.append({'surrogate': s_idx, 't_fb_idx': int(t_fb_idx), 't_cue_idx': int(t_cue_idx), 'status': res.get('status', 'unknown')})
                                continue
                            sur_beta['item_match'][a, b] = np.float32(res['params'].get('item_match', np.nan))
                            sur_beta['memory_match'][a, b] = np.float32(res['params'].get('memory_match', np.nan))
                            sur_beta['rpe_match'][a, b] = np.float32(res['params'].get('rpe_match', np.nan))

                    for term in ('item_match', 'memory_match', 'rpe_match'):
                        x = np.asarray(sur_beta[term], dtype=np.float64)
                        valid = np.isfinite(x)
                        if np.any(valid):
                            mean_prev = np.asarray(null_mean[term], dtype=np.float64)
                            M2_prev = np.asarray(null_M2[term], dtype=np.float64)
                            counts_prev = np.asarray(null_counts[term], dtype=np.int32)
                            idx = np.where(valid)
                            for i0, i1 in zip(idx[0], idx[1]):
                                cnt_new = int(counts_prev[i0, i1]) + 1
                                val = float(x[i0, i1])
                                delta = val - float(mean_prev[i0, i1])
                                mean_prev[i0, i1] = float(mean_prev[i0, i1]) + delta / float(cnt_new)
                                M2_prev[i0, i1] = float(M2_prev[i0, i1]) + delta * (val - float(mean_prev[i0, i1]))
                                counts_prev[i0, i1] = cnt_new
                            null_mean[term] = mean_prev.astype(np.float32)
                            null_M2[term] = M2_prev.astype(np.float32)
                            null_counts[term] = counts_prev.astype(np.int32)

                        emp = np.asarray(pairwise_results['beta_maps'][term], dtype=np.float64)
                        extreme_counts[term] += (valid & np.isfinite(emp) & (np.abs(x) >= np.abs(emp))).astype(np.int32)

                pairwise_null = {
                    'null_mean': {},
                    'null_std': {},
                    'z_beta': {},
                    'p_beta': {},
                    'fdr_mask': {},
                    'fail_df': pd.DataFrame(fail_rows),
                }
                for term in ('item_match', 'memory_match', 'rpe_match'):
                    counts = np.asarray(null_counts[term], dtype=np.int32)
                    denom = np.maximum(counts.astype(np.float64), 1.0)
                    std = np.sqrt(np.asarray(null_M2[term], dtype=np.float64) / denom)
                    std = np.where(counts > 0, np.maximum(std, 1e-8), np.nan)
                    emp = np.asarray(pairwise_results['beta_maps'][term], dtype=np.float64)
                    mu = np.asarray(null_mean[term], dtype=np.float64)
                    z = (emp - mu) / std
                    p = (extreme_counts[term].astype(np.float64) + 1.0) / (denom + 1.0)
                    pairwise_null['null_mean'][term] = mu.astype(np.float32)
                    pairwise_null['null_std'][term] = std.astype(np.float32)
                    pairwise_null['z_beta'][term] = z.astype(np.float32)
                    pairwise_null['p_beta'][term] = p.astype(np.float32)
                    pairwise_null['fdr_mask'][term] = fdr_correct_timegrid(p)

                pairwise_null['fail_df'].to_csv(os.path.join(OUT_DIR, 'pairwise_surrogate_failures.csv'), index=False)
                np.savez_compressed(
                    os.path.join(OUT_DIR, 'pairwise_null_stats.npz'),
                    feedback_times=fb_t.astype(np.float32),
                    cue_times=cue_t.astype(np.float32),
                    null_mean_item_match=np.asarray(pairwise_null['null_mean']['item_match'], dtype=np.float32),
                    null_mean_memory_match=np.asarray(pairwise_null['null_mean']['memory_match'], dtype=np.float32),
                    null_mean_rpe_match=np.asarray(pairwise_null['null_mean']['rpe_match'], dtype=np.float32),
                    null_std_item_match=np.asarray(pairwise_null['null_std']['item_match'], dtype=np.float32),
                    null_std_memory_match=np.asarray(pairwise_null['null_std']['memory_match'], dtype=np.float32),
                    null_std_rpe_match=np.asarray(pairwise_null['null_std']['rpe_match'], dtype=np.float32),
                    z_item_match=np.asarray(pairwise_null['z_beta']['item_match'], dtype=np.float32),
                    z_memory_match=np.asarray(pairwise_null['z_beta']['memory_match'], dtype=np.float32),
                    z_rpe_match=np.asarray(pairwise_null['z_beta']['rpe_match'], dtype=np.float32),
                    p_item_match=np.asarray(pairwise_null['p_beta']['item_match'], dtype=np.float32),
                    p_memory_match=np.asarray(pairwise_null['p_beta']['memory_match'], dtype=np.float32),
                    p_rpe_match=np.asarray(pairwise_null['p_beta']['rpe_match'], dtype=np.float32),
                    fdr_item_match=np.asarray(pairwise_null['fdr_mask']['item_match'], dtype=np.int8),
                    fdr_memory_match=np.asarray(pairwise_null['fdr_mask']['memory_match'], dtype=np.int8),
                    fdr_rpe_match=np.asarray(pairwise_null['fdr_mask']['rpe_match'], dtype=np.int8),
                )
        """
    )
)

cells.append(
    code(
        """
        # =============================
        # Output index
        # =============================
        print('Output directory:')
        print(' ', os.path.abspath(OUT_DIR))
        print('')
        for fname in sorted(os.listdir(OUT_DIR)):
            print(' ', fname)
        """
    )
)

nb["cells"] = cells
OUT_PATH.write_text(nbf.writes(nb))
print(f"Wrote notebook to: {OUT_PATH}")
