from pathlib import Path
import inspect

import nbformat as nbf


NOTEBOOK_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-ROI_RPECat_Memory_InteractionVariance.ipynb"
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
        # ROI-Wide RPE Category x Memory Interaction Variance

        This notebook quantifies, for each cue-on HFA electrode in a fixed overlap window,
        how much variance is uniquely explained by the full:

        `RPE_cat x memory_outcome`

        interaction block.

        The workflow:

        - uses the exact cue overlap window `(0.37, 0.60)`,
        - analyzes canonical ROIs (`ACC`, `dmPFC`, `dlPFC`, `OFC`, `HPC`, `AMY`),
        - computes one per-electrode metric as interaction-block partial R²,
        - and then tests whether that metric differs by region with a participant-random-intercept mixed model.
        """
    ),
    code(
        """
        import os
        import re
        import warnings
        from itertools import combinations
        from pathlib import Path

        import matplotlib.pyplot as plt
        import mne
        import nbformat
        import numpy as np
        import pandas as pd
        import patsy
        import seaborn as sns
        import statsmodels.formula.api as smf
        from IPython.display import display
        from scipy import stats
        from statsmodels.regression.linear_model import OLS
        from statsmodels.stats.multitest import multipletests
        from tqdm.auto import tqdm

        from plot_config import set_plot_style
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
        ephys_root = base_dir / 'projects/guLab/Salman/EphysAnalyses'

        analysis_name = 'roi_rpecat_memory_interaction_variance'
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

        RPE_MEMORY_OVERLAP_WINDOW = (0.37, 0.60)
        event = 'cue_on'
        condition = 'Day1'
        target_rois = ['ACC', 'dmPFC', 'dlPFC', 'OFC', 'HPC', 'AMY']
        target_fband = 'hfa'
        min_trials = 15
        min_region_electrodes = 10
        min_region_subjects = 3

        reduced_formula = (
            "tfr ~ 1 + "
            "C(rpe_cat, Treatment('neutral')) + "
            "C(memory_outcome, Treatment('miss'))"
        )
        full_formula = (
            "tfr ~ 1 + "
            "C(rpe_cat, Treatment('neutral')) * "
            "C(memory_outcome, Treatment('miss'))"
        )

        results_csv = analysis_root / 'roi_rpecat_memory_interaction_partial_r2_electrode_metrics.csv'
        subject_summary_csv = analysis_root / 'roi_rpecat_memory_interaction_subject_summary.csv'
        region_summary_csv = analysis_root / 'roi_rpecat_memory_interaction_region_summary.csv'
        omnibus_csv = analysis_root / 'roi_rpecat_memory_interaction_region_omnibus.csv'
        pairwise_contrasts_csv = analysis_root / 'roi_rpecat_memory_interaction_region_pairwise_contrasts.csv'
        region_figure_pdf = figure_root / 'roi_rpecat_memory_interaction_partial_r2_by_region.pdf'
        region_figure_png = figure_root / 'roi_rpecat_memory_interaction_partial_r2_by_region.png'

        print('analysis_root:', analysis_root)
        print('target_rois:', target_rois)
        print('RPE_MEMORY_OVERLAP_WINDOW:', RPE_MEMORY_OVERLAP_WINDOW)
        """
    ),
    code(
        """
        def extract_cluster_time_range(stats_path: str, cue_label: str = 'Cue time range', time_label: str = 'Time range'):
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

        if not np.allclose(overlap_window, RPE_MEMORY_OVERLAP_WINDOW):
            raise ValueError(
                f'Parsed overlap {overlap_window} does not match expected {RPE_MEMORY_OVERLAP_WINDOW}'
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
                    'metric': 'Overlap used here',
                    'window_start_s': overlap_window[0],
                    'window_end_s': overlap_window[1],
                },
            ]
        )

        print('Shared subjects:', len(shared_subjects))
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
            (combined_df['condition'] == condition)
        ].copy()

        expected_combined_cols = [
            'participant',
            'condition',
            'trials_mem',
            'trials_gamble',
            'memory_outcome',
            'rpe_cat',
        ]
        missing_combined_cols = [c for c in expected_combined_cols if c not in combined_df.columns]
        if missing_combined_cols:
            raise ValueError(
                f'preprocess_combined_df did not provide required columns: {missing_combined_cols}'
            )

        expected_learn_cols = ['participant', 'rpe_cat']
        missing_learn_cols = [c for c in expected_learn_cols if c not in learn_df.columns]
        if missing_learn_cols:
            raise ValueError(
                f'preprocess_learn_df did not provide required columns: {missing_learn_cols}'
            )

        print('Behavior rows after filtering shared subjects:', len(combined_df))
        display(
            combined_df[
                ['participant', 'condition', 'trials_mem', 'trials_gamble', 'memory_outcome', 'rpe_cat']
            ].head()
        )
        """
    ),
    md("## Export or discover exact-window cue-on HFA CSVs for the canonical ROIs"),
    code(
        """
        def normalize_rpe_category(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip().lower()
            mapping = {
                'neg': 'negative',
                'negative': 'negative',
                'neu': 'neutral',
                'neutral': 'neutral',
                'pos': 'positive',
                'positive': 'positive',
            }
            return mapping.get(text, np.nan)


        def build_memory_metadata_for_subject(subject: str) -> pd.DataFrame:
            memory_df = (
                combined_df[
                    (combined_df['participant'] == subject) &
                    (combined_df['condition'] == condition)
                ]
                .dropna(subset=['trials_mem'])
                .sort_values('trials_mem')
                .reset_index(drop=True)
                .copy()
            )
            if len(memory_df) == 0:
                raise ValueError(f'No Day1 memory metadata found for {subject}')
            memory_df['rpe_cat'] = memory_df['rpe_cat'].apply(normalize_rpe_category)
            return memory_df


        def discover_target_channels(subject: str) -> pd.DataFrame:
            elec_path = ephys_root / subject / 'Day1_reref_elec_df'
            elec_df = pd.read_csv(elec_path)
            if 'salman_region' not in elec_df.columns:
                raise ValueError(f'salman_region missing in {elec_path}')
            roi_df = elec_df[elec_df['salman_region'].isin(target_rois)].copy()
            roi_df = roi_df[['label', 'salman_region']].drop_duplicates().reset_index(drop=True)
            return roi_df


        def export_exact_overlap_csvs_for_subject(subject: str, overwrite: bool = False):
            tfr_path = ephys_root / subject / 'scratch' / 'TFR' / f'{event}-tfr.h5'
            dfs_dir = ephys_root / subject / 'scratch' / 'TFR' / event / 'dfs'
            dfs_dir.mkdir(parents=True, exist_ok=True)

            roi_chan_df = discover_target_channels(subject)
            if len(roi_chan_df) == 0:
                return []

            out_paths = [
                dfs_dir / f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW[1]:.2f}.csv"
                for chan in roi_chan_df['label'].tolist()
            ]
            if (not overwrite) and all(p.exists() for p in out_paths):
                return out_paths

            tfr_obj = mne.time_frequency.read_tfrs(str(tfr_path))
            power_epochs = tfr_obj[0] if isinstance(tfr_obj, list) else tfr_obj
            metadata = build_memory_metadata_for_subject(subject)
            power_epochs.metadata = metadata

            created_paths = []
            for row in tqdm(
                roi_chan_df.itertuples(index=False),
                total=len(roi_chan_df),
                desc=f'Exporting {subject}',
                leave=False,
            ):
                chan = str(row.label)
                roi = str(row.salman_region)
                out_path = dfs_dir / f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW[1]:.2f}.csv"
                if out_path.exists() and not overwrite:
                    created_paths.append(out_path)
                    continue

                pow_df_whole = power_epochs.copy().pick([chan]).to_data_frame()
                pow_df = pow_df_whole[
                    (pow_df_whole['time'] > RPE_MEMORY_OVERLAP_WINDOW[0]) &
                    (pow_df_whole['time'] <= RPE_MEMORY_OVERLAP_WINDOW[1])
                ].copy()
                if len(pow_df) == 0:
                    continue

                pow_df['fband'] = pow_df['freq'].apply(
                    lambda x: 'low_theta' if x < 4 else
                    'high_theta' if (x >= 4) and (x < 8) else
                    'alpha' if (x >= 8) and (x < 13) else
                    'beta' if (x >= 13) and (x < 30) else
                    'gamma' if (x >= 30) and (x < 70) else
                    'hfa'
                )

                pow_df = pow_df.groupby(['fband', 'epoch']).mean(numeric_only=True).reset_index()
                tfr_df = (
                    pow_df.drop(columns=['time', 'freq'])
                    .rename(columns={'epoch': 'trial', chan: 'tfr'})
                )

                metadata_to_merge = power_epochs.metadata.copy()
                metadata_to_merge['trial'] = tfr_df['trial'].unique()
                tfr_df = tfr_df.merge(metadata_to_merge, on=['trial'], how='left')

                tfr_df.insert(0, 'channel', chan)
                tfr_df.insert(0, 'region', roi)
                tfr_df['event'] = event
                tfr_df.to_csv(out_path, index=False)
                created_paths.append(out_path)

            return created_paths


        export_rows = []
        all_exported_paths = []
        for subject in shared_subjects:
            roi_chan_df = discover_target_channels(subject)
            roi_counts = (
                roi_chan_df['salman_region'].value_counts().reindex(target_rois, fill_value=0).to_dict()
                if len(roi_chan_df) > 0 else {roi: 0 for roi in target_rois}
            )
            export_rows.append({
                'Subject': subject,
                **{f'n_{roi}_electrodes': int(roi_counts.get(roi, 0)) for roi in target_rois},
                'n_total_target_electrodes': int(len(roi_chan_df)),
            })
            if len(roi_chan_df) == 0:
                continue
            all_exported_paths.extend(export_exact_overlap_csvs_for_subject(subject))

        subject_export_df = pd.DataFrame(export_rows).sort_values('Subject').reset_index(drop=True)
        subject_export_df.to_csv(subject_summary_csv, index=False)
        print('Exported or discovered exact-window csvs:', len(all_exported_paths))
        display(subject_export_df.head())
        """
    ),
    md("## Load exact-window HFA rows across ROIs"),
    code(
        """
        def load_exact_window_rows(subjects):
            rows = []
            missing = []
            for subject in subjects:
                roi_chan_df = discover_target_channels(subject)
                for row in roi_chan_df.itertuples(index=False):
                    chan = str(row.label)
                    roi = str(row.salman_region)
                    csv_path = (
                        ephys_root / subject / 'scratch' / 'TFR' / event / 'dfs' /
                        f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW[1]:.2f}.csv"
                    )
                    if not csv_path.exists():
                        missing.append({
                            'Subject': subject,
                            'channel': chan,
                            'region': roi,
                            'reason': 'missing_exact_window_csv',
                        })
                        continue

                    df = pd.read_csv(csv_path)
                    df = df[df['fband'] == target_fband].copy()
                    if len(df) == 0:
                        missing.append({
                            'Subject': subject,
                            'channel': chan,
                            'region': roi,
                            'reason': 'no_hfa_rows',
                        })
                        continue

                    df['Subject'] = subject
                    df['electrode'] = chan
                    df['channel'] = chan
                    df['region'] = roi
                    rows.append(df)

            merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            missing_df = pd.DataFrame(missing)
            return merged, missing_df


        exact_df, missing_exact_df = load_exact_window_rows(shared_subjects)
        if len(missing_exact_df) > 0:
            print('Missing exact-window items:')
            display(missing_exact_df.head())

        if len(exact_df) == 0:
            raise ValueError('No exact-window rows were loaded.')

        exact_df = exact_df[
            exact_df['memory_outcome'].isin(['hits', 'misses'])
        ].copy()
        exact_df['memory_outcome'] = exact_df['memory_outcome'].map(
            {'hits': 'hit', 'misses': 'miss'}
        )
        exact_df['rpe_cat'] = exact_df['rpe_cat'].apply(normalize_rpe_category)
        exact_df = exact_df[
            exact_df['region'].isin(target_rois) &
            exact_df['memory_outcome'].isin(['hit', 'miss']) &
            exact_df['rpe_cat'].isin(['negative', 'neutral', 'positive'])
        ].copy()

        exact_df['participant'] = exact_df['Subject']
        exact_df['unique_elec_id'] = (
            exact_df[['Subject', 'electrode']]
            .astype(str)
            .agg('::'.join, axis=1)
        )
        exact_df['rpe_cat'] = pd.Categorical(
            exact_df['rpe_cat'],
            categories=['negative', 'neutral', 'positive'],
            ordered=True,
        )
        exact_df['memory_outcome'] = pd.Categorical(
            exact_df['memory_outcome'],
            categories=['miss', 'hit'],
            ordered=True,
        )

        required_cols = ['participant', 'electrode', 'channel', 'region', 'fband', 'tfr', 'rpe_cat', 'memory_outcome']
        missing_cols = [c for c in required_cols if c not in exact_df.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns in exact-window dataframe: {missing_cols}')

        print('Exact-window HFA rows:', len(exact_df))
        print('Unique electrodes:', exact_df['unique_elec_id'].nunique())
        display(
            exact_df[
                ['participant', 'region', 'electrode', 'trial', 'tfr', 'rpe_cat', 'memory_outcome']
            ].head()
        )
        """
    ),
    md("## Per-electrode interaction-block partial R²"),
    code(
        """
        def fit_interaction_block_models(data: pd.DataFrame):
            model_df = data.copy()
            model_df = model_df.replace([np.inf, -np.inf], np.nan)
            model_df = model_df.dropna(subset=['tfr', 'rpe_cat', 'memory_outcome'])

            y_reduced, X_reduced = patsy.dmatrices(reduced_formula, model_df, return_type='dataframe')
            y_full, X_full = patsy.dmatrices(full_formula, model_df, return_type='dataframe')

            reduced_model = OLS(y_reduced, X_reduced).fit()
            full_model = OLS(y_full, X_full).fit()

            sse_reduced = float(reduced_model.ssr)
            sse_full = float(full_model.ssr)
            if sse_reduced <= 0:
                partial_r2 = np.nan
            else:
                partial_r2 = float(max(sse_reduced - sse_full, 0.0) / sse_reduced)

            df_num = int(X_full.shape[1] - X_reduced.shape[1])
            df_den = int(full_model.df_resid)
            if (df_num <= 0) or (df_den <= 0) or (sse_full < 0):
                f_stat = np.nan
                p_value = np.nan
            else:
                ms_num = max(sse_reduced - sse_full, 0.0) / df_num
                ms_den = sse_full / df_den if df_den > 0 else np.nan
                f_stat = float(ms_num / ms_den) if np.isfinite(ms_den) and ms_den > 0 else np.nan
                p_value = float(stats.f.sf(f_stat, df_num, df_den)) if np.isfinite(f_stat) else np.nan

            return {
                'reduced_model': reduced_model,
                'full_model': full_model,
                'sse_reduced': sse_reduced,
                'sse_full': sse_full,
                'reduced_r2': float(reduced_model.rsquared),
                'full_r2': float(full_model.rsquared),
                'partial_r2_interaction': partial_r2,
                'block_f_stat': f_stat,
                'block_p_value': p_value,
                'block_df_num': df_num,
                'block_df_den': df_den,
            }


        required_count_cols = [
            f'{rpe_cat}_{memory_outcome}'
            for rpe_cat in ['negative', 'neutral', 'positive']
            for memory_outcome in ['hit', 'miss']
        ]

        result_rows = []
        skipped_rows = []
        sanity_result = None
        grouped = exact_df.groupby('unique_elec_id')
        for elec_id, elec_df in tqdm(
            grouped,
            total=exact_df['unique_elec_id'].nunique(),
            desc='Electrodes',
        ):
            elec_df = elec_df.copy()
            count_map = (
                elec_df.groupby(['rpe_cat', 'memory_outcome'])
                .size()
                .to_dict()
            )
            count_map = {
                f'{str(k1)}_{str(k2)}': int(v)
                for (k1, k2), v in count_map.items()
            }

            if len(elec_df) < min_trials:
                skipped_rows.append({
                    'electrode_id': elec_id,
                    'participant': str(elec_df['participant'].iloc[0]),
                    'region': str(elec_df['region'].iloc[0]),
                    'electrode': str(elec_df['electrode'].iloc[0]),
                    'reason': 'too_few_trials',
                    'n_trials': int(len(elec_df)),
                })
                continue

            if any(count_map.get(col, 0) < 1 for col in required_count_cols):
                skipped_rows.append({
                    'electrode_id': elec_id,
                    'participant': str(elec_df['participant'].iloc[0]),
                    'region': str(elec_df['region'].iloc[0]),
                    'electrode': str(elec_df['electrode'].iloc[0]),
                    'reason': 'missing_rpecat_memory_cell',
                    'n_trials': int(len(elec_df)),
                })
                continue

            fit = fit_interaction_block_models(elec_df)
            row = {
                'participant': str(elec_df['participant'].iloc[0]),
                'region': str(elec_df['region'].iloc[0]),
                'electrode': str(elec_df['electrode'].iloc[0]),
                'channel': str(elec_df['channel'].iloc[0]),
                'n_trials': int(len(elec_df)),
                'reduced_r2': fit['reduced_r2'],
                'full_r2': fit['full_r2'],
                'partial_r2_interaction': fit['partial_r2_interaction'],
                'block_f_stat': fit['block_f_stat'],
                'block_p_value': fit['block_p_value'],
                'block_df_num': fit['block_df_num'],
                'block_df_den': fit['block_df_den'],
                'sse_reduced': fit['sse_reduced'],
                'sse_full': fit['sse_full'],
                **{col: int(count_map.get(col, 0)) for col in required_count_cols},
            }
            result_rows.append(row)

            if sanity_result is None:
                sanity_result = {
                    'electrode_id': elec_id,
                    **row,
                }

        results_df = pd.DataFrame(result_rows).sort_values(
            ['region', 'participant', 'electrode']
        ).reset_index(drop=True)
        skipped_df = pd.DataFrame(skipped_rows)

        if len(results_df) == 0:
            raise ValueError('No electrode-level partial R² results were produced.')

        results_df.to_csv(results_csv, index=False)
        print('Saved results_csv:', results_csv)
        display(results_df.head())
        print('Analyzed electrodes:', len(results_df))
        if len(skipped_df) > 0:
            print('Skipped electrodes:')
            display(skipped_df.head())
        """
    ),
    md("## Sanity check"),
    code(
        """
        if sanity_result is None:
            raise ValueError('Sanity result was not captured.')

        sanity_df = pd.DataFrame([sanity_result])[
            [
                'electrode_id',
                'participant',
                'region',
                'electrode',
                'n_trials',
                'reduced_r2',
                'full_r2',
                'partial_r2_interaction',
                'block_f_stat',
                'block_p_value',
                'block_df_num',
                'block_df_den',
                'sse_reduced',
                'sse_full',
            ]
        ]
        display(sanity_df)
        """
    ),
    md("## Region-level summaries and mixed-effects comparison"),
    code(
        """
        region_summary_df = (
            results_df.groupby('region')
            .agg(
                n_electrodes=('electrode', 'count'),
                n_subjects=('participant', 'nunique'),
                mean_partial_r2=('partial_r2_interaction', 'mean'),
                median_partial_r2=('partial_r2_interaction', 'median'),
                sd_partial_r2=('partial_r2_interaction', 'std'),
                sem_partial_r2=('partial_r2_interaction', lambda x: np.nanstd(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
                mean_full_r2=('full_r2', 'mean'),
                mean_reduced_r2=('reduced_r2', 'mean'),
            )
            .reset_index()
        )
        region_summary_df['included_in_mixedlm'] = (
            (region_summary_df['n_electrodes'] >= min_region_electrodes) &
            (region_summary_df['n_subjects'] >= min_region_subjects)
        )
        region_summary_df = region_summary_df.sort_values('mean_partial_r2', ascending=False).reset_index(drop=True)
        region_summary_df.to_csv(region_summary_csv, index=False)
        print('Saved region_summary_csv:', region_summary_csv)
        display(region_summary_df)

        included_regions = [
            roi for roi in target_rois
            if roi in region_summary_df.loc[region_summary_df['included_in_mixedlm'], 'region'].tolist()
        ]
        mixedlm_df = results_df[results_df['region'].isin(included_regions)].copy()
        mixedlm_df['region'] = pd.Categorical(mixedlm_df['region'], categories=included_regions, ordered=True)

        if len(included_regions) < 2:
            raise ValueError(
                'Fewer than two regions met mixed-model inclusion thresholds. '
                f'Included regions: {included_regions}'
            )

        baseline_region = 'ACC' if 'ACC' in included_regions else included_regions[0]

        null_mixedlm = smf.mixedlm(
            'partial_r2_interaction ~ 1',
            data=mixedlm_df,
            groups='participant',
        ).fit(reml=False, method='lbfgs')
        full_mixedlm = smf.mixedlm(
            f"partial_r2_interaction ~ C(region, Treatment('{baseline_region}'))",
            data=mixedlm_df,
            groups='participant',
        ).fit(reml=False, method='lbfgs')

        lr_stat = float(2 * (full_mixedlm.llf - null_mixedlm.llf))
        lr_df = int(len(full_mixedlm.fe_params) - 1)
        lr_p_value = float(stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan
        omnibus_df = pd.DataFrame(
            [{
                'metric': 'partial_r2_interaction',
                'baseline_region': baseline_region,
                'included_regions': ','.join(included_regions),
                'n_electrodes': int(len(mixedlm_df)),
                'n_subjects': int(mixedlm_df['participant'].nunique()),
                'lr_stat': lr_stat,
                'lr_df': lr_df,
                'lr_p_value': lr_p_value,
            }]
        )
        omnibus_df.to_csv(omnibus_csv, index=False)
        print('Saved omnibus_csv:', omnibus_csv)
        display(omnibus_df)

        fe_params = full_mixedlm.fe_params.copy()
        fe_names = list(fe_params.index)
        fe_cov = full_mixedlm.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
        fe_beta = fe_params.to_numpy(dtype=float)
        coef_index = {name: idx for idx, name in enumerate(fe_names)}

        def make_contrast(weights):
            vec = np.zeros(len(fe_names), dtype=float)
            for name, value in weights.items():
                vec[coef_index[name]] = value
            return vec

        def linear_contrast(contrast_vec):
            est = float(contrast_vec @ fe_beta)
            se = float(np.sqrt(max(contrast_vec @ fe_cov @ contrast_vec, 0.0)))
            z_val = est / se if se > 0 else np.nan
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_val)))) if np.isfinite(z_val) else np.nan
            ci_low = est - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = est + 1.96 * se if np.isfinite(se) else np.nan
            return {
                'estimate': est,
                'se': se,
                'z': z_val,
                'p_value': p_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
            }

        def region_mean_weights(region):
            weights = {'Intercept': 1.0}
            if region != baseline_region:
                coef_name = f"C(region, Treatment('{baseline_region}'))[T.{region}]"
                weights[coef_name] = 1.0
            return weights

        pairwise_rows = []
        for region_a, region_b in combinations(included_regions, 2):
            weights_a = region_mean_weights(region_a)
            weights_b = region_mean_weights(region_b)
            contrast_weights = {}
            for key in set(weights_a).union(weights_b):
                contrast_weights[key] = weights_a.get(key, 0.0) - weights_b.get(key, 0.0)
            contrast_vec = make_contrast(contrast_weights)
            out = linear_contrast(contrast_vec)
            pairwise_rows.append({
                'region_a': region_a,
                'region_b': region_b,
                **out,
            })

        pairwise_df = pd.DataFrame(pairwise_rows)
        if len(pairwise_df) > 0:
            reject, p_fdr, _, _ = multipletests(pairwise_df['p_value'].to_numpy(dtype=float), method='fdr_bh')
            pairwise_df['p_value_fdr'] = p_fdr
            pairwise_df['reject_fdr'] = reject
            pairwise_df = pairwise_df.sort_values('p_value').reset_index(drop=True)
        pairwise_df.to_csv(pairwise_contrasts_csv, index=False)
        print('Saved pairwise_contrasts_csv:', pairwise_contrasts_csv)
        display(pairwise_df)
        """
    ),
    md("## ROI distribution figure"),
    code(
        """
        subject_plot_df = (
            mixedlm_df.groupby(['participant', 'region'], as_index=False, observed=True)
            .agg(
                partial_r2_interaction=('partial_r2_interaction', 'mean')
            )
        )
        plot_order = included_regions

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, gridspec_kw={'width_ratios': [1.5, 1.0]})

        sns.barplot(
            data=subject_plot_df,
            x='region',
            y='partial_r2_interaction',
            order=plot_order,
            color='0.75',
            errorbar='se',
            ax=axes[0],
        )
        sns.stripplot(
            data=subject_plot_df,
            x='region',
            y='partial_r2_interaction',
            order=plot_order,
            color='black',
            alpha=0.45,
            jitter=0.22,
            ax=axes[0],
        )
        axes[0].set(
            title='Subject-mean partial R² by ROI',
            xlabel='ROI',
            ylabel='Interaction partial R²',
        )
        axes[0].tick_params(axis='x', rotation=25)

        summary_plot_df = (
            subject_plot_df.groupby('region', as_index=False, observed=True)
            .agg(
                mean_partial_r2=('partial_r2_interaction', 'mean'),
                sem_partial_r2=('partial_r2_interaction', lambda x: np.nanstd(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            )
        )
        summary_plot_df['region'] = pd.Categorical(summary_plot_df['region'], categories=plot_order, ordered=True)
        summary_plot_df = summary_plot_df.sort_values('region')
        axes[1].errorbar(
            x=np.arange(len(summary_plot_df)),
            y=summary_plot_df['mean_partial_r2'].to_numpy(dtype=float),
            yerr=summary_plot_df['sem_partial_r2'].to_numpy(dtype=float),
            fmt='o',
            color='black',
            ecolor='black',
            elinewidth=1.3,
            capsize=4,
        )
        axes[1].set(
            title='ROI mean +/- SEM',
            xlabel='ROI',
            ylabel='Mean partial R²',
        )
        axes[1].set_xticks(np.arange(len(summary_plot_df)))
        axes[1].set_xticklabels(summary_plot_df['region'].tolist(), rotation=25)

        for ax in axes:
            sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(region_figure_pdf, bbox_inches='tight')
        fig.savefig(region_figure_png, bbox_inches='tight')
        display(fig)
        plt.close(fig)

        print('Saved region_figure_pdf:', region_figure_pdf)
        print('Saved region_figure_png:', region_figure_png)
        """
    ),
    md("## Final checks"),
    code(
        """
        print('RPE_MEMORY_OVERLAP_WINDOW:', RPE_MEMORY_OVERLAP_WINDOW)
        print('Target ROIs:', target_rois)
        print('Electrode result rows:', len(results_df))
        print('Unique analyzed participants:', results_df['participant'].nunique())
        print('Unique analyzed electrodes:', results_df[['participant', 'electrode']].drop_duplicates().shape[0])
        print('Mixed-model included regions:', included_regions)
        print('Any duplicate participant/electrode rows:', results_df.duplicated(['participant', 'electrode']).any())
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
