from pathlib import Path
import inspect

import nbformat as nbf


NOTEBOOK_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-ACC_RPE_Memory_Overlap_ElectrodeRegression.ipynb"
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
        # ACC RPE-Memory Overlap Electrode Regression

        This notebook defines the ACC cue-period overlap between:

        - ACC RPE reinstatement
        - ACC hit decoding

        It then regenerates exact cue-on ACC HFA averages for that overlap window and runs
        per-electrode regressions of:

        `HFA ~ zrpe * memory_outcome`
        """
    ),
    code(
        """
        import os
        import re
        import warnings
        from pathlib import Path

        import matplotlib.pyplot as plt
        import mne
        import nbformat
        import numpy as np
        import pandas as pd
        import patsy
        import seaborn as sns
        from matplotlib.backends.backend_pdf import PdfPages
        from IPython.display import display
        from scipy import stats
        from statsmodels.regression.linear_model import OLS
        from tqdm.auto import tqdm

        from plot_config import set_plot_style
        from mv_granger_utils import (
            load_behavior_tables,
            preprocess_combined_df,
            preprocess_learn_df,
        )
        from mv_regression_utils import Subject

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

        analysis_name = 'acc_rpe_memory_overlap_electrode_regression'
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

        RPE_REINSTATEMENT_CUE_WINDOW_ACC = (0.31, 0.72)
        HIT_DECODING_CUE_WINDOW_ACC = (0.37, 0.60)
        RPE_MEMORY_OVERLAP_WINDOW_ACC = (0.37, 0.60)

        event = 'cue_on'
        condition = 'Day1'
        target_roi = 'ACC'
        target_fband = 'hfa'
        n_permutations = 1000
        random_state = 42
        min_trials = 15
        hfa_freq_range = (70.0, 201.0)
        hfa_downsample_factor = 5
        hfa_win_ms = 200
        hfa_smooth_method = 'movmean'
        hfa_artifact_abs_thresh = 10.0

        results_csv = analysis_root / 'acc_rpe_memory_overlap_electrode_regression_results.csv'
        subject_summary_csv = analysis_root / 'acc_rpe_memory_overlap_subject_summary.csv'
        overlap_figure_pdf = figure_root / 'acc_rpe_memory_overlap_zscore_distribution.pdf'
        overlap_figure_png = figure_root / 'acc_rpe_memory_overlap_zscore_distribution.png'
        top_electrodes_csv = analysis_root / 'acc_rpe_memory_overlap_top10_absolute_interaction_electrodes.csv'
        top_electrode_pdf_dir = figure_root / 'top10_absolute_interaction_single_trial_hfa_pdfs'
        top_electrode_pdf_dir.mkdir(parents=True, exist_ok=True)

        rpe_category_order = ['positive', 'neutral', 'negative']
        memory_outcome_order = ['hit', 'miss']
        n_plot_trials = 3

        print('analysis_root:', analysis_root)
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', RPE_MEMORY_OVERLAP_WINDOW_ACC)
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


        rpe_cluster_window, rpe_stats_text = extract_cluster_time_range(
            acc_reinstatement_stats_txt,
            cue_label='Cue time range',
        )
        hit_cluster_window, hit_stats_text = extract_cluster_time_range(
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
        print('RPE reinstatement cue cluster:', rpe_cluster_window)
        print('Hit decoding cue cluster:', hit_cluster_window)
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', overlap_window)
        display(overlap_df)
        display(acc_reinstatement_tbl.sort_values('Subject').reset_index(drop=True))
        display(acc_hit_tbl.sort_values('Subject').reset_index(drop=True))
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
            'phit',
            'zrpe',
        ]
        missing_combined_cols = [c for c in expected_combined_cols if c not in combined_df.columns]
        if missing_combined_cols:
            raise ValueError(
                f'preprocess_combined_df did not provide required columns: {missing_combined_cols}'
            )

        expected_learn_cols = ['participant', 'zrpe', 'rpe_cat']
        missing_learn_cols = [c for c in expected_learn_cols if c not in learn_df.columns]
        if missing_learn_cols:
            raise ValueError(
                f'preprocess_learn_df did not provide required columns: {missing_learn_cols}'
            )

        print('Behavior rows after filtering shared subjects:', len(combined_df))
        display(
            combined_df[
                ['participant', 'condition', 'trials_mem', 'trials_gamble', 'memory_outcome', 'zrpe', 'phit']
            ].head()
        )
        """
    ),
    md("## Discover ACC channels and regenerate exact overlap-window cue-on HFA averages"),
    code(
        """
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
            return memory_df


        def discover_acc_channels(subject: str):
            elec_path = ephys_root / subject / 'Day1_reref_elec_df'
            elec_df = pd.read_csv(elec_path)
            if 'salman_region' not in elec_df.columns:
                raise ValueError(f'salman_region missing in {elec_path}')
            acc_df = elec_df[elec_df['salman_region'] == target_roi].copy()
            return acc_df


        def export_exact_overlap_csvs_for_subject(subject: str, overwrite: bool = False):
            tfr_path = ephys_root / subject / 'scratch' / 'TFR' / f'{event}-tfr.h5'
            dfs_dir = ephys_root / subject / 'scratch' / 'TFR' / event / 'dfs'
            dfs_dir.mkdir(parents=True, exist_ok=True)

            acc_elec_df = discover_acc_channels(subject)
            if len(acc_elec_df) == 0:
                return []

            out_paths = [
                dfs_dir / f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW_ACC[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW_ACC[1]:.2f}.csv"
                for chan in acc_elec_df['label'].tolist()
            ]
            if (not overwrite) and all(p.exists() for p in out_paths):
                return out_paths

            tfr_obj = mne.time_frequency.read_tfrs(str(tfr_path))
            power_epochs = tfr_obj[0] if isinstance(tfr_obj, list) else tfr_obj
            metadata = build_memory_metadata_for_subject(subject)
            power_epochs.metadata = metadata

            created_paths = []
            for chan in tqdm(acc_elec_df['label'].tolist(), desc=f'Exporting {subject}', leave=False):
                out_path = dfs_dir / f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW_ACC[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW_ACC[1]:.2f}.csv"
                if out_path.exists() and not overwrite:
                    created_paths.append(out_path)
                    continue

                pow_df_whole = power_epochs.copy().pick([chan]).to_data_frame()
                pow_df = pow_df_whole[
                    (pow_df_whole['time'] > RPE_MEMORY_OVERLAP_WINDOW_ACC[0]) &
                    (pow_df_whole['time'] <= RPE_MEMORY_OVERLAP_WINDOW_ACC[1])
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
                tfr_df.insert(0, 'region', target_roi)
                tfr_df['event'] = event
                tfr_df.to_csv(out_path, index=False)
                created_paths.append(out_path)

            return created_paths


        export_rows = []
        all_exported_paths = []
        for subject in shared_subjects:
            acc_elec_df = discover_acc_channels(subject)
            export_rows.append({
                'Subject': subject,
                'n_acc_electrodes': int(len(acc_elec_df)),
            })
            if len(acc_elec_df) == 0:
                continue
            all_exported_paths.extend(export_exact_overlap_csvs_for_subject(subject))

        subject_export_df = pd.DataFrame(export_rows).sort_values('Subject').reset_index(drop=True)
        subject_export_df.to_csv(subject_summary_csv, index=False)
        print('Exported or discovered exact-window ACC csvs:', len(all_exported_paths))
        display(subject_export_df)
        """
    ),
    md("## Load exact-window ACC HFA rows and prepare electrode-level regression data"),
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


        def load_exact_window_acc_rows(subjects):
            rows = []
            missing = []
            for subject in subjects:
                acc_elec_df = discover_acc_channels(subject)
                for chan in acc_elec_df['label'].tolist():
                    csv_path = (
                        ephys_root / subject / 'scratch' / 'TFR' / event / 'dfs' /
                        f"{chan}_time_averaged_df_time{RPE_MEMORY_OVERLAP_WINDOW_ACC[0]:.2f}-{RPE_MEMORY_OVERLAP_WINDOW_ACC[1]:.2f}.csv"
                    )
                    if not csv_path.exists():
                        missing.append({'Subject': subject, 'channel': chan, 'reason': 'missing_exact_window_csv'})
                        continue
                    df = pd.read_csv(csv_path)
                    df = df[df['fband'] == target_fband].copy()
                    if len(df) == 0:
                        missing.append({'Subject': subject, 'channel': chan, 'reason': 'no_hfa_rows'})
                        continue

                    df['Subject'] = subject
                    df['electrode'] = chan
                    df['channel'] = chan
                    rows.append(df)

            merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            missing_df = pd.DataFrame(missing)
            return merged, missing_df


        acc_exact_df, missing_exact_df = load_exact_window_acc_rows(shared_subjects)
        if len(missing_exact_df) > 0:
            print('Missing exact-window items:')
            display(missing_exact_df.head())

        if len(acc_exact_df) == 0:
            raise ValueError('No ACC exact-window rows were loaded.')

        acc_exact_df = acc_exact_df[
            acc_exact_df['memory_outcome'].isin(['hits', 'misses'])
        ].copy()
        acc_exact_df['memory_outcome'] = acc_exact_df['memory_outcome'].map(
            {'hits': 'hit', 'misses': 'miss'}
        )
        acc_exact_df['rpe_cat'] = acc_exact_df['rpe_cat'].apply(normalize_rpe_category)
        acc_exact_df['phit'] = acc_exact_df['memory_outcome'].map({'hit': 0.5, 'miss': -0.5})

        acc_exact_df['participant'] = acc_exact_df['Subject']
        acc_exact_df['unique_elec_id'] = (
            acc_exact_df[['Subject', 'electrode']]
            .astype(str)
            .agg('::'.join, axis=1)
        )

        required_cols = ['participant', 'electrode', 'channel', 'region', 'fband', 'tfr', 'zrpe', 'phit']
        missing_cols = [c for c in required_cols if c not in acc_exact_df.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns in exact-window dataframe: {missing_cols}')

        print('ACC exact-window HFA rows:', len(acc_exact_df))
        print('ACC electrodes:', acc_exact_df['unique_elec_id'].nunique())
        display(
            acc_exact_df[
                ['participant', 'electrode', 'trial', 'tfr', 'zrpe', 'rpe_cat', 'memory_outcome', 'phit']
            ].head()
        )
        """
    ),
    md("## Electrode-level permutation regression"),
    code(
        """
        def permutation_interaction_regression(
            data: pd.DataFrame,
            formula: str = 'tfr ~ 1 + zrpe + phit + zrpe:phit',
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
                'null_betas': null_betas,
            }


        regression_rows = []
        sanity_result = None
        for elec_id, elec_df in tqdm(
            acc_exact_df.groupby('unique_elec_id'),
            total=acc_exact_df['unique_elec_id'].nunique(),
            desc='ACC electrodes',
        ):
            elec_df = elec_df.copy()
            elec_df = elec_df.replace([np.inf, -np.inf], np.nan)
            elec_df = elec_df.dropna(subset=['tfr', 'zrpe', 'phit'])

            if len(elec_df) < min_trials:
                continue
            if elec_df['phit'].nunique() < 2:
                continue
            if elec_df['zrpe'].nunique() < 2:
                continue

            result = permutation_interaction_regression(
                elec_df,
                n_permutations=n_permutations,
                random_state=random_state,
            )

            row = {
                'participant': str(elec_df['participant'].iloc[0]),
                'electrode': str(elec_df['electrode'].iloc[0]),
                'channel': str(elec_df['channel'].iloc[0]),
                'region': str(elec_df['region'].iloc[0]),
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
                sanity_result = {
                    'electrode_id': elec_id,
                    **row,
                }

        results_df = pd.DataFrame(regression_rows).sort_values(
            ['participant', 'electrode']
        ).reset_index(drop=True)
        if len(results_df) == 0:
            raise ValueError('No electrode-level regression results were produced.')

        display(results_df.head())
        print('Analyzed ACC electrodes:', len(results_df))
        """
    ),
    md("## Sanity check"),
    code(
        """
        if sanity_result is None:
            raise ValueError('Sanity result was not captured.')

        sanity_df = pd.DataFrame([sanity_result])[
            ['electrode_id', 'participant', 'electrode', 'n_trials', 'raw_weight', 'null_mean', 'null_std', 'z_scored_weight', 'p_value']
        ]
        print('Sanity-check electrode regression:')
        display(sanity_df)
        """
    ),
    md("## Save results and plot ACC z-scored interaction-weight distribution"),
    code(
        """
        if results_df.duplicated(['participant', 'electrode']).any():
            raise ValueError('Duplicate (participant, electrode) rows found in results_df.')

        results_df.to_csv(results_csv, index=False)
        print('Saved results_csv:', results_csv)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        sns.histplot(
            data=results_df,
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
            title='ACC electrode distribution: z-scored zrpe:memory_outcome weights',
            xlabel='Interaction weight z-score (zrpe:phit)',
            ylabel='Electrode count',
        )
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(overlap_figure_pdf, bbox_inches='tight')
        fig.savefig(overlap_figure_png, bbox_inches='tight')
        display(fig)
        plt.close(fig)

        print('Saved overlap_figure_pdf:', overlap_figure_pdf)
        print('Saved overlap_figure_png:', overlap_figure_png)
        """
    ),
    md("## Rank eligible electrodes by absolute interaction magnitude"),
    code(
        """
        condition_count_df = (
            acc_exact_df.dropna(subset=['rpe_cat', 'memory_outcome'])
            .groupby(['participant', 'electrode', 'rpe_cat', 'memory_outcome'])
            .size()
            .rename('n_trials_in_cell')
            .reset_index()
        )

        count_wide_df = (
            condition_count_df.assign(
                count_col=lambda df: df['rpe_cat'] + '_' + df['memory_outcome']
            )
            .pivot_table(
                index=['participant', 'electrode'],
                columns='count_col',
                values='n_trials_in_cell',
                fill_value=0,
                aggfunc='sum',
            )
            .reset_index()
        )
        count_wide_df.columns.name = None

        required_count_cols = [
            f'{rpe_cat}_{memory_outcome}'
            for rpe_cat in ['negative', 'neutral', 'positive']
            for memory_outcome in memory_outcome_order
        ]
        for col in required_count_cols:
            if col not in count_wide_df.columns:
                count_wide_df[col] = 0

        top_electrode_candidates_df = results_df.merge(
            count_wide_df,
            on=['participant', 'electrode'],
            how='left',
        )
        for col in required_count_cols:
            top_electrode_candidates_df[col] = (
                pd.to_numeric(top_electrode_candidates_df[col], errors='coerce')
                .fillna(0)
                .astype(int)
            )

        top_electrode_candidates_df['abs_z_scored_weight'] = (
            top_electrode_candidates_df['z_scored_weight'].abs()
        )
        top_electrode_candidates_df['eligible_for_plotting'] = (
            top_electrode_candidates_df[required_count_cols] >= n_plot_trials
        ).all(axis=1)

        eligible_electrodes_df = top_electrode_candidates_df[
            top_electrode_candidates_df['eligible_for_plotting']
        ].copy()
        eligible_electrodes_df = eligible_electrodes_df.sort_values(
            ['abs_z_scored_weight', 'z_scored_weight', 'participant', 'electrode'],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        eligible_electrodes_df['absolute_rank'] = np.arange(1, len(eligible_electrodes_df) + 1)

        top10_electrodes_df = eligible_electrodes_df.head(10).copy()
        top10_electrodes_df.to_csv(top_electrodes_csv, index=False)

        higher_ranked_ineligible_df = top_electrode_candidates_df[
            ~top_electrode_candidates_df['eligible_for_plotting']
        ].copy()
        higher_ranked_ineligible_df = higher_ranked_ineligible_df.sort_values(
            ['abs_z_scored_weight', 'z_scored_weight', 'participant', 'electrode'],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        print('Eligible electrodes:', len(eligible_electrodes_df))
        print('Selected top electrodes:', len(top10_electrodes_df))
        print('Saved top_electrodes_csv:', top_electrodes_csv)
        display(
            top10_electrodes_df[
                ['absolute_rank', 'participant', 'electrode', 'z_scored_weight', 'abs_z_scored_weight', 'p_value'] + required_count_cols
            ]
        )
        """
    ),
    md("## Plot smoothed cue-on HFA traces for the top eligible electrodes"),
    code(
        """
        def load_full_time_hfa_trials(subject: str, channel: str) -> tuple[pd.DataFrame, np.ndarray]:
            subj = Subject(subject, str(base_dir), str(load_dir), min_trials=min_trials, verbose=False)
            neural_cue = subj._get_or_build_event_neural_data(
                event=event,
                roi=target_roi,
                freq_range=hfa_freq_range,
                downsample_factor=hfa_downsample_factor,
                win_ms=hfa_win_ms,
                smooth_method=hfa_smooth_method,
                artifact_abs_thresh=hfa_artifact_abs_thresh,
            )

            cue_hfa = np.asarray(neural_cue['hfa'], dtype=np.float32)
            cue_times = np.asarray(neural_cue['times'], dtype=np.float32)
            cue_labels = np.asarray(neural_cue['electrode_labels'], dtype=str)

            matches = np.where(cue_labels == str(channel))[0]
            if len(matches) == 0:
                raise ValueError(f'Channel {channel} not found in smoothed cue HFA for {subject}')
            chan_idx = int(matches[0])

            trial_ids = np.arange(cue_hfa.shape[0], dtype=int)
            hfa_df = pd.DataFrame({
                'trial': np.repeat(trial_ids, cue_hfa.shape[2]),
                'time': np.tile(cue_times, cue_hfa.shape[0]),
                'tfr': cue_hfa[:, chan_idx, :].reshape(-1),
            })

            metadata_to_merge = build_memory_metadata_for_subject(subject).copy()
            metadata_to_merge['trial'] = np.arange(len(metadata_to_merge))
            hfa_df = hfa_df.merge(metadata_to_merge, on=['trial'], how='left')
            hfa_df['participant'] = subject
            hfa_df['electrode'] = channel
            hfa_df['memory_outcome'] = hfa_df['memory_outcome'].map({'hits': 'hit', 'misses': 'miss'})
            hfa_df['rpe_cat'] = hfa_df['rpe_cat'].apply(normalize_rpe_category)
            hfa_df = hfa_df[
                hfa_df['memory_outcome'].isin(memory_outcome_order) &
                hfa_df['rpe_cat'].isin(rpe_category_order)
            ].copy()

            if len(hfa_df) == 0:
                raise ValueError(f'No plot-eligible cue-on trials found for {subject} {channel}')

            return hfa_df, np.sort(hfa_df['time'].unique())


        pdf_export_rows = []
        skipped_plot_rows = []
        for _, row in tqdm(
            top10_electrodes_df.iterrows(),
            total=len(top10_electrodes_df),
            desc='Top ACC electrodes',
        ):
            participant = str(row['participant'])
            electrode = str(row['electrode'])

            try:
                hfa_trials_df, _ = load_full_time_hfa_trials(participant, electrode)
            except Exception as exc:
                skipped_plot_rows.append({
                    'participant': participant,
                    'electrode': electrode,
                    'reason': f'load_failed: {exc}',
                })
                continue

            selected_trial_rows = []
            missing_cells = []
            for rpe_cat in rpe_category_order:
                for memory_outcome in memory_outcome_order:
                    cond_trials = (
                        hfa_trials_df[
                            (hfa_trials_df['rpe_cat'] == rpe_cat) &
                            (hfa_trials_df['memory_outcome'] == memory_outcome)
                        ][['trial']]
                        .drop_duplicates()
                        .sort_values('trial')
                    )
                    if len(cond_trials) < n_plot_trials:
                        missing_cells.append(f'{rpe_cat}_{memory_outcome}')
                        continue
                    chosen_trials = cond_trials.head(n_plot_trials)['trial'].tolist()
                    selected_trial_rows.append(
                        hfa_trials_df[
                            (hfa_trials_df['rpe_cat'] == rpe_cat) &
                            (hfa_trials_df['memory_outcome'] == memory_outcome) &
                            (hfa_trials_df['trial'].isin(chosen_trials))
                        ].copy()
                    )

            if missing_cells:
                skipped_plot_rows.append({
                    'participant': participant,
                    'electrode': electrode,
                    'reason': f'missing_cells_after_reload: {missing_cells}',
                })
                continue

            selected_hfa_df = pd.concat(selected_trial_rows, ignore_index=True)
            ymin = float(selected_hfa_df['tfr'].min())
            ymax = float(selected_hfa_df['tfr'].max())
            if not np.isfinite(ymin) or not np.isfinite(ymax):
                skipped_plot_rows.append({
                    'participant': participant,
                    'electrode': electrode,
                    'reason': 'non_finite_plot_limits',
                })
                continue
            if np.isclose(ymin, ymax):
                pad = 1.0
            else:
                pad = 0.05 * (ymax - ymin)
            ylim = (ymin - pad, ymax + pad)

            pdf_path = top_electrode_pdf_dir / f'{row["absolute_rank"]:02d}_{participant}_{electrode}_absolute_interaction_hfa_trials.pdf'
            fig, axes = plt.subplots(
                len(rpe_category_order),
                len(memory_outcome_order),
                figsize=(10, 10),
                dpi=300,
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_2d(axes)
            line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for row_idx, rpe_cat in enumerate(rpe_category_order):
                for col_idx, memory_outcome in enumerate(memory_outcome_order):
                    ax = axes[row_idx, col_idx]
                    panel_df = selected_hfa_df[
                        (selected_hfa_df['rpe_cat'] == rpe_cat) &
                        (selected_hfa_df['memory_outcome'] == memory_outcome)
                    ].copy()
                    trial_order = (
                        panel_df[['trial']]
                        .drop_duplicates()
                        .sort_values('trial')['trial']
                        .tolist()
                    )
                    for trace_idx, trial_id in enumerate(trial_order):
                        trace_df = panel_df[panel_df['trial'] == trial_id].sort_values('time')
                        ax.plot(
                            trace_df['time'].to_numpy(dtype=float),
                            trace_df['tfr'].to_numpy(dtype=float),
                            color=line_colors[trace_idx % len(line_colors)],
                            linewidth=1.6,
                            alpha=0.95,
                            label=f'Trial {int(trial_id)}',
                        )
                    ax.axvspan(
                        RPE_MEMORY_OVERLAP_WINDOW_ACC[0],
                        RPE_MEMORY_OVERLAP_WINDOW_ACC[1],
                        color='0.85',
                        alpha=0.5,
                        zorder=0,
                    )
                    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
                    ax.set_ylim(ylim)
                    ax.set_title(
                        f'{rpe_cat.capitalize()} | {memory_outcome.capitalize()}\\n'
                        f'Trials: {\", \".join(str(int(t)) for t in trial_order)}',
                        fontsize=10,
                    )
                    if row_idx == len(rpe_category_order) - 1:
                        ax.set_xlabel('Cue time (s)')
                    if col_idx == 0:
                        ax.set_ylabel('HFA')
                    sns.despine(ax=ax)

            fig.suptitle(
                f'Rank {int(row[\"absolute_rank\"])} | {participant} {electrode}\\n'
                f'z={row[\"z_scored_weight\"]:.3f}, |z|={row[\"abs_z_scored_weight\"]:.3f}',
                fontsize=13,
                y=0.98,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper right', frameon=False)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            pdf_export_rows.append({
                'absolute_rank': int(row['absolute_rank']),
                'participant': participant,
                'electrode': electrode,
                'pdf_path': str(pdf_path),
            })

        pdf_export_df = pd.DataFrame(pdf_export_rows)
        skipped_plot_df = pd.DataFrame(skipped_plot_rows)

        print('PDFs created:', len(pdf_export_df))
        print('PDF output dir:', top_electrode_pdf_dir)
        if len(pdf_export_df):
            display(pdf_export_df)
        if len(skipped_plot_df):
            print('Skipped during plotting:')
            display(skipped_plot_df)
        """
    ),
    md("## Final checks"),
    code(
        """
        print('RPE_MEMORY_OVERLAP_WINDOW_ACC:', RPE_MEMORY_OVERLAP_WINDOW_ACC)
        print('Results rows:', len(results_df))
        print('Eligible electrodes:', len(eligible_electrodes_df))
        print('Selected electrodes:', len(top10_electrodes_df))
        print('PDFs created:', len(pdf_export_df))
        print('Unique participants:', results_df['participant'].nunique())
        print('Unique electrodes:', results_df['electrode'].nunique())
        print('Exact-window HFA only:', sorted(acc_exact_df['fband'].dropna().unique().tolist()))
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
