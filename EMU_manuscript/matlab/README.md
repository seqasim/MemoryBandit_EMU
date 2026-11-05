# MATLAB Decoding & Activation Pattern Scripts (`matlab/`)

These scripts implement the multivariate pattern classification (MVPA-Light) analyses reported in **Figures 2B, 2D, 3A–E, 5C, S3, S4, and S6**.

## 1. Required MATLAB Toolboxes

| Toolbox | Used by | Environment variable |
|---|---|---|
| [**MVPA-Light**](https://github.com/treder/MVPA-Light) | All decoding and Haufe pattern scripts | `MVPA_LIGHT_DIR` |
| **Statistics and Machine Learning Toolbox** | Built-in MATLAB functions (`quantile`, `discretize`, `ttest`, `bwconncomp`) | (MATLAB built-in) |
| **Parallel Computing Toolbox** | `parfor` permutation loops (`nPerm = 250`) | (MATLAB built-in; falls back to serial `for` if absent) |
| `ecog_fmri_visualization_matlab` | `plot_posthoc_patterns_on_brain_surface.m` (Fig S3 surface plots only) | `ECOG_VIS_DIR` |

All path resolution is handled centrally by [`config_paths.m`](config_paths.m). No scripts contain hardcoded user paths.

## 2. Preparing Input `.mat` TFR Files

The MATLAB scripts read per-subject `.mat` time-frequency files at:
```text
$MEMORYBANDIT_DATA_DIR/projects/guLab/Salman/EphysAnalyses/{subj}/scratch/TFR/{feedback_start,cue_on}-tfr.mat
```
If your data directory contains MNE `.h5` TFR files (`*-tfr.h5`, produced by `notebooks/03_spectral/00_compute_TFR_Fig2A_5A.ipynb`), convert them to `.mat` before running MATLAB:
```bash
python tools/export_tfr_to_mat.py
```

## 3. Running All Manuscript Panels Automatically

In MATLAB (from the `matlab/` directory):
```matlab
setenv('MEMORYBANDIT_DATA_DIR', '/path/to/MemoryBanditData');
setenv('MVPA_LIGHT_DIR', '/path/to/MVPA-Light');
run_manuscript_panels;
```

## 4. Panel-by-Panel Script & Parameter Reference

| Manuscript Panel | Script | Key Config Parameters | Output Files (under `$MEMORYBANDIT_DATA_DIR/scratch/MemoryBandit/MVPA/`) |
|---|---|---|---|
| **Fig 2B** (example ACC electrode HFA by RPE tercile) | `plot_hfa_by_rpe_category.m` | `roi_name='ACC'`, `dependent_var='rpe_category'`, `freq_label='hfa'`, `win_ms=200` | `decoding_feedback/ACC_{subj}_HFA_by_RPE_*.pdf` |
| **Fig 2D** (3-class RPE decoding across 6 ROIs at feedback) | `test_script_feedback_decoding.m` | `ROIs={'OFC','ACC','dmPFC','dlPFC','HPC','AMY'}`, `dependent_vars={'rpe_category'}`, `freq_low=70`, `freq_high=201`, `win_ms=200` | `decoding_feedback/{ROI}_rpe_category_hfa_200_{results.mat,stats.txt,raw.pdf,zscore.pdf}` |
| **Fig S3** (prefrontal Haufe activation patterns) | `test_script_feedback_rpe_posthoc_patterns.m`, `plot_posthoc_patterns_on_brain_surface.m` | `ROIs={'ACC','OFC','dmPFC','dlPFC'}`, `contrasts={'positive_vs_negative','positive_vs_neutral'}` | `decoding_feedback_posthoc_patterns/{ROI}_rpe_category_hfa_200_posthoc_patterns.csv`, `surface_plots/*.png` |
| **Fig S4A** (binary perceptual memorability decoding at feedback) | `test_script_feedback_decoding.m` | `ROIs={'OFC','ACC','dmPFC','dlPFC','HPC','AMY'}`, `dependent_vars={'pm_category'}` (5-fold CV, shrinkage LDA) | `decoding_feedback/{ROI}_pm_category_hfa_200_*` |
| **Fig S4B** (binary reward outcome decoding at feedback) | `test_script_feedback_decoding.m` | `ROIs={'OFC','ACC','dmPFC','dlPFC','HPC','AMY'}`, `dependent_vars={'reward'}` (5-fold CV, shrinkage LDA) | `decoding_feedback/{ROI}_reward_hfa_200_*` |
| **Fig 3A–B** (ACC feedback $\to$ cue cross-decoding & per-subject reactivation) | `test_script_crossdecoding.m` | `ROIs={'ACC'}`, `dependent_vars={'rpe_category'}`, `freq_low=70`, `freq_high=201`, `smooth_win_ms=200` | `crossdecoding/rpe_category/ACC_rpe_category_HFA_{all_acc_subjects.mat,stats.txt,subject_reinstatement.csv}` |
| **Fig 3C** (ACC cross-decoding controls: binary reward & memorability) | `test_script_crossdecoding.m` | `ROIs={'ACC'}`, `dependent_vars={'reward','pm_category'}` | `crossdecoding/{reward,pm_category}/ACC_*_HFA_*` |
| **Fig S6** (feedback $\to$ cue cross-decoding in other 5 ROIs) | `test_script_crossdecoding.m` | `ROIs={'dmPFC','OFC','dlPFC','HPC','AMY'}`, `dependent_vars={'rpe_category'}` | `crossdecoding/rpe_category/{ROI}_rpe_category_HFA_*` |
| **Fig 3D** (ACC HFA memory-choice decoding for old cues: hits vs misses) | `test_script_cue_decoding.m` | `ROIs={'ACC'}`, `freq_low=70`, `freq_high=201`, `freq_label='hfa'`, `win_ms=200`, `memory_mode='hit_miss'` | `decoding_cue/hit_miss/ACC_memory_outcome_hit_miss_hfa_200_*` |
| **Fig 3E** (ACC HFA memory-choice decoding for lure cues: FA vs CR) | `test_script_cue_decoding.m` | `ROIs={'ACC'}`, `freq_low=70`, `freq_high=201`, `freq_label='hfa'`, `win_ms=200`, `memory_mode='fa_cr'` | `decoding_cue/fa_cr/ACC_memory_outcome_fa_cr_hfa_200_*` |
| **Fig 5C** (HPC & ACC theta memory-choice decoding for old cues) | `test_script_cue_decoding.m` | `ROIs={'HPC','ACC'}`, `freq_low=4`, `freq_high=8`, `freq_label='theta'`, `win_ms=25`, `memory_mode='hit_miss'` | `decoding_cue/hit_miss/{HPC,ACC}_memory_outcome_hit_miss_theta_25_*` |
