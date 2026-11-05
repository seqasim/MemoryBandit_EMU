%% =========================================================================
%  RUN_MANUSCRIPT_PANELS
%  Master driver script for all MATLAB (MVPA-Light) manuscript figures.
%
%  Prerequisites:
%    1. Set environment variables before launching MATLAB (or via setenv):
%         setenv('MEMORYBANDIT_DATA_DIR', '/path/to/MemoryBanditData');
%         setenv('MVPA_LIGHT_DIR', '/path/to/MVPA-Light');
%         % Optional (only for Fig S3 brain surface rendering):
%         setenv('ECOG_VIS_DIR', '/path/to/ecog_fmri_visualization_matlab');
%    2. Export per-subject `.h5` TFR files to `.mat` if not already present:
%         python tools/export_tfr_to_mat.py
% =========================================================================

MANUSCRIPT_DRIVER_RUN = true;
config_paths();

%% 1. Figure 2D & Figure S4A-B: Feedback-period HFA decoding across all 6 ROIs
%  - rpe_category (3-class shrinkage LDA, 5-fold CV) -> Figure 2D
%  - pm_category  (binary shrinkage LDA, 5-fold CV)  -> Figure S4A
%  - reward       (binary shrinkage LDA, 5-fold CV)  -> Figure S4B
fprintf('\n=== [1/7] Running Figure 2D & Figure S4 (Feedback HFA decoding) ===\n');
ROIs = {'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY'};
dependent_vars = {'rpe_category', 'pm_category', 'reward'};
freq_low = 70;
freq_high = 201;
freq_label = 'hfa';
win_ms = 200;
run('test_script_feedback_decoding.m');

%% 2. Figure 2B: Example ACC electrode HFA split by RPE tercile
fprintf('\n=== [2/7] Running Figure 2B (ACC HFA traces by RPE category) ===\n');
roi_name = 'ACC';
dependent_var = 'rpe_category';
freq_label = 'hfa';
win_ms = 200;
run('plot_hfa_by_rpe_category.m');

%% 3. Figure S3: Haufe activation patterns for prefrontal RPE encoding
fprintf('\n=== [3/7] Running Figure S3 (Post-hoc Haufe activation patterns) ===\n');
ROIs = {'ACC', 'OFC', 'dmPFC', 'dlPFC'};
run('test_script_feedback_rpe_posthoc_patterns.m');

if ~isempty(getenv('ECOG_VIS_DIR')) && exist(getenv('ECOG_VIS_DIR'), 'dir')
    for c_name = {'positive_vs_negative', 'positive_vs_neutral'}
        roi_name = 'all';
        contrast_name = c_name{1};
        run('plot_posthoc_patterns_on_brain_surface.m');
    end
else
    fprintf('Skipping surface mesh rendering (set ECOG_VIS_DIR to run plot_posthoc_patterns_on_brain_surface.m).\n');
end

%% 4. Figure 3A-C & Figure S6: Feedback -> Recognition Cue cross-decoding
%  - rpe_category across all 6 ROIs -> Figure 3A-B (ACC) & Figure S6 (other ROIs)
%  - reward and pm_category in ACC  -> Figure 3C controls
fprintf('\n=== [4/7] Running Figure 3A-C & Figure S6 (Feedback->Cue cross-decoding) ===\n');
ROIs = {'ACC', 'dmPFC', 'OFC', 'dlPFC', 'HPC', 'AMY'};
dependent_vars = {'rpe_category'};
freq_low = 70;
freq_high = 201;
freq_label = 'HFA';
smooth_win_ms = 200;
run('test_script_crossdecoding.m');

ROIs = {'ACC'};
dependent_vars = {'reward', 'pm_category'};
run('test_script_crossdecoding.m');

%% 5. Figure 3D: ACC HFA decoding of memory choice for OLD cues (hits vs misses)
fprintf('\n=== [5/7] Running Figure 3D (ACC HFA cue decoding: old images, hits vs misses) ===\n');
ROIs = {'ACC'};
freq_low = 70;
freq_high = 201;
freq_label = 'hfa';
win_ms = 200;
memory_mode = 'hit_miss';
run('test_script_cue_decoding.m');

%% 6. Figure 3E: ACC HFA decoding of memory choice for LURE cues (FA vs CR)
fprintf('\n=== [6/7] Running Figure 3E (ACC HFA cue decoding: lure images, FA vs CR) ===\n');
ROIs = {'ACC'};
freq_low = 70;
freq_high = 201;
freq_label = 'hfa';
win_ms = 200;
memory_mode = 'fa_cr';
run('test_script_cue_decoding.m');

%% 7. Figure 5C: Hippocampal (and ACC) theta power decoding of memory choice (old cues)
fprintf('\n=== [7/7] Running Figure 5C (HPC & ACC theta cue decoding: old images, hits vs misses) ===\n');
ROIs = {'HPC', 'ACC'};
freq_low = 4;
freq_high = 8;
freq_label = 'theta';
win_ms = 25;
memory_mode = 'hit_miss';
run('test_script_cue_decoding.m');

fprintf('\nAll MATLAB manuscript decoding panels completed.\n');
