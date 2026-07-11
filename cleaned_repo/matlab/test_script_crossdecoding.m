%% ============================================================
%  FEEDBACK → CUE CROSS-PERIOD DECODING (Flexible Frequency Band)
%  Matches modernized feedback-period decoding pipeline
%  Author: Salman Qasim
% =============================================================

clear; clc;

%% --- FIX: Remove FieldTrip paths to avoid MEX signature issues ---
% FieldTrip's MEX files have code signature issues on macOS
% MATLAB's native statistics functions work fine for MVPA-Light
ft_base = '/Users/salmanqasim/Documents/MATLAB/plot_recon_roi/dependencies/fieldtrip-20240110';
ft_paths_to_remove = {
    fullfile(ft_base, 'external', 'stats');
    fullfile(ft_base, 'src');
    fullfile(ft_base, 'external', 'signal');  % Also remove signal to be safe
};
rng(42, 'twister');

current_path = path;
for i = 1:numel(ft_paths_to_remove)
    if contains(current_path, ft_paths_to_remove{i})
        rmpath(ft_paths_to_remove{i});
        fprintf('Removed from path: %s\n', ft_paths_to_remove{i});
    end
end

%% --- CONFIGURATION ---
rng(42);  % Set seed for reproducibility

base_dir    = '/Volumes/T7 Shield';
load_dir    = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
ephys_dir   = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses');
output_dir = '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/crossdecoding';

subjects = {'MS012','UI001','MS016','MS017','UI002','MS019','MS020','MS022','MS023',...
            'UI003','MS025','MS026','MS028','MS030','UI004','MS035','MS036','UI006','UI007'};

ROIs = {'ACC'};        % can loop over multiple
dependent_vars = {'rpe_category'};  % Options: 'rpe_category', 'reward'
downsample_factor = 5;

% Frequency band configuration
freq_low = 70;                    % lower frequency bound (Hz)
freq_high = 201;                  % upper frequency bound (Hz)
freq_label = 'HFA';               % label for output files
smooth_win_ms = 200;              % smoothing window in ms (adjust per band)
% Example configurations for common bands:
% Theta:  freq_low = 4;   freq_high = 8;   freq_label = 'theta'; smooth_win_ms = 25;
% Alpha:  freq_low = 8;   freq_high = 12;  freq_label = 'alpha'; smooth_win_ms = 50;
% Beta:   freq_low = 12;  freq_high = 30;  freq_label = 'beta';  smooth_win_ms = 75;
% Gamma:  freq_low = 30;  freq_high = 70;  freq_label = 'gamma'; smooth_win_ms = 100;
% HFA:    freq_low = 70;  freq_high = 210; freq_label = 'HFA';   smooth_win_ms = 150;

alpha = 0.05;
nPerm = 5000;

%% --- PLOT AESTHETICS CONFIGURATION ---
plot_cfg = struct();
plot_cfg.font_name = 'Arial';
plot_cfg.font_size_label = 12;
plot_cfg.font_size_tick = 10;
plot_cfg.font_size_roi = 12;
plot_cfg.tick_dir = 'out';
plot_cfg.line_width = 1;
plot_cfg.contour_width = 2;
plot_cfg.contour_color = 'w';
plot_cfg.clim = [31 40];           % Color limits (accuracy in %)
plot_cfg.xticks = 0:0.2:1.0;       % X-axis tick positions
plot_cfg.yticks = 0:0.5:1.5;       % Y-axis tick positions
plot_cfg.tick_length = [0.025 0.025];
plot_cfg.colormap = 'parula';      % 'parula', 'viridis', 'jet', etc.
class_names = {'negative', 'neutral', 'positive'};

%% --- LOAD BEHAVIOR DATA ONCE (optimization) ---
fprintf('Loading behavioral data...\n');
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
learn_df.pm_category = categorical(learn_df.DPRIME > median(learn_df.DPRIME,'omitnan'), [false true], {'low','high'});
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- LOOP OVER DEPENDENT VARIABLES ---
for d = 1:numel(dependent_vars)
    dependent_var = dependent_vars{d};
    fprintf('\n========================================\n');
    fprintf('Processing dependent variable: %s\n', dependent_var);
    fprintf('========================================\n');

    % Set chance level and color limits based on dependent variable
    switch dependent_var
        case 'reward'
            chance = 0.5;
            plot_cfg.clim = [45 60];  % Adjust for binary classification
        case 'pm_category'
            chance = 0.5;
            plot_cfg.clim = [45 60];  % Binary classification
        case 'rpe_category'
            chance = 1/3;
            plot_cfg.clim = [31 40];  % Original multiclass limits
        otherwise
            error('Unknown dependent variable: %s', dependent_var);
    end

    % Update output directory to include dependent variable
    output_dir_var = fullfile(output_dir, dependent_var);
    if ~exist(output_dir_var, 'dir')
        mkdir(output_dir_var);
    end

    %% --- STORAGE FOR ALL ROIs (for combined plotting) ---
    all_acc_rois = cell(1, numel(ROIs));
    sig_mask_rois = cell(1, numel(ROIs));
    time_fb_rois = cell(1, numel(ROIs));
    time_cue_rois = cell(1, numel(ROIs));
    valid_rois = false(1, numel(ROIs));

    %% --- LOOP OVER ROIs ---
for r = 1:numel(ROIs)
    roi = ROIs{r};
    fprintf('\n===== ROI: %s =====\n', roi);

    all_acc = [];
    diagnostic_feedback_subjects = cell(1, numel(subjects));
    diagnostic_cue_subjects = cell(1, numel(subjects));
    diagnostic_y_subjects = cell(1, numel(subjects));
    sig_window_confusion_subjects = [];
    sig_window_precision_subjects = [];
    sig_window_recall_subjects = [];
    sig_window_f1_subjects = [];
    sig_window_cell_mask = [];
    sig_window_stats_table = [];
    subj_ids = {};  % Track which subjects have valid data for this ROI
    n_elec_subj = zeros(1, numel(subjects));  % Track electrode count per subject

    %% --- LOOP OVER SUBJECTS ---
    for s = 1:numel(subjects)
        subj = subjects{s};
        fprintf('  Subject: %s\n', subj);

        %% --- SELECT SUBJECT DATA ---
        subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
        subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
        subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);

        %% Process dependent variable
        switch dependent_var
            case 'rpe_category'
                % Process RPE categories using tercile split within subject
                edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
                subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});
                edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
                subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative', 'neutral', 'positive'});
            case 'reward'
                % Reward is already binary (0 or 1) in learn_df
                % No processing needed
            case 'pm_category'
                % pm_category is precomputed globally from learn_df.DPRIME
                % No subject-specific preprocessing needed
        end

        %% Trial mapping and remove fast RTs
        enc_trials = subj_combined_df.trials_gamble;
        ret_trials = subj_combined_df.trials_mem;
%         memory_mask = strcmp(subj_combined_df.hits, 'False');
%         mask_rt = subj_combined_df.gamble_rt_gamble >= 0.3;
%         enc_trials = enc_trials(memory_mask);
%         ret_trials = ret_trials(memory_mask);

        %% --- LOAD ELECTRODES ---
        elec_df = readtable(fullfile(ephys_dir, subj, 'Day1_reref_elec_df'));
        roi_elec = elec_df.label(strcmp(elec_df.salman_region, roi));
        if isempty(roi_elec)
            warning('No %s electrodes for %s', roi, subj);
            continue;
        end
        n_elec_subj(s) = numel(roi_elec);  % Track electrode count

        %% --- LOAD TFR DATA ---
        tfr_dir = fullfile(ephys_dir, subj, 'scratch', 'TFR');
        fb = load(fullfile(tfr_dir, 'feedback_start-tfr.mat'));
        cue = load(fullfile(tfr_dir, 'cue_on-tfr.mat'));

        feedback_tfr.powspctrm = fb.powspctrm;
        feedback_tfr.freq = fb.freqs;
        feedback_tfr.time = fb.times;
        feedback_tfr.label = fb.ch_names;
        feedback_tfr.fsample = fb.sfreq;

        cue_tfr.powspctrm = cue.powspctrm;
        cue_tfr.freq = cue.freqs;
        cue_tfr.time = cue.times;
        cue_tfr.label = cue.ch_names;
        cue_tfr.fsample = cue.sfreq;

        %% --- SELECT ROI ELECTRODES ---
        [~, idx_fb] = ismember(roi_elec, feedback_tfr.label);
        [~, idx_cue] = ismember(roi_elec, cue_tfr.label);
        feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
        cue_tfr.powspctrm = cue_tfr.powspctrm(:, idx_cue, :, :);

        %% --- COMPUTE FREQUENCY BAND POWER ---
        freq_mask = feedback_tfr.freq >= freq_low & feedback_tfr.freq <= freq_high;
        feedback_band = squeeze(mean(feedback_tfr.powspctrm(:,:,freq_mask,:),3));
        cue_band = squeeze(mean(cue_tfr.powspctrm(:,:,freq_mask,:),3));

        %% --- SMOOTH FREQUENCY BAND TIME SERIES ---
        fs = feedback_tfr.fsample;
        win_size = round(fs * smooth_win_ms / 1000);
        feedback_band = movmean(feedback_band, win_size, 3);
        cue_band = movmean(cue_band, win_size, 3);

        Ndims = ndims(feedback_band);
        if Ndims < 3
            feedback_band = reshape(feedback_band, [size(feedback_band, 1), 1, size(feedback_band, 2)]);
            cue_band = reshape(cue_band, [size(cue_band, 1), 1, size(cue_band, 2)]);
        end

        %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
        z_thresh = 10;

        % Process feedback HFA
        [nTrials_fb, nElec_fb, nTime_fb] = size(feedback_band);
        for trial = 1:nTrials_fb
            for elec = 1:nElec_fb
                trace = squeeze(feedback_band(trial, elec, :));
                bad_idx = abs(trace) > z_thresh;

                if any(bad_idx)
                    good_idx = find(~bad_idx);
                    bad_idx_list = find(bad_idx);

                    if numel(good_idx) >= 2
                        trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                    else
                        trace(trace > z_thresh) = z_thresh;
                        trace(trace < -z_thresh) = -z_thresh;
                    end

                    feedback_band(trial, elec, :) = trace;
                end
            end
        end

        % Process cue HFA
        [nTrials_cue, nElec_cue, nTime_cue] = size(cue_band);
        for trial = 1:nTrials_cue
            for elec = 1:nElec_cue
                trace = squeeze(cue_band(trial, elec, :));
                bad_idx = abs(trace) > z_thresh;

                if any(bad_idx)
                    good_idx = find(~bad_idx);
                    bad_idx_list = find(bad_idx);

                    if numel(good_idx) >= 2
                        trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                    else
                        trace(trace > z_thresh) = z_thresh;
                        trace(trace < -z_thresh) = -z_thresh;
                    end

                    cue_band(trial, elec, :) = trace;
                end
            end
        end


        %% --- DOWNSAMPLE ---
        feedback_band = feedback_band(enc_trials,:,1:downsample_factor:end);
        cue_band = cue_band(ret_trials,:,1:downsample_factor:end);
        time_fb = feedback_tfr.time(1:downsample_factor:end);
        time_cue = cue_tfr.time(1:downsample_factor:end);

        %% --- LABELS ---
        switch dependent_var
            case 'rpe_category'
                y_train_cat = subj_learn_df.rpe_category(enc_trials);
                y_train = grp2idx(y_train_cat); % 1=neg, 2=neutral, 3=pos
            case 'reward'
                y_train = double(subj_learn_df.reward(enc_trials)); % Binary: 0 or 1
            case 'pm_category'
                y_train = double(subj_learn_df.pm_category(enc_trials)); % Binary: low or high
            otherwise
                error('Unknown dependent variable: %s', dependent_var);
        end

        %% --- CROSS-DECODE (TRAIN FEEDBACK → TEST CUE) ---
        cfg = [];
        switch dependent_var
            case 'rpe_category'
                cfg.classifier = 'multiclass_lda';
                cfg.metric = 'acc';
                cfg.prior = 'uniform';
%                 cfg.reg = 'ridge';
                cfg.reg = 'shrink';
            case 'reward'
                cfg.classifier = 'lda';
                cfg.metric = 'acc';
                cfg.reg = 'shrink';
            case 'pm_category'
                cfg.classifier = 'lda';
                cfg.metric = 'acc';
                cfg.reg = 'shrink';
            otherwise
                error('Unknown dependent variable: %s', dependent_var);
        end
        cfg.cv = 'none'; % I don't need CV here because I am testing on a different period.

        [nTrials, nElec, nT_fb] = size(feedback_band);
        [~, ~, nT_cue] = size(cue_band);
        acc_matrix = zeros(nT_fb, nT_cue);
        is_rpe_category = strcmp(dependent_var, 'rpe_category');

        parfor t_fb = 1:nT_fb
            Xtrain = squeeze(feedback_band(:,:,t_fb));
            Xtrain = zscore(Xtrain, 0, 1);

            for t_cue = 1:nT_cue
                Xtest = squeeze(cue_band(:,:,t_cue));
                Xtest = zscore(Xtest, 0, 1);

                acc_matrix(t_fb,t_cue) = mv_classify(cfg, Xtrain, y_train, Xtest, y_train);
            end
        end

        all_acc(:,:,s) = acc_matrix;
        if is_rpe_category
            diagnostic_feedback_subjects{s} = feedback_band;
            diagnostic_cue_subjects{s} = cue_band;
            diagnostic_y_subjects{s} = y_train;
        end
        subj_ids{s} = subj;  % Track subject ID
    end

    %% --- REMOVE EMPTY SUBJECTS ---
    keep = squeeze(any(any(all_acc,1),2));
    all_acc = all_acc(:,:,keep);
    if strcmp(dependent_var, 'rpe_category')
        diagnostic_feedback_subjects = diagnostic_feedback_subjects(keep);
        diagnostic_cue_subjects = diagnostic_cue_subjects(keep);
        diagnostic_y_subjects = diagnostic_y_subjects(keep);
    end
    subj_ids = subj_ids(keep);  % Keep only subjects with valid data
    n_elec_subj = n_elec_subj(keep);  % Keep electrode counts for valid subjects
    fprintf('Kept %d subjects\n', sum(keep));

    %% --- GROUP-LEVEL STATS (vs. Chance) ---
    % chance is set in the dependent variable loop
    nSubj_valid = size(all_acc, 3);

    % Vectorized t-test computation (much faster than nested loops)
    group_mean = mean(all_acc, 3);
    group_std = std(all_acc, 0, 3);
    tvals = (group_mean - chance) ./ (group_std / sqrt(nSubj_valid));
    tvals(~isfinite(tvals)) = 0;

    % Compute p-values (one-tailed, right)
    pvals = 1 - tcdf(tvals, nSubj_valid - 1);

    %% --- CLUSTER-BASED PERMUTATION TEST ---
    % Threshold at t corresponding to p<0.05 (one-tailed)
%     t_thresh = tinv(1-alpha, sum(keep)-1);
    t_thresh = 1;
    sig_mask = tvals > t_thresh;

    % Label clusters
    cc = bwconncomp(sig_mask, 8);
    cluster_tsum = cellfun(@(x) sum(tvals(x)), cc.PixelIdxList);

    % Null distribution via random sign-flip permutations (parallelized)
    % Pre-generate all random flips for reproducibility with parfor
    all_flips = (randi([0 1], [nPerm, nSubj_valid]) * 2 - 1);
    centered_acc = all_acc - chance;
    max_tsum = zeros(nPerm, 1);

    parfor p = 1:nPerm
        flips = all_flips(p, :);
        perm_data = centered_acc .* reshape(flips, 1, 1, []);
        perm_mean = mean(perm_data, 3);
        perm_std = std(perm_data, 0, 3);
        perm_t = perm_mean ./ (perm_std / sqrt(nSubj_valid));
        perm_t(~isfinite(perm_t)) = 0;
        perm_mask = perm_t > t_thresh;
        cc_perm = bwconncomp(perm_mask, 8);
        if cc_perm.NumObjects > 0
            max_tsum(p) = max(cellfun(@(x) sum(perm_t(x)), cc_perm.PixelIdxList));
        else
            max_tsum(p) = 0;
        end
    end

    % Compute cluster p-values
    cluster_pvals = arrayfun(@(x) mean(max_tsum >= x), cluster_tsum);
    sig_clusters = find(cluster_pvals < alpha);
    sig_mask_final = false(size(sig_mask));
    for c = sig_clusters
        sig_mask_final(cc.PixelIdxList{c}) = true;
    end

    %% --- STORE RESULTS FOR COMBINED PLOTTING ---
    all_acc_rois{r} = all_acc;
    sig_mask_rois{r} = sig_mask_final;
    time_fb_rois{r} = time_fb;
    time_cue_rois{r} = time_cue;
    valid_rois(r) = true;

    %% --- INDIVIDUAL ROI PLOT (optional, with updated aesthetics) ---
    fig_individual = figure('Position', [200 200 500 450]);
    set(fig_individual, 'Color', 'w');
    set(fig_individual, 'Renderer', 'painters');  % Better for PDF export

    imagesc(time_cue, time_fb, mean(all_acc, 3) * 100, plot_cfg.clim);
    axis xy;
    colormap(plot_cfg.colormap);

    % Colorbar
    cb = colorbar;
    cb.Label.String = 'classification accuracy (%)';
    cb.Label.FontName = plot_cfg.font_name;
    cb.Label.FontSize = plot_cfg.font_size_label;
    cb.FontName = plot_cfg.font_name;
    cb.FontSize = plot_cfg.font_size_tick;
    cb.TickDirection = plot_cfg.tick_dir;
    cb.LineWidth = plot_cfg.line_width;

    % Axis labels
    xlabel('time since recognition cue (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    ylabel('time since feedback (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);

    % Tick configuration
    set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
    set(gca, 'TickDir', plot_cfg.tick_dir);
    set(gca, 'LineWidth', plot_cfg.line_width);
    set(gca, 'XTick', plot_cfg.xticks);
    set(gca, 'YTick', plot_cfg.yticks);
    set(gca, 'TickLength', plot_cfg.tick_length);
    box off;

    % ROI label in bottom-left corner
    text(0.05, 0.08, roi, 'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
        'FontSize', plot_cfg.font_size_roi + 2, 'FontWeight', 'bold', 'Color', plot_cfg.contour_color);

    % Significant cluster contours
    hold on;
    if any(sig_mask_final(:))
        contour(time_cue, time_fb, double(sig_mask_final), [0.5 0.5], plot_cfg.contour_color, 'LineWidth', plot_cfg.contour_width);
    end
    hold off;

    print(fig_individual, fullfile(output_dir_var, sprintf('%s_%s_%s_crossdecoding.pdf', roi, dependent_var, freq_label)), '-dpdf', '-r300');
    close(fig_individual);

    %% --- SIGNIFICANT-WINDOW 3-CLASS DIAGNOSTICS ---
    if strcmp(dependent_var, 'rpe_category')
        sig_window_cell_mask = sig_mask_final;
        if any(sig_mask_final(:))
            [sig_window_confusion_subjects, sig_window_precision_subjects, sig_window_recall_subjects, sig_window_f1_subjects, sig_window_stats_table] = ...
                save_cross_significant_window_diagnostics(output_dir_var, roi, dependent_var, freq_label, ...
                diagnostic_feedback_subjects, diagnostic_cue_subjects, diagnostic_y_subjects, sig_mask_final, class_names, plot_cfg, cfg);
        else
            fprintf('No significant feedback-cue cells for %s - skipping 3-class diagnostic summaries.\n', roi);
        end
    end

    save(fullfile(output_dir_var, sprintf('%s_%s_%s_all_acc_subjects.mat', roi, dependent_var, freq_label)), 'all_acc', 'time_fb', 'time_cue', 'subj_ids', 'n_elec_subj', 'sig_mask_final', 'cluster_pvals', 'dependent_var', 'chance', 'freq_low', 'freq_high', 'freq_label', 'sig_window_confusion_subjects', 'sig_window_precision_subjects', 'sig_window_recall_subjects', 'sig_window_f1_subjects', 'sig_window_cell_mask', 'sig_window_stats_table', 'class_names', '-v7.3');

    %% --- SAVE DETAILED STATISTICAL REPORT ---
    txt_filename = fullfile(output_dir_var, sprintf('%s_%s_%s_crossdecoding_stats.txt', roi, dependent_var, freq_label));

    fid = fopen(txt_filename, 'w');

    fprintf(fid, 'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
    fprintf(fid, '======================================================\n\n');

    %% --- GENERAL INFORMATION ---
    fprintf(fid, 'ROI: %s\n', roi);
    fprintf(fid, 'Dependent variable: %s\n', dependent_var);
    fprintf(fid, 'Number of subjects: %d\n', sum(keep));
    fprintf(fid, 'Smoothing window: %d ms\n', smooth_win_ms);
    fprintf(fid, 'Downsample factor: %d\n', downsample_factor);
    fprintf(fid, 'Time resolution (feedback): %.6f s\n', mean(diff(time_fb)));
    fprintf(fid, 'Time resolution (cue): %.6f s\n\n', mean(diff(time_cue)));

    %% --- DECODING CONFIGURATION ---
    fprintf(fid, 'Classifier: %s\n', cfg.classifier);
    fprintf(fid, 'Metric: %s\n', format_metric_for_report(cfg.metric));
    if isfield(cfg, 'prior')
        fprintf(fid, 'Prior: %s\n', cfg.prior);
    end
    if isfield(cfg, 'reg')
        fprintf(fid, 'Regularization: %s\n', cfg.reg);
    end
    fprintf(fid, 'Cross-validation: %s (train on feedback, test on cue)\n', cfg.cv);
    fprintf(fid, 'Frequency range: %d-%d Hz (%s)\n', freq_low, freq_high, freq_label);
    fprintf(fid, 'Z-scoring: per timepoint across trials (independent normalization)\n\n');

    %% --- STATISTICAL TEST DETAILS ---
    fprintf(fid, 'Statistical test: sign-flip permutation\n');
    fprintf(fid, 'Test statistic: t-test\n');
    fprintf(fid, 'Multiple comparison correction: cluster\n');
    fprintf(fid, 'Alpha level: %.4f\n', alpha);
    fprintf(fid, 'Number of permutations: %d\n\n', nPerm);

    fprintf(fid, 'Group-level design: within\n');
    fprintf(fid, 'Statistic: t-test\n');
    fprintf(fid, 'Tail: 1\n');
    fprintf(fid, 'Null hypothesis value: %.4f\n', chance);
    fprintf(fid, 'Permutation test: sign-flip\n');
    fprintf(fid, 'Cluster statistic: maxsum\n');
    fprintf(fid, 'Cluster-forming threshold: tinv(1-alpha, n-1) = %.4f\n', t_thresh);
    fprintf(fid, 'Cluster connectivity: 8-connected (2D)\n\n');

    %% --- CLUSTER RESULTS ---
    n_sig_clusters = numel(sig_clusters);
    fprintf(fid, 'Number of significant clusters: %d\n\n', n_sig_clusters);

    if n_sig_clusters == 0
        fprintf(fid, 'No significant clusters detected.\n');
    else
        for c = 1:numel(sig_clusters)
            cid = sig_clusters(c);
            [rows, cols] = ind2sub(size(sig_mask), cc.PixelIdxList{cid});

            t_fb_start = time_fb(min(rows));
            t_fb_end = time_fb(max(rows));
            t_cue_start = time_cue(min(cols));
            t_cue_end = time_cue(max(cols));

            fprintf(fid, 'Cluster %d\n', c);
            fprintf(fid, '  Feedback time range: %.4f – %.4f s\n', t_fb_start, t_fb_end);
            fprintf(fid, '  Cue time range: %.4f – %.4f s\n', t_cue_start, t_cue_end);
            fprintf(fid, '  Cluster-level p-value: %.6f\n', cluster_pvals(cid));
            fprintf(fid, '\n');
        end
    end

    fclose(fid);
    fprintf('Saved statistical summary:\n%s\n', txt_filename);

    %% --- SUBJECT-LEVEL REINSTATEMENT SCORES ---
    if any(sig_mask_final(:))
        nSubj = size(all_acc, 3);
        subj_reinstatement = zeros(nSubj, 1);
        subj_reinstatement_effect = zeros(nSubj, 1);  % accuracy - chance

        for subj_idx = 1:nSubj
            subj_acc = all_acc(:,:,subj_idx);
            % Mean accuracy within significant cluster
            subj_reinstatement(subj_idx) = mean(subj_acc(sig_mask_final));
            % Effect size: accuracy above chance
            subj_reinstatement_effect(subj_idx) = mean(subj_acc(sig_mask_final)) - chance;
        end

        % Display results to console
        fprintf('\nSubject-level reinstatement scores for %s:\n', roi);
        fprintf('%-10s\t%s\t\t%s\t\t%s\n', 'Subject', 'Mean Acc', 'Effect', 'nElec');
        for subj_idx = 1:nSubj
            fprintf('%-10s\t%.4f\t\t%.4f\t\t%d\n', subj_ids{subj_idx}, subj_reinstatement(subj_idx), subj_reinstatement_effect(subj_idx), n_elec_subj(subj_idx));
        end

        % Check correlation between reinstatement and electrode count
        [r_elec, p_elec] = corr(subj_reinstatement_effect, n_elec_subj(:), 'Type', 'Spearman');
        fprintf('\nCorrelation between reinstatement effect and electrode count:\n');
        fprintf('  Spearman rho = %.4f, p = %.4f\n', r_elec, p_elec);
        if p_elec < 0.05
            warning('Significant correlation between reinstatement and electrode count - consider controlling for this in downstream analyses.');
        end

        % Save as CSV with subject IDs and electrode counts
        subj_table = table(subj_ids(:), subj_reinstatement, subj_reinstatement_effect, n_elec_subj(:), ...
            'VariableNames', {'Subject', 'MeanAccuracy', 'ReinstateEffect', 'NumElectrodes'});
        writetable(subj_table, fullfile(output_dir_var, sprintf('%s_%s_%s_subject_reinstatement.csv', roi, dependent_var, freq_label)));
        fprintf('Saved subject reinstatement scores:\n%s\n', ...
            fullfile(output_dir_var, sprintf('%s_%s_%s_subject_reinstatement.csv', roi, dependent_var, freq_label)));

        % Bar plot of subject contributions
        fig_bar = figure('Position', [100 100 700 400]);
        set(fig_bar, 'Color', 'w');
        set(fig_bar, 'Renderer', 'painters');
        bar(subj_reinstatement_effect, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
        hold on;
        yline(0, 'k--', 'LineWidth', plot_cfg.line_width);
        xlabel('Subject', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
        ylabel('Reinstatement Effect (Accuracy - Chance)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
        title(sprintf('%s: Subject-Level Reinstatement', roi), 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label + 2);
        set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        set(gca, 'TickDir', plot_cfg.tick_dir);
        set(gca, 'LineWidth', plot_cfg.line_width);
        set(gca, 'TickLength', plot_cfg.tick_length);
        xticks(1:nSubj);
        xticklabels(subj_ids);
        xtickangle(45);
        xlim([0.5 nSubj+0.5]);
        box off;
        hold off;
        print(fig_bar, fullfile(output_dir_var, sprintf('%s_%s_%s_subject_reinstatement.pdf', roi, dependent_var, freq_label)), '-dpdf', '-r300');
        close(fig_bar);
    else
        fprintf('\nNo significant clusters found for %s - skipping subject-level reinstatement.\n', roi);
    end
end

    %% ============================================================
    %  COMBINED SUBPLOT FIGURE (Publication-quality)
    % =============================================================

% Identify which ROIs have valid data
valid_idx = find(valid_rois);
nValidROIs = numel(valid_idx);

if nValidROIs > 0
    fprintf('\n===== Creating combined subplot figure =====\n');

    % Figure sizing: adjust width based on number of ROIs
    subplot_width = 220;  % pixels per subplot
    fig_width = subplot_width * nValidROIs + 120;  % extra space for colorbar
    fig_height = 350;

    fig_combined = figure('Position', [100 100 fig_width fig_height]);
    set(fig_combined, 'Color', 'w');
    set(fig_combined, 'Renderer', 'painters');  % Better for PDF export

    % Create tiled layout for better control
    t = tiledlayout(1, nValidROIs, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:nValidROIs
        r = valid_idx(i);
        roi_name = ROIs{r};

        nexttile;

        % Get data for this ROI
        acc_data = mean(all_acc_rois{r}, 3) * 100;  % Convert to percentage
        sig_mask = sig_mask_rois{r};
        t_cue = time_cue_rois{r};
        t_fb = time_fb_rois{r};

        % Plot heatmap
        imagesc(t_cue, t_fb, acc_data, plot_cfg.clim);
        axis xy;
        colormap(plot_cfg.colormap);

        % Axis configuration
        set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        set(gca, 'TickDir', plot_cfg.tick_dir);
        set(gca, 'LineWidth', plot_cfg.line_width);
        set(gca, 'XTick', plot_cfg.xticks);
        set(gca, 'YTick', plot_cfg.yticks);
        set(gca, 'TickLength', plot_cfg.tick_length);
        box off;

        % X-axis label for all subplots
        xlabel('time since recognition cue (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label - 1);

        % Y-axis label only for first subplot
        if i == 1
            ylabel('time since feedback (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label - 1);
        else
            set(gca, 'YTickLabel', []);
        end

        % ROI label in bottom-left corner
        text(0.06, 0.06, roi_name, 'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
            'FontSize', plot_cfg.font_size_roi, 'FontWeight', 'bold', 'Color', plot_cfg.contour_color, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

        % Significant cluster contours
        hold on;
        if any(sig_mask(:))
            contour(t_cue, t_fb, double(sig_mask), [0.5 0.5], plot_cfg.contour_color, 'LineWidth', plot_cfg.contour_width);
        end
        hold off;
    end

    % Shared colorbar
    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Label.String = 'classification accuracy (%)';
    cb.Label.FontName = plot_cfg.font_name;
    cb.Label.FontSize = plot_cfg.font_size_label - 1;
    cb.FontName = plot_cfg.font_name;
    cb.FontSize = plot_cfg.font_size_tick;
    cb.TickDirection = plot_cfg.tick_dir;
    cb.LineWidth = plot_cfg.line_width;

    % Save combined figure
    print(fig_combined, fullfile(output_dir_var, sprintf('all_ROIs_%s_%s_crossdecoding_combined.pdf', dependent_var, freq_label)), '-dpdf', '-r300');
    saveas(fig_combined, fullfile(output_dir_var, sprintf('all_ROIs_%s_%s_crossdecoding_combined.fig', dependent_var, freq_label)));
    fprintf('Saved combined figure:\n%s\n', fullfile(output_dir_var, sprintf('all_ROIs_%s_%s_crossdecoding_combined.pdf', dependent_var, freq_label)));
    close(fig_combined);
    end
end  % End dependent variable loop

function [confusion_subjects, precision_subjects, recall_subjects, f1_subjects, stats_table] = save_cross_significant_window_diagnostics( ...
    output_dir, roi_name, dependent_var, freq_label, diagnostic_feedback_subjects, diagnostic_cue_subjects, ...
    diagnostic_y_subjects, sig_mask, class_names, plot_cfg, base_cfg)

    [sig_fb_idx, sig_cue_idx] = find(sig_mask);
    n_sig_cells = numel(sig_fb_idx);
    n_subjects = numel(diagnostic_feedback_subjects);
    n_classes = numel(class_names);
    confusion_subjects = nan(n_classes, n_classes, n_subjects);
    precision_subjects = nan(n_subjects, n_classes);
    recall_subjects = nan(n_subjects, n_classes);
    f1_subjects = nan(n_subjects, n_classes);

    diagnostic_cfg = base_cfg;
    diagnostic_cfg.metric = {'confusion', 'precision', 'recall', 'f1'};
    diagnostic_cfg.feedback = 0;

    for subj_idx = 1:n_subjects
        subject_confusion = nan(n_classes, n_classes, n_sig_cells);
        subject_precision = nan(n_sig_cells, n_classes);
        subject_recall = nan(n_sig_cells, n_classes);
        subject_f1 = nan(n_sig_cells, n_classes);

        for cell_idx = 1:n_sig_cells
            Xtrain = squeeze(diagnostic_feedback_subjects{subj_idx}(:,:,sig_fb_idx(cell_idx)));
            Xtrain = zscore(Xtrain, 0, 1);
            Xtest = squeeze(diagnostic_cue_subjects{subj_idx}(:,:,sig_cue_idx(cell_idx)));
            Xtest = zscore(Xtest, 0, 1);
            y = diagnostic_y_subjects{subj_idx};

            diag_perf = mv_classify(diagnostic_cfg, Xtrain, y, Xtest, y);
            subject_confusion(:,:,cell_idx) = standardize_cross_confusion(diag_perf{1}, n_classes);
            subject_precision(cell_idx,:) = standardize_cross_class_metric(diag_perf{2}, n_classes);
            subject_recall(cell_idx,:) = standardize_cross_class_metric(diag_perf{3}, n_classes);
            subject_f1(cell_idx,:) = standardize_cross_class_metric(diag_perf{4}, n_classes);
        end

        confusion_subjects(:,:,subj_idx) = mean(subject_confusion, 3, 'omitnan');
        precision_subjects(subj_idx,:) = mean(subject_precision, 1, 'omitnan');
        recall_subjects(subj_idx,:) = mean(subject_recall, 1, 'omitnan');
        f1_subjects(subj_idx,:) = mean(subject_f1, 1, 'omitnan');
    end

    mean_confusion = mean(confusion_subjects, 3, 'omitnan');
    mean_precision = mean(precision_subjects, 1, 'omitnan');
    sem_precision = std(precision_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);
    mean_recall = mean(recall_subjects, 1, 'omitnan');
    sem_recall = std(recall_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);
    mean_f1 = mean(f1_subjects, 1, 'omitnan');
    sem_f1 = std(f1_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);

    plot_confusion_summary(mean_confusion, class_names, ...
        sprintf('%s: significant-window cross-decoding confusion', roi_name), ...
        fullfile(output_dir, sprintf('%s_%s_%s_crossdecoding_significant_window_confusion_summary.pdf', roi_name, dependent_var, freq_label)), ...
        plot_cfg);
    plot_class_metric_summary(mean_recall, sem_recall, mean_precision, sem_precision, mean_f1, sem_f1, class_names, ...
        sprintf('%s: significant-window cross-decoding class metrics', roi_name), ...
        fullfile(output_dir, sprintf('%s_%s_%s_crossdecoding_significant_window_class_metrics.pdf', roi_name, dependent_var, freq_label)), ...
        plot_cfg);

    metric_summary = table(class_names(:), mean_recall(:), sem_recall(:), mean_precision(:), sem_precision(:), mean_f1(:), sem_f1(:), ...
        'VariableNames', {'Class', 'MeanRecall', 'SEMRecall', 'MeanPrecision', 'SEMPrecision', 'MeanF1', 'SEMF1'});
    writetable(metric_summary, fullfile(output_dir, sprintf('%s_%s_%s_crossdecoding_significant_window_class_metrics.csv', roi_name, dependent_var, freq_label)));

    confusion_table = array2table(mean_confusion, 'VariableNames', matlab.lang.makeValidName(strcat('Predicted_', class_names)), ...
        'RowNames', matlab.lang.makeValidName(strcat('True_', class_names)));
    writetable(confusion_table, fullfile(output_dir, sprintf('%s_%s_%s_crossdecoding_significant_window_confusion_summary.csv', roi_name, dependent_var, freq_label)), ...
        'WriteRowNames', true);

    stats_table = calculate_class_paired_tests(recall_subjects, precision_subjects, f1_subjects, confusion_subjects, class_names);
    writetable(stats_table, fullfile(output_dir, sprintf('%s_%s_%s_crossdecoding_significant_window_paired_tests.csv', roi_name, dependent_var, freq_label)));
end

function confusion_out = standardize_cross_confusion(confusion_in, n_classes)
    confusion_out = squeeze(confusion_in);
    if ~isequal(size(confusion_out), [n_classes n_classes])
        error('Expected %dx%d confusion matrix, got size [%s]', n_classes, n_classes, num2str(size(confusion_out)));
    end
end

function metric_out = standardize_cross_class_metric(metric_in, n_classes)
    metric_out = squeeze(metric_in);
    if numel(metric_out) ~= n_classes
        error('Expected one value per class, got %d values', numel(metric_out));
    end
    metric_out = reshape(metric_out, 1, n_classes);
end

function plot_confusion_summary(confusion_matrix, class_names, plot_title, output_file, plot_cfg)
    fig = figure('Position', [200 200 460 400], 'Color', 'w');
    imagesc(confusion_matrix, [0 1]);
    axis square;
    colormap(parula);
    cb = colorbar;
    cb.Label.String = 'proportion';
    set(gca, 'XTick', 1:numel(class_names), 'XTickLabel', class_names, ...
        'YTick', 1:numel(class_names), 'YTickLabel', class_names, ...
        'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick, ...
        'TickDir', plot_cfg.tick_dir, 'TickLength', plot_cfg.tick_length);
    xlabel('predicted class', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    ylabel('true class', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    title(plot_title, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    for ii = 1:numel(class_names)
        for jj = 1:numel(class_names)
            text(jj, ii, sprintf('%.2f', confusion_matrix(ii,jj)), ...
                'HorizontalAlignment', 'center', 'FontName', plot_cfg.font_name, ...
                'FontSize', plot_cfg.font_size_tick, 'Color', 'w');
        end
    end
    exportgraphics(fig, output_file, 'Resolution', 300);
    close(fig);
end

function plot_class_metric_summary(mean_recall, sem_recall, mean_precision, sem_precision, mean_f1, sem_f1, class_names, plot_title, output_file, plot_cfg)
    metric_values = [mean_recall(:), mean_precision(:), mean_f1(:)];
    metric_errors = [sem_recall(:), sem_precision(:), sem_f1(:)];
    fig = figure('Position', [200 200 640 420], 'Color', 'w');
    bar(metric_values);
    hold on;
    ngroups = size(metric_values, 1);
    nbars = size(metric_values, 2);
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for metric_idx = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*metric_idx-1) * groupwidth / (2*nbars);
        errorbar(x, metric_values(:,metric_idx), metric_errors(:,metric_idx), 'k', ...
            'LineStyle', 'none', 'LineWidth', plot_cfg.line_width);
    end
    hold off;
    ylim([0 1]);
    set(gca, 'XTick', 1:numel(class_names), 'XTickLabel', class_names, ...
        'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick, ...
        'TickDir', plot_cfg.tick_dir, 'TickLength', plot_cfg.tick_length);
    ylabel('score', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    title(plot_title, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    legend({'Recall', 'Precision', 'F1'}, 'Location', 'best', 'Box', 'off');
    box off;
    exportgraphics(fig, output_file, 'Resolution', 300);
    close(fig);
end

function stats_table = calculate_class_paired_tests(recall_subjects, precision_subjects, f1_subjects, confusion_subjects, class_names)
    positive_idx = find(strcmp(class_names, 'positive'), 1);
    comparison_idx = [find(strcmp(class_names, 'neutral'), 1), find(strcmp(class_names, 'negative'), 1)];
    comparison_names = {'neutral', 'negative'};
    metric_names = {'Recall', 'Precision', 'F1', 'ConfusionDiagonal'};

    metric_col = {};
    contrast_col = {};
    positive_class_col = {};
    comparison_class_col = {};
    n_col = [];
    positive_mean_col = [];
    comparison_mean_col = [];
    mean_diff_col = [];
    t_col = [];
    df_col = [];
    p_col = [];

    for comparison_num = 1:numel(comparison_idx)
        class_idx = comparison_idx(comparison_num);
        metric_data = {
            recall_subjects(:, positive_idx), recall_subjects(:, class_idx);
            precision_subjects(:, positive_idx), precision_subjects(:, class_idx);
            f1_subjects(:, positive_idx), f1_subjects(:, class_idx);
            squeeze(confusion_subjects(positive_idx, positive_idx, :)), squeeze(confusion_subjects(class_idx, class_idx, :))
        };

        for metric_idx = 1:numel(metric_names)
            positive_values = metric_data{metric_idx, 1};
            comparison_values = metric_data{metric_idx, 2};
            valid = isfinite(positive_values) & isfinite(comparison_values);

            if sum(valid) >= 2
                [~, p_value, ~, stats] = ttest(positive_values(valid), comparison_values(valid));
                t_value = stats.tstat;
                df_value = stats.df;
            else
                p_value = NaN;
                t_value = NaN;
                df_value = NaN;
            end

            metric_col{end+1, 1} = metric_names{metric_idx}; %#ok<AGROW>
            contrast_col{end+1, 1} = sprintf('positive_vs_%s', comparison_names{comparison_num}); %#ok<AGROW>
            positive_class_col{end+1, 1} = 'positive'; %#ok<AGROW>
            comparison_class_col{end+1, 1} = comparison_names{comparison_num}; %#ok<AGROW>
            n_col(end+1, 1) = sum(valid); %#ok<AGROW>
            positive_mean_col(end+1, 1) = mean(positive_values(valid), 'omitnan'); %#ok<AGROW>
            comparison_mean_col(end+1, 1) = mean(comparison_values(valid), 'omitnan'); %#ok<AGROW>
            mean_diff_col(end+1, 1) = mean(positive_values(valid) - comparison_values(valid), 'omitnan'); %#ok<AGROW>
            t_col(end+1, 1) = t_value; %#ok<AGROW>
            df_col(end+1, 1) = df_value; %#ok<AGROW>
            p_col(end+1, 1) = p_value; %#ok<AGROW>
        end
    end

    stats_table = table(metric_col, contrast_col, positive_class_col, comparison_class_col, n_col, ...
        positive_mean_col, comparison_mean_col, mean_diff_col, t_col, df_col, p_col, ...
        'VariableNames', {'Metric', 'Contrast', 'PositiveClass', 'ComparisonClass', 'N', ...
        'PositiveMean', 'ComparisonMean', 'MeanDifference', 'T', 'DF', 'PValue'});
end

function metric_text = format_metric_for_report(metric)
    if iscell(metric)
        metric_text = strjoin(metric, ', ');
    else
        metric_text = metric;
    end
end
