%% --- CONFIGURATION ---
path_cfg = config_paths();
base_dir = path_cfg.base_dir;
load_dir = path_cfg.load_dir;
output_root = fullfile(path_cfg.mvpa_dir, 'decoding_cue');

rng(42, 'twister');


% 'OFC', 'ACC', 'dmPFC',     'dlPFC', 'HPC', 'AMY', 'Temporal', 'PFC'
if ~exist('ROIs', 'var') || isempty(ROIs)
    ROIs = {'HPC'};
end
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

downsample_factor = 5;

% Frequency band configuration
if ~exist('freq_low', 'var'), freq_low = 4; end
if ~exist('freq_high', 'var'), freq_high = 8; end
if ~exist('freq_label', 'var'), freq_label = 'theta'; end
if ~exist('win_ms', 'var'), win_ms = 25; end

permutation = 1;
nPerm = 250;
alpha = 0.05;

plot_cfg = struct();
plot_cfg.font_name = 'Arial';
plot_cfg.font_size_label = 12;
plot_cfg.font_size_tick = 10;
plot_cfg.font_size_roi = 12;
plot_cfg.line_width = 1.5;
plot_cfg.chance_width = 1;
plot_cfg.tick_dir = 'out';
plot_cfg.tick_length = [0.025 0.025];
plot_cfg.roi_colors = containers.Map( ...
    {'OFC', 'ACC', 'HPC', 'dmPFC', 'dlPFC', 'AMY'}, ...
    {[1.00 0.73 0.37], [0.73 0.31 0.73], [0.42 0.36 1.00], [0.45 0.16 0.10], [1.00 0.20 0.18], [0.58 0.18 0.98]});

% Memory decoding configuration
if ~exist('memory_mode', 'var') || isempty(memory_mode)
    memory_mode = 'hit_miss'; % 'all4', 'hit_miss' (old cues), or 'fa_cr' (lure cues)
end

%% --- LOAD TABLES ONCE ---
combined_df = readtable(path_cfg.combined_csv);
cols = ["hits","misses","false_alarms","correct_rejections"];

% Convert 'True'/'False' strings to logical matrix
X = strcmp(combined_df{:,cols}, 'True');

% Find which column is True
[~, idx] = max(X, [], 2);

% Store label
combined_df.memory_outcome = categorical(idx, 1:numel(cols), cols);

%% --- LOOP OVER DEPENDENT VARIABLES ---
dependent_vars = {'memory_outcome'};
for d = 1:numel(dependent_vars)
    dependent_var = dependent_vars{d};
    fprintf('Processing variable: %s (%s)\n', dependent_var, memory_mode);
    roi_plot_data = cell(1, numel(ROIs));

    output_dir = fullfile(output_root, memory_mode);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% --- LOOP OVER ROIs ---
    for r = 1:numel(ROIs)
        if strcmp(ROIs{r}, 'PFC')
            roi = {'OFC', 'ACC', 'dlPFC', 'dmPFC'};
        elseif strcmp(ROIs{r}, 'MTL')
            roi = {'HPC', 'AMY', 'PHC', 'EC'};
        else
            roi = ROIs{r};
        end

        fprintf('Processing ROI: %s\n', ROIs{r});

        all_acc_subjects = [];
        all_perm_subjects = [];
        all_results_subjects = cell(1, numel(subjects));
        n_elec_subj = zeros(1, numel(subjects));
        class_names = {};
        chancelevel = NaN;
        cue_tfr = struct();

        %% --- LOOP OVER SUBJECTS ---
        for s = 1:numel(subjects)
            subj = subjects{s};
            fprintf('  Subject: %s\n', subj);

            %% --- SELECT SUBJECT DATA ---
            subj_combined_df = combined_df(strcmp(combined_df.condition, 'Day1') & strcmp(combined_df.participant, subj), :);
            subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_mem), :);

            if isempty(subj_combined_df)
                warning('No cue trials for subject %s', subj);
                continue;
            end

            if ~ismember(dependent_var, subj_combined_df.Properties.VariableNames)
                warning('Field %s not found for subject %s', dependent_var, subj);
                continue;
            end

            [y, keep_mask, class_names_subject] = map_memory_outcome_labels(subj_combined_df.(dependent_var), memory_mode);
            cue_trials = subj_combined_df.trials_mem;

            cue_trials = cue_trials(keep_mask);
            y = y(keep_mask);

            valid_trial_mask = ~isnan(cue_trials) & cue_trials > 0;
            cue_trials = cue_trials(valid_trial_mask);
            y = y(valid_trial_mask);

            if isempty(cue_trials) || isempty(y)
                warning('No valid labeled cue trials for subject %s', subj);
                continue;
            end

            %% Load electrode info
            elec_df = readtable(fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'Day1_reref_elec_df'));
            roi_electrodes = elec_df.label(find(matches(elec_df.salman_region, roi)));
            if isempty(roi_electrodes)
                warning('No electrodes found for %s in %s', subj, ROIs{r});
                continue;
            end

            %% Load cue TFR data
            filepath = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'scratch', 'TFR');
            cue = load(fullfile(filepath, 'cue_on-tfr.mat'));
            cue_tfr.powspctrm = cue.powspctrm;
            cue_tfr.freq = cue.freqs;
            cue_tfr.time = cue.times;
            cue_tfr.label = cue.ch_names;
            cue_tfr.fsample = cue.sfreq;

            %% Select ROI electrodes
            [~, idx_cue] = ismember(roi_electrodes, cue_tfr.label);
            valid_elec_mask = idx_cue > 0;
            idx_cue = idx_cue(valid_elec_mask);

            if isempty(idx_cue)
                warning('No matching cue electrodes found for %s in %s', subj, ROIs{r});
                continue;
            end

            n_elec_subj(s) = numel(idx_cue);
            cue_tfr.powspctrm = cue_tfr.powspctrm(:, idx_cue, :, :);

            %% Extract frequency band power
            freq_mask = cue_tfr.freq >= freq_low & cue_tfr.freq <= freq_high;
            cue_band = squeeze(mean(cue_tfr.powspctrm(:, :, freq_mask, :), 3));
            Ndims = ndims(cue_band);
            if Ndims < 3
                cue_band = reshape(cue_band, [size(cue_band, 1), 1, size(cue_band, 2)]);
            end

            %% --- SMOOTH (VECTORIZE) ---
            fs = cue_tfr.fsample;
            win_size = round(fs * (win_ms / 1000));
            cue_band_smooth = movmean(cue_band, win_size, 3);

            %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
            z_thresh = 10;
            [nTrials_raw, nElec, ~] = size(cue_band_smooth);

            for trial = 1:nTrials_raw
                for elec = 1:nElec
                    trace = squeeze(cue_band_smooth(trial, elec, :));
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

                        cue_band_smooth(trial, elec, :) = trace;
                    end
                end
            end

            if max(cue_trials) > size(cue_band_smooth, 1)
                warning('Cue trial index exceeds available TFR trials for subject %s', subj);
                continue;
            end

            %% Select cue trials
            cue_band_smooth = cue_band_smooth(cue_trials, :, :);

            %% Downsample
            cue_band_ds = cue_band_smooth(:, :, 1:downsample_factor:end);

            %% Define labels and chance level
            switch memory_mode
                case 'all4'
                    chancelevel = 0.25;
                case {'hit_miss', 'fa_cr'}
                    chancelevel = 0.5;
                otherwise
                    error('Unknown memory_mode: %s', memory_mode);
            end

            %% --- MVPA-LIGHT CONFIGURATION ---
            cfg = [];
            switch memory_mode
                case 'all4'
                    cfg.classifier = 'multiclass_lda';
                    cfg.metric = 'acc';
                    cfg.prior = 'uniform';
                    cfg.reg = 'shrink';
                case {'hit_miss', 'fa_cr'}
                    cfg.classifier = 'lda';
                    cfg.reg = 'shrink';
                    cfg.metric = 'acc';
            end
            cfg.feedback = 0;
            cfg.cv = 'kfold';
            cfg.k = 5;

            %% --- TIME-RESOLVED DECODING ---
            [perf, results] = mv_classify_across_time(cfg, cue_band_ds, y);

            if isempty(all_acc_subjects)
                num_timepoints = size(perf, 1);
                all_acc_subjects = zeros(num_timepoints, numel(subjects));
                all_perm_subjects = zeros(num_timepoints, nPerm, numel(subjects));
            end

            all_acc_subjects(:, s) = perf;
            all_results_subjects{s} = results;
            class_names = class_names_subject;

            %% --- PERMUTATIONS (PARALLEL) ---
            if permutation == 1
                perm_metrics = zeros(size(perf, 1), nPerm);
                parfor p = 1:nPerm
                    y_perm = y(randperm(length(y)));
                    [perf_perm, ~] = mv_classify_across_time(cfg, cue_band_ds, y_perm);
                    perm_metrics(:, p) = perf_perm;
                end
                all_perm_subjects(:, :, s) = perm_metrics;
            end
        end

        if isempty(all_acc_subjects)
            warning('No usable decoding results for ROI %s', ROIs{r});
            continue;
        end

        %% --- REMOVE SUBJECTS WITH ALL-ZERO ACCURACY ---
        subjects_to_keep = any(all_acc_subjects ~= 0, 1);
        all_acc_subjects = all_acc_subjects(:, subjects_to_keep);
        all_results_subjects = all_results_subjects(subjects_to_keep);
        n_elec_subj = n_elec_subj(subjects_to_keep);
        subjects_used = subjects(subjects_to_keep);

        if isempty(all_acc_subjects)
            warning('No non-empty subject results for ROI %s', ROIs{r});
            continue;
        end

        if permutation == 1
            all_perm_subjects = all_perm_subjects(:, :, subjects_to_keep);

            %% --- ZSCORE AGAINST PERMUTATIONS (VECTORIZE) ---
            mu_perm = mean(all_perm_subjects, 2);
            sigma_perm = std(all_perm_subjects, 0, 2);
            zscore_acc = (all_acc_subjects - squeeze(mu_perm)) ./ squeeze(sigma_perm);

            for n = 1:length(all_results_subjects)
                all_results_subjects{1, n}.perf = zscore_acc(:, n);
            end

            mean_z = mean(zscore_acc, 2);
            sem_z = std(zscore_acc, 0, 2) / sqrt(size(zscore_acc, 2));
        end

        time_vec = cue_tfr.time(1:downsample_factor:end);

        %% --- GROUP-LEVEL STATS (vs. Chance) ---
        if permutation == 1
            data_for_stats = zscore_acc;
            null_value = 0;
        else
            data_for_stats = all_acc_subjects;
            null_value = chancelevel;
        end

        nSubj = size(data_for_stats, 2);
        nTime = size(data_for_stats, 1);

        tvals = zeros(nTime, 1);
        pvals = zeros(nTime, 1);

        for t = 1:nTime
            [~, p, ~, stats] = ttest(data_for_stats(t, :), null_value, 'Tail', 'right');
            tvals(t) = stats.tstat;
            pvals(t) = p;
        end

        %% --- CLUSTER-BASED PERMUTATION TEST ---
        t_thresh = 1;
        sig_mask = tvals > t_thresh;

        cc = bwconncomp(sig_mask, 4);
        cluster_tsum = cellfun(@(x) sum(tvals(x)), cc.PixelIdxList);

        nPerm_stat = 1000;
        max_tsum = zeros(nPerm_stat, 1);
        for p = 1:nPerm_stat
            flips = (randi([0 1], [1, nSubj]) * 2 - 1);
            perm_data = data_for_stats - null_value;
            perm_data = perm_data .* flips;
            perm_t = mean(perm_data, 2) ./ (std(perm_data, 0, 2) / sqrt(nSubj));
            perm_mask = perm_t > t_thresh;
            cc_perm = bwconncomp(perm_mask, 4);
            if cc_perm.NumObjects > 0
                max_tsum(p) = max(cellfun(@(x) sum(perm_t(x)), cc_perm.PixelIdxList));
            else
                max_tsum(p) = 0;
            end
        end

        cluster_pvals = arrayfun(@(x) mean(max_tsum >= x), cluster_tsum);
        sig_clusters = find(cluster_pvals < alpha);
        sig_mask_final = false(size(sig_mask));
        for c = sig_clusters
            sig_mask_final(cc.PixelIdxList{c}) = true;
        end
        n_sig_clusters = numel(sig_clusters);

        raw_plot_mean = mean(all_acc_subjects, 2);
        raw_plot_sem = std(all_acc_subjects, 0, 2) / sqrt(nSubj);
        raw_chance_line = chancelevel;

        if permutation == 1
            z_plot_mean = mean_z;
            z_plot_sem = sem_z;
            z_chance_line = 0;
        else
            z_plot_mean = [];
            z_plot_sem = [];
            z_chance_line = [];
        end

        roi_plot_data{r} = struct( ...
            'roi_name', ROIs{r}, ...
            'time_vec', time_vec, ...
            'sig_mask', sig_mask_final, ...
            'raw_mean', raw_plot_mean, ...
            'raw_sem', raw_plot_sem, ...
            'raw_chance_line', raw_chance_line, ...
            'z_mean', z_plot_mean, ...
            'z_sem', z_plot_sem, ...
            'z_chance_line', z_chance_line);

        %% --- PLOT RESULTS ---
        save_roi_plot(output_dir, ROIs{r}, dependent_var, memory_mode, freq_label, win_ms, time_vec, ...
            raw_plot_mean, raw_plot_sem, raw_chance_line, sig_mask_final, 'raw');
        if permutation == 1
            save_roi_plot(output_dir, ROIs{r}, dependent_var, memory_mode, freq_label, win_ms, time_vec, ...
                z_plot_mean, z_plot_sem, z_chance_line, sig_mask_final, 'zscore');
        end

        %% --- SAVE DETAILED STATISTICAL REPORT ---
        txt_filename = fullfile(output_dir, ...
            [ROIs{r} '_' dependent_var '_' memory_mode '_' freq_label '_' num2str(win_ms) '_stats.txt']);

        fid = fopen(txt_filename, 'w');

        fprintf(fid, 'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
        fprintf(fid, '======================================================\n\n');

        %% --- GENERAL INFORMATION ---
        fprintf(fid, 'ROI: %s\n', ROIs{r});
        fprintf(fid, 'Dependent variable: %s\n', dependent_var);
        fprintf(fid, 'Memory mode: %s\n', memory_mode);
        fprintf(fid, 'Classes: %s\n', strjoin(class_names, ', '));
        fprintf(fid, 'Number of subjects: %d\n', nSubj);
        fprintf(fid, 'Smoothing window: %d ms\n', win_ms);
        fprintf(fid, 'Downsample factor: %d\n', downsample_factor);
        fprintf(fid, 'Time resolution: %.6f s\n\n', mean(diff(time_vec)));

        %% --- DECODING CONFIGURATION ---
        fprintf(fid, 'Metric: %s\n', cfg.metric);
        fprintf(fid, 'Frequency range: %d-%d Hz (%s)\n\n', freq_low, freq_high, freq_label);

        %% --- STATISTICAL TEST DETAILS ---
        fprintf(fid, 'Statistical test: sign-flip permutation\n');
        fprintf(fid, 'Test statistic: t-test\n');
        fprintf(fid, 'Multiple comparison correction: cluster\n');
        fprintf(fid, 'Alpha level: %.4f\n', alpha);
        fprintf(fid, 'Number of permutations: %d\n\n', nPerm_stat);

        fprintf(fid, 'Group-level design: within\n');
        fprintf(fid, 'Statistic: t-test\n');
        fprintf(fid, 'Tail: 1\n');
        if permutation == 1
            fprintf(fid, 'Null hypothesis value: %.4f (z-scored)\n', 0);
        else
            fprintf(fid, 'Null hypothesis value: %.4f\n', chancelevel);
        end
        fprintf(fid, 'Permutation test: sign-flip\n');
        fprintf(fid, 'Cluster statistic: maxsum\n');
        fprintf(fid, 'Cluster-forming threshold: %.4f\n', t_thresh);
        fprintf(fid, 'Cluster connectivity: 4-connected (1D)\n\n');

        %% --- CLUSTER RESULTS ---
        fprintf(fid, 'Number of significant clusters: %d\n\n', n_sig_clusters);

        if n_sig_clusters == 0
            fprintf(fid, 'No significant clusters detected.\n');
        else
            for c = 1:numel(sig_clusters)
                cid = sig_clusters(c);
                cluster_idx = cc.PixelIdxList{cid};

                t_start = time_vec(min(cluster_idx));
                t_end = time_vec(max(cluster_idx));

                fprintf(fid, 'Cluster %d\n', c);
                fprintf(fid, '  Time range: %.4f - %.4f s\n', t_start, t_end);
                fprintf(fid, '  Cluster-level p-value: %.6f\n', cluster_pvals(cid));
                fprintf(fid, '\n');
            end
        end

        fclose(fid);

        fprintf('Saved statistical summary:\n%s\n', txt_filename);

        %% --- SUBJECT-LEVEL REINSTATEMENT SCORES ---
        if any(sig_mask_final(:))
            nSubj_kept = size(all_acc_subjects, 2);
            subj_reinstatement = zeros(nSubj_kept, 1);
            subj_reinstatement_effect = zeros(nSubj_kept, 1);

            for subj_idx = 1:nSubj_kept
                subj_acc = all_acc_subjects(:, subj_idx);
                subj_reinstatement(subj_idx) = mean(subj_acc(sig_mask_final));
                subj_reinstatement_effect(subj_idx) = subj_reinstatement(subj_idx) - chancelevel;
            end

            subj_table = table(subjects_used(:), subj_reinstatement, subj_reinstatement_effect, n_elec_subj(:), ...
                'VariableNames', {'Subject', 'MeanAccuracy', 'ReinstateEffect', 'NumElectrodes'});
            csv_filename = fullfile(output_dir, ...
                sprintf('%s_%s_%s_%s_subject_reinstatement.csv', ROIs{r}, dependent_var, memory_mode, freq_label));
            writetable(subj_table, csv_filename);
            fprintf('Saved subject reinstatement scores:\n%s\n', csv_filename);
        else
            fprintf('\nNo significant clusters found for %s - skipping subject-level reinstatement.\n', ROIs{r});
        end

        %% --- SAVE DECODING RESULTS ---
        mat_filename = fullfile(output_dir, [ROIs{r} '_' dependent_var '_' memory_mode '_' freq_label '_' num2str(win_ms) '_results.mat']);
        save(mat_filename, 'all_acc_subjects', 'subjects_used', 'time_vec', ...
             'chancelevel', 'win_ms', 'downsample_factor', 'dependent_var', 'cfg', ...
             'freq_low', 'freq_high', 'freq_label', 'memory_mode', 'class_names');
        fprintf('Saved decoding results:\n%s\n', mat_filename);
    end

    create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, memory_mode, freq_label, win_ms, 'raw');
    if permutation == 1
        create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, memory_mode, freq_label, win_ms, 'zscore');
    end
end

function [y, keep_mask, class_names] = map_memory_outcome_labels(raw_labels, memory_mode)
    labels = normalize_memory_outcome_strings(raw_labels);
    keep_mask = false(numel(labels), 1);
    y = nan(numel(labels), 1);

    switch memory_mode
        case 'all4'
            canonical_labels = {'hits', 'misses', 'false_alarms', 'correct_rejections'};
            for i = 1:numel(labels)
                match_idx = find(strcmp(labels{i}, canonical_labels), 1);
                if ~isempty(match_idx)
                    keep_mask(i) = true;
                    y(i) = match_idx;
                end
            end
            class_names = canonical_labels;
        case 'hit_miss'
            class_names = {'hits', 'misses'};
            for i = 1:numel(labels)
                if strcmp(labels{i}, 'hits')
                    keep_mask(i) = true;
                    y(i) = 1;
                elseif strcmp(labels{i}, 'misses')
                    keep_mask(i) = true;
                    y(i) = 2;
                end
            end
        case 'fa_cr'
            class_names = {'false_alarms', 'correct_rejections'};
            for i = 1:numel(labels)
                if strcmp(labels{i}, 'false_alarms')
                    keep_mask(i) = true;
                    y(i) = 1;
                elseif strcmp(labels{i}, 'correct_rejections')
                    keep_mask(i) = true;
                    y(i) = 2;
                end
            end
        otherwise
            error('Unknown memory_mode: %s', memory_mode);
    end
end

function labels = normalize_memory_outcome_strings(raw_labels)
    if istable(raw_labels)
        raw_labels = table2array(raw_labels);
    end

    if iscategorical(raw_labels) || ischar(raw_labels)
        raw_labels = cellstr(raw_labels);
    elseif isstring(raw_labels)
        raw_labels = cellstr(raw_labels);
    elseif isnumeric(raw_labels) || islogical(raw_labels)
        raw_labels = arrayfun(@num2str, raw_labels, 'UniformOutput', false);
    end

    if ~iscell(raw_labels)
        raw_labels = num2cell(raw_labels);
    end

    labels = cell(numel(raw_labels), 1);
    for i = 1:numel(raw_labels)
        value = raw_labels{i};

        if isempty(value)
            labels{i} = '';
            continue;
        end

        if isstring(value)
            if ismissing(value)
                labels{i} = '';
                continue;
            end
            value = char(value);
        elseif iscell(value)
            if isempty(value)
                labels{i} = '';
                continue;
            end
            value = value{1};
        elseif isnumeric(value) || islogical(value)
            value = num2str(value);
        end

        if isempty(value)
            labels{i} = '';
            continue;
        end

        value = lower(strtrim(value));
        value = regexprep(value, '[\s\-]+', '_');

        switch value
            case {'hit', 'hits'}
                labels{i} = 'hits';
            case {'miss', 'misses'}
                labels{i} = 'misses';
            case {'false_alarm', 'false_alarms', 'fa', 'fas'}
                labels{i} = 'false_alarms';
            case {'correct_rejection', 'correct_rejections', 'cr', 'crs'}
                labels{i} = 'correct_rejections';
            otherwise
                labels{i} = value;
        end
    end
end

function save_roi_plot(output_dir, roi_name, dependent_var, memory_mode, freq_label, win_ms, time_vec, plot_mean, plot_sem, chance_line, sig_mask, plot_mode)
    figure('Position', [200 200 800 400]);
    y_min = min(plot_mean - plot_sem);
    y_max = max(plot_mean + plot_sem);
    y_pad = 0.08 * max(eps, y_max - y_min);
    y_limits = [y_min - y_pad, y_max + 2 * y_pad];

    hold on;
    fill([time_vec, fliplr(time_vec)], ...
         [plot_mean + plot_sem; flipud(plot_mean - plot_sem)]', ...
         [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    plot(time_vec, plot_mean, 'b', 'LineWidth', 2);
    yline(chance_line, 'k--', 'LineWidth', 1);
    shade_significance_bar(time_vec, sig_mask, y_limits, [0.2 0.2 0.8]);

    xlabel('Time (s)');
    if strcmp(plot_mode, 'zscore')
        ylabel('Z-score');
    else
        ylabel('Accuracy');
    end
    title(sprintf('%s: %s decoding (%s, %s)', roi_name, dependent_var, memory_mode, plot_mode));
    set(gca, 'FontSize', 10, 'FontName', 'Arial');
    ylim(y_limits);
    hold off;

    suffix = '';
    if strcmp(plot_mode, 'zscore')
        suffix = '_zscore';
    end
    f = gcf;
    exportgraphics(f, fullfile(output_dir, [roi_name '_' dependent_var '_' memory_mode '_' freq_label '_' num2str(win_ms) suffix '.pdf']), 'Resolution', 300);
end

function create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, memory_mode, freq_label, win_ms, plot_mode)
    valid_idx = find(~cellfun(@isempty, roi_plot_data));
    if isempty(valid_idx)
        return;
    end

    if strcmp(plot_mode, 'zscore')
        scale_factor = 1;
    else
        scale_factor = 100;
    end

    all_lower = [];
    all_upper = [];
    for idx = valid_idx
        roi_data = roi_plot_data{idx};
        [plot_mean, plot_sem] = get_plot_fields(roi_data, plot_mode);
        all_lower = [all_lower; scale_factor * (plot_mean - plot_sem)];
        all_upper = [all_upper; scale_factor * (plot_mean + plot_sem)];
    end

    y_min = min(all_lower);
    y_max = max(all_upper);
    y_pad = 0.08 * max(eps, y_max - y_min);
    y_limits = [y_min - y_pad, y_max + 2 * y_pad];

    nPlots = numel(valid_idx);
    nCols = min(3, nPlots);
    nRows = ceil(nPlots / nCols);

    fig = figure('Position', [100 100 420 * nCols 240 * nRows]);
    set(fig, 'Color', 'w');
    tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ii = 1:nPlots
        idx = valid_idx(ii);
        roi_data = roi_plot_data{idx};
        nexttile;

        if isKey(plot_cfg.roi_colors, roi_data.roi_name)
            roi_color = plot_cfg.roi_colors(roi_data.roi_name);
        else
            roi_color = [0.2 0.2 0.2];
        end

        [plot_mean_raw, plot_sem_raw, chance_line_raw] = get_plot_fields(roi_data, plot_mode);
        plot_mean = scale_factor * plot_mean_raw;
        plot_sem = scale_factor * plot_sem_raw;
        chance_line = scale_factor * chance_line_raw;

        hold on;
        fill([roi_data.time_vec, fliplr(roi_data.time_vec)], ...
             [plot_mean + plot_sem; flipud(plot_mean - plot_sem)]', ...
             roi_color, 'FaceAlpha', 0.28, 'EdgeColor', 'none');
        plot(roi_data.time_vec, plot_mean, 'Color', roi_color, 'LineWidth', plot_cfg.line_width);
        yline(chance_line, 'k--', 'LineWidth', plot_cfg.chance_width);

        shade_significance_bar(roi_data.time_vec, roi_data.sig_mask, y_limits, roi_color);

        xlim([roi_data.time_vec(1), roi_data.time_vec(end)]);
        ylim(y_limits);
        set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        set(gca, 'TickDir', plot_cfg.tick_dir, 'TickLength', plot_cfg.tick_length);
        box off;

        text(0.01, 0.06, roi_data.roi_name, 'Units', 'normalized', ...
            'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_roi, ...
            'Color', [0.2 0.2 0.2], 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');

        if ii > (nRows - 1) * nCols
            xlabel('time since cue (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
        end
        if mod(ii - 1, nCols) == 0
            if strcmp(plot_mode, 'zscore')
                ylabel('z-score', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
            else
                ylabel('classification accuracy (%)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
            end
        else
            set(gca, 'YTickLabel', []);
        end
        hold off;
    end

    combined_filename = fullfile(output_dir, sprintf('all_ROIs_%s_%s_%s_%d_combined_%s.pdf', dependent_var, memory_mode, freq_label, win_ms, plot_mode));
    exportgraphics(fig, combined_filename, 'Resolution', 300);
    fprintf('Saved combined figure:\n%s\n', combined_filename);
end

function [plot_mean, plot_sem, chance_line] = get_plot_fields(roi_data, plot_mode)
    if strcmp(plot_mode, 'zscore')
        plot_mean = roi_data.z_mean;
        plot_sem = roi_data.z_sem;
        chance_line = roi_data.z_chance_line;
    else
        plot_mean = roi_data.raw_mean;
        plot_sem = roi_data.raw_sem;
        chance_line = roi_data.raw_chance_line;
    end
end

function shade_significance_bar(time_vec, sig_mask, y_limits, roi_color)
    if ~any(sig_mask)
        return;
    end

    y_range = y_limits(2) - y_limits(1);
    bar_height = 0.025 * y_range;
    bar_bottom = y_limits(2) - 1.15 * bar_height;
    segments = mask_to_segments(sig_mask);

    for k = 1:size(segments, 1)
        x_start = time_vec(segments(k, 1));
        x_end = time_vec(segments(k, 2));
        patch([x_start x_end x_end x_start], ...
              [bar_bottom bar_bottom bar_bottom + bar_height bar_bottom + bar_height], ...
              roi_color, 'FaceAlpha', 1, 'EdgeColor', 'none');
    end
end

function segments = mask_to_segments(sig_mask)
    idx = find(sig_mask(:)');
    if isempty(idx)
        segments = zeros(0, 2);
        return;
    end

    breakpoints = [1, find(diff(idx) > 1) + 1, numel(idx) + 1];
    segments = zeros(numel(breakpoints) - 1, 2);
    for i = 1:numel(breakpoints) - 1
        seg_idx = idx(breakpoints(i):breakpoints(i + 1) - 1);
        segments(i, :) = [seg_idx(1), seg_idx(end)];
    end
end
