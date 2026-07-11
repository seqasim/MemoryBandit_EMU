%% --- POST HOC ELECTRODE PATTERNS FOR FEEDBACK RPE DECODING ---
% This script loads saved multiclass rpe_category feedback-decoding results,
% reconstructs the significant time window from the companion stats file,
% and computes descriptive binary LDA weights and Haufe activation patterns
% for positive-vs-neutral and positive-vs-negative contrasts.

%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
results_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback/');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback_posthoc_patterns/');

ROIs = {'ACC', 'OFC', 'dmPFC', 'dlPFC'};
dependent_var = 'rpe_category';

downsample_factor = 5;
freq_low = 70;
freq_high = 201;
freq_label = 'hfa';
win_ms = 200;
contrasts = {'positive_vs_neutral', 'positive_vs_negative'};

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));

for r = 1:numel(ROIs)
    roi_name = ROIs{r};
    fprintf('Processing ROI: %s\n', roi_name);

    results_file = fullfile(results_dir, ...
        sprintf('%s_%s_%s_%d_results.mat', roi_name, dependent_var, freq_label, win_ms));
    stats_file = fullfile(results_dir, ...
        sprintf('%s_%s_%s_%d_stats.txt', roi_name, dependent_var, freq_label, win_ms));

    if ~exist(results_file, 'file')
        warning('Results file not found for ROI %s: %s', roi_name, results_file);
        continue;
    end
    if ~exist(stats_file, 'file')
        warning('Stats file not found for ROI %s: %s', roi_name, stats_file);
        continue;
    end

    result_data = load(results_file);
    if ~isfield(result_data, 'subjects_used') || ~isfield(result_data, 'time_vec')
        warning('Results file for ROI %s is missing required fields.', roi_name);
        continue;
    end

    cluster_ranges = parse_cluster_ranges(stats_file);
    if isempty(cluster_ranges)
        fprintf('  No significant clusters found in stats file. Skipping ROI.\n');
        continue;
    end

    sig_mask = false(size(result_data.time_vec));
    for c = 1:size(cluster_ranges, 1)
        sig_mask = sig_mask | ...
            (result_data.time_vec >= cluster_ranges(c, 1) & result_data.time_vec <= cluster_ranges(c, 2));
    end

    if ~any(sig_mask)
        warning('Cluster ranges for ROI %s did not overlap saved time vector.', roi_name);
        continue;
    end

    roi_output = struct();
    roi_output.roi_name = roi_name;
    roi_output.dependent_var = dependent_var;
    roi_output.freq_low = freq_low;
    roi_output.freq_high = freq_high;
    roi_output.freq_label = freq_label;
    roi_output.win_ms = win_ms;
    roi_output.downsample_factor = downsample_factor;
    roi_output.time_vec = result_data.time_vec;
    roi_output.sig_mask = sig_mask;
    roi_output.cluster_ranges = cluster_ranges;
    roi_output.subjects_from_decoding = result_data.subjects_used;
    roi_output.contrasts = struct();

    csv_rows = {};

    for c = 1:numel(contrasts)
        contrast_name = contrasts{c};
        [class_a, class_b] = get_contrast_categories(contrast_name);

        contrast_struct = struct();
        contrast_struct.name = contrast_name;
        contrast_struct.class_a = class_a;
        contrast_struct.class_b = class_b;
        contrast_struct.subjects = {};
        contrast_struct.subject_electrodes = {};
        contrast_struct.subject_coords = {};
        contrast_struct.subject_weights = {};
        contrast_struct.subject_patterns = {};
        contrast_struct.subject_trial_counts = [];
        contrast_struct.subject_window_data = {};
        contrast_struct.group_all_pattern_values = [];
        contrast_struct.group_all_weight_values = [];

        for s = 1:numel(result_data.subjects_used)
            subj = result_data.subjects_used{s};
            fprintf('  %s | %s\n', subj, contrast_name);

            subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
            if isempty(subj_learn_df)
                warning('No learn_df rows found for subject %s', subj);
                continue;
            end

            roi = get_roi_definition(roi_name);

            elec_file = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'Day1_reref_elec_df');
            if ~exist(elec_file, 'file')
                warning('Electrode table missing for subject %s', subj);
                continue;
            end
            elec_df = readtable(elec_file);
            roi_electrodes = elec_df.label(find(matches(elec_df.salman_region, roi)));
            if isempty(roi_electrodes)
                continue;
            end

            tfr_file = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'scratch', 'TFR', 'feedback_start-tfr.mat');
            if ~exist(tfr_file, 'file')
                warning('Feedback TFR missing for subject %s', subj);
                continue;
            end
            fb = load(tfr_file);
            feedback_tfr = struct();
            feedback_tfr.powspctrm = fb.powspctrm;
            feedback_tfr.freq = fb.freqs;
            feedback_tfr.time = fb.times;
            feedback_tfr.label = fb.ch_names;
            feedback_tfr.fsample = fb.sfreq;

            [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
            valid_elec_mask = idx_fb > 0;
            idx_fb = idx_fb(valid_elec_mask);
            roi_electrodes = roi_electrodes(valid_elec_mask);
            if isempty(idx_fb)
                continue;
            end

            feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
            roi_elec_df = elec_df(find(matches(elec_df.salman_region, roi)), :);
            roi_elec_df = roi_elec_df(valid_elec_mask, :);

            freq_mask = feedback_tfr.freq >= freq_low & feedback_tfr.freq <= freq_high;
            feedback_band = squeeze(mean(feedback_tfr.powspctrm(:, :, freq_mask, :), 3));
            if ndims(feedback_band) < 3
                feedback_band = reshape(feedback_band, [size(feedback_band, 1), 1, size(feedback_band, 2)]);
            end

            fs = feedback_tfr.fsample;
            win_size = round(fs * (win_ms / 1000));
            feedback_band_smooth = movmean(feedback_band, win_size, 3);
            feedback_band_smooth = interpolate_artifacts(feedback_band_smooth, 10);

            encoding_trials = subj_learn_df.trials;
            if isempty(encoding_trials) || max(encoding_trials) > size(feedback_band_smooth, 1)
                warning('Invalid encoding trials for subject %s', subj);
                continue;
            end

            edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
            subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, ...
                'categorical', {'negative', 'neutral', 'positive'});

            feedback_band_smooth = feedback_band_smooth(encoding_trials, :, :);
            y_all = subj_learn_df.rpe_category;
            y_all = y_all(encoding_trials);
            y_all_str = string(y_all);

            feedback_band_ds = feedback_band_smooth(:, :, 1:downsample_factor:end);
            time_vec_ds = feedback_tfr.time(1:downsample_factor:end);

            if numel(time_vec_ds) ~= numel(result_data.time_vec)
                if numel(time_vec_ds) < numel(result_data.time_vec)
                    warning('Time-vector mismatch for subject %s in ROI %s', subj, roi_name);
                    continue;
                end
                time_match_mask = false(size(time_vec_ds));
                for t = 1:numel(result_data.time_vec)
                    [~, idx_nearest] = min(abs(time_vec_ds - result_data.time_vec(t)));
                    time_match_mask(idx_nearest) = sig_mask(t) | time_match_mask(idx_nearest);
                end
                sig_mask_subject = time_match_mask;
            else
                sig_mask_subject = sig_mask;
            end

            X_window = squeeze(mean(feedback_band_ds(:, :, sig_mask_subject), 3));
            if isvector(X_window)
                X_window = reshape(X_window, [size(feedback_band_ds, 1), numel(roi_electrodes)]);
            end

            class_mask = y_all_str == class_a | y_all_str == class_b;
            if ~any(class_mask)
                continue;
            end

            y_pair = y_all_str(class_mask);
            X_pair = X_window(class_mask, :);

            n_class_a = sum(y_pair == class_a);
            n_class_b = sum(y_pair == class_b);
            if n_class_a == 0 || n_class_b == 0
                continue;
            end

            y_binary = nan(size(y_pair));
            y_binary(y_pair == class_a) = 1;
            y_binary(y_pair == class_b) = 2;

            param = mv_get_hyperparameter('lda');
            param.reg = 'shrink';
            param.lambda = 'auto';
            param.form = 'auto';
            param.prob = 0;
            param.scale = 0;

            cf = train_lda(param, X_pair, y_binary);
            pattern = mv_stat_activation_pattern(cf, X_pair, y_binary);

            coords = extract_coordinate_table(roi_elec_df);
            coords = coords(1:numel(roi_electrodes), :);

            contrast_struct.subjects{end+1, 1} = subj;
            contrast_struct.subject_electrodes{end+1, 1} = roi_electrodes;
            contrast_struct.subject_coords{end+1, 1} = coords;
            contrast_struct.subject_weights{end+1, 1} = cf.w(:);
            contrast_struct.subject_patterns{end+1, 1} = pattern(:);
            contrast_struct.subject_trial_counts(end+1, :) = [n_class_a, n_class_b];
            contrast_struct.subject_window_data{end+1, 1} = X_pair;
            contrast_struct.group_all_pattern_values = [contrast_struct.group_all_pattern_values; pattern(:)];
            contrast_struct.group_all_weight_values = [contrast_struct.group_all_weight_values; cf.w(:)];

            cluster_start = min(result_data.time_vec(sig_mask));
            cluster_end = max(result_data.time_vec(sig_mask));

            for e = 1:numel(roi_electrodes)
                csv_rows(end+1, :) = build_csv_row( ...
                    roi_name, contrast_name, subj, roi_electrodes{e}, ...
                    pattern(e), cf.w(e), cluster_start, cluster_end, coords, e); %#ok<AGROW>
            end
        end

        if ~isempty(contrast_struct.group_all_pattern_values)
            contrast_struct.group_mean_pattern = mean(contrast_struct.group_all_pattern_values, 'omitnan');
            contrast_struct.group_sem_pattern = std(contrast_struct.group_all_pattern_values, 0, 'omitnan') / ...
                sqrt(sum(~isnan(contrast_struct.group_all_pattern_values)));
            contrast_struct.group_mean_weight = mean(contrast_struct.group_all_weight_values, 'omitnan');
            contrast_struct.group_sem_weight = std(contrast_struct.group_all_weight_values, 0, 'omitnan') / ...
                sqrt(sum(~isnan(contrast_struct.group_all_weight_values)));
        else
            contrast_struct.group_mean_pattern = NaN;
            contrast_struct.group_sem_pattern = NaN;
            contrast_struct.group_mean_weight = NaN;
            contrast_struct.group_sem_weight = NaN;
        end

        roi_output.contrasts.(contrast_name) = contrast_struct;
    end

    mat_file = fullfile(output_dir, sprintf('%s_%s_%s_%d_posthoc_patterns.mat', ...
        roi_name, dependent_var, freq_label, win_ms));
    save(mat_file, 'roi_output');
    fprintf('Saved ROI post hoc patterns: %s\n', mat_file);

    if ~isempty(csv_rows)
        csv_table = cell2table(csv_rows, 'VariableNames', ...
            {'ROI', 'Contrast', 'Subject', 'Electrode', 'PatternValue', 'WeightValue', ...
             'ClusterStart', 'ClusterEnd', 'CoordX', 'CoordY', 'CoordZ'});
        csv_file = fullfile(output_dir, sprintf('%s_%s_%s_%d_posthoc_patterns.csv', ...
            roi_name, dependent_var, freq_label, win_ms));
        writetable(csv_table, csv_file);
        fprintf('Saved ROI post hoc CSV: %s\n', csv_file);
    end
end

function cluster_ranges = parse_cluster_ranges(stats_file)
    file_text = fileread(stats_file);
    lines = splitlines(string(file_text));
    cluster_ranges = [];

    for i = 1:numel(lines)
        line = strtrim(lines(i));
        if startsWith(line, 'Time range:')
            values = regexp(line, '[-+]?\d*\.?\d+', 'match');
            if numel(values) >= 2
                cluster_ranges(end+1, :) = [str2double(values{1}), str2double(values{2})]; %#ok<AGROW>
            end
        end
    end
end

function roi = get_roi_definition(roi_name)
    if strcmp(roi_name, 'PFC')
        roi = {'OFC', 'ACC', 'dlPFC', 'dmPFC'};
    elseif strcmp(roi_name, 'MTL')
        roi = {'HPC', 'AMY', 'PHC', 'EC'};
    else
        roi = roi_name;
    end
end

function data_out = interpolate_artifacts(data_in, z_thresh)
    data_out = data_in;
    [nTrials, nElec, ~] = size(data_out);

    for trial = 1:nTrials
        for elec = 1:nElec
            trace = squeeze(data_out(trial, elec, :));
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

                data_out(trial, elec, :) = trace;
            end
        end
    end
end

function [class_a, class_b] = get_contrast_categories(contrast_name)
    switch contrast_name
        case 'positive_vs_neutral'
            class_a = "positive";
            class_b = "neutral";
        case 'positive_vs_negative'
            class_a = "positive";
            class_b = "negative";
        otherwise
            error('Unknown contrast: %s', contrast_name);
    end
end

function coords = extract_coordinate_table(elec_df)
    coord_candidates = {
        {'mni_x', 'mni_y', 'mni_z'}
        {'MNI_x', 'MNI_y', 'MNI_z'}
        {'x', 'y', 'z'}
        {'X', 'Y', 'Z'}
    };

    coords = nan(height(elec_df), 3);
    for i = 1:numel(coord_candidates)
        cols = coord_candidates{i};
        if all(ismember(cols, elec_df.Properties.VariableNames))
            coords = [elec_df.(cols{1}), elec_df.(cols{2}), elec_df.(cols{3})];
            return;
        end
    end
end

function row = build_csv_row(roi_name, contrast_name, subj, electrode, pattern_value, weight_value, cluster_start, cluster_end, coords, idx)
    row = {
        roi_name, ...
        contrast_name, ...
        subj, ...
        electrode, ...
        pattern_value, ...
        weight_value, ...
        cluster_start, ...
        cluster_end, ...
        coords(idx, 1), ...
        coords(idx, 2), ...
        coords(idx, 3)};
end
