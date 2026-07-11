%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback_univariate/');

% 'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY', 'Temporal','PFC'
ROIs = {'OFC'}; % , 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY', 'Temporal','PFC'
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

downsample_factor = 5;

% Frequency band configuration
freq_low = 70;
freq_high = 201;
freq_label = 'hfa';
win_ms = 150;

permutation = 0;
nPerm = 250;
alpha = 0.05;

%% --- LOAD TABLES ONCE ---
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
    
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
learn_df.Q = learn_df.rpe + learn_df.reward;

combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- LOOP OVER DEPENDENT VARIABLES ---
% dependent_vars = {'reward', 'phit'};
dependent_vars = {'rpe_category'};
for d = 1:numel(dependent_vars)
    dependent_var = dependent_vars{d};
    fprintf('Processing variable: %s\n', dependent_var);

    if strcmp(dependent_var, 'rpe')
        permutation = 1;
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
        n_elec_subjects = zeros(1, numel(subjects));

        %% --- LOOP OVER SUBJECTS ---
        for s = 1:numel(subjects)
            subj = subjects{s};
            fprintf('  Subject: %s\n', subj);

            %% --- SELECT SUBJECT DATA ---
            subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
            subj_combined_df = combined_df(strcmp(combined_df.condition, 'Day1') & strcmp(combined_df.participant, subj), :);
            subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);

            subj_combined_df.phit = double(strcmp(subj_combined_df.hits, 'True'));

            edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
            subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});

            edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
            subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative', 'neutral', 'positive'});

            if strcmp(dependent_var, 'phit')
                encoding_trials = subj_combined_df.trials_gamble;
            else
                encoding_trials = subj_learn_df.trials;
            end

            if isempty(encoding_trials)
                warning('No trials for subject %s', subj);
                continue;
            end

            %% Load electrode info
            elec_df = readtable(fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'Day1_reref_elec_df'));
            roi_electrodes = elec_df.label(find(matches(elec_df.salman_region, roi)));
            if isempty(roi_electrodes)
                warning('No electrodes found for %s in %s', subj, ROIs{r});
                continue;
            end

            %% Load feedback TFR data
            filepath = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', subj, 'scratch', 'TFR');
            fb = load(fullfile(filepath, 'feedback_start-tfr.mat'));
            feedback_tfr.powspctrm = fb.powspctrm;
            feedback_tfr.freq = fb.freqs;
            feedback_tfr.time = fb.times;
            feedback_tfr.label = fb.ch_names;
            feedback_tfr.fsample = fb.sfreq;

            %% Select ROI electrodes
            [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
            idx_fb = idx_fb(idx_fb > 0);
            if isempty(idx_fb)
                warning('No matching electrodes found for %s in %s', subj, ROIs{r});
                continue;
            end
            feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);

            %% Extract frequency band power
            freq_mask = feedback_tfr.freq >= freq_low & feedback_tfr.freq <= freq_high;
            feedback_band = squeeze(mean(feedback_tfr.powspctrm(:, :, freq_mask, :), 3));
            if ndims(feedback_band) < 3
                feedback_band = reshape(feedback_band, [size(feedback_band, 1), 1, size(feedback_band, 2)]);
            end

            %% --- SMOOTH ---
            fs = feedback_tfr.fsample;
            win_size = round(fs * (win_ms / 1000));
            feedback_band_smooth = movmean(feedback_band, win_size, 3);

            %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
            feedback_band_smooth = interpolate_artifacts(feedback_band_smooth, 10);

            %% Select encoding trials
            if max(encoding_trials) > size(feedback_band_smooth, 1)
                warning('Trial index exceeds available feedback trials for subject %s', subj);
                continue;
            end
            feedback_band_smooth = feedback_band_smooth(encoding_trials, :, :);

            %% Downsample
            feedback_band_ds = feedback_band_smooth(:, :, 1:downsample_factor:end);

            %% Define labels
            switch dependent_var
                case 'reward'
                    y = subj_learn_df.(dependent_var);
                    chancelevel = 0.5;
                case 'rpe'
                    y = subj_learn_df.(dependent_var);
                    chancelevel = 0;
                case 'phit'
                    y = subj_combined_df.(dependent_var);
                    chancelevel = 0.5;
                case 'rpe_category'
                    y = subj_learn_df.(dependent_var);
                    chancelevel = 0.33;
                otherwise
                    error('Unknown dependent variable: %s', dependent_var);
            end
            y = double(y);

            %% --- MVPA-LIGHT CONFIGURATION ---
            cfg = [];
            switch dependent_var
                case 'reward'
                    cfg.classifier = 'lda';
                    cfg.metric = 'acc';
                    cfg.cv = 'leaveout';
                case 'phit'
                    cfg.classifier = 'lda';
                    cfg.metric = 'acc';
                    cfg.cv = 'leaveout';
                case 'rpe_category'
                    cfg.classifier = 'multiclass_lda';
                    cfg.metric = 'acc';
                    cfg.prior = 'uniform';
                    cfg.reg = 'ridge';
                    cfg.cv = 'leaveout';
                case 'rpe'
                    cfg.reg = 'ridge';
                    cfg.metric = 'r_squared';
                    cfg.hyperparameter = [];
                    cfg.hyperparameter = 'lambda';
                    cfg.lambda = 'auto';
                    cfg.dimension_names = {'samples' 'channels', 'time points'};
            end
            cfg.feedback = 0;

            is_regression = strcmp(dependent_var, 'rpe');
            [perf, results, perm_metrics, n_elec_used] = run_univariate_across_time(feedback_band_ds, y, cfg, is_regression, permutation == 1, nPerm);

            if isempty(all_acc_subjects)
                num_timepoints = size(perf, 1);
                all_acc_subjects = zeros(num_timepoints, numel(subjects));
                if permutation == 1
                    all_perm_subjects = zeros(num_timepoints, nPerm, numel(subjects));
                end
            end

            all_acc_subjects(:, s) = perf;
            all_results_subjects{s} = results;
            n_elec_subjects(s) = n_elec_used;

            if permutation == 1
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
        n_elec_subjects = n_elec_subjects(subjects_to_keep);

        if isempty(all_acc_subjects)
            warning('No non-empty subject results for ROI %s', ROIs{r});
            continue;
        end

        if permutation == 1
            all_perm_subjects = all_perm_subjects(:, :, subjects_to_keep);

            %% --- ZSCORE AGAINST PERMUTATIONS ---
            mu_perm = mean(all_perm_subjects, 2);
            sigma_perm = std(all_perm_subjects, 0, 2);
            zscore_acc = (all_acc_subjects - squeeze(mu_perm)) ./ squeeze(sigma_perm);

            for n = 1:length(all_results_subjects)
                all_results_subjects{1, n}.perf = zscore_acc(:, n);
            end

            mean_z = mean(zscore_acc, 2);
            sem_z = std(zscore_acc, 0, 2) / sqrt(size(zscore_acc, 2));
        end

        time_vec = feedback_tfr.time(1:downsample_factor:end);

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

        %% --- PLOT RESULTS ---
        figure('Position', [200 200 800 400]);
        hold on;

        if permutation == 1
            plot_mean = mean_z;
            plot_sem = sem_z;
            chance_line = 0;
            ylabel_text = 'Z-score';
        else
            plot_mean = mean(all_acc_subjects, 2);
            plot_sem = std(all_acc_subjects, 0, 2) / sqrt(nSubj);
            chance_line = chancelevel;
            ylabel_text = 'Accuracy';
        end

        fill([time_vec, fliplr(time_vec)], ...
             [plot_mean + plot_sem; flipud(plot_mean - plot_sem)]', ...
             [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(time_vec, plot_mean, 'b', 'LineWidth', 2);
        yline(chance_line, 'k--', 'LineWidth', 1);

        if any(sig_mask_final)
            sig_times = time_vec(sig_mask_final);
            yl = ylim;
            for i = 1:length(sig_times)
                plot([sig_times(i) sig_times(i)], yl, 'r', 'LineWidth', 2);
            end
        end

        xlabel('Time (s)');
        ylabel(ylabel_text);
        title(sprintf('%s: %s decoding', ROIs{r}, dependent_var));
        set(gca, 'FontSize', 10, 'FontName', 'Arial');
        hold off;

        f = gcf;
        exportgraphics(f, fullfile(output_dir, [ROIs{r} '_' dependent_var '_' freq_label '_' num2str(win_ms) '.pdf']), 'Resolution', 300);

        %% --- SAVE DETAILED STATISTICAL REPORT ---
        txt_filename = fullfile(output_dir, [ROIs{r} '_' dependent_var '_' freq_label '_' num2str(win_ms) '_stats.txt']);
        fid = fopen(txt_filename, 'w');

        fprintf(fid, 'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
        fprintf(fid, '======================================================\n\n');
        fprintf(fid, 'ROI: %s\n', ROIs{r});
        fprintf(fid, 'Dependent variable: %s\n', dependent_var);
        fprintf(fid, 'Number of subjects: %d\n', nSubj);
        fprintf(fid, 'Mean electrodes per subject: %.2f\n', mean(n_elec_subjects));
        fprintf(fid, 'Electrode aggregation: within-subject mean\n');
        fprintf(fid, 'Smoothing window: %d ms\n', win_ms);
        fprintf(fid, 'Downsample factor: %d\n', downsample_factor);
        fprintf(fid, 'Time resolution: %.6f s\n\n', mean(diff(time_vec)));

        fprintf(fid, 'Metric: %s\n', cfg.metric);
        fprintf(fid, 'Frequency range: %d-%d Hz (%s)\n\n', freq_low, freq_high, freq_label);

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
                fprintf(fid, '  Time range: %.4f – %.4f s\n', t_start, t_end);
                fprintf(fid, '  Cluster-level p-value: %.6f\n\n', cluster_pvals(cid));
            end
        end

        fclose(fid);
        fprintf('Saved statistical summary:\n%s\n', txt_filename);

        %% --- SAVE DECODING RESULTS ---
        mat_filename = fullfile(output_dir, [ROIs{r} '_' dependent_var '_' freq_label '_' num2str(win_ms) '_results.mat']);
        subjects_used = subjects(subjects_to_keep);
        save(mat_filename, 'all_acc_subjects', 'subjects_used', 'time_vec', ...
             'chancelevel', 'win_ms', 'downsample_factor', 'dependent_var', 'cfg', ...
             'freq_low', 'freq_high', 'freq_label', 'n_elec_subjects');
        fprintf('Saved decoding results:\n%s\n', mat_filename);
    end
end

function [perf, results, perm_metrics, n_elec] = run_univariate_across_time(X, y, cfg, is_regression, do_permutation, nPerm)
    n_elec = size(X, 2);
    elec_perf = [];

    for e = 1:n_elec
        X_e = X(:, e, :);
        if is_regression
            perf_e = mv_regress(cfg, X_e, y);
        else
            perf_e = mv_classify_across_time(cfg, X_e, y);
        end
        elec_perf(:, e) = perf_e(:);
    end

    perf = mean(elec_perf, 2);
    results = struct('perf', perf, 'n_electrodes', n_elec);

    if do_permutation
        n_time = size(elec_perf, 1);
        perm_metrics = zeros(n_time, nPerm);
        parfor p = 1:nPerm
            y_perm = y(randperm(length(y)));
            perm_elec = zeros(n_time, n_elec);
            for e = 1:n_elec
                X_e = X(:, e, :);
                if is_regression
                    perf_perm = mv_regress(cfg, X_e, y_perm);
                else
                    perf_perm = mv_classify_across_time(cfg, X_e, y_perm);
                end
                perm_elec(:, e) = perf_perm(:);
            end
            perm_metrics(:, p) = mean(perm_elec, 2);
        end
    else
        perm_metrics = [];
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
