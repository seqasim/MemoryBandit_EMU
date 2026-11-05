%% --- CONFIGURATION ---
path_cfg = config_paths();
base_dir = path_cfg.base_dir;
load_dir = path_cfg.load_dir;
output_dir = fullfile(path_cfg.mvpa_dir, 'decoding_feedback');

rng(42, 'twister');


% ROIs = {['dlPFC', 'OFC', 'ACC', 'dmPFC']};  % list of ROIs
% 'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY', 'Temporal','PFC'
if ~exist('ROIs', 'var') || isempty(ROIs)
    ROIs = {'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY'};
end
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

downsample_factor = 5;            % match your Python version

% Frequency band configuration
if ~exist('freq_low', 'var'), freq_low = 70; end
if ~exist('freq_high', 'var'), freq_high = 201; end
if ~exist('freq_label', 'var'), freq_label = 'hfa'; end
if ~exist('win_ms', 'var'), win_ms = 200; end
% Example configurations for common bands:
% Theta:  freq_low = 4;   freq_high = 8;   freq_label = 'theta'; win_ms = 25;
% Alpha:  freq_low = 8;   freq_high = 12;  freq_label = 'alpha'; win_ms = 50;
% Beta:   freq_low = 12;  freq_high = 30;  freq_label = 'beta';  win_ms = 75;
% Gamma:  freq_low = 30;  freq_high = 70;  freq_label = 'gamma'; win_ms = 100;
% HFA:    freq_low = 70;  freq_high = 200; freq_label = 'HFA';   win_ms = 150;

permutation = 1;
nPerm = 250;                         % number of permutations for significance
alpha = 0.05;                     % cluster-level significance
% thresh = 0.5;
% min_cluster_pts = round(0.05 * 100);  % e.g., fs=250 => 38 points

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
class_names = {'negative', 'neutral', 'positive'};

%% --- LOAD TABLES ONCE ---
learn_df = readtable(path_cfg.learn_csv);

% compute Q:
learn_df.Q = learn_df.rpe + learn_df.reward;

learn_df.pm_category = categorical(learn_df.DPRIME > median(learn_df.DPRIME,'omitnan'), [false true], {'low','high'});

combined_df = readtable(path_cfg.combined_csv);

%% --- LOOP OVER DEPENDENT VARIABLES ---
% dependent_vars = {'reward', 'phit', 'pm_category'};  % 'reward' or 'rpe_category'
if ~exist('dependent_vars', 'var') || isempty(dependent_vars)
    dependent_vars = {'rpe_category'};
end
for d = 1:numel(dependent_vars)
    dependent_var = dependent_vars{d};
    fprintf('Processing variable: %s\n', dependent_var);

    roi_plot_data = cell(1, numel(ROIs));

    % Force permutations on for RPE
    if strcmp(dependent_var, 'rpe')
        permutation = 1;
    end

    %% --- LOOP OVER ROIs ---
    for r = 1:numel(ROIs)
%         roi = ROIs{r};
        if strcmp(ROIs{r}, 'PFC')
            roi = {'OFC', 'ACC', 'dlPFC', 'dmPFC'};
        elseif strcmp(ROIs{r}, 'MTL')
            roi = {'HPC', 'AMY', 'PHC', 'EC'};
        else
            roi = ROIs{r};
        end

        fprintf('Processing ROI: %s\n', ROIs{r} );

        % Preallocate arrays for all subjects
        max_timepoints = 1000; % placeholder; will be adjusted dynamically
        all_acc_subjects = [];
        all_perm_subjects = [];
        diagnostic_feedback_subjects = cell(1, numel(subjects));
        diagnostic_y_subjects = cell(1, numel(subjects));
        sig_window_confusion_subjects = [];
        sig_window_precision_subjects = [];
        sig_window_recall_subjects = [];
        sig_window_f1_subjects = [];
        sig_window_time_mask = [];
        sig_window_stats_table = [];
        all_results_subjects = cell(1, numel(subjects));  % store results struct
        n_elec_subj = zeros(1, numel(subjects));

        %% --- LOOP OVER SUBJECTS ---
        for s = 1:numel(subjects)
            subj = subjects{s};
            fprintf('  Subject: %s\n', subj);

            %% --- SELECT SUBJECT DATA ---
            subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
            subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
            subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);

            %% Add memory information
            subj_combined_df.phit = double(strcmp(subj_combined_df.hits, 'True'));

            %% Process RPE categories using tercile split within subject
            edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
            subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});

            edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
            subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative', 'neutral', 'positive'});

            %% Trial mapping and remove fast RTs
            if strcmp(dependent_var, 'phit')
                encoding_trials = subj_combined_df.trials_gamble;
%                 mask_rt = subj_combined_df.gamble_rt_gamble >= 0.3;
%                 encoding_trials = encoding_trials(mask_rt);
            else
                encoding_trials = subj_learn_df.trials;
%                 mask_rt = subj_learn_df.gamble_rt >= 0.3;
%                 encoding_trials = encoding_trials(mask_rt);
            end

%             %% Trial mapping
%             if strcmp(dependent_var, 'phit')
%                 encoding_trials = subj_combined_df.trials_gamble;
%             else
%                 encoding_trials = subj_learn_df.trials;
%             end

            if isempty(encoding_trials)
                warning('No trials for subject %s', subj);
                continue;
            end

            %% Load electrode info
            elec_df = readtable(fullfile(base_dir, 'projects','guLab','Salman','EphysAnalyses', subj, 'Day1_reref_elec_df'));
%             roi_electrodes = elec_df.label(strcmp(elec_df.salman_region, roi));
            roi_electrodes = elec_df.label(find(matches(elec_df.salman_region,roi)));
            if isempty(roi_electrodes)
                warning('No electrodes found for %s in %s', subj, ROIs{r} );
                continue;
            end

            %% Load feedback TFR data
            filepath = fullfile(base_dir,'projects','guLab','Salman','EphysAnalyses',subj,'scratch','TFR');
            fb = load(fullfile(filepath, 'feedback_start-tfr.mat'));
            feedback_tfr.powspctrm = fb.powspctrm;
            feedback_tfr.freq = fb.freqs;
            feedback_tfr.time = fb.times;
            feedback_tfr.label = fb.ch_names;
            feedback_tfr.fsample = fb.sfreq;

            %% Select ROI electrodes
            [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
            valid_elec_mask = idx_fb > 0;
            idx_fb = idx_fb(valid_elec_mask);
            if isempty(idx_fb)
                warning('No matching feedback electrodes found for %s in %s', subj, ROIs{r});
                continue;
            end
            n_elec_subj(s) = numel(idx_fb);
            feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);

            %% Extract frequency band power
            freq_mask = feedback_tfr.freq >= freq_low & feedback_tfr.freq <= freq_high;
            feedback_band = squeeze(mean(feedback_tfr.powspctrm(:, :, freq_mask, :), 3));  % trials x electrodes x time
            Ndims = ndims(feedback_band);
            if Ndims < 3
                feedback_band = reshape(feedback_band, [size(feedback_band, 1), 1, size(feedback_band, 2)]);
            end
            %% --- SMOOTH (VECTORIZE) ---
            fs = feedback_tfr.fsample;
            win_size = round(fs * (win_ms/1000));
            feedback_band_smooth = movmean(feedback_band, win_size, 3);  % vectorized smoothing

            %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
            z_thresh = 10;
            [nTrials_raw, nElec, nTime] = size(feedback_band_smooth);

            for trial = 1:nTrials_raw
                for elec = 1:nElec
                    trace = squeeze(feedback_band_smooth(trial, elec, :));
                    bad_idx = abs(trace) > z_thresh;

                    if any(bad_idx)
                        good_idx = find(~bad_idx);
                        bad_idx_list = find(bad_idx);

                        if numel(good_idx) >= 2
                            % Interpolate bad points from good neighbors
                            trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                        else
                            % Fallback: clip if not enough good points to interpolate
                            trace(trace > z_thresh) = z_thresh;
                            trace(trace < -z_thresh) = -z_thresh;
                        end

                        feedback_band_smooth(trial, elec, :) = trace;
                    end
                end
            end

            %% Select encoding trials
            feedback_band_smooth = feedback_band_smooth(encoding_trials,:,:);

            %% Downsample
            feedback_band_ds = feedback_band_smooth(:,:,1:downsample_factor:end);

        %% Define labels
        switch dependent_var
            case 'reward'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.5;
            case 'rpe'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0;
            case 'phit'
                y = subj_combined_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.5;
            case 'rpe_category'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.33;
            case 'pm_category'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.5;
        end
            y = double(y);

            %% --- MVPA-LIGHT CONFIGURATION ---
            cfg = [];
            switch dependent_var
                case 'reward'
                    cfg.classifier = 'lda'; cfg.metric = 'acc';
                    cfg.reg = 'shrink';
                    cfg.cv = 'kfold';
                    cfg.k = 5;
                case 'phit'
                    cfg.classifier = 'lda'; cfg.metric = 'acc';
                    cfg.reg = 'shrink';
                    cfg.cv = 'kfold';
                    cfg.k = 5;
                case 'pm_category'
                    cfg.classifier = 'lda'; cfg.metric = 'acc';
                    cfg.reg = 'shrink';
                    cfg.cv = 'kfold';
                    cfg.k = 5;
                case 'rpe_category'
                    cfg.classifier = 'multiclass_lda';
                    cfg.metric = 'acc';
                    cfg.prior = 'uniform';
%                     cfg.reg = 'ridge';
                    cfg.reg = 'shrink';
%                     cfg.cv = 'leaveout';
                    cfg.cv = 'kfold';
                    cfg.k = 5;
                case 'rpe'
%                     cfg.reg = 'ridge';
                    cfg.reg = 'shrink';
                    cfg.metric = 'r_squared';
                    cfg.hyperparameter = [];
                    cfg.hyperparameter = 'lambda';
                    cfg.lambda = 'auto';
                    cfg.dimension_names = {'samples' 'channels', 'time points'};
            end
            cfg.feedback = 0;

            %% --- TIME-RESOLVED DECODING ---
            if strcmp(dependent_var, 'rpe')
                [perf, results] = mv_regress(cfg, feedback_band_ds, y);
            else
                [perf, results] = mv_classify_across_time(cfg, feedback_band_ds, y);
            end

            is_rpe_category = strcmp(dependent_var, 'rpe_category');

            % Preallocate arrays if first subject
            if isempty(all_acc_subjects)
                num_timepoints = size(perf,1);
                all_acc_subjects = zeros(num_timepoints, numel(subjects));
                all_perm_subjects = zeros(num_timepoints, nPerm, numel(subjects));
            end
            all_acc_subjects(:,s) = perf;
            if is_rpe_category
                diagnostic_feedback_subjects{s} = feedback_band_ds;
                diagnostic_y_subjects{s} = y;
            end
            all_results_subjects{s} = results;

            %% --- PERMUTATIONS (PARALLEL) ---
            if permutation == 1
                perm_metrics = zeros(size(perf,1), nPerm);
                parfor p = 1:nPerm
                    y_perm = y(randperm(length(y)));
                    if strcmp(dependent_var, 'rpe')
                        [perf_perm, ~] = mv_regress(cfg, feedback_band_ds, y_perm);
                    else
                        cfg_perm = cfg;
                        cfg_perm.metric = 'acc';
                        [perf_perm, ~] = mv_classify_across_time(cfg_perm, feedback_band_ds, y_perm);
                    end
                    perm_metrics(:,p) = perf_perm;
                end
                all_perm_subjects(:,:,s) = perm_metrics;
            end

        end

        %% --- REMOVE SUBJECTS WITH ALL-ZERO ACCURACY ---
        subjects_to_keep = any(all_acc_subjects ~= 0, 1);
        all_acc_subjects = all_acc_subjects(:,subjects_to_keep);
        if strcmp(dependent_var, 'rpe_category')
            diagnostic_feedback_subjects = diagnostic_feedback_subjects(subjects_to_keep);
            diagnostic_y_subjects = diagnostic_y_subjects(subjects_to_keep);
        end
        all_results_subjects = all_results_subjects(subjects_to_keep);
        n_elec_subj = n_elec_subj(subjects_to_keep);
        subjects_used = subjects(subjects_to_keep);

        if permutation == 1
            all_perm_subjects = all_perm_subjects(:, :, subjects_to_keep);

            %% --- ZSCORE AGAINST PERMUTATIONS (VECTORIZE) ---
            mu_perm = mean(all_perm_subjects,2);
            sigma_perm = std(all_perm_subjects,0,2);
            zscore_acc = (all_acc_subjects - squeeze(mu_perm)) ./ squeeze(sigma_perm);

            for n = 1:length(all_results_subjects)
                all_results_subjects{1, n}.perf = zscore_acc(:, n);
            end

            % Average across subjects
            mean_z = mean(zscore_acc,2);
            sem_z  = std(zscore_acc,0,2)/sqrt(size(zscore_acc,2));
    %
        end
%         time_vec = feedback_tfr.time(1:downsample_factor:end);
%         cfg_stat = [];
%         cfg_stat.metric          =  cfg.metric;
%         cfg_stat.test            = 'permutation';
%         cfg_stat.correctm        = 'cluster';  % correction method is cluster
%         cfg_stat.n_permutations  = 1000;
%         % Clusterstatistic is the actual statistic used for the clustertest.
%         % Normally the default value 'maxum' is used, we are setting it here
%         % explicitly for clarity. Maxsum adds up the statistics calculated at each
%         % time point (the latter are set below using cfg_stat.statistic)
%         cfg_stat.clusterstatistic = 'maxsum';
%         cfg_stat.alpha           = 0.05; % use standard significance threshold of 5%
%         % Level 2 stats design: we have to choose between within-subject and
%         % between-subjects. Between-subjects is relevant when there is two
%         % different experimental groups (eg patients vs controls) and we want to
%         % investigate whether their MVPA results are significantly different. Here,
%         % we have only one group and we want to see whether the AUC is
%         % significantly different from a null value, hence the statistical design
%         % is within-subject
%         cfg_stat.design          = 'within';
%         % cfg_stat.statistic defines how the difference between the AUC values and
%         % the null is calculated at each time point (across subjects).
%         % We can choose t-test or its nonparametric counterpart Wilcoxon test. We
%         % choose Wilcoxon here.
%         cfg_stat.statistic       = 'wilcoxon';
%         % The null value for AUC (corresponding to a random classifier) is 0.5
%         if permutation == 1
%             cfg_stat.null = 0;   % z-scored null
%         else
%             cfg_stat.null = chancelevel;
%         end
%         % clustercritval is a parameter that needs to be selected by the user. In a
%         % Level 2 (group level) test, it represents the critical cutoff value for
%         % the statistic. Here, we selected Wilcoxon, so clustercritval corresponds
%         % to the cutoff value for the z-statistic which is obtained by a normal
%         % approximation.
%         cfg_stat.clustercritval= 1.645;
%         cfg_stat.tail = 1;
%         % z-val = 1.65 corresponds to uncorrected p-value = 0.1
%         % z-val = 1.96 corresponds to uncorrected p-value = 0.05
%         % z-val = 2.58 corresponds to uncorrected p-value = 0.01
%         % Note that these z-values correspond to a two-sided test. In many cases it
%         % suffices to use a one-sided test. In this case, the following cutoff
%         % values are obtained:
%         % z-val = 1.282 corresponds to uncorrected p-value = 0.1
%         % z-val = 1.645 corresponds to uncorrected p-value = 0.05
%         % z-val = 2.326 corresponds to uncorrected p-value = 0.01
%         stat_level2 = mv_statistics(cfg_stat, all_results_subjects);
%         result_average = mv_combine_results(all_results_subjects, 'average');
%         result_average.perf_std{1, 1} = result_average.perf_std{1, 1} / sqrt(length(all_results_subjects));  % turn this into standard error for plots
%         mv_plot_result(result_average, time_vec, 'mask', stat_level2.mask)
%     %         grid on;
%     %         set(gcf,'units','centimeters','position',[xini,yini,xsize,ysize]);
%         set(gca,'FontSize',10,'FontName','Arial');
%     %         print("title",'-depsc2');
%     %         saveas(gcf, fullfile(output_dir,[roi '_phit.eps']));
%         f = gcf;
%         exportgraphics(f,fullfile(output_dir,[ROIs{r} '_' dependent_var '_' num2str(win_ms) '.pdf']),'Resolution',300)
                time_vec = feedback_tfr.time(1:downsample_factor:end);

        %% --- GROUP-LEVEL STATS (vs. Chance) ---
        % Use z-scored values if permutations are enabled, otherwise use raw values
        if permutation == 1
            data_for_stats = zscore_acc;
            null_value = 0;
        else
            data_for_stats = all_acc_subjects;
            null_value = chancelevel;
        end
        % data_for_stats is [timepoints x subjects]
        nSubj = size(data_for_stats, 2);
        nTime = size(data_for_stats, 1);

        tvals = zeros(nTime, 1);
        pvals = zeros(nTime, 1);

        for t = 1:nTime
            [~, p, ~, stats] = ttest(data_for_stats(t,:), null_value, 'Tail', 'right');
            tvals(t) = stats.tstat;
            pvals(t) = p;
        end

        %% --- CLUSTER-BASED PERMUTATION TEST ---
        % Threshold at t corresponding to p<0.05 (one-tailed)
%         t_thresh = tinv(1 - alpha, nSubj - 1);
        t_thresh = 1;
        sig_mask = tvals > t_thresh;

        % Label clusters (1D connectivity)
        cc = bwconncomp(sig_mask, 4);
        cluster_tsum = cellfun(@(x) sum(tvals(x)), cc.PixelIdxList);

        % Null distribution via random sign-flip permutations
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

        % Compute cluster p-values
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
        save_roi_plot(output_dir, ROIs{r}, dependent_var, freq_label, win_ms, time_vec, ...
            raw_plot_mean, raw_plot_sem, raw_chance_line, sig_mask_final, 'raw');
        if permutation == 1
            save_roi_plot(output_dir, ROIs{r}, dependent_var, freq_label, win_ms, time_vec, ...
                z_plot_mean, z_plot_sem, z_chance_line, sig_mask_final, 'zscore');
        end
        %% --- SAVE DETAILED STATISTICAL REPORT ---
        txt_filename = fullfile(output_dir, ...
            [ROIs{r} '_' dependent_var '_' freq_label '_' num2str(win_ms) '_stats.txt']);

        fid = fopen(txt_filename,'w');

        fprintf(fid,'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
        fprintf(fid,'======================================================\n\n');

        %% --- GENERAL INFORMATION ---
        fprintf(fid,'ROI: %s\n', ROIs{r});
        fprintf(fid,'Dependent variable: %s\n', dependent_var);
        fprintf(fid,'Number of subjects: %d\n', nSubj);
        fprintf(fid,'Smoothing window: %d ms\n', win_ms);
        fprintf(fid,'Downsample factor: %d\n', downsample_factor);
        fprintf(fid,'Time resolution: %.6f s\n\n', mean(diff(time_vec)));

        %% --- DECODING CONFIGURATION ---
%         fprintf(fid,'Classifier: %s\n', cfg.classifier);
        fprintf(fid,'Metric: %s\n', format_metric_for_report(cfg.metric));
%         fprintf(fid,'Cross-validation: %s\n', cfg.cv);
        fprintf(fid,'Frequency range: %d-%d Hz (%s)\n\n', freq_low, freq_high, freq_label);

        %% --- STATISTICAL TEST DETAILS ---
        fprintf(fid,'Statistical test: sign-flip permutation\n');
        fprintf(fid,'Test statistic: t-test\n');
        fprintf(fid,'Multiple comparison correction: cluster\n');
        fprintf(fid,'Alpha level: %.4f\n', alpha);
        fprintf(fid,'Number of permutations: %d\n\n', nPerm_stat);

        fprintf(fid,'Group-level design: within\n');
        fprintf(fid,'Statistic: t-test\n');
        fprintf(fid,'Tail: 1\n');
        if permutation == 1
            fprintf(fid,'Null hypothesis value: %.4f (z-scored)\n', 0);
        else
            fprintf(fid,'Null hypothesis value: %.4f\n', chancelevel);
        end
        fprintf(fid,'Permutation test: sign-flip\n');
        fprintf(fid,'Cluster statistic: maxsum\n');
        fprintf(fid,'Cluster-forming threshold: %.4f\n', t_thresh);
        fprintf(fid,'Cluster connectivity: 4-connected (1D)\n\n');

        %% --- CLUSTER RESULTS ---
        fprintf(fid,'Number of significant clusters: %d\n\n', n_sig_clusters);

        if n_sig_clusters == 0
            fprintf(fid,'No significant clusters detected.\n');
        else
            for c = 1:numel(sig_clusters)
                cid = sig_clusters(c);
                cluster_idx = cc.PixelIdxList{cid};

                t_start = time_vec(min(cluster_idx));
                t_end   = time_vec(max(cluster_idx));

                fprintf(fid,'Cluster %d\n', c);
                fprintf(fid,'  Time range: %.4f – %.4f s\n', t_start, t_end);
                fprintf(fid,'  Cluster-level p-value: %.6f\n', cluster_pvals(cid));
                fprintf(fid,'\n');
            end
        end

        fclose(fid);

        fprintf('Saved statistical summary:\n%s\n', txt_filename);

        %% --- SUBJECT-LEVEL CLUSTER SCORES ---
        if any(sig_mask_final(:))
            nSubj_kept = size(all_acc_subjects, 2);
            subj_cluster_metric = zeros(nSubj_kept, 1);
            subj_cluster_effect = zeros(nSubj_kept, 1);

            for subj_idx = 1:nSubj_kept
                subj_acc = all_acc_subjects(:, subj_idx);
                subj_cluster_metric(subj_idx) = mean(subj_acc(sig_mask_final));
                subj_cluster_effect(subj_idx) = subj_cluster_metric(subj_idx) - chancelevel;
            end

            if is_accuracy_metric(cfg.metric)
                subj_table = table(subjects_used(:), subj_cluster_metric, subj_cluster_effect, n_elec_subj(:), ...
                    'VariableNames', {'Subject', 'MeanAccuracy', 'ReinstateEffect', 'NumElectrodes'});
            else
                subj_table = table(subjects_used(:), subj_cluster_metric, subj_cluster_effect, n_elec_subj(:), ...
                    'VariableNames', {'Subject', 'MeanMetric', 'EffectAboveNull', 'NumElectrodes'});
            end

            csv_filename = fullfile(output_dir, ...
                sprintf('%s_%s_%s_subject_reinstatement.csv', ROIs{r}, dependent_var, freq_label));
            writetable(subj_table, csv_filename);
            fprintf('Saved subject reinstatement scores:\n%s\n', csv_filename);
        else
            fprintf('\nNo significant clusters found for %s - skipping subject-level reinstatement.\n', ROIs{r});
        end

        %% --- SIGNIFICANT-WINDOW 3-CLASS DIAGNOSTICS ---
        if strcmp(dependent_var, 'rpe_category')
            sig_window_time_mask = sig_mask_final;
            if any(sig_mask_final(:))
                [sig_window_confusion_subjects, sig_window_precision_subjects, sig_window_recall_subjects, sig_window_f1_subjects, sig_window_stats_table] = ...
                    save_feedback_significant_window_diagnostics(output_dir, ROIs{r}, dependent_var, freq_label, win_ms, ...
                    diagnostic_feedback_subjects, diagnostic_y_subjects, sig_mask_final, class_names, plot_cfg, cfg);
            else
                fprintf('No significant time points for %s - skipping 3-class diagnostic summaries.\n', ROIs{r});
            end
        end

        %% --- SAVE DECODING RESULTS ---
        mat_filename = fullfile(output_dir, [ROIs{r} '_' dependent_var '_' freq_label '_' num2str(win_ms) '_results.mat']);
        save(mat_filename, 'all_acc_subjects', 'subjects_used', 'time_vec', ...
             'chancelevel', 'win_ms', 'downsample_factor', 'dependent_var', 'cfg', ...
             'freq_low', 'freq_high', 'freq_label', 'sig_window_confusion_subjects', ...
             'sig_window_precision_subjects', 'sig_window_recall_subjects', ...
             'sig_window_f1_subjects', 'sig_window_time_mask', 'sig_window_stats_table', 'class_names');
        fprintf('Saved decoding results:\n%s\n', mat_filename);

    end

    create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, freq_label, win_ms, 'raw');
    if permutation == 1
        create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, freq_label, win_ms, 'zscore');
    end
end

function [confusion_subjects, precision_subjects, recall_subjects, f1_subjects, stats_table] = save_feedback_significant_window_diagnostics( ...
    output_dir, roi_name, dependent_var, freq_label, win_ms, diagnostic_feedback_subjects, diagnostic_y_subjects, ...
    sig_mask, class_names, plot_cfg, base_cfg)

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
        X_sig = diagnostic_feedback_subjects{subj_idx}(:,:,sig_mask);
        y_sig = diagnostic_y_subjects{subj_idx};
        [diag_perf, ~] = mv_classify_across_time(diagnostic_cfg, X_sig, y_sig);

        confusion_subjects(:,:,subj_idx) = mean(standardize_feedback_confusion(diag_perf{1}, n_classes), 3, 'omitnan');
        precision_subjects(subj_idx,:) = mean(standardize_feedback_class_metric(diag_perf{2}, n_classes), 1, 'omitnan');
        recall_subjects(subj_idx,:) = mean(standardize_feedback_class_metric(diag_perf{3}, n_classes), 1, 'omitnan');
        f1_subjects(subj_idx,:) = mean(standardize_feedback_class_metric(diag_perf{4}, n_classes), 1, 'omitnan');
    end

    mean_confusion = mean(confusion_subjects, 3, 'omitnan');
    mean_precision = mean(precision_subjects, 1, 'omitnan');
    sem_precision = std(precision_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);
    mean_recall = mean(recall_subjects, 1, 'omitnan');
    sem_recall = std(recall_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);
    mean_f1 = mean(f1_subjects, 1, 'omitnan');
    sem_f1 = std(f1_subjects, 0, 1, 'omitnan') ./ sqrt(n_subjects);

    plot_confusion_summary(mean_confusion, class_names, ...
        sprintf('%s: significant-window confusion', roi_name), ...
        fullfile(output_dir, sprintf('%s_%s_%s_%d_significant_window_confusion_summary.pdf', roi_name, dependent_var, freq_label, win_ms)), ...
        plot_cfg);
    plot_class_metric_summary(mean_recall, sem_recall, mean_precision, sem_precision, mean_f1, sem_f1, class_names, ...
        sprintf('%s: significant-window class metrics', roi_name), ...
        fullfile(output_dir, sprintf('%s_%s_%s_%d_significant_window_class_metrics.pdf', roi_name, dependent_var, freq_label, win_ms)), ...
        plot_cfg);

    metric_summary = table(class_names(:), mean_recall(:), sem_recall(:), mean_precision(:), sem_precision(:), mean_f1(:), sem_f1(:), ...
        'VariableNames', {'Class', 'MeanRecall', 'SEMRecall', 'MeanPrecision', 'SEMPrecision', 'MeanF1', 'SEMF1'});
    writetable(metric_summary, fullfile(output_dir, sprintf('%s_%s_%s_%d_significant_window_class_metrics.csv', roi_name, dependent_var, freq_label, win_ms)));

    confusion_table = array2table(mean_confusion, 'VariableNames', matlab.lang.makeValidName(strcat('Predicted_', class_names)), ...
        'RowNames', matlab.lang.makeValidName(strcat('True_', class_names)));
    writetable(confusion_table, fullfile(output_dir, sprintf('%s_%s_%s_%d_significant_window_confusion_summary.csv', roi_name, dependent_var, freq_label, win_ms)), ...
        'WriteRowNames', true);

    stats_table = calculate_class_paired_tests(recall_subjects, precision_subjects, f1_subjects, confusion_subjects, class_names);
    writetable(stats_table, fullfile(output_dir, sprintf('%s_%s_%s_%d_significant_window_paired_tests.csv', roi_name, dependent_var, freq_label, win_ms)));
end

function confusion_out = standardize_feedback_confusion(confusion_in, n_classes)
    confusion_in = squeeze(confusion_in);
    sz = size(confusion_in);
    if isequal(sz, [n_classes n_classes])
        confusion_out = reshape(confusion_in, n_classes, n_classes, 1);
    elseif numel(sz) >= 3 && sz(1) == n_classes && sz(2) == n_classes
        confusion_out = confusion_in;
    elseif numel(sz) >= 3 && sz(end-1) == n_classes && sz(end) == n_classes
        confusion_out = permute(confusion_in, [2 3 1]);
    else
        error('Could not identify class dimensions in confusion output of size [%s]', num2str(sz));
    end
end

function metric_out = standardize_feedback_class_metric(metric_in, n_classes)
    metric_in = squeeze(metric_in);
    sz = size(metric_in);
    if isvector(metric_in)
        if numel(metric_in) ~= n_classes
            error('Expected one value per class, got %d values', numel(metric_in));
        end
        metric_out = reshape(metric_in, 1, n_classes);
    elseif sz(2) == n_classes
        metric_out = metric_in;
    elseif sz(1) == n_classes
        metric_out = metric_in';
    else
        error('Could not identify class dimension in metric output of size [%s]', num2str(sz));
    end
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

function tf = is_accuracy_metric(metric)
    if iscell(metric)
        tf = any(ismember(metric, {'acc', 'accuracy'}));
    else
        tf = any(strcmp(metric, {'acc', 'accuracy'}));
    end
end

function save_roi_plot(output_dir, roi_name, dependent_var, freq_label, win_ms, time_vec, plot_mean, plot_sem, chance_line, sig_mask, plot_mode)
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
    title(sprintf('%s: %s decoding (%s)', roi_name, dependent_var, plot_mode));
    set(gca, 'FontSize', 10, 'FontName', 'Arial');
    ylim(y_limits);
    hold off;

    suffix = '';
    if strcmp(plot_mode, 'zscore')
        suffix = '_zscore';
    end
    f = gcf;
    exportgraphics(f, fullfile(output_dir, [roi_name '_' dependent_var '_' freq_label '_' num2str(win_ms) suffix '.pdf']), 'Resolution', 300);
    close(f);
end

function create_combined_decoding_plot(roi_plot_data, plot_cfg, output_dir, dependent_var, freq_label, win_ms, plot_mode)
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

%         text(0.01, 0.06, roi_data.roi_name, 'Units', 'normalized', ...
%             'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_roi, ...
%             'Color', [0.2 0.2 0.2], 'HorizontalAlignment', 'left', 'VerticalAlignment', 'center');

        if ii > (nRows - 1) * nCols
            xlabel('time since feedback (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
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

    combined_filename = fullfile(output_dir, sprintf('all_ROIs_%s_%s_%d_combined_%s.pdf', dependent_var, freq_label, win_ms, plot_mode));
    exportgraphics(fig, combined_filename, 'Resolution', 300);
    fprintf('Saved combined figure:\n%s\n', combined_filename);
    close(fig);
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
