%% --- PLOT HFA BY RPE CATEGORY FOR ALL SUBJECTS WITH ACC ELECTRODES ---
% This script loads decoding results from test_script_feedback_decoding.m,
% and plots smoothed HFA split by RPE category for each ACC electrode
% for every subject that has ACC electrodes.

%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback/');

roi_name = 'ACC';
dependent_var = 'rpe_category';
win_ms = 200;
downsample_factor = 5;

%% --- LOAD DECODING RESULTS ---
results_file = fullfile(output_dir, [roi_name '_' dependent_var '_' num2str(win_ms) '_results.mat']);
if ~exist(results_file, 'file')
    error('Results file not found: %s\nRun test_script_feedback_decoding.m first.', results_file);
end
load(results_file, 'all_acc_subjects', 'subjects_used', 'time_vec', 'chancelevel');
fprintf('Loaded decoding results from: %s\n', results_file);

%% --- LOAD BEHAVIORAL DATA (once) ---
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));

%% --- RPE CATEGORY COLORS ---
categories = {'negative', 'neutral', 'positive'};
colors = [0.8 0.2 0.2;   % red for negative
          0.5 0.5 0.5;   % gray for neutral  
          0.2 0.6 0.2];  % green for positive

%% --- LOOP OVER ALL SUBJECTS ---
n_subjects = numel(subjects_used);
fprintf('Processing %d subjects...\n', n_subjects);

for subj_idx = 1:n_subjects
    current_subject = subjects_used{subj_idx};
    fprintf('\n--- Subject %d/%d: %s ---\n', subj_idx, n_subjects, current_subject);
    
    %% --- LOAD ELECTRODE INFO ---
    elec_file = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', current_subject, 'Day1_reref_elec_df');
    if ~exist(elec_file, 'file')
        fprintf('  Electrode file not found, skipping.\n');
        continue;
    end
    elec_df = readtable(elec_file);
    roi_electrodes = elec_df.label(matches(elec_df.salman_region, roi_name));
    n_electrodes = numel(roi_electrodes);
    
    if n_electrodes == 0
        fprintf('  No %s electrodes found, skipping.\n', roi_name);
        continue;
    end
    fprintf('  Found %d %s electrodes: %s\n', n_electrodes, roi_name, strjoin(roi_electrodes, ', '));
    
    %% --- GET SUBJECT BEHAVIORAL DATA ---
    subj_learn_df = learn_df(strcmp(learn_df.participant, current_subject), :);
    
    % Compute RPE categories using tercile split
    edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
    subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});
    
    encoding_trials = subj_learn_df.trials;
    rpe_labels = subj_learn_df.rpe_category;
    
    %% --- LOAD TFR DATA ---
    filepath = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses', current_subject, 'scratch', 'TFR');
    tfr_file = fullfile(filepath, 'feedback_start-tfr.mat');
    if ~exist(tfr_file, 'file')
        fprintf('  TFR file not found, skipping.\n');
        continue;
    end
    fb = load(tfr_file);
    feedback_tfr.powspctrm = fb.powspctrm;
    feedback_tfr.freq = fb.freqs;
    feedback_tfr.time = fb.times;
    feedback_tfr.label = fb.ch_names;
    feedback_tfr.fsample = fb.sfreq;
    
    %% --- SELECT ROI ELECTRODES ---
    [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
    feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
    
    %% --- EXTRACT HFA (70-200 Hz) ---
    hfa_mask = feedback_tfr.freq >= 70 & feedback_tfr.freq <= 201;
    feedback_hfa = squeeze(mean(feedback_tfr.powspctrm(:, :, hfa_mask, :), 3));  % trials x electrodes x time
    
    % Handle case of single electrode
    if ndims(feedback_hfa) < 3
        feedback_hfa = reshape(feedback_hfa, [size(feedback_hfa, 1), 1, size(feedback_hfa, 2)]);
    end
    
    %% --- SMOOTH HFA ---
    fs = feedback_tfr.fsample;
    win_size = round(fs * (win_ms / 1000));
    feedback_hfa_smooth = movmean(feedback_hfa, win_size, 3);
    
    %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
    z_thresh = 10;
    [nTrials_raw, nElec_raw, nTime_raw] = size(feedback_hfa_smooth);
    
    for trial = 1:nTrials_raw
        for elec = 1:nElec_raw
            trace = squeeze(feedback_hfa_smooth(trial, elec, :));
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
                
                feedback_hfa_smooth(trial, elec, :) = trace;
            end
        end
    end
    
    %% --- SELECT ENCODING TRIALS ---
    feedback_hfa_smooth = feedback_hfa_smooth(encoding_trials, :, :);
    rpe_labels_subset = rpe_labels(encoding_trials);
    
    %% --- DOWNSAMPLE ---
    feedback_hfa_ds = feedback_hfa_smooth(:, :, 1:downsample_factor:end);
    time_vec_plot = feedback_tfr.time(1:downsample_factor:end);
    
    %% --- PLOT SETUP ---
    n_cols = ceil(sqrt(n_electrodes));
    n_rows = ceil(n_electrodes / n_cols);
    
    fig = figure('Position', [100 100 300*n_cols 250*n_rows], 'Color', 'w', 'Visible', 'off');
    sgtitle(sprintf('%s: HFA by RPE Category (Subject: %s)', roi_name, current_subject), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    %% --- PLOT EACH ELECTRODE ---
    for e = 1:n_electrodes
        subplot(n_rows, n_cols, e);
        hold on;
        
        electrode_hfa = squeeze(feedback_hfa_ds(:, e, :));  % trials x time
        
        legend_handles = [];
        for c = 1:numel(categories)
            cat_mask = rpe_labels_subset == categories{c};
            cat_hfa = electrode_hfa(cat_mask, :);
            
            mean_hfa = mean(cat_hfa, 1);
            sem_hfa = std(cat_hfa, 0, 1) / sqrt(sum(cat_mask));
            
            % Plot shaded SEM
            fill([time_vec_plot, fliplr(time_vec_plot)], ...
                 [mean_hfa + sem_hfa, fliplr(mean_hfa - sem_hfa)], ...
                 colors(c, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            
            % Plot mean line
            h = plot(time_vec_plot, mean_hfa, 'Color', colors(c, :), 'LineWidth', 2);
            legend_handles = [legend_handles, h];
        end
        
        % Formatting
        xlabel('Time (s)');
        ylabel('HFA Power');
        title(roi_electrodes{e}, 'Interpreter', 'none');
        xline(0, 'k--', 'LineWidth', 1);
        set(gca, 'FontSize', 10, 'FontName', 'Arial');
        
        % Add legend to first subplot only
        if e == 1
            legend(legend_handles, categories, 'Location', 'best');
        end
        
        hold off;
    end
    
    %% --- SAVE FIGURE ---
    output_filename = fullfile(output_dir, sprintf('%s_%s_HFA_by_RPE_%s.pdf', roi_name, current_subject, dependent_var));
    exportgraphics(fig, output_filename, 'Resolution', 300);
    fprintf('  Saved figure: %s\n', output_filename);
    close(fig);
end

fprintf('\n=== Done processing all subjects ===\n');

