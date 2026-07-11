%% ============================================================
%  ACC ↔ dmPFC SUBJECT-LEVEL REINSTATEMENT CORRELATION
%  Reads ROI reinstatement CSVs from test_script_crossdecoding.m outputs
%  and optionally relates reinstatement to subject-level dprime
% =============================================================

clear; clc;

%% --- CONFIGURATION ---
base_output_dir = '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/crossdecoding';
load_dir = '/Volumes/T7 Shield/work/qasims01/MemoryBanditData/EMU';
combined_df_file = fullfile(load_dir, 'full_df_RWH.csv');
dependent_var = 'rpe_category';
freq_label = 'HFA';
acc_roi = 'ACC';
dmpfc_roi = 'dmPFC';
metric_field = 'ReinstateEffect';   % or 'MeanAccuracy'
corr_type = 'Spearman';             % 'Spearman' or 'Pearson'
run_secondary_pearson = true;
run_dprime_correlations = true;
dprime_field = 'dprime';
label_points = true;

plot_cfg = struct();
plot_cfg.font_name = 'Arial';
plot_cfg.font_size_label = 12;
plot_cfg.font_size_tick = 10;
plot_cfg.line_width = 1.2;
plot_cfg.marker_size = 70;
plot_cfg.marker_face = [0.2 0.45 0.75];
plot_cfg.marker_edge = [0.1 0.1 0.1];
plot_cfg.fit_line = [0.2 0.2 0.2];

%% --- INPUT / OUTPUT PATHS ---
input_dir = fullfile(base_output_dir, dependent_var);

acc_csv = fullfile(input_dir, sprintf('%s_%s_%s_subject_reinstatement.csv', acc_roi, dependent_var, freq_label));
dmpfc_csv = fullfile(input_dir, sprintf('%s_%s_%s_subject_reinstatement.csv', dmpfc_roi, dependent_var, freq_label));

if ~isfile(acc_csv)
    error('ACC reinstatement CSV not found:\n%s', acc_csv);
end

if ~isfile(dmpfc_csv)
    error('dmPFC reinstatement CSV not found:\n%s', dmpfc_csv);
end

output_prefix = sprintf('%s_%s_%s_vs_%s_%s', dependent_var, freq_label, acc_roi, dmpfc_roi, metric_field);
merged_csv = fullfile(input_dir, [output_prefix '_merged_subjects.csv']);
summary_txt = fullfile(input_dir, [output_prefix '_correlation_summary.txt']);
plot_pdf = fullfile(input_dir, [output_prefix '_scatter.pdf']);
acc_dprime_pdf = fullfile(input_dir, [output_prefix '_ACC_vs_dprime_scatter.pdf']);
dmpfc_dprime_pdf = fullfile(input_dir, [output_prefix '_dmPFC_vs_dprime_scatter.pdf']);

%% --- LOAD ROI TABLES ---
acc_table = readtable(acc_csv);
dmpfc_table = readtable(dmpfc_csv);

required_vars = {'Subject', metric_field, 'NumElectrodes'};
assert_required_columns(acc_table, required_vars, acc_roi);
assert_required_columns(dmpfc_table, required_vars, dmpfc_roi);

acc_table = renamevars(acc_table, ...
    {'Subject', metric_field, 'NumElectrodes'}, ...
    {'Subject', ['ACC_' metric_field], 'ACC_NumElectrodes'});

dmpfc_table = renamevars(dmpfc_table, ...
    {'Subject', metric_field, 'NumElectrodes'}, ...
    {'Subject', ['dmPFC_' metric_field], 'dmPFC_NumElectrodes'});

%% --- INNER JOIN ON OVERLAPPING SUBJECTS ---
merged_table = innerjoin(acc_table, dmpfc_table, 'Keys', 'Subject');

if isempty(merged_table)
    error('No overlapping subjects were found between %s and %s.', acc_roi, dmpfc_roi);
end

acc_metric_col = ['ACC_' metric_field];
dmpfc_metric_col = ['dmPFC_' metric_field];

valid_mask = ~isnan(merged_table.(acc_metric_col)) & ~isnan(merged_table.(dmpfc_metric_col));
merged_table = merged_table(valid_mask, :);

if isempty(merged_table)
    error('No overlapping subjects had valid %s values in both ROIs.', metric_field);
end

%% --- LOAD AND COLLAPSE DPRIME ---
if run_dprime_correlations
    if ~isfile(combined_df_file)
        error('combined_df file not found:\n%s', combined_df_file);
    end

    combined_df = readtable(combined_df_file);
    required_combined_vars = {'participant', 'condition', dprime_field};
    assert_required_columns(combined_df, required_combined_vars, 'combined_df');

    combined_df = combined_df(strcmp(combined_df.condition, 'Day1'), :);
    dprime_table = collapse_subject_metric(combined_df, 'participant', dprime_field);
    dprime_table = renamevars(dprime_table, {'participant', dprime_field}, {'Subject', 'dprime'});

    merged_table = innerjoin(merged_table, dprime_table, 'Keys', 'Subject');

    if isempty(merged_table)
        error('No overlapping subjects were found across ROI reinstatement tables and dprime.');
    end
end

%% --- SAVE MERGED SUBJECT TABLE ---
writetable(merged_table, merged_csv);
fprintf('Saved merged subject table:\n%s\n', merged_csv);

%% --- ANALYSES ---
main_result = run_correlation_analysis(merged_table, acc_metric_col, dmpfc_metric_col, corr_type, run_secondary_pearson);
main_result.label_x = sprintf('%s %s', acc_roi, metric_field);
main_result.label_y = sprintf('%s %s', dmpfc_roi, metric_field);
main_result.title = sprintf('%s vs %s reinstatement (%s)', acc_roi, dmpfc_roi, dependent_var);
main_result.output_pdf = plot_pdf;

dprime_results = struct([]);
if run_dprime_correlations
    acc_dprime_result = run_correlation_analysis(merged_table, acc_metric_col, 'dprime', corr_type, run_secondary_pearson);
    acc_dprime_result.label_x = sprintf('%s %s', acc_roi, metric_field);
    acc_dprime_result.label_y = 'dprime';
    acc_dprime_result.title = sprintf('%s reinstatement vs dprime (%s)', acc_roi, dependent_var);
    acc_dprime_result.output_pdf = acc_dprime_pdf;

    dmpfc_dprime_result = run_correlation_analysis(merged_table, dmpfc_metric_col, 'dprime', corr_type, run_secondary_pearson);
    dmpfc_dprime_result.label_x = sprintf('%s %s', dmpfc_roi, metric_field);
    dmpfc_dprime_result.label_y = 'dprime';
    dmpfc_dprime_result.title = sprintf('%s reinstatement vs dprime (%s)', dmpfc_roi, dependent_var);
    dmpfc_dprime_result.output_pdf = dmpfc_dprime_pdf;

    dprime_results = [acc_dprime_result, dmpfc_dprime_result];
end

%% --- TEXT SUMMARY ---
fid = fopen(summary_txt, 'w');
fprintf(fid, 'ACC-dmPFC SUBJECT-LEVEL REINSTATEMENT CORRELATION\n');
fprintf(fid, '================================================\n\n');

fprintf(fid, 'Analysis type: subject-level association\n');
fprintf(fid, 'Interpretation note: this is correlational and does not imply temporal or causal prediction.\n\n');

fprintf(fid, 'Dependent variable: %s\n', dependent_var);
fprintf(fid, 'Frequency label: %s\n', freq_label);
fprintf(fid, 'Metric used: %s\n', metric_field);
fprintf(fid, 'Primary correlation type: %s\n', corr_type);
fprintf(fid, 'ACC ROI file: %s\n', acc_csv);
fprintf(fid, 'dmPFC ROI file: %s\n', dmpfc_csv);
if run_dprime_correlations
    fprintf(fid, 'combined_df file: %s\n', combined_df_file);
    fprintf(fid, 'dprime field: %s\n', dprime_field);
end
fprintf(fid, '\n');

fprintf(fid, 'Merged subjects across available inputs: %d\n', height(merged_table));
fprintf(fid, 'Subjects available after merging:\n');
for i = 1:height(merged_table)
    fprintf(fid, '  %s\n', merged_table.Subject{i});
end
fprintf(fid, '\n');

write_result_block(fid, sprintf('%s vs %s', acc_roi, dmpfc_roi), main_result, merged_table, acc_metric_col, dmpfc_metric_col);

if run_dprime_correlations
    write_result_block(fid, sprintf('%s vs dprime', acc_roi), dprime_results(1), merged_table, acc_metric_col, 'dprime');
    write_result_block(fid, sprintf('%s vs dprime', dmpfc_roi), dprime_results(2), merged_table, dmpfc_metric_col, 'dprime');
end

fclose(fid);
fprintf('Saved correlation summary:\n%s\n', summary_txt);

%% --- SCATTER PLOTS ---
make_scatter_plot(main_result, plot_cfg, label_points);
fprintf('Saved scatter plot:\n%s\n', main_result.output_pdf);

if run_dprime_correlations
    make_scatter_plot(dprime_results(1), plot_cfg, label_points);
    fprintf('Saved scatter plot:\n%s\n', dprime_results(1).output_pdf);
    make_scatter_plot(dprime_results(2), plot_cfg, label_points);
    fprintf('Saved scatter plot:\n%s\n', dprime_results(2).output_pdf);
end

%% --- CONSOLE SUMMARY ---
fprintf('\nACC-dmPFC correlation summary\n');
fprintf('-----------------------------\n');
print_console_result(sprintf('%s vs %s', acc_roi, dmpfc_roi), main_result, corr_type, run_secondary_pearson);
if run_dprime_correlations
    print_console_result(sprintf('%s vs dprime', acc_roi), dprime_results(1), corr_type, run_secondary_pearson);
    print_console_result(sprintf('%s vs dprime', dmpfc_roi), dprime_results(2), corr_type, run_secondary_pearson);
end

function result = run_correlation_analysis(tbl, x_col, y_col, corr_type, run_secondary_pearson)
    valid_mask = ~isnan(tbl.(x_col)) & ~isnan(tbl.(y_col));
    analysis_tbl = tbl(valid_mask, :);
    n = height(analysis_tbl);

    if n < 3
        error('Only %d subjects with valid values for %s vs %s. Need at least 3.', n, x_col, y_col);
    end

    x = analysis_tbl.(x_col);
    y = analysis_tbl.(y_col);
    [rho_main, p_main] = corr(x, y, 'Type', corr_type, 'Rows', 'complete');

    if run_secondary_pearson
        [rho_pearson, p_pearson] = corr(x, y, 'Type', 'Pearson', 'Rows', 'complete');
    else
        rho_pearson = NaN;
        p_pearson = NaN;
    end

    result = struct();
    result.x_col = x_col;
    result.y_col = y_col;
    result.analysis_tbl = analysis_tbl;
    result.n = n;
    result.corr_type = corr_type;
    result.rho_main = rho_main;
    result.p_main = p_main;
    result.rho_pearson = rho_pearson;
    result.p_pearson = p_pearson;
end

function tbl_out = collapse_subject_metric(tbl, subject_col, metric_col)
    [groups, subjects] = findgroups(tbl.(subject_col));
    metric_values = splitapply(@collect_unique_metric, tbl.(metric_col), groups);
    tbl_out = table(subjects, metric_values, 'VariableNames', {subject_col, metric_col});
end

function metric_value = collect_unique_metric(values)
    values = values(~isnan(values));
    unique_values = unique(values);

    if isempty(unique_values)
        metric_value = NaN;
    elseif numel(unique_values) == 1
        metric_value = unique_values;
    else
        error('Subject has multiple non-identical dprime values: %s', mat2str(unique_values'));
    end
end

function make_scatter_plot(result, plot_cfg, label_points)
    fig = figure('Position', [200 200 650 500]);
    set(fig, 'Color', 'w');
    set(fig, 'Renderer', 'painters');

    x_values = result.analysis_tbl.(result.x_col);
    y_values = result.analysis_tbl.(result.y_col);

    scatter(x_values, y_values, plot_cfg.marker_size, ...
        'MarkerFaceColor', plot_cfg.marker_face, ...
        'MarkerEdgeColor', plot_cfg.marker_edge, ...
        'LineWidth', plot_cfg.line_width);
    hold on;

    coeffs = polyfit(x_values, y_values, 1);
    xfit = linspace(min(x_values), max(x_values), 100);
    yfit = polyval(coeffs, xfit);
    plot(xfit, yfit, 'Color', plot_cfg.fit_line, 'LineWidth', 1.5);

    if label_points
        for i = 1:result.n
            text(x_values(i), y_values(i), [' ' result.analysis_tbl.Subject{i}], ...
                'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        end
    end

    xlabel(result.label_x, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    ylabel(result.label_y, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    title(result.title, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label + 1);
    set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
    set(gca, 'LineWidth', plot_cfg.line_width);
    box off;
    grid on;

    annotation_text = sprintf('%s rho = %.3f, p = %.3g, n = %d', ...
        result.corr_type, ...
        result.rho_main, result.p_main, result.n);
    text(0.05, 0.95, annotation_text, 'Units', 'normalized', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);

    print(fig, result.output_pdf, '-dpdf', '-r300');
end

function write_result_block(fid, title_text, result, merged_table, x_col, y_col)
    fprintf(fid, '%s\n', title_text);
    fprintf(fid, '%s\n', repmat('-', 1, length(title_text)));
    fprintf(fid, 'Valid subjects used: %d\n', result.n);
    fprintf(fid, 'Subjects included:\n');
    for i = 1:result.n
        fprintf(fid, '  %s\n', result.analysis_tbl.Subject{i});
    end
    fprintf(fid, 'Primary result\n');
    fprintf(fid, '  rho = %.6f\n', result.rho_main);
    fprintf(fid, '  p = %.6f\n', result.p_main);
    if ~isnan(result.rho_pearson)
        fprintf(fid, 'Secondary Pearson\n');
        fprintf(fid, '  r = %.6f\n', result.rho_pearson);
        fprintf(fid, '  p = %.6f\n', result.p_pearson);
    end
    fprintf(fid, 'Per-subject values\n');
    for i = 1:result.n
        fprintf(fid, '  %s\t%s=%.6f\t%s=%.6f\n', ...
            result.analysis_tbl.Subject{i}, x_col, result.analysis_tbl.(x_col)(i), ...
            y_col, result.analysis_tbl.(y_col)(i));
    end

    if ismember('ACC_NumElectrodes', merged_table.Properties.VariableNames) && ismember('dmPFC_NumElectrodes', merged_table.Properties.VariableNames)
        fprintf(fid, 'Electrode counts are available in merged table columns ACC_NumElectrodes and dmPFC_NumElectrodes.\n');
    end
    fprintf(fid, '\n');
end

function print_console_result(title_text, result, corr_type, run_secondary_pearson)
    fprintf('%s: rho = %.4f, p = %.4f, n = %d (%s)\n', ...
        title_text, result.rho_main, result.p_main, result.n, corr_type);
    if run_secondary_pearson
        fprintf('  Pearson r = %.4f, p = %.4f\n', result.rho_pearson, result.p_pearson);
    end
end

function assert_required_columns(tbl, required_vars, table_name)
    missing_vars = required_vars(~ismember(required_vars, tbl.Properties.VariableNames));
    if ~isempty(missing_vars)
        error('%s is missing required columns: %s', table_name, strjoin(missing_vars, ', '));
    end
end
