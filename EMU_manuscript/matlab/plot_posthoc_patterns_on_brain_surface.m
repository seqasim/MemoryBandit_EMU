    %% Plot post hoc electrode patterns on a cortical surface
% This script wraps the ecog_fmri_visualization_matlab toolbox to project
% electrode-level post hoc activation patterns from CSV files onto an
% inflated or pial cortical mesh.

%% --- CONFIGURATION ---

path_cfg = config_paths();
if ~exist('roi_name', 'var') || isempty(roi_name), roi_name = 'all'; end
toolbox_dir = path_cfg.ecog_vis_dir;
csv_dir = fullfile(path_cfg.mvpa_dir, 'decoding_feedback_posthoc_patterns');
output_dir = fullfile(csv_dir, 'surface_plots');
if ~exist('contrast_name', 'var') || isempty(contrast_name), contrast_name = 'positive_vs_neutral'; end
subject_filter = {};                 % e.g. {'MS012','MS016'} or {}

value_field = 'PatternValue';        % or 'WeightValue'
aggregate_mode = 'mean_by_coordinate';
coord_round_digits = 3;

surface_type = 'inflated';          % 'inflated' or 'pial'
hemisphere_mode = 'both';           % 'auto', 'left', 'right', 'both', 'combined'
patch_size = 15;
overlap_method = 'mean';            % 'mean', 'max', 'sum'
transparency = 0.75;
use_roi_defaults = true;           % if true, ROI presets below override top-level viz settings

signed_map = true;
colormap_name = 'piyg';             % signed default; use 'viridis' for magnitude plots
normalization_mode = 'auto';        % 'auto', 'none', 'zscore_within_subject_roi', 'maxabs_within_subject_roi'
caxis_mode = 'percentile';          % 'percentile' or 'manual'
caxis_percentile = 90;
caxis_limits = [];                  % used if caxis_mode = 'manual'
render_mode = 'multipanel';         % 'single' or 'multipanel'
panel_set = 'canonical_6';          % currently supports 'canonical_6'
brain_scope = 'combined_cortical';  % 'combined_cortical' or 'roi_name'
shared_colorbar = true;
panel_titles = {'Frontal', 'Dorsal', 'Left Lateral', 'Right Lateral', 'Left Medial', 'Right Medial'};

save_png = true;
save_pdf = false;
figure_color = 'w';
renderer_name = 'opengl';
brain_transparency = 0;          % 0 = opaque, higher values = more transparent
brain_gray_level = 0.92;         % lighter base cortex color
brain_face_lighting = 'gouraud'; % 'gouraud', 'flat', or 'none'
brain_ambient_strength = 0.60;
brain_diffuse_strength = 0.25;
brain_specular_strength = 0.05;

standard_brain_data_file = fullfile(toolbox_dir, 'demo_data.mat');
standard_curvature_file = '';
use_curvature_underlay = true;
curvature_scale = 0.35;            % smaller values make the sulcal underlay less contrasty

mesh_cache_file = '';
fs_subject_dir = '';
fs_dir = '';                        % optional FreeSurfer install directory
cache_loaded_mesh = false;

view_positions = struct();
view_positions.left = [-90 -20];
view_positions.right = [90 -20];
view_positions.combined = [180 15];

if use_roi_defaults
    roi_defaults = get_roi_surface_defaults(roi_name);
    surface_type = roi_defaults.surface_type;
    patch_size = roi_defaults.patch_size;
    transparency = roi_defaults.transparency;
    signed_map = roi_defaults.signed_map;
    colormap_name = roi_defaults.colormap_name;
    use_curvature_underlay = roi_defaults.use_curvature_underlay;
    hemisphere_mode = roi_defaults.hemisphere_mode;
    view_positions = roi_defaults.view_positions;
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

addpath(genpath(toolbox_dir));

if strcmpi(render_mode, 'multipanel') && strcmpi(brain_scope, 'combined_cortical')
    roi_list = {'OFC', 'dmPFC', 'dlPFC', 'ACC'};
else
    roi_list = get_roi_list(roi_name);
end
data_tbl = load_roi_tables(csv_dir, roi_list);
required_cols = {'ROI', 'Contrast', 'Subject', 'Electrode', value_field, 'CoordX', 'CoordY', 'CoordZ'};
missing_cols = required_cols(~ismember(required_cols, data_tbl.Properties.VariableNames));
if ~isempty(missing_cols)
    error('CSV is missing required columns: %s', strjoin(missing_cols, ', '));
end

row_mask = ismember(data_tbl.ROI, roi_list) & strcmp(data_tbl.Contrast, contrast_name);
if ~isempty(subject_filter)
    row_mask = row_mask & ismember(data_tbl.Subject, subject_filter);
end

plot_tbl = data_tbl(row_mask, :);
plot_tbl = plot_tbl(~isnan(plot_tbl.(value_field)) & ...
                    ~isnan(plot_tbl.CoordX) & ...
                    ~isnan(plot_tbl.CoordY) & ...
                    ~isnan(plot_tbl.CoordZ), :);

if isempty(plot_tbl)
    error('No rows found for ROI=%s, Contrast=%s after filtering.', roi_name, contrast_name);
end

normalization_mode = resolve_normalization_mode(normalization_mode, roi_list);
plot_tbl = normalize_plot_table(plot_tbl, value_field, normalization_mode);

if ~signed_map
    plot_tbl.(value_field) = abs(plot_tbl.(value_field));
end

plot_tbl = aggregate_pattern_rows(plot_tbl, value_field, aggregate_mode, coord_round_digits);

if isempty(plot_tbl)
    error('No rows remain after aggregation.');
end

[brain_data, curvature_data, mesh_source_label] = load_brain_mesh( ...
    standard_brain_data_file, standard_curvature_file, ...
    mesh_cache_file, fs_subject_dir, fs_dir, cache_loaded_mesh);
mesh_by_hemi = select_meshes(brain_data, surface_type);

[left_tbl, right_tbl, combined_tbl, hemis_to_plot] = split_by_hemisphere(plot_tbl, hemisphere_mode);

all_values = plot_tbl.(value_field);
plot_limits = compute_color_limits(all_values, signed_map, caxis_mode, caxis_percentile, caxis_limits);
plot_cmap = get_plot_colormap(colormap_name, signed_map);

if strcmpi(render_mode, 'multipanel')
    create_multipanel_figure(mesh_by_hemi, curvature_data, left_tbl, right_tbl, combined_tbl, ...
        roi_list, contrast_name, value_field, plot_cmap, plot_limits, output_dir, ...
        save_png, save_pdf, surface_type, use_curvature_underlay, curvature_scale, ...
        brain_transparency, brain_gray_level, brain_face_lighting, ...
        brain_ambient_strength, brain_diffuse_strength, brain_specular_strength, ...
        patch_size, overlap_method, transparency, figure_color, renderer_name, ...
        panel_set, shared_colorbar, panel_titles, normalization_mode);
else
    for h = 1:numel(hemis_to_plot)
        hemi = hemis_to_plot{h};
        fig = figure('Color', figure_color, 'Renderer', renderer_name, 'Position', [100 100 900 700]);
        ax = axes('Parent', fig);

        ok = render_surface_panel(ax, hemi, mesh_by_hemi, curvature_data, left_tbl, right_tbl, combined_tbl, ...
            surface_type, use_curvature_underlay, curvature_scale, brain_transparency, ...
            brain_gray_level, brain_face_lighting, brain_ambient_strength, brain_diffuse_strength, ...
            brain_specular_strength, patch_size, overlap_method, transparency, ...
            plot_cmap, plot_limits, figure_color, value_field);
        if ~ok
            close(fig);
            continue;
        end

        title(ax, sprintf('%s | %s | %s', strjoin(roi_list, '+'), contrast_name, upper_first(hemi)), 'Interpreter', 'none');
        cb = colorbar(ax);
        if signed_map
            cb.Label.String = sprintf('%s (%s)', value_field, 'signed');
        else
            cb.Label.String = sprintf('|%s|', value_field);
        end

        output_stub = sprintf('%s_%s_%s_%s_%s%s', ...
            strjoin(roi_list, '_'), contrast_name, value_field, surface_type, hemi, normalization_suffix(normalization_mode));
        output_stub = regexprep(output_stub, '[^A-Za-z0-9_\-]', '_');

        if save_png
            exportgraphics(fig, fullfile(output_dir, [output_stub '.png']), 'Resolution', 300);
        end
        if save_pdf
            exportgraphics(fig, fullfile(output_dir, [output_stub '.pdf']), 'ContentType', 'vector');
        end
    end
end

function create_multipanel_figure(mesh_by_hemi, curvature_data, left_tbl, right_tbl, combined_tbl, ...
    roi_list, contrast_name, value_field, plot_cmap, plot_limits, output_dir, ...
    save_png, save_pdf, surface_type, use_curvature_underlay, curvature_scale, ...
    brain_transparency, brain_gray_level, brain_face_lighting, ...
    brain_ambient_strength, brain_diffuse_strength, brain_specular_strength, ...
    patch_size, overlap_method, transparency, figure_color, renderer_name, ...
    panel_set, shared_colorbar, panel_titles, normalization_mode)

    panel_specs = get_panel_specs(panel_set, panel_titles);
    for i = 1:numel(panel_specs)
        preset = panel_specs(i);
        fig = figure('Color', figure_color, 'Renderer', renderer_name, 'Position', [100 100 900 700]);
        ax = axes('Parent', fig, 'Position', [0.03 0.05 0.82 0.88]);
        ok = render_surface_panel(ax, preset.hemi, mesh_by_hemi, curvature_data, left_tbl, right_tbl, combined_tbl, ...
            surface_type, use_curvature_underlay, curvature_scale, brain_transparency, ...
            brain_gray_level, brain_face_lighting, brain_ambient_strength, brain_diffuse_strength, ...
            brain_specular_strength, patch_size, overlap_method, transparency, ...
            plot_cmap, plot_limits, figure_color, value_field);
        if ~ok
            close(fig);
            continue;
        end
        apply_panel_view(ax, preset);
        title(ax, preset.title, 'Interpreter', 'none', 'FontWeight', 'bold');

        if shared_colorbar
            cb = colorbar(ax);
            cb.Units = 'normalized';
            cb.Position = [0.88 0.17 0.02 0.66];
            cb.Label.String = sprintf('%s (%s)', value_field, 'signed');
        end

        output_stub = sprintf('%s_%s_%s_%s_%s%s', ...
            strjoin(roi_list, '_'), contrast_name, value_field, surface_type, preset.file_tag, normalization_suffix(normalization_mode));
        output_stub = regexprep(output_stub, '[^A-Za-z0-9_\-]', '_');

        if save_png
            exportgraphics(fig, fullfile(output_dir, [output_stub '.png']), 'Resolution', 300);
        end
        if save_pdf
            exportgraphics(fig, fullfile(output_dir, [output_stub '.pdf']), 'ContentType', 'vector');
        end
    end
end

function ok = render_surface_panel(ax, hemi, mesh_by_hemi, curvature_data, left_tbl, right_tbl, combined_tbl, ...
    surface_type, use_curvature_underlay, curvature_scale, brain_transparency, ...
    brain_gray_level, brain_face_lighting, brain_ambient_strength, brain_diffuse_strength, ...
    brain_specular_strength, patch_size, overlap_method, transparency, ...
    plot_cmap, plot_limits, figure_color, value_field)

    ok = true;
    axes(ax);

    switch hemi
        case 'left'
            hemi_tbl = left_tbl;
            mesh = mesh_by_hemi.left;
            view_pos = [-90 -20];
        case 'right'
            hemi_tbl = right_tbl;
            mesh = mesh_by_hemi.right;
            view_pos = [90 -20];
        case 'combined'
            hemi_tbl = combined_tbl;
            mesh = merge_meshes(mesh_by_hemi.left, mesh_by_hemi.right);
            view_pos = [180 15];
        otherwise
            error('Unknown hemisphere: %s', hemi);
    end

    if isempty(hemi_tbl)
        ok = false;
        return;
    end

    setappdata(ax, 'panel_mesh_limits', get_mesh_limits(mesh, 0.02));

    underlay = [];
    if use_curvature_underlay && ~strcmp(hemi, 'combined')
        underlay = get_curvature_values(curvature_data, hemi, surface_type, mesh, curvature_scale);
    end

    if isempty(underlay)
        h_brain = plot_mesh_brain(mesh, view_pos, [], brain_transparency, base_brain_colormap(64, brain_gray_level));
    else
        h_brain = plot_mesh_brain(mesh, view_pos, underlay, brain_transparency, gray(64));
        caxis(ax, [-1 1]);
    end
    apply_brain_style(h_brain, brain_face_lighting, brain_ambient_strength, ...
        brain_diffuse_strength, brain_specular_strength);
    if strcmp(surface_type, 'inflated')
        set(ax, 'Color', figure_color);
    end

    coords = [hemi_tbl.CoordX, hemi_tbl.CoordY, hemi_tbl.CoordZ];
    values = hemi_tbl.(value_field);
    plot_data_on_mesh(coords, values, 'surface', ...
        'patchsize', patch_size, ...
        'overlapmethod', overlap_method, ...
        'transparency', transparency, ...
        'colormap', plot_cmap);

    colormap(ax, plot_cmap);
    caxis(ax, plot_limits);
end

function panel_specs = get_panel_specs(panel_set, panel_titles)
    switch lower(panel_set)
        case 'canonical_6'
            panel_specs = struct( ...
                'hemi', {'combined', 'combined', 'combined', 'combined', 'left', 'right'}, ...
                'title', panel_titles, ...
                'view', {[180 0], [180 90], [-90 0], [90 0], [-90 0], [90 0]}, ...
                'zoom', {1.05, 1.00, 1.10, 1.10, 1.18, 1.18}, ...
                'file_tag', {'frontal', 'dorsal', 'left_lateral', 'right_lateral', 'left_medial', 'right_medial'});
        otherwise
            error('Unknown panel_set: %s', panel_set);
    end
end

function apply_panel_view(ax, preset)
    axes(ax);
    view(ax, preset.view);
    mesh_limits = getappdata(ax, 'panel_mesh_limits');
    if ~isempty(mesh_limits)
        xlim(ax, mesh_limits.x);
        ylim(ax, mesh_limits.y);
        zlim(ax, mesh_limits.z);
    end
    axis(ax, 'manual');
    axis(ax, 'vis3d');
    daspect(ax, [1 1 1]);
    try
        camzoom(ax, preset.zoom);
    catch
        camzoom(preset.zoom);
    end
    axis(ax, 'off');
    camlight('headlight');
end

function mesh_limits = get_mesh_limits(mesh, pad_frac)
    if nargin < 2
        pad_frac = 0.02;
    end

    mins = min(mesh.vertices, [], 1);
    maxs = max(mesh.vertices, [], 1);
    span = maxs - mins;
    pad = max(span * pad_frac, eps);

    mesh_limits = struct();
    mesh_limits.x = [mins(1) - pad(1), maxs(1) + pad(1)];
    mesh_limits.y = [mins(2) - pad(2), maxs(2) + pad(2)];
    mesh_limits.z = [mins(3) - pad(3), maxs(3) + pad(3)];
end

function tbl = load_roi_tables(csv_dir, roi_list)
    expected_vars = {'ROI', 'Contrast', 'Subject', 'Electrode', ...
        'PatternValue', 'WeightValue', 'ClusterStart', 'ClusterEnd', ...
        'CoordX', 'CoordY', 'CoordZ'};
    tbls = cell(numel(roi_list), 1);
    for i = 1:numel(roi_list)
        csv_file = fullfile(csv_dir, sprintf('%s_rpe_category_hfa_200_posthoc_patterns.csv', roi_list{i}));
        if ~exist(csv_file, 'file')
            error('CSV file not found for ROI %s: %s', roi_list{i}, csv_file);
        end

        tbl_i = readtable(csv_file);
        missing_vars = expected_vars(~ismember(expected_vars, tbl_i.Properties.VariableNames));
        if ~isempty(missing_vars)
            error('CSV file for ROI %s is missing required variables: %s', ...
                roi_list{i}, strjoin(missing_vars, ', '));
        end

        tbl_i = tbl_i(:, expected_vars);
        tbls{i} = tbl_i;
    end
    tbl = vertcat(tbls{:});
end

function roi_list = get_roi_list(roi_name)
    switch lower(roi_name)
        case 'cortical'
            roi_list = {'OFC', 'dmPFC', 'dlPFC'};
        case 'all'
            roi_list = {'OFC', 'dmPFC', 'dlPFC', 'ACC'};
        otherwise
            roi_list = {roi_name};
    end
end

function csv_dir = csv_dir_from_stack()
    path_cfg = config_paths();
    csv_dir = fullfile(path_cfg.mvpa_dir, 'decoding_feedback_posthoc_patterns');
end

function roi_list = discover_available_rois(csv_dir)
    files = dir(fullfile(csv_dir, '*_rpe_category_hfa_200_posthoc_patterns.csv'));
    if isempty(files)
        error('No ROI posthoc CSV files found in %s', csv_dir);
    end

    roi_list = cell(numel(files), 1);
    suffix = '_rpe_category_hfa_200_posthoc_patterns.csv';
    for i = 1:numel(files)
        roi_list{i} = extractBefore(files(i).name, suffix);
    end
    roi_list = unique(roi_list, 'stable');
end

function tbl_out = aggregate_pattern_rows(tbl_in, value_field, aggregate_mode, coord_round_digits)
    tbl = tbl_in;
    switch aggregate_mode
        case 'mean_by_coordinate'
            x_round = round(tbl.CoordX, coord_round_digits);
            y_round = round(tbl.CoordY, coord_round_digits);
            z_round = round(tbl.CoordZ, coord_round_digits);
            coord_key = strcat(string(x_round), "_", string(y_round), "_", string(z_round));
            [group_idx, group_keys] = findgroups(coord_key); %#ok<ASGLU>

            agg_val = splitapply(@mean, tbl.(value_field), group_idx);
            agg_x = splitapply(@mean, tbl.CoordX, group_idx);
            agg_y = splitapply(@mean, tbl.CoordY, group_idx);
            agg_z = splitapply(@mean, tbl.CoordZ, group_idx);
            agg_n = splitapply(@numel, tbl.(value_field), group_idx);
            agg_electrode = splitapply(@(c) c(1), tbl.Electrode, group_idx);
            agg_subject = splitapply(@(c) {strjoin(unique(string(c)), '|')}, tbl.Subject, group_idx);

            tbl_out = table(agg_x, agg_y, agg_z, agg_val, agg_n, agg_electrode, agg_subject, ...
                'VariableNames', {'CoordX', 'CoordY', 'CoordZ', value_field, 'NumRows', 'Electrode', 'Subjects'});
        otherwise
            error('Unknown aggregate_mode: %s', aggregate_mode);
    end
end

function mode_out = resolve_normalization_mode(mode_in, roi_list)
    switch lower(mode_in)
        case 'auto'
            if numel(roi_list) > 1
                mode_out = 'zscore_within_subject_roi';
            else
                mode_out = 'none';
            end
        case {'none', 'zscore_within_subject_roi', 'maxabs_within_subject_roi'}
            mode_out = lower(mode_in);
        otherwise
            error('Unknown normalization_mode: %s', mode_in);
    end
end

function tbl_out = normalize_plot_table(tbl_in, value_field, normalization_mode)
    tbl_out = tbl_in;
    if strcmp(normalization_mode, 'none')
        return;
    end

    group_key = strcat(string(tbl_out.ROI), "||", string(tbl_out.Subject));
    group_idx = findgroups(group_key);
    values = tbl_out.(value_field);
    values_norm = nan(size(values));

    unique_groups = unique(group_idx(~isnan(group_idx)));
    for i = 1:numel(unique_groups)
        g = unique_groups(i);
        mask = group_idx == g;
        group_values = values(mask);

        switch normalization_mode
            case 'zscore_within_subject_roi'
                mu = mean(group_values, 'omitnan');
                sigma = std(group_values, 0, 'omitnan');
                if ~isfinite(sigma) || sigma == 0
                    group_values_norm = zeros(size(group_values));
                else
                    group_values_norm = (group_values - mu) ./ sigma;
                end
            case 'maxabs_within_subject_roi'
                denom = max(abs(group_values), [], 'omitnan');
                if ~isfinite(denom) || denom == 0
                    group_values_norm = zeros(size(group_values));
                else
                    group_values_norm = group_values ./ denom;
                end
            otherwise
                error('Unknown normalization_mode: %s', normalization_mode);
        end

        values_norm(mask) = group_values_norm;
    end

    values_norm(~isfinite(values_norm)) = 0;
    tbl_out.(value_field) = values_norm;
end

function suffix = normalization_suffix(normalization_mode)
    switch lower(normalization_mode)
        case 'none'
            suffix = '';
        case 'zscore_within_subject_roi'
            suffix = '_zscore_subject_roi';
        case 'maxabs_within_subject_roi'
            suffix = '_maxabs_subject_roi';
        otherwise
            error('Unknown normalization_mode: %s', normalization_mode);
    end
end

function [brain_data, curvature_data, mesh_source_label] = load_brain_mesh( ...
    standard_brain_data_file, standard_curvature_file, ...
    mesh_cache_file, fs_subject_dir, fs_dir, cache_loaded_mesh)
    mesh_source_label = '';
    curvature_data = struct();

    if ~isempty(standard_brain_data_file)
        loaded = load(standard_brain_data_file);
        if isfield(loaded, 'brain_data')
            brain_data = loaded.brain_data;
        else
            brain_data = loaded;
        end

        if isfield(loaded, 'vertex_data')
            curvature_data = loaded.vertex_data;
        end
        if ~isempty(standard_curvature_file)
            curv_loaded = load(standard_curvature_file);
            if isfield(curv_loaded, 'vertex_data')
                curvature_data = curv_loaded.vertex_data;
            else
                curvature_data = curv_loaded;
            end
        end

        mesh_source_label = 'standardized mesh';
        return;
    end

    if ~isempty(mesh_cache_file)
        loaded = load(mesh_cache_file);
        if isfield(loaded, 'brain_data')
            brain_data = loaded.brain_data;
        else
            brain_data = loaded;
        end
        if isfield(loaded, 'vertex_data')
            curvature_data = loaded.vertex_data;
        end
        mesh_source_label = 'cached mesh';
        return;
    end

    if isempty(fs_subject_dir)
        error(['Set standard_brain_data_file, mesh_cache_file, or fs_subject_dir. ', ...
               'A cortical mesh is required for plotting.']);
    end

    if isempty(fs_dir)
        brain_data = read_freesurfer_brain(fs_subject_dir);
    else
        brain_data = read_freesurfer_brain(fs_subject_dir, fs_dir);
    end
    mesh_source_label = 'FreeSurfer mesh';

    if cache_loaded_mesh
        [cache_dir, cache_name] = fileparts(fs_subject_dir);
        if isempty(cache_dir)
            cache_dir = pwd;
        end
        cache_path = fullfile(cache_dir, [cache_name '_brain_data_cache.mat']);
        save(cache_path, 'brain_data', '-v7.3');
    end
end

function underlay = get_curvature_values(curvature_data, hemi, surface_type, mesh, curvature_scale)
    underlay = [];
    if isempty(curvature_data) || strcmp(surface_type, 'pial')
        return;
    end

    if isfield(curvature_data, hemi) && isfield(curvature_data.(hemi), 'vertex_curvature_index')
        underlay = -sign(curvature_data.(hemi).vertex_curvature_index);
    elseif strcmp(hemi, 'left') && isfield(curvature_data, 'left') && isfield(curvature_data.left, 'vertex_curvature_index')
        underlay = -sign(curvature_data.left.vertex_curvature_index);
    elseif strcmp(hemi, 'right') && isfield(curvature_data, 'right') && isfield(curvature_data.right, 'vertex_curvature_index')
        underlay = -sign(curvature_data.right.vertex_curvature_index);
    end

    if ~isempty(underlay) && numel(underlay) ~= size(mesh.vertices, 1)
        underlay = [];
        return;
    end

    if ~isempty(underlay)
        underlay = curvature_scale * underlay;
    end
end

function mesh_by_hemi = select_meshes(brain_data, surface_type)
    switch surface_type
        case 'inflated'
            left_field = 'inflated_left';
            right_field = 'inflated_right';
        case 'pial'
            left_field = 'pial_left';
            right_field = 'pial_right';
        otherwise
            error('Unknown surface_type: %s', surface_type);
    end

    if ~isfield(brain_data, left_field) || ~isfield(brain_data, right_field)
        error('brain_data is missing fields required for surface_type=%s', surface_type);
    end

    mesh_by_hemi = struct();
    mesh_by_hemi.left = brain_data.(left_field);
    mesh_by_hemi.right = brain_data.(right_field);
end

function [left_tbl, right_tbl, combined_tbl, hemis_to_plot] = split_by_hemisphere(tbl, hemisphere_mode)
    left_tbl = tbl(tbl.CoordX < 0, :);
    right_tbl = tbl(tbl.CoordX > 0, :);
    combined_tbl = tbl;

    switch hemisphere_mode
        case 'auto'
            hemis_to_plot = {};
            if ~isempty(left_tbl)
                hemis_to_plot{end+1} = 'left'; %#ok<AGROW>
            end
            if ~isempty(right_tbl)
                hemis_to_plot{end+1} = 'right'; %#ok<AGROW>
            end
        case 'left'
            right_tbl = right_tbl([], :);
            hemis_to_plot = {'left'};
        case 'right'
            left_tbl = left_tbl([], :);
            hemis_to_plot = {'right'};
        case 'both'
            hemis_to_plot = {'left', 'right'};
        case 'combined'
            hemis_to_plot = {'combined'};
        otherwise
            error('Unknown hemisphere_mode: %s', hemisphere_mode);
    end

    if isempty(hemis_to_plot)
        error('No hemisphere contains valid electrodes after filtering.');
    end
end

function mesh = merge_meshes(left_mesh, right_mesh)
    mesh = struct();
    mesh.vertices = [left_mesh.vertices; right_mesh.vertices];
    mesh.faces = [left_mesh.faces; right_mesh.faces + size(left_mesh.vertices, 1)];
end

function plot_limits = compute_color_limits(values, signed_map, caxis_mode, caxis_percentile, caxis_limits)
    switch caxis_mode
        case 'percentile'
            if signed_map
                vmax = prctile(abs(values), caxis_percentile);
                plot_limits = [-vmax vmax];
            else
                vmax = prctile(values, caxis_percentile);
                plot_limits = [0 vmax];
            end
        case 'manual'
            if numel(caxis_limits) ~= 2
                error('Manual caxis_mode requires caxis_limits = [min max].');
            end
            plot_limits = caxis_limits;
        otherwise
            error('Unknown caxis_mode: %s', caxis_mode);
    end

    if any(~isfinite(plot_limits)) || diff(plot_limits) <= 0
        error('Computed invalid color limits: [%g %g]', plot_limits(1), plot_limits(2));
    end
end

function cmap = get_plot_colormap(colormap_name, signed_map)
    switch lower(colormap_name)
        case 'redblue'
            cmap = redblue_colormap(64);
        case 'piyg'
            cmap = piyg_colormap(64);
        case 'spectral'
            cmap = spectral_colormap(64);
        case 'viridis'
            cmap = viridis_fallback(64);
        case 'parula'
            cmap = parula(64);
        otherwise
            if signed_map
                cmap = piyg_colormap(64);
            else
                cmap = viridis_fallback(64);
            end
    end
end

function cmap = base_brain_colormap(n, brain_gray_level)
    if nargin < 1
        n = 64;
    end
    if nargin < 2
        brain_gray_level = 0.92;
    end
    base_color = repmat(brain_gray_level, n, 3);
    cmap = base_color;
end

function apply_brain_style(h_brain, face_lighting, ambient_strength, diffuse_strength, specular_strength)
    if isempty(h_brain) || ~isgraphics(h_brain)
        return;
    end

    set(h_brain, ...
        'FaceLighting', face_lighting, ...
        'AmbientStrength', ambient_strength, ...
        'DiffuseStrength', diffuse_strength, ...
        'SpecularStrength', specular_strength);

    material dull;
    axes(ancestor(h_brain, 'axes'));
    camlight('headlight');
end

function txt = upper_first(txt)
    txt = char(txt);
    txt(1) = upper(txt(1));
end

function cmap = redblue_colormap(n)
    if nargin < 1
        n = 64;
    end

    if mod(n, 2) == 0
        n_neg = n / 2;
        n_pos = n / 2;
        neg = [linspace(0, 1, n_neg)', linspace(0, 1, n_neg)', ones(n_neg, 1)];
        pos = [ones(n_pos, 1), linspace(1, 0, n_pos)', linspace(1, 0, n_pos)'];
        cmap = [neg; pos];
        cmap(n_neg, :) = [1 1 1];
        cmap(n_neg + 1, :) = [1 1 1];
    else
        n_side = floor(n / 2);
        neg = [linspace(0, 1, n_side)', linspace(0, 1, n_side)', ones(n_side, 1)];
        mid = [1 1 1];
        pos = [ones(n_side, 1), linspace(1, 0, n_side)', linspace(1, 0, n_side)'];
        cmap = [neg; mid; pos];
    end

    cmap = min(max(cmap, 0), 1);
end

function cmap = viridis_fallback(n)
    if nargin < 1
        n = 64;
    end

    if exist('viridis', 'file') == 2
        cmap = viridis(n);
        return;
    end

    anchor = [ ...
        0.267004 0.004874 0.329415
        0.282327 0.094955 0.417331
        0.253935 0.265254 0.529983
        0.206756 0.371758 0.553117
        0.163625 0.471133 0.558148
        0.127568 0.566949 0.550556
        0.134692 0.658636 0.517649
        0.266941 0.748751 0.440573
        0.477504 0.821444 0.318195
        0.741388 0.873449 0.149561];

    x_anchor = linspace(0, 1, size(anchor, 1));
    xq = linspace(0, 1, n);
    cmap = interp1(x_anchor, anchor, xq, 'linear');
end

function cmap = spectral_colormap(n)
    if nargin < 1
        n = 64;
    end

    anchor = [ ...
        0.620 0.004 0.259
        0.835 0.243 0.310
        0.957 0.427 0.263
        0.992 0.682 0.380
        0.996 0.878 0.545
        1.000 1.000 0.749
        0.902 0.961 0.596
        0.671 0.867 0.643
        0.400 0.761 0.647
        0.196 0.533 0.741
        0.369 0.310 0.635];

    x_anchor = linspace(0, 1, size(anchor, 1));
    xq = linspace(0, 1, n);
    cmap = interp1(x_anchor, anchor, xq, 'linear');
    cmap = min(max(cmap, 0), 1);
end

function cmap = piyg_colormap(n)
    if nargin < 1
        n = 64;
    end

    anchor = [ ...
        0.557 0.004 0.322
        0.773 0.106 0.490
        0.871 0.467 0.682
        0.945 0.714 0.855
        0.992 0.878 0.937
        0.969 0.969 0.969
        0.902 0.961 0.816
        0.722 0.882 0.525
        0.498 0.737 0.255
        0.302 0.573 0.129
        0.153 0.392 0.098];

    x_anchor = linspace(0, 1, size(anchor, 1));
    xq = linspace(0, 1, n);
    cmap = interp1(x_anchor, anchor, xq, 'linear');
    cmap = min(max(cmap, 0), 1);
end

function roi_defaults = get_roi_surface_defaults(roi_name)
    roi_defaults = struct();
    roi_defaults.surface_type = 'inflated';
    roi_defaults.patch_size = 8;
    roi_defaults.transparency = 0.10;
    roi_defaults.signed_map = true;
    roi_defaults.colormap_name = 'redblue';
    roi_defaults.use_curvature_underlay = true;
    roi_defaults.hemisphere_mode = 'both';
    roi_defaults.view_positions = struct('left', [-90 -20], 'right', [90 -20], 'combined', [180 15]);

    switch upper(roi_name)
        case 'OFC'
            roi_defaults.surface_type = 'pial';
            roi_defaults.patch_size = 8;
            roi_defaults.transparency = 0.10;
            roi_defaults.signed_map = true;
            roi_defaults.colormap_name = 'piyg';
            roi_defaults.use_curvature_underlay = false;
            roi_defaults.hemisphere_mode = 'both';
            roi_defaults.view_positions.left = [-90 -90];
            roi_defaults.view_positions.right = [90 -90];
        case {'DMPFC', 'DLPFC'}
            roi_defaults.surface_type = 'pial';
            roi_defaults.patch_size = 8;
            roi_defaults.transparency = 0.10;
            roi_defaults.signed_map = true;
            roi_defaults.colormap_name = 'piyg';
            roi_defaults.use_curvature_underlay = false;
            roi_defaults.hemisphere_mode = 'both';
            roi_defaults.view_positions.left = [-90 90];
            roi_defaults.view_positions.right = [90 90];
        case 'ACC'
            roi_defaults.surface_type = 'pial';
            roi_defaults.patch_size = 8;
            roi_defaults.transparency = 0.10;
            roi_defaults.signed_map = true;
            roi_defaults.colormap_name = 'piyg';
            roi_defaults.use_curvature_underlay = true;
            roi_defaults.hemisphere_mode = 'both';
            roi_defaults.view_positions.left = [-90 0];
            roi_defaults.view_positions.right = [90 0];
        case 'CORTICAL'
            roi_defaults.surface_type = 'pial';
            roi_defaults.patch_size = 8;
            roi_defaults.transparency = 0.10;
            roi_defaults.signed_map = true;
            roi_defaults.colormap_name = 'piyg';
            roi_defaults.use_curvature_underlay = false;
            roi_defaults.hemisphere_mode = 'both';
            roi_defaults.view_positions.left = [-90 25];
            roi_defaults.view_positions.right = [90 25];
            roi_defaults.view_positions.combined = [180 15];
        case 'ALL'
            roi_defaults.surface_type = 'pial';
            roi_defaults.patch_size = 8;
            roi_defaults.transparency = 0.10;
            roi_defaults.signed_map = true;
            roi_defaults.colormap_name = 'piyg';
            roi_defaults.use_curvature_underlay = false;
            roi_defaults.hemisphere_mode = 'both';
            roi_defaults.view_positions.left = [-90 10];
            roi_defaults.view_positions.right = [90 10];
            roi_defaults.view_positions.combined = [180 10];
    end
end
