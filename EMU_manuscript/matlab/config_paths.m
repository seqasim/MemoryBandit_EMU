function cfg = config_paths()
% CONFIG_PATHS Resolve data and toolbox paths for MemoryBandit EMU MATLAB scripts.
%
% Environment variables (all optional; sensible defaults provided):
%   MEMORYBANDIT_DATA_DIR : Root of the MemoryBandit data directory
%                           (default: ~/MemoryBanditData)
%   MVPA_LIGHT_DIR        : Path to MVPA-Light toolbox checkout
%   FIELDTRIP_DIR         : Path to FieldTrip toolbox (if present, conflicting
%                           stats/signal MEX folders are removed from path)
%   ECOG_VIS_DIR          : Path to ecog_fmri_visualization_matlab toolbox
%                           (used only by plot_posthoc_patterns_on_brain_surface.m)

    base_dir = getenv('MEMORYBANDIT_DATA_DIR');
    if isempty(base_dir)
        home_dir = getenv('HOME');
        if isempty(home_dir)
            home_dir = getenv('USERPROFILE');
        end
        base_dir = fullfile(home_dir, 'MemoryBanditData');
    end

    cfg = struct();
    cfg.base_dir  = base_dir;
    cfg.load_dir  = fullfile(base_dir, 'work', 'qasims01', 'MemoryBanditData', 'EMU');
    cfg.ephys_dir = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses');
    cfg.mvpa_dir  = fullfile(base_dir, 'scratch', 'MemoryBandit', 'MVPA');

    % Resolve behavior CSV paths
    learn_v2 = fullfile(cfg.load_dir, 'learn_df_RWH_v2.csv');
    if exist(learn_v2, 'file')
        cfg.learn_csv = learn_v2;
    else
        cfg.learn_csv = fullfile(cfg.load_dir, 'learn_df_RWH.csv');
    end

    full_v2 = fullfile(cfg.load_dir, 'full_df_RWH_v2.csv');
    if exist(full_v2, 'file')
        cfg.combined_csv = full_v2;
    else
        cfg.combined_csv = fullfile(cfg.load_dir, 'full_df_RWH.csv');
    end

    cfg.mvpa_light_dir = getenv('MVPA_LIGHT_DIR');
    if ~isempty(cfg.mvpa_light_dir) && exist(cfg.mvpa_light_dir, 'dir')
        addpath(genpath(cfg.mvpa_light_dir));
        if exist('startup_MVPA_Light', 'file') && ~exist('mv_classify_across_time', 'file')
            startup_MVPA_Light;
        end
    end

    cfg.fieldtrip_dir = getenv('FIELDTRIP_DIR');
    cfg.ecog_vis_dir  = getenv('ECOG_VIS_DIR');
end
