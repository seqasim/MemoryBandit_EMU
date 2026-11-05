# Data Directory Layout & Schemas (`$MEMORYBANDIT_DATA_DIR`)

All Python, MATLAB, and R scripts in this package resolve input and scratch paths relative to the environment variable `MEMORYBANDIT_DATA_DIR` (see `membandit/paths.py` and `matlab/config_paths.m`).

---

## 1. Directory Tree under `$MEMORYBANDIT_DATA_DIR`

```text
$MEMORYBANDIT_DATA_DIR/
├── work/
│   └── qasims01/
│       └── MemoryBanditData/
│           ├── EMU/
│           │   ├── learn_df_RWH.csv                # EMU 2-arm bandit learning trials (n=19 subjects x 80 trials)
│           │   ├── full_df_RWH.csv                 # EMU combined recognition + learning table
│           │   ├── learn_df_RWH_v2.csv             # Exported with hierarchical pyEM RW alpha, beta, rpe, rpe_cat
│           │   └── full_df_RWH_v2.csv              # Exported with hierarchical pyEM RW alpha, beta, rpe, rpe_cat
│           └── Prolific/
│               └── DelayedMemoryTest/
│                   ├── learn_df_prolific_v2.csv    # Online cohort learning trials (n=200 good-DM subjects x 80 trials)
│                   ├── full_data_df_prolific_v2.csv# Online cohort recognition table
│                   └── demographics.csv            # Online cohort Prolific demographic export (Age, Sex)
├── projects/
│   └── guLab/
│       └── Salman/
│           ├── EMU/
│           │   └── MemoryBanditsDatabase.xlsx      # Cohort roster & inclusion flags (MSSMCode, Age, Gender, Site)
│           └── EphysAnalyses/
│               ├── {subj}/                         # One folder per EMU participant (e.g. MS012..UI007; n=19)
│               │   ├── Day1_reref_elec_df          # CSV table of bipolar-rereferenced electrodes & ROI labels
│               │   ├── neural/
│               │   │   └── Day1/
│               │   │       ├── baseline_start-epo.fif      # Fixation cross epochs (decision-making task)
│               │   │       ├── feedback_start-epo.fif      # Reward feedback epochs (0 to 1.5 s + 1.0 s buffer)
│               │   │       ├── baseline_start_mem-epo.fif  # Fixation cross epochs (recognition memory task)
│               │   │       ├── cue_on-epo.fif              # Recognition cue epochs (0 to 1.0 s + 1.0 s buffer)
│               │   │       ├── *_IED_df.csv                # Detected interictal epileptiform discharge timestamps
│               │   │       └── *_artifact_df.csv           # Detected sharp transient artifact timestamps
│               │   └── scratch/
│               │       ├── TFR/
│               │       │   ├── feedback_start-tfr.h5       # Baselined Morlet TFR (MNE EpochsTFR HDF5)
│               │       │   ├── cue_on-tfr.h5               # Baselined Morlet TFR (MNE EpochsTFR HDF5)
│               │       │   ├── feedback_start-tfr.mat      # MATLAB export via tools/export_tfr_to_mat.py
│               │       │   └── cue_on-tfr.mat              # MATLAB export via tools/export_tfr_to_mat.py
│               │       └── neural_cache/                   # Cached smoothed HFA / spectral arrays (membandit.io.Subject)
│               └── Connectivity/
│                   └── scratch/
│                       ├── ppc_hpc_acc_rpe_cutrev/cue_on/  # Cached per-subject PPC surrogates & ppc_pairs_cue_on.csv
│                       ├── psi_hpc_acc_rpe_hitmiss/cue_on/ # Cached per-subject PSI surrogates & all_psi_pairs_*.csv
│                       └── mv_gcpac_rpe/                   # Cached per-subject MV GC-PAC & all_gcpac_subject_*.csv
└── scratch/
    └── MemoryBandit/
        ├── MVPA/
        │   ├── decoding_feedback/                  # MATLAB Fig 2B, 2D, S4 outputs (*_results.mat, *_stats.txt, *.pdf)
        │   ├── decoding_feedback_posthoc_patterns/ # MATLAB Fig S3 Haufe pattern CSVs & surface plots
        │   ├── crossdecoding/                      # MATLAB Fig 3A-C, S6 outputs & ACC_rpe_category_HFA_subject_reinstatement.csv
        │   └── decoding_cue/                       # MATLAB Fig 3D, 3E, 5C outputs (hit_miss/ and fa_cr/)
        └── RSA_MVTemplate/
            ├── trial_level/                        # Per-subject cross-temporal RSA NPZ caches
            └── group_level/                        # Group RSA pickles (e.g. group_results_ACC_to_HPC.pkl)
```

---

## 2. File Schemas

### 2.1 EMU Learning Table (`learn_df_RWH.csv` / `learn_df_RWH_v2.csv`)

One row per trial of the 80-trial 2-arm bandit task per participant ($19 \times 80 = 1{,}520$ rows):

| Column | Type | Description |
|---|---|---|
| `participant` | `str` | De-identified participant code (`MS012`, `UI001`, …) |
| `trials` | `int` | 1-based bandit trial index (`1..80`) |
| `block` | `int` | Reversal block index (`1..5`; reversals at trials `16, 33, 48, 65`) |
| `prob` | `float` | Underlying win probability of the chosen deck (`0.8` or `0.2`) |
| `gamble` | `int` | Deck choice (`0` or `1`) |
| `male`, `female` | `int` | Binary indicator (`0`/`1`) for drawing from the male or female face deck |
| `gamble_rt` | `float` | Choice reaction time (seconds) |
| `reward` | `int` | Binary outcome (`1` = win, `0` = loss) |
| `DPRIME` | `float` | Normed perceptual memorability ($d'$) of the trial-unique face image |
| `rpe` | `float` | Trial-wise reward prediction error ($\delta_t = r_t - Q_t^c$) from the hierarchical RW model |
| `rpe_cat` | `str` | Within-participant RPE tercile (`negative`, `neutral`, `positive`) |
| `alpha`, `beta` | `float` | Participant-level hierarchical RW learning rate ($\alpha$) and inverse temperature ($\beta$) |
| `good_dm_performance` | `bool` | Decision-making inclusion flag |

### 2.2 EMU Combined Recognition + Learning Table (`full_df_RWH.csv` / `full_df_RWH_v2.csv`)

One row per recognition memory trial (40 studied images + 40 novel lure images on `Day1`):

| Column | Type | Description |
|---|---|---|
| `participant` | `str` | De-identified participant code |
| `condition` | `str` | Session label (`Day1` / `day_1` for immediate recognition) |
| `trials_mem` | `int` | 1-based recognition trial index (`1..80`) |
| `trials_gamble` | `float` | 1-based bandit encoding trial index (`1..80` for studied `old` cues; `NaN` for novel `lure` cues) |
| `hits`, `misses`, `false_alarms`, `correct_rejections` | `bool` | Recognition outcome indicators |
| `dprime` | `float` | Participant-level recognition sensitivity ($d'$; `> 0.25` for all 19 included EMU participants) |
| `DPRIME` | `float` | Normed image-level perceptual memorability ($d'$) |
| `rpe`, `rpe_cat`, `alpha`, `beta` | `float`/`str` | Encoding-trial RW RPE and participant-level RW parameters (`NaN` on novel lure trials) |

### 2.3 Online Cohort Tables (`learn_df_prolific_v2.csv`, `full_data_df_prolific_v2.csv`)

Produced by `notebooks/01_behavior/00_prolific_preprocessing.ipynb`. Follows the same schema as the EMU tables with `learn_trials` for the 1-based learning trial index and `image_dprime` for image-level perceptual memorability ($d'$).

### 2.4 Electrode Localization Table (`{subj}/Day1_reref_elec_df`)

CSV file (no file extension) with one row per bipolar-rereferenced intracranial channel:

| Column | Type | Description |
|---|---|---|
| `label` | `str` | Bipolar channel name matching `epochs.ch_names` / `tfr.ch_names` |
| `salman_region` | `str` | Anatomical ROI (`OFC`, `ACC`, `dmPFC`, `dlPFC`, `HPC`, `AMY`, …) |
| `hemisphere` | `str` | Hemisphere (`l` / `r` or `left` / `right`) |
| `mni_x`, `mni_y`, `mni_z` (or `x`, `y`, `z`) | `float` | MNI coordinates (mm) from LeGUI / Yale Brain Atlas localization |

### 2.5 MNE Epochs (`*-epo.fif`) & TFR Files (`*-tfr.h5` / `*-tfr.mat`)

- **`{event}-epo.fif`**: MNE `Epochs` at `500 Hz`, bipolar re-referenced, 60 Hz notch-filtered, with a `1.0 s` buffer before/after the analysis window (`feedback_start`: `[0, 1.5] s`; `cue_on`: `[0, 1.0] s`; fixation baselines: `[0, 0.75] s`).
- **`{event}-tfr.h5`**: MNE `EpochsTFR` containing fixation-baselined z-scored power across 30 log-spaced frequencies (`2–200 Hz`, 30 log-spaced Morlet cycles `3–10`), with IED and gradient artifact windows ($\pm 100\text{ ms}$) masked as `NaN`.
- **`{event}-tfr.mat`**: Exported from `*-tfr.h5` by `python tools/export_tfr_to_mat.py` with fields `powspctrm` (`n_trials x n_channels x n_freqs x n_times`), `freqs`, `times`, `ch_names`, and `sfreq`.
