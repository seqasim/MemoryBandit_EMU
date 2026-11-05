# Hippocampal-Cingulate Dynamics in the Human Brain Link Reinforcement-Learning and Memory — Manuscript Reproducibility Package

> ## **DISCLAIMER**
>
> 1. **Not for exact figure reproduction without the empirical dataset:** This directory provides the analysis and plotting code accompanying the manuscript, but **it cannot reproduce the exact figures or numerical results in the paper on its own** because doing so requires the empirical patient and behavioral dataset, which is **not publicly shared at this time** due to human-subjects privacy restrictions (see Data Availability in the manuscript).
> 2. **Potential translation errors:** In refactoring and packaging our internal research codebase into a self-contained repository for general/public use, **some errors or discrepancies may have been introduced in translation**. The authors would be happy to address and resolve any issues upon notification (please open a GitHub issue or contact the corresponding author at `salman.qasim@rutgers.edu`).

---

This directory contains the self-contained Python, MATLAB, and R analysis code accompanying:

> **Hippocampal-cingulate dynamics in the human brain link reinforcement-learning and memory**  
> Salman E. Qasim, Fedor Panov, Lizbeth Nunez, Ariane E. Rhone, Hiroto Kawasaki, Christopher Kovach, Christopher Garcia, Brian Dlouhy, Xiaosi Gu\*, and Ignacio Saez\*

---

## Quick Navigation

| Document | Contents |
|---|---|
| **[`FIGURE_MAP.md`](FIGURE_MAP.md)** | Complete panel-by-panel mapping of **Figures 1–5**, **Figures S1–S13**, **Tables S1–S2**, and in-text statistics to exact notebooks, MATLAB scripts, and R scripts |
| **[`DATA_LAYOUT.md`](DATA_LAYOUT.md)** | Expected directory structure under `$MEMORYBANDIT_DATA_DIR` and column schemas for behavioral CSVs, electrode tables, MNE epochs, and TFR files |
| **[`matlab/README.md`](matlab/README.md)** | Detailed instructions for running the MVPA-Light MATLAB decoding and activation-pattern scripts |

---

## Repository Structure

```text
EMU_manuscript/
├── README.md                  # This guide: system requirements, installation, demo, run order
├── FIGURE_MAP.md              # Figure/table/stat -> code & section lookup table
├── DATA_LAYOUT.md             # Expected data directory layout & file schemas
├── LICENSE                    # MIT License
├── pyproject.toml             # Installable membandit + rlmodels_k99 + pyem package config
├── requirements.txt           # Pinned Python dependencies
├── environment.yml            # Conda environment specification
├── membandit/                 # Core Python library (behavior, I/O, RSA, phase sync, PAC, oscillation)
├── external/
│   ├── rlmodels_k99/          # Vendored hierarchical RL modeling library (used by 01_behavior/)
│   └── pyem/                  # Vendored pyEM (pinned commit c0d09807)
├── notebooks/
│   ├── 01_behavior/           # Fig 1B, 1D-G, Fig S1, Fig S2A-D, Table S1
│   ├── 02_electrodes/         # Fig 1H (glass brain & LFP traces), Table S2
│   ├── 03_spectral/           # TFR compute, Fig 2A, Fig 5A, Fig 5B, theta bursts (BOSC), Fig S5 (ripples)
│   ├── 04_decoding_rsa_followups/ # Fig 4A-B, Fig S7, Fig S8, Fig S9-S11
│   └── 05_theta_sync_pac/     # Fig 5D-F (PPC & PSI), Fig S12 (MV GC-PAC), Fig S13
├── matlab/                    # MVPA-Light decoding & Haufe activation patterns (Fig 2B, 2D, 3A-E, 5C, S3, S4, S6)
├── R/
│   ├── glmer_recognition_emu.R    # Fig 1G binomial mixed-effects recognition model (EMU, n=19)
│   ├── glmer_recognition_online.R # Fig S2D binomial mixed-effects recognition model (online, n=193)
│   └── lmer_neural.R              # Fig 5B, 5E, 5F, S12 linear mixed-effects neural models
├── tools/
│   ├── run_demo.py            # Self-contained synthetic demo (no external data required; ~5-10 s)
│   ├── export_tfr_to_mat.py   # Convert per-subject MNE *-tfr.h5 files to *-tfr.mat for MATLAB
│   └── check_package.py       # Static verification of imports, notebooks, and path portability
└── tests/
    └── test_smoke.py          # Automated test suite verifying package integrity
```

---

## 1. System Requirements

### Operating Systems & Tested Versions
- **Operating systems:** macOS (tested on macOS Darwin 25.3.0 / arm64) and Linux (Ubuntu 22.04 x86\_64).
- **Python:** Python $\ge 3.10$ (tested on Python 3.10, 3.11, and 3.12).
  - Core Python packages and tested versions: `numpy==2.5.1` ($\ge 1.24$), `scipy==1.18.0` ($\ge 1.11$), `pandas==3.0.3` ($\ge 2.0$), `matplotlib==3.11.0` ($\ge 3.7$), `seaborn==0.13.2` ($\ge 0.13$), `scikit-learn==1.9.0` ($\ge 1.3$), `statsmodels==0.14.6` ($\ge 0.14$), `mne==1.12.1` ($\ge 1.6$), `mne-connectivity==0.8.1` ($\ge 0.8$), `nibabel>=5.1`, `nilearn>=0.10`, `openpyxl>=3.1`, `h5py>=3.9`, `fooof>=1.1`, `neurodsp>=2.2`, and `pyem` (commit `c0d09807`, vendored in `external/pyem/`).
- **R:** R $\ge 4.2$ (tested on R 4.4) with packages `lme4` ($\ge 1.1$) and `lmerTest` ($\ge 3.1$).
- **MATLAB:** MATLAB R2021b or newer (tested on MATLAB R2021b) with the Statistics and Machine Learning Toolbox, Parallel Computing Toolbox, and [MVPA-Light](https://github.com/treder/MVPA-Light) (commit `1cc9944`).

### Non-Standard Hardware
- **No non-standard hardware (such as GPUs) is required.** All scripts, mixed-effects models, and plotting pipelines run on a standard desktop or laptop CPU (16 GB RAM recommended). Full 500-surrogate connectivity or 250-permutation RSA batch runs across all 19 participants benefit from multi-core CPU parallelization (`n_jobs`), or run in seconds from precomputed per-subject caches.

---

## 2. Installation Guide

### Option A: Conda + pip (Recommended)

```bash
cd EMU_manuscript
conda env create -f environment.yml
conda activate membandit-emu
```

### Option B: Python virtual environment (`venv`)

```bash
cd EMU_manuscript
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[oscillation,notebooks,dev]"
```

`pip install -e .` installs `membandit` (the iEEG/behavior toolkit), `rlmodels_k99` (the vendored hierarchical RL modeling library), and `pyem` (vendored under `external/pyem/`).

- **Typical install time on a standard desktop/laptop computer:** **~2–5 minutes.**

### External & Non-PyPI Dependencies by Analysis Tier

| Analysis Tier | Panels Covered | Additional Requirement | Setup Instructions |
|---|---|---|---|
| **Tier 1: Core Python + R** | Fig 1B, 1D–H, 2A, 4A–B, 5A–B, 5D–F, S1, S2, S5, S7–S11, S13, Tables S1–S2 | R (`lme4`, `lmerTest`) | In R: `install.packages(c("lme4", "lmerTest"))` |
| **Tier 2: Multivariate GC-PAC** | Fig S12 (`05_theta_sync_pac/FigS12_PAC.ipynb`) | Patched `tensorpac` fork supporting unequal channel counts across regions in multivariate Gaussian-copula PAC | `git clone https://github.com/seqasim/tensorpac.git && cd tensorpac && git checkout 33dcbf5 && pip install -e .` (or set `export TENSORPAC_ROOT=/path/to/tensorpac`) |
| **Tier 3: BOSC Theta Bursts** | In-text theta-burst prevalence (`03_spectral/Text_theta_bursts_BOSC.ipynb`) | `LFPAnalysis` (`eBOSC_wrapper`) | `git clone https://github.com/seqasim/LFPAnalysis.git && cd LFPAnalysis && git checkout f0ca5b4` and `export LFPANALYSIS_ROOT=/path/to/LFPAnalysis` |
| **Tier 4: MATLAB MVPA Decoding** | Fig 2B, 2D, 3A–E, 5C, S3, S4, S6 (`matlab/*.m`) | MATLAB (R2021b+) + [**MVPA-Light**](https://github.com/treder/MVPA-Light) (`1cc9944`) + optional `ecog_fmri_visualization_matlab` (for Fig S3 3D brain surface rendering) | Set `MVPA_LIGHT_DIR` (and optional `ECOG_VIS_DIR`) before running `matlab/run_manuscript_panels.m` (see [`matlab/README.md`](matlab/README.md)) |

---

## 3. Demo (Synthetic Data — No External Data Drive Required)

Because human intracranial EEG recordings cannot be publicly distributed due to patient privacy regulations, we provide a self-contained synthetic demo script (`tools/run_demo.py`) and static verification suite (`tools/check_package.py`) that run out of the box without `$MEMORYBANDIT_DATA_DIR`:

```bash
cd EMU_manuscript
python tools/run_demo.py
python tools/check_package.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

- **What `tools/run_demo.py` runs:**
  1. Simulates 2-arm bandit reversal-learning behavior ($N = 10$ synthetic agents $\times 80$ trials) under the Rescorla–Wagner model and fits the hierarchical Laplace-EM model via `rlmodels_k99` + `pyem`.
  2. Preprocesses the simulated trial table via `membandit.behavior` (computing within-subject RPE terciles `negative`/`neutral`/`positive` and Gelman-standardized predictors).
  3. Synthesizes multi-electrode encoding (`feedback_start`) and retrieval (`cue_on`) HFA arrays and runs the cross-temporal `ACC -> HPC` representational similarity analysis (Euclidean RDMs, Spearman $\rho_a$, remembered minus forgotten contrast, and trial-label permutation z-scoring) via `membandit.rsa`.
- **Expected output:** Console summary of parameter recovery correlations ($r(\alpha), r(\beta)$), RPE tercile counts, and cross-temporal RSA heatmap dimensions/z-scores, plus `results/demo/demo_preprocessed_learn_df.csv`.
- **Expected run time on a standard desktop/laptop computer:** **~5–10 seconds.**

---

## 4. Instructions for Use & Manuscript Reproduction

### 4.1 Configuring the Data Directory

All Python, R, and MATLAB scripts resolve data paths from a single environment variable, `MEMORYBANDIT_DATA_DIR`:

```bash
export MEMORYBANDIT_DATA_DIR="/path/to/MemoryBanditData"
```

See **[`DATA_LAYOUT.md`](DATA_LAYOUT.md)** for the directory structure and column schemas expected under `$MEMORYBANDIT_DATA_DIR`. Continuous LFP pre-processing (bipolar re-referencing, 60 Hz notch filtering, resampling to 500 Hz, 25–80 Hz 5-SD interictal epileptiform discharge detection, 5-SD gradient artifact detection, 100 ms artifact padding removal, photodiode/TTL synchronization, epoching with 1,000 ms buffer, and LeGUI / Yale Brain Atlas electrode localization) is performed upstream using [`LFPAnalysis`](https://github.com/seqasim/LFPAnalysis) (`f0ca5b4`).

### 4.2 Step-by-Step Manuscript Reproduction Order

1. `notebooks/01_behavior/00_prolific_preprocessing.ipynb` — builds `learn_df_prolific_v2.csv` and `full_data_df_prolific_v2.csv` ($N = 208 \to 200 \to 193$).
2. `notebooks/01_behavior/Fig1B_S2A_reversal_curves.ipynb` — generates **Fig 1B (top)** and **Fig S2A**.
3. `notebooks/01_behavior/Fig1B_1F_cohort_comparison.ipynb` — generates **Fig 1B (bottom)** (`log RT` and `% correct` densities + Welch *t*-tests) and **Fig 1F** ($d'$ density + Welch *t*-test).
4. `notebooks/01_behavior/Fig1DEG_S1_TableS1_modeling_EMU.ipynb` — runs parameter recovery (**Fig S1**), 6-model pyEM comparison (**Fig 1D**, **Table S1**), RW $\alpha/\beta$ density (**Fig 1E**), exports `learn_df_RWH_v2.csv` / `full_df_RWH_v2.csv`, and exports the Day-1 recognition table for R.
   - Run `Rscript R/glmer_recognition_emu.R` to fit the binomial `glmer` model, then run the final cell of the notebook to plot **Fig 1G**.
5. `notebooks/01_behavior/FigS2BCD_modeling_online.ipynb` — fits the 6 candidate models on the online cohort (**Fig S2B–C**) and exports the Day-1 recognition table.
   - Run `Rscript R/glmer_recognition_online.R`, then run the final cell of the notebook to plot **Fig S2D**.
6. `notebooks/02_electrodes/Fig1H_TableS2_electrodes.ipynb` — generates **Table S2**, the MNI glass-brain electrode map (**Fig 1H left**), and example filtered LFP traces per ROI (**Fig 1H right**).
7. `notebooks/03_spectral/00_compute_TFR_Fig2A_5A.ipynb` — computes fixation-baselined Morlet wavelet TFRs (`*-tfr.h5`, 30 log-spaced frequencies 2–200 Hz, 30 log-spaced cycles 3–10) and plots **Fig 2A** and **Fig 5A**.
8. `python tools/export_tfr_to_mat.py` — exports per-subject `feedback_start-tfr.mat` and `cue_on-tfr.mat` for MATLAB MVPA-Light scripts.
9. `notebooks/03_spectral/Fig5B_spectra.ipynb` — computes 1/f-corrected PSDs across ROIs and plots **Fig 5B**.
10. `notebooks/03_spectral/Text_theta_bursts_BOSC.ipynb` — computes eBOSC theta-burst prevalence across RPE classes in HPC and ACC (in-text *t*-tests).
11. `notebooks/03_spectral/FigS5_ripples.ipynb` — detects hippocampal ripples and generates **Fig S5A–B** and RPE-class statistical comparisons.
12. In MATLAB, run `matlab/run_manuscript_panels.m` (see [`matlab/README.md`](matlab/README.md)), which executes:
    - `test_script_feedback_decoding.m` (**Fig 2D**, **Fig S4A–B**)
    - `plot_hfa_by_rpe_category.m` (**Fig 2B**)
    - `test_script_feedback_rpe_posthoc_patterns.m` + `plot_posthoc_patterns_on_brain_surface.m` (**Fig S3**)
    - `test_script_crossdecoding.m` (**Fig 3A–C**, **Fig S6**; exports `ACC_rpe_category_HFA_subject_reinstatement.csv`)
    - `test_script_cue_decoding.m` (**Fig 3D** old cues, **Fig 3E** lure cues, **Fig 5C** HPC & ACC theta)
13. `notebooks/04_decoding_rsa_followups/Fig4_S8_RSA.ipynb` — runs within-ROI (**Fig 4A**) and cross-region (**Fig 4B**, **Fig S8**) feedback-to-cue HFA RSA and exports `group_results_ACC_to_HPC.pkl`.
14. `notebooks/04_decoding_rsa_followups/FigS7_ACC_reactivation_covariates.ipynb` — correlates subject-level ACC RPE reactivation against electrode count, feedback/cue HFA, and RW $\alpha/\beta$ (**Fig S7A–B**).
15. `notebooks/04_decoding_rsa_followups/FigS9_S10_S11_ACC_HPC_RSA_covariates.ipynb` — correlates subject-level ACC–HPC hit RSA against electrode counts and RW $\alpha/\beta$ (**Fig S9A–B**), recognition $d'$ (**Fig S10**), and ACC RPE reactivation (**Fig S11**).
16. `notebooks/05_theta_sync_pac/Fig5DEF_S13_PPC_PSI.ipynb` — computes surrogate-normalized HPC–ACC theta PPC (**Fig 5E**), trial-wise HPC$\to$ACC theta PSI (**Fig 5F**), exemplar theta traces (**Fig 5D**), and subject-level positive-RPE PPC correlations with ACC reactivation and ACC–HPC RSA (**Fig S13A–B**).
17. `notebooks/05_theta_sync_pac/FigS12_PAC.ipynb` — computes surrogate-normalized multivariate Gaussian-copula PAC (**Fig S12A–B**).
18. `Rscript R/lmer_neural.R` — runs the R `lme4::lmer` mixed-effects models for **Fig 5B**, **Fig 5E**, **Fig 5F**, and **Fig S12**.

---

## License

Released under the [MIT License](LICENSE). The vendored `pyEM` library under `external/pyem/` is distributed under its original [CC BY-NC-SA 4.0 license](external/pyem/LICENSE).
