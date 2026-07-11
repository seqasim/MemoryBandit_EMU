# MemoryBandit EMU — manuscript analysis bundle

Publication-oriented copies of analysis notebooks and a unified Python package (`manuscript_utils/`) consolidated from `mv_regression_utils.py`, `mv_granger_utils.py`, `bosc_theta_alpha_rpe_utils.py`, and `likelihood_ratio_test.py`. **Original notebooks under `Notebooks/` are not modified**; work here only.

## Layout

| Path | Purpose |
|------|---------|
| `manuscript_utils/` | Python package (behavior loading, multivariate RSA/encoding, Granger, BOSC, stats helpers) |
| `notebooks/` | Cleaned Jupyter copies wired to `manuscript_utils` |
| `matlab/` | MATLAB scripts copied verbatim from `MVPA-Light/projects/MemoryBandit` (cross-decoding / validation) |
| `external/` | Placeholder dirs + instructions for **non-pip** deps (`LFPAnalysis`, Bayesian model modules, MATLAB toolboxes); see [external/README.md](external/README.md) |

## Environment & dependencies

### Python version

Python **3.10+** recommended.

### Pip install (full stack)

From `manuscript_repo/`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes the core scientific stack, Jupyter, stats/ML, electrophysiology (**mne**, **mne-connectivity**, **neurodsp**, **fooof**, **tensorpac**, **nibabel**), and Bayesian tooling (**pymc**, **bambi**, **arviz**, **pytensor**, **numpyro**, **jax**, **jaxlib**, **xarray**, **dask**).

### JAX / JAXLIB (platform-specific)

**jax** and **jaxlib** wheels differ by OS and accelerator (CPU vs CUDA vs Apple Metal). If `pip install -r requirements.txt` fails or you need GPU/Metal, follow the official install guide: <https://jax.readthedocs.io/en/latest/installation.html> and adjust versions to match your platform.

### Data paths

Set data root (optional; defaults to `/Volumes/T7 Shield`):

```bash
export MEMORYBANDIT_DATA_DIR="/path/to/your/data/root"
```

EMU behavior tables are expected at:

`$MEMORYBANDIT_DATA_DIR/work/qasims01/MemoryBanditData/EMU/`  
(files `learn_df_RWH.csv`, `full_df_RWH.csv`).

In code: `from manuscript_utils.paths import get_base_dir, get_emu_behavior_dir`.

### External (non-pip) dependencies

Full instructions: **[external/README.md](external/README.md)**.

| Dependency | Needed for | Notebooks / code |
|------------|------------|------------------|
| **LFPAnalysis** (lab package) | BOSC `eBOSC_wrapper` | `bosc_theta_alpha_rpe_utils`; `MemoryBandit-CrossRegionMV-GCPAC`, `MemoryBandit-HippocampalRippleDetection`, `MemoryBandit-TFRPlotsAndStatistics`, `MemoryBanditsEMU_Behavior`; optional cells in `BayesianRegression` |
| **Bayesian model modules** (`HierarchicalRW*`, `RWModel*`, `model_utils`, etc.) | Hierarchical RL fits | `BayesianRegression`, `EMU_Behav`, `MemoryBanditsEMU_Behavior`, `ProlificSubjectsBehavior_Processing-Delay` |
| **`plot_config.py`** | Shared matplotlib style | Drop in parent `Notebooks/`; first notebook cell adds it to `sys.path` when present |
| **MVPA-Light** (MATLAB) | All `mv_*` calls in `matlab/*.m` | Required to run MATLAB decoding scripts |
| **FieldTrip** (MATLAB) | Cross-decoding scripts only | `test_script_crossdecoding.m`, `test_script_crossdecoding_univariate.m` (edit `ft_base` paths) |

You can skip **LFPAnalysis** / **Bayesian modules** / **MATLAB** if you only run notebooks that do not import them.

## Running notebooks

1. Open Jupyter from a cwd where the first path cell resolves `manuscript_repo` (e.g. open the project at `.../MemoryBandit_EMU/Notebooks` and run notebooks under `manuscript_repo/notebooks/`), **or** adjust `sys.path` in the first cell.
2. The first code cell prepends `manuscript_repo` to `sys.path` and, if present, the parent `Notebooks/` folder for `plot_config.py`.

## `manuscript_utils` overview

| Module | Contents |
|--------|----------|
| `behavior` | `load_behavior_tables`, `preprocess_learn_df`, `preprocess_combined_df`, unified `rpe_cat` labels (`negative` / `neutral` / `positive`), `resolve_trial_for_event`, `normalize_rpe_cat_labels` |
| `multivariate` | Large module: `Subject`, TFR GLM, RSA, reactivation, multinomial / LRT maps, ROI oscillation |
| `granger` | Multivariate Granger causality, trial maps, mixed-effects plots |
| `bosc_theta_alpha_rpe_utils` | BOSC pipeline (theta/alpha) |
| `oscillation` | Re-exports BOSC + key ROI PSD / FOOOF entry points |
| `stats` | `electrode_likelihood_ratio_test`, `compute_null_stats_one_sided`, MNE cluster tests |
| `plotting` | Common `plot_*` re-exports from `granger` and `multivariate` |
| `paths` | `MEMORYBANDIT_DATA_DIR`, EMU paths |

Import examples:

```python
from manuscript_utils.behavior import load_behavior_tables, preprocess_learn_df, preprocess_combined_df
from manuscript_utils.multivariate import Subject, load_subject_data
from manuscript_utils.granger import compute_multivariate_gc
```

## Figure → notebook mapping (main text & supplements)

Approximate cell indices shift after the inserted path/title cells (+2 at the top). Use the notebook outline / search (`savefig`, `to_csv`) for exact export cells.

| Manuscript | Panels | Notebook | Notes |
|------------|--------|------------|--------|
| Fig 1 | B, C, S1 | `notebooks/ProlificSubjectsBehavior_Processing-Delay.ipynb` | Online cohort, model comparison |
| Fig 1 | F | `notebooks/BayesianRegression.ipynb` | Mixed-effects posteriors (online) |
| Fig 2 | A–D | `notebooks/MemoryBanditsEMU_Behavior.ipynb`, `notebooks/EMU_Behav.ipynb` | EMU behavior |
| Fig 2 | D | `notebooks/BayesianRegression.ipynb` | EMU memory regression |
| Fig 2 | E, F; Table 2 | `notebooks/MemoryBandit-ParticipantElectrodeSummary.ipynb` | Electrode coverage |
| Fig 3 | A | `notebooks/MemoryBandit-TFRPlotsAndStatistics.ipynb` | Example TFR |
| Fig 3 | B–D | `notebooks/HFA_RPE_RSA_MVTemplate.ipynb` | RPE decoding, time courses |
| Fig 3 | S2, S3, S5 | `notebooks/HFA_RPE_RSA_MVTemplate.ipynb` | Posthoc patterns, memorability decode, other PF ROIs |
| Fig S4 | | `notebooks/MemoryBandit-HippocampalRippleDetection.ipynb` | Ripple rates |
| Fig 4 | A–C | `notebooks/HFA_RPE_RSA_MVTemplate.ipynb` | Cross-period decoding / reinstatement |
| Fig 4 | D, S6 | `notebooks/MemoryBandit-ACC_Reinstatement_SubjectCovariates.ipynb` | Covariates, RL params |
| Fig 5 | A–C | `notebooks/MemoryBandit_ROI_OscillationComparison.ipynb` | Theta / spectra / bursts |
| Fig 5 | D, S7 | `notebooks/MemoryBandit-MultivariateGrangerCausality_RPE.ipynb` | MVGC RPE |
| Fig 5 | E | `notebooks/MemoryBandit-ACC_dmPFC_Reinstatement_GC_RPE.ipynb` | Reinstatement vs GC |
| Fig S8 | | `notebooks/MemoryBandit-CrossRegionMV-GCPAC.ipynb` | Phase–amplitude coupling |
| Fig 6 | A–C | `notebooks/MemoryBandit-ACC_HPC_HitsRSA_GC_RPE.ipynb` | Cross-region RSA hit–miss |
| Fig S9–S11 | | `notebooks/MemoryBandit-ACC_HPC_HitsRSA_GC_RPE.ipynb` | Between-ROI RSA supplements |

## MATLAB (`matlab/`)

Scripts are **unmodified** copies for decoding / surface plots used alongside Python MVPA:

- `test_script_feedback_decoding.m`, `test_script_cue_decoding.m`, `test_script_crossdecoding.m`, etc.
- See each file’s header comments for usage.

## Conventions enforced in this bundle

- **`learn_df` / `combined_df`:** Prefer `load_behavior_tables` + `preprocess_learn_df` / `preprocess_combined_df` from `manuscript_utils.behavior` so columns and **`rpe_cat`** labels match across analyses.
- **`rpe_cat`:** `negative`, `neutral`, `positive` for both learn and combined tables.
- **Trial column resolution:** `manuscript_utils.behavior.resolve_trial_for_event`.

## Maintenance

- Re-run `patch_notebooks.py` only on fresh copies from `Notebooks/` if you need to re-apply path/import rewrites after updating sources.
