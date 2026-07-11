# External (non-PyPI) dependencies

These are **not** installed by `pip install -r ../requirements.txt`. Place lab-specific code or MATLAB toolboxes in the subfolders below (or install from your lab’s git remote), then add them to your path as described.

## `LFPAnalysis/`

**Purpose:** Lab Python package used for BOSC (`eBOSC_wrapper`). Imported as:

```python
from LFPAnalysis.oscillation_utils import eBOSC_wrapper
```

**Used by:** `manuscript_utils/bosc_theta_alpha_rpe_utils.py`, and notebooks:

- `MemoryBandit-CrossRegionMV-GCPAC.ipynb`
- `MemoryBandit-HippocampalRippleDetection.ipynb`
- `MemoryBandit-TFRPlotsAndStatistics.ipynb`
- `MemoryBanditsEMU_Behavior.ipynb`
- `BayesianRegression.ipynb` (if BOSC-related cells run)

**Setup options:**

1. Clone or copy your lab’s `LFPAnalysis` repository **into** `external/LFPAnalysis/` so the package root contains `LFPAnalysis/__init__.py` (or equivalent).
2. Or: `pip install -e /path/to/your/LFPAnalysis` from wherever you keep the repo.
3. Ensure the package is importable before running notebooks, e.g. prepend to `sys.path` in a notebook cell (after the existing path cell). Use `Path.cwd()` or an absolute path — **`__file__` is not defined in Jupyter**:

   ```python
   import sys
   from pathlib import Path
   _repo = Path.cwd().resolve()  # or Path("/path/to/manuscript_repo")
   _ext = _repo / "external" / "LFPAnalysis"
   if _ext.is_dir():
       sys.path.insert(0, str(_ext.parent))  # adjust for your package layout
   ```

   Adjust the inserted path to match your actual package layout (`site-packages` style vs. src layout).

## `bayes_models/`

**Purpose:** Local helper modules for hierarchical / RL model fitting used by the Bayesian behavior notebooks (not on PyPI). Typical filenames (copy from your lab repo):

- `BayesModel.py`, `EWAModel.py`, `HierarchicalBayes.py`
- `HierarchicalRW.py`, `HierarchicalRW_Fictive1.py`, `HierarchicalRW_Signed.py`, `HierarchicalRW_Unsigned.py`
- `HierarchicalVKF.py`, `HierarchicalWSLS.py`
- `RWModel.py`, `RWModel_Fictive1.py`, `RWModel_Fictive2.py`, `RWModel_Signed.py`, `RWModel_Unsigned.py`
- `VKFModel.py`, `WSLSModel.py`
- `model_utils.py`, `param_recov_utils.py`, `model_recov_utils.py`

**Used by:** `BayesianRegression.ipynb`, `EMU_Behav.ipynb`, `MemoryBanditsEMU_Behavior.ipynb`, `ProlificSubjectsBehavior_Processing-Delay.ipynb`

**Setup:** Add `external/bayes_models` to `sys.path` in the notebook (same pattern as above), or run notebooks with `PYTHONPATH` including that directory:

```bash
export PYTHONPATH="/path/to/manuscript_repo/external/bayes_models:$PYTHONPATH"
```

## `matlab_toolboxes/`

**Purpose:** MATLAB code in `../matlab/` depends on:

| Toolbox | Role | Scripts |
|---------|------|---------|
| **MVPA-Light** | `mv_classify`, `mv_regress`, `mv_classify_across_time`, `mv_get_hyperparameter`, `mv_stat_activation_pattern`, etc. | All `test_script_*.m` under `matlab/` |
| **FieldTrip** | Paths referenced for cross-decoding scripts | `test_script_crossdecoding.m`, `test_script_crossdecoding_univariate.m` |

**Upstream references:**

- MVPA-Light: <https://github.com/treder/MVPA-Light>
- FieldTrip: <https://www.fieldtriptoolbox.org/>

**Typical MATLAB setup:**

```matlab
addpath(genpath('/path/to/MVPA-Light'));
addpath(genpath('/path/to/fieldtrip-YYYYMMDD'));
ft_defaults;  % FieldTrip, if required by your workflow
```

The copied `.m` files may contain **machine-specific** `ft_base = '...'` paths; replace those with your local FieldTrip install.

---

## Empty placeholder dirs

Subfolders `LFPAnalysis/`, `bayes_models/`, and `matlab_toolboxes/` may contain only `.gitkeep` until you populate them. They are **not** versioned with proprietary lab code.
