# External (non-PyPI) dependencies

These are **not** installed by `pip install -e .`. Place lab-specific code or MATLAB
toolboxes in the subfolders below (or install from your lab’s git remote).

## `LFPAnalysis/`

**Purpose:** Lab Python package for BOSC (`eBOSC_wrapper`).

```python
from LFPAnalysis.oscillation_utils import eBOSC_wrapper  # used inside membandit.extras.bosc
```

**Used by:** `membandit.extras.bosc`; curated notebooks
`MemoryBandit-CrossRegionMV-GCPAC.ipynb`, `MemoryBandit-HippocampalRippleDetection.ipynb`,
`MemoryBandit-TFRPlotsAndStatistics.ipynb`, and some behavior notebooks.

**Setup:** Clone into `external/LFPAnalysis/` or `pip install -e /path/to/LFPAnalysis`, then
ensure it is on `PYTHONPATH` / `sys.path`.

## `bayes_models/`

**Purpose:** Local hierarchical / RL model modules (`HierarchicalRW*`, `RWModel*`,
`model_utils`, etc.) used by Bayesian behavior notebooks.

**Setup:**

```bash
export PYTHONPATH="/path/to/cleaned_repo/external/bayes_models:$PYTHONPATH"
```

## `matlab_toolboxes/`

Scripts in `../matlab/` need:

| Toolbox | Role |
|---------|------|
| **MVPA-Light** | Decoding / cross-decoding / activation patterns |
| **FieldTrip** | Cross-decoding scripts (edit machine-specific `ft_base` paths) |

- MVPA-Light: https://github.com/treder/MVPA-Light
- FieldTrip: https://www.fieldtriptoolbox.org/
