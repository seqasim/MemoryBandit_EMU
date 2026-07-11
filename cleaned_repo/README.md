# MemoryBandit EMU — cleaned analysis package

Installable Python package (`membandit`) plus curated notebooks for the MemoryBandit
intracranial EEG (EMU) manuscript analyses. Built as a cleaned reorganization of the
legacy `Notebooks/` tree.

**Start here for analyses:** [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)

## Layout

| Path | Purpose |
|------|---------|
| `membandit/` | Installable package (behavior, I/O, regression, RSA, Granger, …) |
| `notebooks/` | Curated manuscript notebooks (14) |
| `archive/` | Remaining analyses by family + DDM/Stan + notebook generators |
| `matlab/` | MVPA-Light MATLAB scripts |
| `external/` | Placeholders / instructions for non-PyPI deps |
| `tools/` | Build / notebook import-rewrite helpers |

## Quick start

```bash
cd cleaned_repo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks,oscillation,pac,bayes]"
# or: pip install -r requirements.txt && pip install -e .
```

Set your data root (optional; defaults to `/Volumes/T7 Shield`):

```bash
export MEMORYBANDIT_DATA_DIR="/path/to/your/data/root"
```

```python
from membandit import Subject, load_behavior_tables, set_plot_style, paths
from membandit.behavior import preprocess_learn_df, preprocess_combined_df

set_plot_style()
learn_df, combined_df = load_behavior_tables(paths.get_emu_behavior_dir())
learn_df = preprocess_learn_df(learn_df)
```

## External (non-pip) dependencies

See [external/README.md](external/README.md) for **LFPAnalysis** (BOSC), Bayesian model
modules, and MATLAB toolboxes (MVPA-Light, FieldTrip).

You can skip lab-only packages if you only run notebooks that do not import them.
`membandit.extras.bosc` imports `LFPAnalysis` **lazily** so `import membandit` works without it.

## Conventions

- **`rpe_cat`:** `negative` / `neutral` / `positive` (canonical in `membandit.behavior`)
- **Trial columns:** `membandit.behavior.resolve_trial_for_event`
- **Behavior tables:** prefer `load_behavior_tables` + `preprocess_*_df`

## Maintenance tools

```bash
# Rebuild public facades / _impl from Notebooks/ sources (developers)
python tools/build_membandit_modules.py

# Rewrite legacy imports in a notebook copy
python tools/rewrite_notebook_imports.py path/to/notebook.ipynb
```

## License

MIT (see [LICENSE](LICENSE)). Confirm with the project owner before public release if needed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
