# Contributing

## Development setup

```bash
cd cleaned_repo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"
```

## Code organization

- Public APIs live in `membandit/*.py` (facades).
- Full implementations live in `membandit/_impl/` (patched copies of the original monoliths).
- Rebuild facades after changing the build map:

```bash
python tools/build_membandit_modules.py
```

## Checks (no data required)

```bash
python -c "import membandit"
python -m pyflakes membandit
python -c "
from pathlib import Path
import nbformat
for p in list(Path('notebooks').glob('*.ipynb')) + list(Path('archive').rglob('*.ipynb')):
    nbformat.read(p, as_version=4)
print('notebooks readable')
"
```

## Notebooks

- Curated manuscript notebooks: `notebooks/`
- Variants / extras: `archive/`
- When copying a new notebook from legacy `Notebooks/`, run
  `tools/rewrite_notebook_imports.py` on it.

## Pull requests

- Prefer small, focused PRs (one analysis family or one module).
- Do not commit data artifacts (`.fif`, fit caches, `.venv/`).
- Do not require `/Volumes/T7 Shield` for CI; keep imports data-free at module load.
