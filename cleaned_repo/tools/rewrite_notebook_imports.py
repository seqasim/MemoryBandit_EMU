#!/usr/bin/env python3
"""
Rewrite notebook imports from legacy Notebooks/ module names to membandit.*.

Also prepends a standard setup cell (sys.path + plot style).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbformat.validator import validate

SETUP_CELL = '''# membandit setup
import os
import sys
from pathlib import Path

# Ensure cleaned_repo (parent of membandit/) is on sys.path when running from notebooks/
_here = Path.cwd().resolve()
for _candidate in [_here, *_here.parents]:
    if (_candidate / "membandit" / "__init__.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

# Optional: set MEMORYBANDIT_DATA_DIR to your data root (defaults to /Volumes/T7 Shield)
# os.environ.setdefault("MEMORYBANDIT_DATA_DIR", "/path/to/data/root")

from membandit.plotting_style import set_plot_style
set_plot_style()
'''

# Map legacy module names -> preferred membandit module for "import X" / "from X import"
MODULE_MAP = {
    "mv_regression_utils": "membandit._impl.multivariate",
    "mv_granger_utils": "membandit.granger",
    "bosc_theta_alpha_rpe_utils": "membandit.extras.bosc",
    "rpe_cat_mvpa_utils": "membandit.extras.mvpa",
    "likelihood_ratio_test": "membandit.stats",
    "plot_config": "membandit.plotting_style",
}

# Symbol-aware remaps for common from-imports (optional refinement)
SYMBOL_MODULE = {
    # behavior
    "load_behavior_tables": "membandit.behavior",
    "preprocess_learn_df": "membandit.behavior",
    "preprocess_combined_df": "membandit.behavior",
    "normalize_rpe_cat_labels": "membandit.behavior",
    "resolve_trial_for_event": "membandit.behavior",
    # io
    "Subject": "membandit.io",
    "load_subject_data": "membandit.io",
    "load_subject_epochs": "membandit.io",
    "get_roi_channels": "membandit.io",
    "select_top_electrode_per_roi": "membandit.io",
    "make_multivariate_indices": "membandit.io",
    # style
    "set_plot_style": "membandit.plotting_style",
    # LRT
    "electrode_likelihood_ratio_test": "membandit.stats",
}

MARKER = "# membandit setup"


def _rewrite_from_import(match: re.Match) -> str:
    mod = match.group(1)
    names_blob = match.group(2)
    if mod not in MODULE_MAP:
        return match.group(0)

    # Parse imported names (simple split; handles multiline poorly — caller flattens)
    raw_names = []
    for part in names_blob.replace("\n", " ").split(","):
        part = part.strip()
        if not part or part == "(" or part == ")":
            continue
        # "name as alias"
        name = part.split(" as ")[0].strip().strip("()")
        if name:
            raw_names.append(part.strip())

    # Group by destination module when possible
    groups: dict[str, list[str]] = {}
    for part in raw_names:
        base = part.split(" as ")[0].strip().strip("()")
        dest = SYMBOL_MODULE.get(base, MODULE_MAP[mod])
        groups.setdefault(dest, []).append(part)

    lines = []
    for dest, parts in groups.items():
        if len(parts) == 1:
            lines.append(f"from {dest} import {parts[0]}")
        else:
            joined = ", ".join(parts)
            lines.append(f"from {dest} import {joined}")
    return "\n".join(lines)


def patch_source_text(text: str) -> str:
    # from X import Y
    text = re.sub(
        r"from\s+(mv_regression_utils|mv_granger_utils|bosc_theta_alpha_rpe_utils|rpe_cat_mvpa_utils|likelihood_ratio_test|plot_config)\s+import\s+\((.*?)\)",
        _rewrite_from_import,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"from\s+(mv_regression_utils|mv_granger_utils|bosc_theta_alpha_rpe_utils|rpe_cat_mvpa_utils|likelihood_ratio_test|plot_config)\s+import\s+([^\n]+)",
        _rewrite_from_import,
        text,
    )
    # import X / import X as Y
    for old, new in MODULE_MAP.items():
        text = re.sub(
            rf"\bimport\s+{re.escape(old)}\b(\s+as\s+\w+)?",
            lambda m, n=new: f"import {n}{m.group(1) or ''}",
            text,
        )
    return text


def ensure_setup_cell(nb: nbformat.NotebookNode) -> None:
    if nb.cells and MARKER in "".join(nb.cells[0].get("source", "")):
        return
    cell = nbformat.v4.new_code_cell(SETUP_CELL)
    nb.cells.insert(0, cell)


def rewrite_notebook(path: Path, inplace: bool = True) -> Path:
    nb = nbformat.read(path, as_version=4)
    # Cell ids require nbformat >= 4.5
    if nb.get("nbformat_minor", 0) < 5:
        nb["nbformat_minor"] = 5
    ensure_setup_cell(nb)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cell["source"] = patch_source_text(src)
    try:
        validate(nb)
    except Exception:
        # Some legacy notebooks have extra metadata; still write after rewrite.
        pass
    if inplace:
        nbformat.write(nb, path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path, help="Notebook paths to rewrite")
    args = parser.parse_args(argv)
    for nb_path in args.notebooks:
        rewrite_notebook(nb_path)
        print(f"rewrote {nb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
