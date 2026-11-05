#!/usr/bin/env python3
"""
Static verification tool for the ``EMU_manuscript`` reproducibility package.

Checks:
  1. ``membandit`` and ``rlmodels_k99`` import cleanly.
  2. Every notebook under ``notebooks/`` validates with ``nbformat`` and starts
     with a standard manuscript metadata header cell.
  3. No hardcoded machine-specific absolute paths (``/Volumes/T7 Shield``,
     ``/Users/<user>``, ``/sc/arion``) exist in any Python, notebook, MATLAB, or R file.
  4. Every script/notebook referenced in ``FIGURE_MAP.md`` exists on disk.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "membandit_check_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("MNE_HOME", _CACHE_DIR)
os.environ.setdefault("HOME", _CACHE_DIR)

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_PATTERNS = (
    re.compile(r"/Volumes/T7 Shield"),
    re.compile(r"/Users/salmanqasim"),
    re.compile(r"/sc/arion"),
)


def check_imports() -> list[str]:
    errors: list[str] = []
    for extra in (REPO_ROOT, REPO_ROOT / "external"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    try:
        import membandit  # noqa: F401
        from membandit import behavior, paths, plotting_style, subject_covariates  # noqa: F401
    except Exception as exc:
        errors.append(f"Failed to import membandit: {exc}")

    try:
        import pyem  # noqa: F401
        import rlmodels_k99  # noqa: F401
        from rlmodels_k99 import registry  # noqa: F401
    except Exception as exc:
        errors.append(f"Failed to import rlmodels_k99 / pyem: {exc}")

    return errors


def check_notebooks() -> list[str]:
    errors: list[str] = []
    nb_paths = sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))
    if not nb_paths:
        errors.append("No notebooks found under notebooks/")
        return errors

    for p in nb_paths:
        rel = p.relative_to(REPO_ROOT)
        try:
            nb = nbformat.read(str(p), as_version=4)
            nbformat.validate(nb)
        except Exception as exc:
            errors.append(f"{rel}: nbformat validation failed: {exc}")
            continue
        if not nb.cells or nb.cells[0].cell_type != "markdown" or "Manuscript panels" not in nb.cells[0].source:
            errors.append(f"{rel}: first cell is not a standard manuscript header table")
    return errors


def check_no_hardcoded_paths() -> list[str]:
    errors: list[str] = []
    exts = {".py", ".ipynb", ".m", ".R"}
    for p in sorted(REPO_ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in exts:
            continue
        if any(part.startswith(".") or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.resolve() == Path(__file__).resolve() or p.name == "test_smoke.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in FORBIDDEN_PATTERNS:
            m = pat.search(text)
            if m:
                errors.append(f"{p.relative_to(REPO_ROOT)}: matched forbidden path pattern {pat.pattern!r}")
    return errors


def check_figure_map_links() -> list[str]:
    errors: list[str] = []
    fmap = REPO_ROOT / "FIGURE_MAP.md"
    if not fmap.is_file():
        errors.append("FIGURE_MAP.md is missing")
        return errors
    text = fmap.read_text(encoding="utf-8")
    ref_pattern = re.compile(r"`((?:notebooks|matlab|R|tools)/[^`\s]+\.(?:ipynb|m|R|py))`")
    refs = sorted(set(ref_pattern.findall(text)))
    if not refs:
        errors.append("No code paths found in FIGURE_MAP.md")
    for ref in refs:
        if not (REPO_ROOT / ref).is_file():
            errors.append(f"FIGURE_MAP.md references missing file: {ref}")
    return errors


def main() -> int:
    all_errors: list[str] = []
    for name, fn in [
        ("imports", check_imports),
        ("notebooks", check_notebooks),
        ("hardcoded_paths", check_no_hardcoded_paths),
        ("figure_map_links", check_figure_map_links),
    ]:
        errs = fn()
        if errs:
            for e in errs:
                print(f"[FAIL:{name}] {e}")
            all_errors.extend(errs)
        else:
            print(f"[PASS] {name}")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
