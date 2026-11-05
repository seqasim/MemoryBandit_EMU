"""
Automated smoke tests for the ``EMU_manuscript`` reproducibility package.

Runs with either ``pytest`` or ``python -m unittest discover -s tests -p 'test_*.py'``
without requiring ``MEMORYBANDIT_DATA_DIR`` to be mounted.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tools import check_package  # noqa: E402


class TestManuscriptPackageSmoke(unittest.TestCase):
    def test_imports(self) -> None:
        errors = check_package.check_imports()
        self.assertFalse(errors, "\n".join(errors))

    def test_notebooks_valid_and_documented(self) -> None:
        errors = check_package.check_notebooks()
        self.assertFalse(errors, "\n".join(errors))

    def test_no_hardcoded_machine_paths(self) -> None:
        errors = check_package.check_no_hardcoded_paths()
        self.assertFalse(errors, "\n".join(errors))

    def test_figure_map_references_exist(self) -> None:
        errors = check_package.check_figure_map_links()
        self.assertFalse(errors, "\n".join(errors))

    def test_r_scripts_parse(self) -> None:
        r_scripts = sorted((REPO_ROOT / "R").glob("*.R"))
        self.assertGreaterEqual(len(r_scripts), 3, "Expected at least 3 R scripts in R/")
        for r_file in r_scripts:
            res = subprocess.run(
                ["Rscript", "-e", f"parse(file={str(r_file)!r})"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, f"{r_file.name} failed R parse:\n{res.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
