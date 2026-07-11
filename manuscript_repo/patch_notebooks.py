#!/usr/bin/env python3
"""One-shot patcher for manuscript_repo notebooks (import paths + manuscript_utils)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"

PATH_CELL = """# Manuscript publication copy — add manuscript_utils to path
import sys
from pathlib import Path
_here = Path.cwd().resolve()
for _p in (_here, _here.parent, _here.parent.parent):
    if (_p / "manuscript_utils" / "__init__.py").is_file():
        sys.path.insert(0, str(_p))
        break
_notebooks = _p.parent
if (_notebooks / "plot_config.py").is_file():
    sys.path.insert(0, str(_notebooks))
"""

TITLES: dict[str, str] = {
    "HFA_RPE_RSA_MVTemplate.ipynb": "# HFA / cross-region RSA & reinstatement (MV template)\n\n**Manuscript figures:** Fig 3B–D, Fig 4A–C, Fig S2, S3, S5.\n",
    "MemoryBandit-TFRPlotsAndStatistics.ipynb": "# Time–frequency plots and statistics\n\n**Manuscript figures:** Fig 3A.\n",
    "BayesianRegression.ipynb": "# Bayesian / hierarchical regression (behavior + memory)\n\n**Manuscript figures:** Fig 1F, Fig 2D, Table 2 (related statistics).\n",
    "ProlificSubjectsBehavior_Processing-Delay.ipynb": "# Prolific cohort behavior processing\n\n**Manuscript figures:** Fig 1B, 1C, Fig S1.\n",
    "EMU_Behav.ipynb": "# EMU intracranial cohort — behavior assembly\n\n**Manuscript figures:** Fig 2A–D (behavior panels).\n",
    "MemoryBanditsEMU_Behavior.ipynb": "# EMU behavior summaries\n\n**Manuscript figures:** Fig 2A–D.\n",
    "MemoryBandit-ParticipantElectrodeSummary.ipynb": "# Participant electrode summary\n\n**Manuscript figures:** Fig 2E–F; Table 2.\n",
    "MemoryBandit-HippocampalRippleDetection.ipynb": "# Hippocampal ripple detection\n\n**Manuscript figures:** Fig S4.\n",
    "MemoryBandit-ACC_Reinstatement_SubjectCovariates.ipynb": "# ACC reinstatement — subject covariates\n\n**Manuscript figures:** Fig 4D, Fig S6.\n",
    "MemoryBandit_ROI_OscillationComparison.ipynb": "# ROI oscillation comparison (PSD / FOOOF)\n\n**Manuscript figures:** Fig 5A–C.\n",
    "MemoryBandit-MultivariateGrangerCausality_RPE.ipynb": "# Multivariate Granger causality (RPE)\n\n**Manuscript figures:** Fig 5D, Fig S7.\n",
    "MemoryBandit-ACC_dmPFC_Reinstatement_GC_RPE.ipynb": "# ACC–dmPFC reinstatement & Granger (RPE)\n\n**Manuscript figures:** Fig 5E.\n",
    "MemoryBandit-CrossRegionMV-GCPAC.ipynb": "# Cross-region multivariate PAC (GCPAC)\n\n**Manuscript figures:** Fig S8.\n",
    "MemoryBandit-ACC_HPC_HitsRSA_GC_RPE.ipynb": "# ACC–HPC RSA & Granger (hits/misses, RPE)\n\n**Manuscript figures:** Fig 6A–C, Fig S9–S11.\n",
}


def _lines_to_source(s: str) -> list[str]:
    lines = s.splitlines(keepends=True)
    return lines if lines else ["\n"]


def patch_source_text(text: str) -> str:
    text = text.replace("from mv_regression_utils import", "from manuscript_utils.multivariate import")
    text = text.replace("import mv_regression_utils as", "import manuscript_utils.multivariate as")
    text = text.replace("import mv_regression_utils\n", "import manuscript_utils.multivariate as mv_regression_utils\n")
    text = text.replace("mv_regression_utils.", "mv_regression_utils.")  # noop
    text = text.replace("from mv_granger_utils import", "from manuscript_utils.granger import")
    text = text.replace("import mv_granger_utils as", "import manuscript_utils.granger as")
    text = text.replace("from bosc_theta_alpha_rpe_utils import", "from manuscript_utils.bosc_theta_alpha_rpe_utils import")
    text = text.replace("import bosc_theta_alpha_rpe_utils", "import manuscript_utils.bosc_theta_alpha_rpe_utils as bosc_theta_alpha_rpe_utils")
    return text


def main() -> None:
    for nb_path in sorted(NB_DIR.glob("*.ipynb")):
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        # Path cell at index 0
        path_cell = {
            "cell_type": "code",
            "metadata": {},
            "source": _lines_to_source(PATH_CELL),
            "outputs": [],
            "execution_count": None,
        }
        if nb["cells"] and "manuscript_utils" in "".join(nb["cells"][0].get("source", [])):
            pass  # already patched
        else:
            nb["cells"].insert(0, path_cell)

        # Title markdown at index 1
        title = TITLES.get(nb_path.name)
        if title:
            title_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": _lines_to_source(title),
            }
            # Avoid duplicate title if re-run
            if len(nb["cells"]) > 1 and nb["cells"][1].get("cell_type") == "markdown":
                first_md = "".join(nb["cells"][1].get("source", []))
                if "Manuscript figures" in first_md and "Manuscript publication copy" in "".join(nb["cells"][0].get("source", [])):
                    nb["cells"][1] = title_cell
                else:
                    nb["cells"].insert(1, title_cell)
            else:
                nb["cells"].insert(1, title_cell)

        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            if isinstance(src, str):
                text = src
            else:
                text = "".join(src)
            new_text = patch_source_text(text)
            if isinstance(src, str):
                cell["source"] = new_text
            else:
                cell["source"] = _lines_to_source(new_text) if "\n" in new_text else [new_text]

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("patched", nb_path.name)


if __name__ == "__main__":
    main()
