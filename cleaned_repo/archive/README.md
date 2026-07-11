# Archive index

This folder holds analyses that are **not** part of the curated manuscript notebook set in
[`../notebooks/`](../notebooks/), but remain useful for exploration, smoketests, and alternate
condition forks. Imports have been rewritten to `membandit.*`.

Prefer the curated notebooks + [`../ANALYSIS_GUIDE.md`](../ANALYSIS_GUIDE.md) for publication
workflows. Use this archive when you need a variant, smoketest, or deeper dive.

## Notebook families (`notebooks/`)

| Family | Contents |
|--------|----------|
| `behavior_ddm/` | Prolific Bayesian regression variant; UAB site ephys fork |
| `tfr_spectral/` | TFR viewers, ITC, cue theta power, TFR GLM smoketest, EMU ephys loader |
| `oscillation_psd_fooof/` | PSD/FOOOF, BOSC RPE, OFC beta, cue band-power overlap |
| `timeresolved_regression/` | Legacy + template time-resolved regression generations |
| `mv_uv_reactivation/` | MV/UV encoding–reactivation drivers and reinstatement variants |
| `mvpa_decoding_mi/` | RPE-category MVPA, multivariate MI / MIC |
| `rsa/` | Pre-template HFA RSA; ACC–HPC phase RSA |
| `granger/` | Base / reward / hit–miss / univariate GC forks |
| `pac/` | Within-/cross-region PAC + smoketests |
| `connectivity/` | wPLI, PSI, canonical coherence, retrieval connectivity |
| `electrode_roi_regression/` | Electrode GLMs, RPE×memory overlap, ROI interaction variance |
| `integration/` | Cross-metric correlation hubs (decoding × GC × RSA) |

## Other archive areas

| Path | Contents |
|------|----------|
| `ddm/` | Stan DDM / RL-DDM models and notebooks — see [`ddm/DDM_README.md`](ddm/DDM_README.md) |
| `notebook_generators/` | One-off `_build_*.py` / `_insert_*.py` scripts that generated some notebooks |

## Ripple detection

The manuscript ripple notebook lives in the curated set:
`../notebooks/MemoryBandit-HippocampalRippleDetection.ipynb`.
