# Drift-diffusion / RL-DDM models (archive)

Stan models and exploratory notebooks for recognition-memory and learning-phase DDMs.
These were **not** required for the main manuscript figure pipeline; kept here for reuse.

## Stan models

| File | Model |
|------|-------|
| `recog_ddm.stan` | Hierarchical Wiener DDM for recognition RTs (subject-level `v`, `a`, `t0`) |
| `recog_ddm_rpe.stan` | Recognition DDM with trial-wise RPE modulating drift |
| `recog_ddm_reward.stan` | Recognition DDM with trial-wise reward modulating drift |
| `recog_ddm_pm.stan` | Recognition DDM with trial-wise memorability / d′ modulating drift |
| `rl_ddm.stan` | RL-DDM for learning: Q-learning + Wiener likelihood on gamble RTs |
| `rl_ddm_driftRPE.stan` | RL-DDM variant where RPE uses drift-as-confidence |

## Notebooks

| Notebook | Role |
|----------|------|
| `stan_test.ipynb` | Behavior feature engineering + Stan prep |
| `stan_test_recogDDM.ipynb` | Recognition DDM fits |
| `stan_test_driftRPE.ipynb` | Drift–RPE RL-DDM variant |
| `ddm_test.ipynb` | HSSM / HDDM prototyping |

## How to fit

1. Install CmdStan / CmdStanPy (or your preferred Stan interface).
2. Point data paths via `MEMORYBANDIT_DATA_DIR` (see package `membandit.paths`).
3. Open the relevant notebook under this folder; models are co-located as `*.stan`.

Empty `recog_ddm*/` and `rl_ddm*/` output directories from the original `Notebooks/` tree are
intentionally **not** copied (they held local fit artifacts).
