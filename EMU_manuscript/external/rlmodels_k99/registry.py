"""Canonical ordered registry of retained models and candidates."""

from __future__ import annotations

try:
    from pyem import EMModel
except ImportError:
    EMModel = None  # type: ignore[assignment,misc]

from rlmodels_k99.models import (
    bayes_fixed_reward_model,
    bayes_model,
    rw_model,
    rw_side_bias_model,
    rw_signed_model,
    rw_sticky_model,
    rwf_model,
    rwus_model,
    vkf_model,
    wsls_model,
)
from rlmodels_k99.models.bayes_variants import (
    bayes_constrained_model,
    bayes_symmetric_model,
    bayes_two_param_model,
)
from rlmodels_k99.models.rw_variants import rw_asym_model, rw_signed_asym_model
from rlmodels_k99.models.rwus_upe import rwus_upe_model
from rlmodels_k99.models.sdt_recognition import (
    sdt_zrpe_day_2_model,
    sdt_zrpe_day_3_model,
)
from rlmodels_k99.models.vkf_variants import (
    vkf_fixed_v0_model,
    vkf_omega_model,
    vkf_w0_model,
)
from rlmodels_k99.parameters import MODEL_PARAM_NAMES, xforms_for

# Canonical comparison set IDs (legacy IDs preserved for notebooks / audits).
MODEL_REGISTRY = {
    "wsls": wsls_model,
    "rw": rw_model,
    "rwf": rwf_model,
    "rw_signed": rw_signed_model,
    "rw_side_bias": rw_side_bias_model,
    "rw_sticky": rw_sticky_model,
    "rwus": rwus_model,
    "vkf": vkf_model,
    "bayes": bayes_model,
    "bayes_fixed_reward": bayes_fixed_reward_model,
    "rw_asym": rw_asym_model,
}

# Opt-in / sensitivity IDs — not deleted; not in the two-parameter comparison set.
CANDIDATE_REGISTRY = {
    "sdt_zrpe_day_2": sdt_zrpe_day_2_model,
    "sdt_zrpe_day_3": sdt_zrpe_day_3_model,
    "bayes_constrained": bayes_constrained_model,
    "bayes_symmetric": bayes_symmetric_model,
    "bayes_two_param": bayes_two_param_model,
    "rwus_upe": rwus_upe_model,
    "vkf_omega": vkf_omega_model,
    "vkf_fixed_v0": vkf_fixed_v0_model,
    "vkf_w0": vkf_w0_model,
    "rw_signed_asym": rw_signed_asym_model,
}

ALL_MODEL_REGISTRY = {**MODEL_REGISTRY, **CANDIDATE_REGISTRY}

# Legacy MODEL_ORDER preserved for backward-compatible notebook paths.
MODEL_ORDER = [
    "wsls",
    "rw",
    "rwf",
    "rw_signed",
    "rw_side_bias",
    "rw_sticky",
    "rwus",
    "vkf",
    "bayes",
    "bayes_fixed_reward",
    "rw_asym",
]

# Final manuscript comparison set (~2 free params identifiable on this task).
# One VKF only: vkf_omega by default; redesign_v2 benchmarks vs vkf_w0.
PRIMARY_MODEL_ORDER = [
    "wsls",
    "rw",
    "rwf",
    "vkf_omega",
    "bayes_fixed_reward",
    "rw_asym",
]

# Sensitivity / historical models retained for audit, not default comparison.
SENSITIVITY_MODEL_ORDER = [
    "rwus_upe",
    "rw_side_bias",
    "rw_sticky",
    "rw_signed",
    "rw_signed_asym",
    "rwus",
    "vkf",
    "vkf_fixed_v0",
    "vkf_w0",
    "bayes",
    "bayes_constrained",
    "bayes_symmetric",
    "bayes_two_param",
]

CANDIDATE_ORDER = [
    "bayes_constrained",
    "bayes_symmetric",
    "bayes_two_param",
    "rwus_upe",
    "vkf_omega",
    "vkf_fixed_v0",
    "vkf_w0",
    "rw_signed_asym",
    "rw_asym",
]

# Equal-design redesign benchmark: comparison set + VKF head-to-head.
REDESIGN_BENCHMARK_ORDER = [
    "wsls",
    "rw",
    "rwf",
    "vkf_omega",
    "vkf_w0",
    "bayes_fixed_reward",
    "rw_asym",
]


def list_models() -> list[str]:
    return list(MODEL_ORDER)


def list_primary_models() -> list[str]:
    return list(PRIMARY_MODEL_ORDER)


def list_sensitivity_models() -> list[str]:
    return list(SENSITIVITY_MODEL_ORDER)


def list_candidates() -> list[str]:
    return list(CANDIDATE_ORDER)


def model_tier(name: str) -> str:
    """Return 'primary', 'sensitivity', 'candidate', or 'legacy'."""
    if name in PRIMARY_MODEL_ORDER:
        return "primary"
    if name in SENSITIVITY_MODEL_ORDER:
        return "sensitivity"
    if name in CANDIDATE_ORDER:
        return "candidate"
    if name in MODEL_ORDER:
        return "legacy"
    raise KeyError(name)


def get_model(name: str):
    if name not in ALL_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}; choose from {MODEL_ORDER + CANDIDATE_ORDER}"
        )
    return ALL_MODEL_REGISTRY[name]


def make_em_model(name: str, all_data=None):
    if EMModel is None:
        raise ImportError(
            "pyem is required for make_em_model; install via: "
            "pip install git+https://github.com/shawnrhoads/pyEM.git@c0d09807e26e7abaf3170366ca9f8a7e827ec063"
        )
    spec = get_model(name)
    names = MODEL_PARAM_NAMES[name]
    return EMModel(
        all_data=all_data,
        fit_func=spec.fit,
        param_names=names,
        param_xform=xforms_for(names),
        simulate_func=spec.sim,
    )
