"""Parameter spaces: transforms, bounds, and hierarchical generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np
from pyem.utils.math import alpha2norm, beta2norm, norm2alpha, norm2beta
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm

from rlmodels_k99.numerics import softplus

GeneratorMode = Literal["hierarchical_latent", "stress_natural", "legacy_natural"]


def identity(x):
    return x


def softplus_xform(x):
    return float(softplus(x)) if np.ndim(x) == 0 else softplus(x)


def inv_softplus(y, eps: float = 1e-12):
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, eps)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


@dataclass(frozen=True)
class ParamDef:
    name: str
    xform: Callable
    inv_xform: Callable
    low: float
    high: float
    # Realistic hierarchical population on the *latent* (Gaussian) scale.
    latent_mu: float
    latent_sd: float
    # Optional natural-scale stress sampler (broad / misspecified).
    stress_sample: Callable[[int, np.random.Generator], np.ndarray] | None = None


def _sample_alpha(n: int, rng: np.random.Generator, lo=0.05, hi=0.95) -> np.ndarray:
    a_lo, a_hi = beta_dist.cdf([lo, hi], 1.1, 1.1)
    u = rng.random(n)
    return beta_dist.ppf(a_lo + u * (a_hi - a_lo), 1.1, 1.1)


def _sample_beta(n: int, rng: np.random.Generator, lo=0.75, hi=10.0) -> np.ndarray:
    a, b = (lo - 0.0) / 2.0, (hi - 0.0) / 2.0
    return truncnorm(a, b, loc=0.0, scale=2.0).rvs(size=n, random_state=rng)


def _sample_unit(n: int, rng: np.random.Generator, a=1.1, b=1.1) -> np.ndarray:
    return rng.beta(a, b, size=n)


def _sample_positive(n: int, rng: np.random.Generator, scale=2.0, hi=15.0) -> np.ndarray:
    x = np.abs(rng.normal(0.0, scale, size=n))
    return np.clip(x, 1e-3, hi)


PARAM_REGISTRY: dict[str, ParamDef] = {
    # beta ~ 3–5 typical; latent around beta2norm(4) ≈ -1.39 under norm2beta max=20.
    "beta": ParamDef(
        "beta",
        norm2beta,
        beta2norm,
        1e-5,
        20.0,
        latent_mu=float(beta2norm(4.0)),
        latent_sd=0.55,
        stress_sample=_sample_beta,
    ),
    "alpha": ParamDef(
        "alpha",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.35)),
        latent_sd=0.75,
        stress_sample=_sample_alpha,
    ),
    "epsilon": ParamDef(
        "epsilon",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.25)),
        latent_sd=0.6,
        stress_sample=lambda n, rng: _sample_unit(n, rng, 1.0, 1.5),
    ),
    "lambda": ParamDef(
        "lambda",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.3)),
        latent_sd=0.7,
        stress_sample=_sample_alpha,
    ),
    "v0": ParamDef(
        "v0",
        softplus_xform,
        inv_softplus,
        1e-5,
        np.inf,
        latent_mu=float(inv_softplus(1.0)),
        latent_sd=0.8,
        stress_sample=_sample_positive,
    ),
    "omega": ParamDef(
        "omega",
        softplus_xform,
        inv_softplus,
        1e-5,
        np.inf,
        latent_mu=float(inv_softplus(1.0)),
        latent_sd=0.8,
        stress_sample=_sample_positive,
    ),
    # p_switch typically small (~0.05–0.15).
    "p_switch": ParamDef(
        "p_switch",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.08)),
        latent_sd=0.55,
        stress_sample=lambda n, rng: _sample_unit(n, rng, 1.0, 4.0),
    ),
    # p_reward should be >0.5 for "correct arm more rewarding".
    "p_reward": ParamDef(
        "p_reward",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.8)),
        latent_sd=0.55,
        stress_sample=lambda n, rng: _sample_unit(n, rng, 1.1, 1.1),
    ),
    "alpha_pos": ParamDef(
        "alpha_pos",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.4)),
        latent_sd=0.75,
        stress_sample=_sample_alpha,
    ),
    "alpha_neg": ParamDef(
        "alpha_neg",
        norm2alpha,
        alpha2norm,
        0.0,
        1.0,
        latent_mu=float(alpha2norm(0.25)),
        latent_sd=0.75,
        stress_sample=_sample_alpha,
    ),
    # Side bias / stickiness live on the natural (logit) scale via identity.
    "bias": ParamDef(
        "bias",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=0.0,
        latent_sd=0.75,
        stress_sample=lambda n, rng: rng.normal(0.0, 1.0, size=n),
    ),
    "stickiness": ParamDef(
        "stickiness",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=0.5,
        latent_sd=0.75,
        stress_sample=lambda n, rng: rng.normal(0.5, 1.0, size=n),
    ),
    # Latent mid-point of signed learning rates: α± = σ(m ± d).
    "alpha_mean": ParamDef(
        "alpha_mean",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=float(alpha2norm(0.3)),
        latent_sd=0.75,
        stress_sample=lambda n, rng: rng.normal(float(alpha2norm(0.3)), 1.0, size=n),
    ),
    # Latent half-difference of signed learning rates.
    "alpha_asym": ParamDef(
        "alpha_asym",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=float(0.5 * (alpha2norm(0.4) - alpha2norm(0.25))),
        latent_sd=0.5,
        stress_sample=lambda n, rng: rng.normal(0.0, 0.75, size=n),
    ),
    # Free asymmetry for two-parameter rw_asym (identity on latent scale).
    "d": ParamDef(
        "d",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=0.0,
        latent_sd=0.5,
        stress_sample=lambda n, rng: rng.normal(0.0, 0.75, size=n),
    ),
    # Recognition SDT: sensitivity d' (mapped via norm2beta like beta).
    "dprime": ParamDef(
        "dprime",
        norm2beta,
        beta2norm,
        1e-5,
        20.0,
        latent_mu=float(beta2norm(1.5)),
        latent_sd=0.55,
        stress_sample=_sample_beta,
    ),
    # Response criterion (real-valued, identity latent).
    "criterion": ParamDef(
        "criterion",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=0.0,
        latent_sd=0.75,
        stress_sample=lambda n, rng: rng.normal(0.0, 0.75, size=n),
    ),
    # zRPE → d′ slope on studied trials (identity latent).
    "b_zrpe": ParamDef(
        "b_zrpe",
        identity,
        identity,
        -np.inf,
        np.inf,
        latent_mu=0.0,
        latent_sd=0.35,
        stress_sample=lambda n, rng: rng.normal(0.0, 0.35, size=n),
    ),
}

# Aliases for condition-indexed recognition SDT params (sdt_zrpe_day_2 / _3).
for _c in range(1, 4):
    PARAM_REGISTRY[f"dprime_{_c}"] = PARAM_REGISTRY["dprime"]
    PARAM_REGISTRY[f"criterion_{_c}"] = PARAM_REGISTRY["criterion"]
    PARAM_REGISTRY[f"b_zrpe_{_c}"] = PARAM_REGISTRY["b_zrpe"]


MODEL_PARAM_NAMES: dict[str, list[str]] = {
    "wsls": ["epsilon"],
    "rw": ["beta", "alpha"],
    "rwf": ["beta", "alpha"],
    "rw_signed": ["beta", "alpha_pos", "alpha_neg"],
    "rw_side_bias": ["beta", "alpha", "bias"],
    "rw_sticky": ["beta", "alpha", "stickiness"],
    "rwus": ["beta", "alpha"],
    "rwus_upe": ["beta", "alpha"],
    "vkf": ["beta", "lambda", "v0", "omega"],
    "vkf_omega": ["beta", "omega"],
    "vkf_fixed_v0": ["beta", "lambda", "omega"],
    "vkf_w0": ["beta", "omega"],
    "rw_signed_asym": ["beta", "alpha_mean", "alpha_asym"],
    "rw_asym": ["beta", "d"],
    "bayes": ["beta", "p_switch", "p_reward"],
    "bayes_fixed_reward": ["beta", "p_switch"],
    "bayes_constrained": ["beta", "p_switch", "p_reward"],
    "bayes_symmetric": ["beta", "p_switch", "p_reward"],
    "bayes_two_param": ["p_switch", "p_reward"],
    "sdt_zrpe_day_2": [
        "dprime_1", "criterion_1", "b_zrpe_1",
        "dprime_2", "criterion_2", "b_zrpe_2",
    ],
    "sdt_zrpe_day_3": [
        "dprime_1", "criterion_1", "b_zrpe_1",
        "dprime_2", "criterion_2", "b_zrpe_2",
        "dprime_3", "criterion_3", "b_zrpe_3",
    ],
}


@dataclass(frozen=True)
class ParameterSpace:
    names: tuple[str, ...]

    @property
    def defs(self) -> list[ParamDef]:
        return [PARAM_REGISTRY[n] for n in self.names]

    def xforms(self) -> list[Callable]:
        return [d.xform for d in self.defs]

    def inv_xforms(self) -> list[Callable]:
        return [d.inv_xform for d in self.defs]

    def to_natural(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=float)
        out = np.empty_like(latent, dtype=float)
        for j, d in enumerate(self.defs):
            out[..., j] = d.xform(latent[..., j])
        return out

    def to_latent(self, natural: np.ndarray) -> np.ndarray:
        natural = np.asarray(natural, dtype=float)
        out = np.empty_like(natural, dtype=float)
        for j, d in enumerate(self.defs):
            out[..., j] = d.inv_xform(natural[..., j])
        return out

    def prior_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, variance) on the latent scale for pyEM GaussianPrior."""
        mu = np.array([d.latent_mu for d in self.defs], dtype=float)
        var = np.array([d.latent_sd**2 for d in self.defs], dtype=float)
        return mu, var

    def sample(
        self,
        n_subjects: int,
        *,
        seed: int | None = None,
        mode: GeneratorMode = "hierarchical_latent",
        latent_mu: np.ndarray | None = None,
        latent_sd: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (natural_params, latent_params) with shape (n, nparams)."""
        rng = np.random.default_rng(seed)
        nparams = len(self.names)
        if mode == "hierarchical_latent":
            mu = np.asarray(latent_mu if latent_mu is not None else [d.latent_mu for d in self.defs], float)
            sd = np.asarray(latent_sd if latent_sd is not None else [d.latent_sd for d in self.defs], float)
            if mu.shape != (nparams,) or sd.shape != (nparams,):
                raise ValueError("latent_mu/sd must have shape (nparams,)")
            latent = rng.normal(mu, sd, size=(n_subjects, nparams))
            natural = self.to_natural(latent)
            # Clip to open bounds for softplus/logistic stability.
            for j, d in enumerate(self.defs):
                col = natural[:, j]
                if np.isfinite(d.low):
                    col = np.maximum(col, d.low + 1e-8)
                if np.isfinite(d.high):
                    col = np.minimum(col, d.high - 1e-8)
                natural[:, j] = col
            latent = self.to_latent(natural)
            return natural, latent
        if mode in ("stress_natural", "legacy_natural"):
            cols = []
            for d in self.defs:
                if d.stress_sample is None:
                    raise ValueError(f"No stress sampler for {d.name}")
                cols.append(d.stress_sample(n_subjects, rng))
            natural = np.column_stack(cols)
            return natural, self.to_latent(natural)
        raise ValueError(f"Unknown generator mode {mode!r}")


def space_for(names: Sequence[str]) -> ParameterSpace:
    return ParameterSpace(names=tuple(names))


def build_params(
    names: Sequence[str],
    n_subjects: int,
    seed: int | None = None,
    mode: GeneratorMode = "hierarchical_latent",
    model_name: str | None = None,
) -> tuple[list[str], list[Callable], np.ndarray]:
    """Backward-compatible builder returning (names, xforms, natural_params)."""
    space = space_for(names)
    natural, _ = space.sample(n_subjects, seed=seed, mode=mode)
    if model_name == "bayes_constrained":
        natural[:, 1] = np.clip(natural[:, 1], 1e-4, 0.49)
        natural[:, 2] = np.clip(natural[:, 2], 0.51, 1.0 - 1e-4)
    return list(names), space.xforms(), natural


def xforms_for(names: Sequence[str]) -> list[Callable]:
    return space_for(names).xforms()


def inv_xforms_for(names: Sequence[str]) -> list[Callable]:
    return space_for(names).inv_xforms()
