# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Observation model for the QA agent.

`Observation` is the per-fraction summary that downstream inference
consumes. `ObservationModel` is responsible for two things:

1. *Encoding* — turn raw inputs into a compact `Observation`. The
   default implementation is intentionally a permissive
   dict-to-dataclass mapping; richer encoders should subclass and
   plug in pymedphys' `_gamma`, `_trf`, `_icom`, `_dicom` modules.
2. *Likelihood* — score ``log p(observation | phenotype)`` as a sum
   of independent per-channel terms (Gaussian for continuous
   channels, Bernoulli for the plan-hash check). Channels left as
   ``None`` contribute zero, so partial QA workflows are
   first-class.

Conditioning is on the discrete phenotype only; continuous error
parameters in `PhenotypeState` are ignored at this stage and will
be wired in once the variational form for the continuous factor is
chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Any

from .phenotypes import Phenotype, PhenotypeState

_LOG_2PI = log(2.0 * pi)


@dataclass(frozen=True)
class Observation:
    """Compact per-fraction observation summary.

    Fields are optional so partial QA workflows (e.g. log-file-only)
    are first-class. ``None`` means "channel not available this
    fraction" and is marginalised out of the likelihood.
    """

    gamma_pass_rate: float | None = None
    mean_mlc_residual_mm: float | None = None
    output_ratio: float | None = None
    setup_residual_mm: float | None = None
    gating_dropouts: int | None = None
    plan_hash_ok: bool | None = None


@dataclass(frozen=True)
class GaussianParams:
    mean: float
    std: float


@dataclass(frozen=True)
class BernoulliParams:
    p_true: float


@dataclass(frozen=True)
class LikelihoodParams:
    """Per-phenotype likelihood parameters for each channel.

    Each inner dict must be exhaustive over `Phenotype`. Defaults
    are illustrative starting points calibrated against typical
    clinical tolerances (TG-218-style); real deployments should
    refit them from local data.
    """

    gamma_pass_rate: dict[Phenotype, GaussianParams]
    mean_mlc_residual_mm: dict[Phenotype, GaussianParams]
    output_ratio: dict[Phenotype, GaussianParams]
    setup_residual_mm: dict[Phenotype, GaussianParams]
    gating_dropouts: dict[Phenotype, GaussianParams]
    plan_hash_ok: dict[Phenotype, BernoulliParams]


def default_likelihood_params() -> LikelihoodParams:
    """Illustrative defaults; replace with locally-fitted values."""

    return LikelihoodParams(
        gamma_pass_rate={
            Phenotype.NOMINAL: GaussianParams(98.0, 1.5),
            Phenotype.MLC_DEGRADED: GaussianParams(88.0, 4.0),
            Phenotype.OUTPUT_DRIFT: GaussianParams(92.0, 3.0),
            Phenotype.SETUP_ERROR: GaussianParams(85.0, 5.0),
            Phenotype.GATING_FAULT: GaussianParams(80.0, 6.0),
            Phenotype.COLLISION_RISK: GaussianParams(95.0, 3.0),
            Phenotype.PLAN_CORRUPTION: GaussianParams(70.0, 10.0),
        },
        mean_mlc_residual_mm={
            Phenotype.NOMINAL: GaussianParams(0.2, 0.1),
            Phenotype.MLC_DEGRADED: GaussianParams(1.5, 0.5),
            Phenotype.OUTPUT_DRIFT: GaussianParams(0.2, 0.1),
            Phenotype.SETUP_ERROR: GaussianParams(0.2, 0.1),
            Phenotype.GATING_FAULT: GaussianParams(0.3, 0.2),
            Phenotype.COLLISION_RISK: GaussianParams(0.2, 0.1),
            Phenotype.PLAN_CORRUPTION: GaussianParams(0.5, 0.3),
        },
        output_ratio={
            Phenotype.NOMINAL: GaussianParams(1.000, 0.005),
            Phenotype.MLC_DEGRADED: GaussianParams(1.000, 0.005),
            Phenotype.OUTPUT_DRIFT: GaussianParams(1.020, 0.010),
            Phenotype.SETUP_ERROR: GaussianParams(1.000, 0.005),
            Phenotype.GATING_FAULT: GaussianParams(1.000, 0.010),
            Phenotype.COLLISION_RISK: GaussianParams(1.000, 0.005),
            Phenotype.PLAN_CORRUPTION: GaussianParams(1.000, 0.020),
        },
        setup_residual_mm={
            Phenotype.NOMINAL: GaussianParams(0.5, 0.3),
            Phenotype.MLC_DEGRADED: GaussianParams(0.5, 0.3),
            Phenotype.OUTPUT_DRIFT: GaussianParams(0.5, 0.3),
            Phenotype.SETUP_ERROR: GaussianParams(5.0, 2.0),
            Phenotype.GATING_FAULT: GaussianParams(1.0, 0.5),
            Phenotype.COLLISION_RISK: GaussianParams(1.0, 0.5),
            Phenotype.PLAN_CORRUPTION: GaussianParams(0.5, 0.3),
        },
        gating_dropouts={
            Phenotype.NOMINAL: GaussianParams(0.0, 0.5),
            Phenotype.MLC_DEGRADED: GaussianParams(0.0, 0.5),
            Phenotype.OUTPUT_DRIFT: GaussianParams(0.0, 0.5),
            Phenotype.SETUP_ERROR: GaussianParams(0.0, 0.5),
            Phenotype.GATING_FAULT: GaussianParams(8.0, 4.0),
            Phenotype.COLLISION_RISK: GaussianParams(0.0, 0.5),
            Phenotype.PLAN_CORRUPTION: GaussianParams(0.0, 0.5),
        },
        plan_hash_ok={
            Phenotype.NOMINAL: BernoulliParams(0.999),
            Phenotype.MLC_DEGRADED: BernoulliParams(0.99),
            Phenotype.OUTPUT_DRIFT: BernoulliParams(0.99),
            Phenotype.SETUP_ERROR: BernoulliParams(0.99),
            Phenotype.GATING_FAULT: BernoulliParams(0.99),
            Phenotype.COLLISION_RISK: BernoulliParams(0.99),
            Phenotype.PLAN_CORRUPTION: BernoulliParams(0.05),
        },
    )


def _gaussian_log(x: float, params: GaussianParams) -> float:
    z = (x - params.mean) / params.std
    return -0.5 * z * z - log(params.std) - 0.5 * _LOG_2PI


def _bernoulli_log(observed: bool, p_true: float) -> float:
    # Clip to avoid log(0) when a phenotype assigns probability 0/1.
    p = min(max(p_true, 1e-9), 1.0 - 1e-9)
    return log(p) if observed else log(1.0 - p)


class ObservationModel:
    """Encodes raw inputs into `Observation` and scores likelihoods."""

    def __init__(self, params: LikelihoodParams | None = None) -> None:
        self._params = params or default_likelihood_params()

    @property
    def params(self) -> LikelihoodParams:
        return self._params

    def encode(self, raw: dict[str, Any]) -> Observation:
        """Permissive dict-to-`Observation` mapping.

        Recognised keys mirror `Observation` fields. Unknown keys are
        ignored. Subclass to plug in real encoders backed by
        pymedphys' `_gamma`, `_trf`, `_icom`, `_dicom` modules.
        """

        return Observation(
            gamma_pass_rate=raw.get("gamma_pass_rate"),
            mean_mlc_residual_mm=raw.get("mean_mlc_residual_mm"),
            output_ratio=raw.get("output_ratio"),
            setup_residual_mm=raw.get("setup_residual_mm"),
            gating_dropouts=raw.get("gating_dropouts"),
            plan_hash_ok=raw.get("plan_hash_ok"),
        )

    def log_likelihood(self, observation: Observation, state: PhenotypeState) -> float:
        """``log p(observation | phenotype)`` as a sum over channels.

        Continuous error parameters in `state` are not yet used.
        """

        p = state.phenotype
        total = 0.0

        if observation.gamma_pass_rate is not None:
            total += _gaussian_log(
                observation.gamma_pass_rate, self._params.gamma_pass_rate[p]
            )
        if observation.mean_mlc_residual_mm is not None:
            total += _gaussian_log(
                observation.mean_mlc_residual_mm,
                self._params.mean_mlc_residual_mm[p],
            )
        if observation.output_ratio is not None:
            total += _gaussian_log(
                observation.output_ratio, self._params.output_ratio[p]
            )
        if observation.setup_residual_mm is not None:
            total += _gaussian_log(
                observation.setup_residual_mm,
                self._params.setup_residual_mm[p],
            )
        if observation.gating_dropouts is not None:
            total += _gaussian_log(
                float(observation.gating_dropouts),
                self._params.gating_dropouts[p],
            )
        if observation.plan_hash_ok is not None:
            total += _bernoulli_log(
                observation.plan_hash_ok,
                self._params.plan_hash_ok[p].p_true,
            )

        return total
