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
2. *Likelihood* — score ``log p(observation | fault_mode)`` as a sum
   of independent per-channel terms (Gaussian for continuous
   channels, Bernoulli for the plan-hash check). Channels left as
   ``None`` contribute zero, so partial QA workflows are
   first-class.

Conditioning is on the discrete fault mode only; continuous error
parameters in `LatentState` are ignored at this stage and will
be wired in once the variational form for the continuous factor is
chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Any

from .fault_modes import FaultMode, LatentState

_LOG_2PI = log(2.0 * pi)

_GAUSSIAN_CHANNELS: tuple[str, ...] = (
    "gamma_pass_rate",
    "mean_mlc_residual_mm",
    "output_ratio",
    "setup_residual_mm",
    "gating_dropouts",
)


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
    """Per-fault-mode likelihood parameters for each channel.

    Each inner dict must be exhaustive over `FaultMode`. Defaults
    are illustrative starting points calibrated against typical
    clinical tolerances (TG-218-style); real deployments should
    refit them from local data.
    """

    gamma_pass_rate: dict[FaultMode, GaussianParams]
    mean_mlc_residual_mm: dict[FaultMode, GaussianParams]
    output_ratio: dict[FaultMode, GaussianParams]
    setup_residual_mm: dict[FaultMode, GaussianParams]
    gating_dropouts: dict[FaultMode, GaussianParams]
    plan_hash_ok: dict[FaultMode, BernoulliParams]


DEFAULT_LIKELIHOOD_PARAMS = LikelihoodParams(
    gamma_pass_rate={
        FaultMode.NOMINAL: GaussianParams(98.0, 1.5),
        FaultMode.MLC_DEGRADED: GaussianParams(88.0, 4.0),
        FaultMode.OUTPUT_DRIFT: GaussianParams(92.0, 3.0),
        FaultMode.SETUP_ERROR: GaussianParams(85.0, 5.0),
        FaultMode.GATING_FAULT: GaussianParams(80.0, 6.0),
        FaultMode.COLLISION_RISK: GaussianParams(95.0, 3.0),
        FaultMode.PLAN_CORRUPTION: GaussianParams(70.0, 10.0),
    },
    mean_mlc_residual_mm={
        FaultMode.NOMINAL: GaussianParams(0.2, 0.1),
        FaultMode.MLC_DEGRADED: GaussianParams(1.5, 0.5),
        FaultMode.OUTPUT_DRIFT: GaussianParams(0.2, 0.1),
        FaultMode.SETUP_ERROR: GaussianParams(0.2, 0.1),
        FaultMode.GATING_FAULT: GaussianParams(0.3, 0.2),
        FaultMode.COLLISION_RISK: GaussianParams(0.2, 0.1),
        FaultMode.PLAN_CORRUPTION: GaussianParams(0.5, 0.3),
    },
    output_ratio={
        FaultMode.NOMINAL: GaussianParams(1.000, 0.005),
        FaultMode.MLC_DEGRADED: GaussianParams(1.000, 0.005),
        FaultMode.OUTPUT_DRIFT: GaussianParams(1.020, 0.010),
        FaultMode.SETUP_ERROR: GaussianParams(1.000, 0.005),
        FaultMode.GATING_FAULT: GaussianParams(1.000, 0.010),
        FaultMode.COLLISION_RISK: GaussianParams(1.000, 0.005),
        FaultMode.PLAN_CORRUPTION: GaussianParams(1.000, 0.020),
    },
    setup_residual_mm={
        FaultMode.NOMINAL: GaussianParams(0.5, 0.3),
        FaultMode.MLC_DEGRADED: GaussianParams(0.5, 0.3),
        FaultMode.OUTPUT_DRIFT: GaussianParams(0.5, 0.3),
        FaultMode.SETUP_ERROR: GaussianParams(5.0, 2.0),
        FaultMode.GATING_FAULT: GaussianParams(1.0, 0.5),
        FaultMode.COLLISION_RISK: GaussianParams(1.0, 0.5),
        FaultMode.PLAN_CORRUPTION: GaussianParams(0.5, 0.3),
    },
    gating_dropouts={
        FaultMode.NOMINAL: GaussianParams(0.0, 0.5),
        FaultMode.MLC_DEGRADED: GaussianParams(0.0, 0.5),
        FaultMode.OUTPUT_DRIFT: GaussianParams(0.0, 0.5),
        FaultMode.SETUP_ERROR: GaussianParams(0.0, 0.5),
        FaultMode.GATING_FAULT: GaussianParams(8.0, 4.0),
        FaultMode.COLLISION_RISK: GaussianParams(0.0, 0.5),
        FaultMode.PLAN_CORRUPTION: GaussianParams(0.0, 0.5),
    },
    plan_hash_ok={
        FaultMode.NOMINAL: BernoulliParams(0.999),
        FaultMode.MLC_DEGRADED: BernoulliParams(0.99),
        FaultMode.OUTPUT_DRIFT: BernoulliParams(0.99),
        FaultMode.SETUP_ERROR: BernoulliParams(0.99),
        FaultMode.GATING_FAULT: BernoulliParams(0.99),
        FaultMode.COLLISION_RISK: BernoulliParams(0.99),
        FaultMode.PLAN_CORRUPTION: BernoulliParams(0.05),
    },
)


def _gaussian_log(x: float, params: GaussianParams) -> float:
    z = (x - params.mean) / params.std
    return -0.5 * z * z - log(params.std) - 0.5 * _LOG_2PI


def _bernoulli_log(observed: bool, p_true: float) -> float:
    # Clip to avoid log(0) when a fault mode assigns probability 0/1.
    p = min(max(p_true, 1e-9), 1.0 - 1e-9)
    return log(p) if observed else log(1.0 - p)


class ObservationModel:
    """Encodes raw inputs into `Observation` and scores likelihoods."""

    def __init__(self, params: LikelihoodParams | None = None) -> None:
        self._params = params if params is not None else DEFAULT_LIKELIHOOD_PARAMS

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

    def log_likelihood(self, observation: Observation, state: LatentState) -> float:
        """``log p(observation | fault_mode)`` as a sum over channels.

        Continuous error parameters in `state` are not yet used.
        """

        fm = state.fault_mode
        total = 0.0

        for channel in _GAUSSIAN_CHANNELS:
            value = getattr(observation, channel)
            if value is not None:
                total += _gaussian_log(float(value), getattr(self._params, channel)[fm])

        if observation.plan_hash_ok is not None:
            total += _bernoulli_log(
                observation.plan_hash_ok, self._params.plan_hash_ok[fm].p_true
            )

        return total
