# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""QA phenotypes: factored hidden state for the patient-QA agent.

The phenotype splits into a discrete failure mode and a small vector
of continuous error parameters. This factoring keeps inference
tractable while retaining enough dosimetric resolution to drive
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phenotype(str, Enum):
    """Discrete top-level QA phenotype.

    Membership is exhaustive over the failure modes the agent is
    expected to discriminate. Add new modes only when an observation
    channel exists that can update belief over them.
    """

    NOMINAL = "nominal"
    MLC_DEGRADED = "mlc_degraded"
    OUTPUT_DRIFT = "output_drift"
    SETUP_ERROR = "setup_error"
    GATING_FAULT = "gating_fault"
    COLLISION_RISK = "collision_risk"
    PLAN_CORRUPTION = "plan_corruption"


@dataclass(frozen=True)
class ContinuousErrors:
    """Low-dimensional continuous error parameters.

    Units are deliberately physical so observation models stay
    interpretable. All defaults correspond to a nominally-delivered
    fraction.
    """

    mlc_leaf_bias_mm: float = 0.0
    output_drift_pct: float = 0.0
    setup_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    setup_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class PhenotypeState:
    """Factored hidden state: discrete phenotype + continuous errors."""

    phenotype: Phenotype = Phenotype.NOMINAL
    errors: ContinuousErrors = field(default_factory=ContinuousErrors)
