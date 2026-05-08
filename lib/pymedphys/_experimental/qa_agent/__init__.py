# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Experimental active-inference / POMDP agent for patient QA.

Hidden state is a factored "QA phenotype" (discrete failure mode plus
low-dimensional continuous error parameters). Observations come from
gamma analysis, TRF/iCOM logs, DICOM RT records, and EPID/IGRT
residuals. Actions are QA decisions (approve, hold, remeasure,
recalibrate, replan, escalate). Action selection minimises expected
free energy in the Friston sense, balancing pragmatic value (meeting
tolerances) against epistemic value (information gain about the
phenotype).

This package is a scaffold: interfaces are stable, numerics are
deliberately stubbed. It is decision-support only and must not drive
clinical approval autonomously.
"""

from .agent import QAAgent, QAAgentConfig
from .belief import Belief, BeliefUpdater
from .observation_model import Observation, ObservationModel
from .phenotypes import Phenotype, PhenotypeState
from .policy import Action, Policy

__all__ = [
    "Action",
    "Belief",
    "BeliefUpdater",
    "Observation",
    "ObservationModel",
    "Phenotype",
    "PhenotypeState",
    "Policy",
    "QAAgent",
    "QAAgentConfig",
]
