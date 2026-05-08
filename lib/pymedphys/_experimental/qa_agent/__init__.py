# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Friston-style active-inference / POMDP agent for patient QA.

Hidden state is a factored "QA latent state" combining a discrete
fault mode (which failure is occurring) and continuous error
parameters (how much). Observations come from gamma analysis,
TRF/iCOM logs, DICOM RT records, and EPID/IGRT residuals. Actions
are QA decisions (approve, hold, remeasure, recalibrate, replan,
escalate). Action selection minimises expected free energy,
balancing pragmatic value (meeting tolerances) against epistemic
value (information gain about the latent state).

This package is a scaffold: interfaces are stable, numerics are
deliberately stubbed where appropriate. It is decision-support
only and must not drive clinical approval autonomously.
"""

from .agent import QAAgent, QAAgentConfig
from .belief import Belief, BeliefUpdater
from .fault_modes import FaultMode, LatentState
from .observation_model import Observation, ObservationModel
from .policy import Action, Policy

__all__ = [
    "Action",
    "Belief",
    "BeliefUpdater",
    "FaultMode",
    "LatentState",
    "Observation",
    "ObservationModel",
    "Policy",
    "QAAgent",
    "QAAgentConfig",
]
