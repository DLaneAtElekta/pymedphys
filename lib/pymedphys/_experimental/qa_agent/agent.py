# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-fraction orchestrator for the patient-QA agent.

Glues together :class:`ObservationModel`, :class:`BeliefUpdater`,
and :class:`Policy` into a single ``observe -> infer -> act`` loop.
The agent maintains a `Belief` across fractions; each call to
:meth:`QAAgent.step` consumes one fraction's raw inputs and returns
a recommendation plus the updated belief.

Decision-support only. The recommendation is advisory; clinical
approval remains with the physicist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .belief import Belief, BeliefUpdater
from .observation_model import Observation, ObservationModel
from .policy import EFEDecomposition, Policy


@dataclass
class QAAgentConfig:
    """Configuration knobs surfaced to the caller."""

    preferences: dict[str, float] = field(default_factory=dict)


@dataclass
class QAStepResult:
    """Output of one fraction's `observe -> infer -> act` cycle."""

    observation: Observation
    posterior: Belief
    recommendation: EFEDecomposition
    alternatives: list[EFEDecomposition]


class QAAgent:
    """Active-inference agent for per-fraction patient QA."""

    def __init__(
        self,
        observation_model: ObservationModel,
        belief_updater: BeliefUpdater,
        policy: Policy,
        config: QAAgentConfig | None = None,
        prior: Belief | None = None,
    ) -> None:
        self._observation_model = observation_model
        self._belief_updater = belief_updater
        self._policy = policy
        self._config = config or QAAgentConfig()
        self._belief = prior or Belief()

    @property
    def belief(self) -> Belief:
        return self._belief

    def step(self, raw_inputs: dict[str, Any]) -> QAStepResult:
        observation = self._observation_model.encode(raw_inputs)
        self._belief = self._belief_updater.update(self._belief, observation)
        evaluated = self._policy.evaluate(self._belief)
        recommendation = min(evaluated, key=lambda d: d.total)
        return QAStepResult(
            observation=observation,
            posterior=self._belief,
            recommendation=recommendation,
            alternatives=evaluated,
        )
