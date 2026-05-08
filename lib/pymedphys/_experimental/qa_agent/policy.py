# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Action selection by expected-free-energy minimisation.

Expected free energy decomposes into two terms:

* **Pragmatic value**: how well predicted observations under an
  action match prior preferences `C(o)` (tolerance compliance,
  workflow cost).
* **Epistemic value**: expected information gain about the
  phenotype state — i.e. reduction in `Belief` entropy.

The agent picks the action minimising their sum. "Request
remeasurement" emerges naturally when the posterior is ambiguous
because it has high epistemic value despite weak pragmatic value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .belief import Belief


class Action(str, Enum):
    APPROVE_FRACTION = "approve_fraction"
    HOLD_FOR_REVIEW = "hold_for_review"
    REQUEST_REMEASUREMENT = "request_remeasurement"
    TRIGGER_RECALIBRATION = "trigger_recalibration"
    REPLAN = "replan"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class EFEDecomposition:
    """Per-action breakdown of expected free energy.

    Surfaced to the physicist alongside the recommendation so the
    *why* of an action is auditable.
    """

    action: Action
    pragmatic: float
    epistemic: float

    @property
    def total(self) -> float:
        return self.pragmatic + self.epistemic


class Policy:
    """EFE-based policy.

    Stubbed. The reference implementation should, for each action:

    1. Roll the belief forward one step under the transition model
       conditioned on that action.
    2. Sample / integrate predicted observations under that
       posterior.
    3. Score pragmatic value against `preferences` and epistemic
       value as belief-entropy reduction.

    The action with minimum total EFE is returned.
    """

    def __init__(
        self,
        preferences: dict[str, float] | None = None,
        actions: tuple[Action, ...] = tuple(Action),
    ) -> None:
        self._preferences = preferences or {}
        self._actions = actions

    def evaluate(self, belief: Belief) -> list[EFEDecomposition]:
        raise NotImplementedError("Policy.evaluate is stubbed")

    def select(self, belief: Belief) -> EFEDecomposition:
        evaluated = self.evaluate(belief)
        return min(evaluated, key=lambda d: d.total)
