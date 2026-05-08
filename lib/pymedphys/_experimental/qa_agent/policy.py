# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Action selection by expected-free-energy minimisation.

EFE for action ``a`` decomposes into:

* **Pragmatic value** — expected mismatch between predicted outcomes
  and the agent's prior preferences ``C(o)``. Encoded here as the
  expected per-fault-mode outcome cost of taking ``a``: approving
  while truly non-nominal is expensive; replanning while truly
  nominal is wasteful.
* **Epistemic value** — expected information gain about the
  fault mode. Encoded here as ``-info_factor[a] * H[q(s)]``: actions
  that re-observe the system (remeasurement, recalibration) reduce
  belief entropy, while approve/escalate do not.

Total EFE = pragmatic + epistemic; the policy picks the minimum.
"Request remeasurement" emerges naturally when the posterior is
ambiguous because its epistemic term dominates despite a non-zero
pragmatic cost.

This is the discrete-factor approximation. A principled
implementation rolls the belief forward through a transition model
conditioned on the action and integrates predicted observations;
that is the natural next extension and slots in behind the same
``Policy.evaluate`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .belief import Belief
from .fault_modes import FaultMode


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


DEFAULT_INFO_FACTOR: dict[Action, float] = {
    Action.APPROVE_FRACTION: 0.0,
    Action.HOLD_FOR_REVIEW: 0.3,
    Action.REQUEST_REMEASUREMENT: 1.0,
    Action.TRIGGER_RECALIBRATION: 0.6,
    Action.REPLAN: 0.4,
    Action.ESCALATE: 0.5,
}
"""Fraction of QA evidence each action effectively re-observes."""


# Per-action, per-fault-mode pragmatic cost. Approving a non-nominal fault mode
# is the dominant patient-safety cost; intervening when truly nominal is a
# smaller workflow cost. Tune against local clinical preferences.
_FLAT_HOLD_COST = dict.fromkeys(FaultMode, 1.0)
_FLAT_REMEASURE_COST = dict.fromkeys(FaultMode, 1.5)

DEFAULT_OUTCOME_COST: dict[Action, dict[FaultMode, float]] = {
    Action.APPROVE_FRACTION: {
        FaultMode.NOMINAL: 0.0,
        FaultMode.MLC_DEGRADED: 5.0,
        FaultMode.OUTPUT_DRIFT: 4.0,
        FaultMode.SETUP_ERROR: 6.0,
        FaultMode.GATING_FAULT: 7.0,
        FaultMode.COLLISION_RISK: 10.0,
        FaultMode.PLAN_CORRUPTION: 12.0,
    },
    Action.HOLD_FOR_REVIEW: _FLAT_HOLD_COST,
    Action.REQUEST_REMEASUREMENT: _FLAT_REMEASURE_COST,
    Action.TRIGGER_RECALIBRATION: {
        FaultMode.NOMINAL: 3.0,
        FaultMode.MLC_DEGRADED: 1.5,
        FaultMode.OUTPUT_DRIFT: 0.5,
        FaultMode.SETUP_ERROR: 3.0,
        FaultMode.GATING_FAULT: 2.0,
        FaultMode.COLLISION_RISK: 3.0,
        FaultMode.PLAN_CORRUPTION: 3.0,
    },
    Action.REPLAN: {
        FaultMode.NOMINAL: 5.0,
        FaultMode.MLC_DEGRADED: 3.0,
        FaultMode.OUTPUT_DRIFT: 4.0,
        FaultMode.SETUP_ERROR: 4.0,
        FaultMode.GATING_FAULT: 4.0,
        FaultMode.COLLISION_RISK: 3.0,
        FaultMode.PLAN_CORRUPTION: 1.0,
    },
    Action.ESCALATE: {
        FaultMode.NOMINAL: 2.0,
        FaultMode.MLC_DEGRADED: 1.5,
        FaultMode.OUTPUT_DRIFT: 1.5,
        FaultMode.SETUP_ERROR: 1.5,
        FaultMode.GATING_FAULT: 1.5,
        FaultMode.COLLISION_RISK: 1.0,
        FaultMode.PLAN_CORRUPTION: 1.0,
    },
}


class Policy:
    """EFE-based policy over the discrete fault-mode factor.

    Pragmatic term: ``E_q(s)[outcome_cost[a][s]]``.
    Epistemic term: ``-info_factor[a] * H[q(s)]`` (negative cost,
    so high-entropy beliefs prefer informative actions).
    """

    def __init__(
        self,
        info_factor: dict[Action, float] | None = None,
        outcome_cost: dict[Action, dict[FaultMode, float]] | None = None,
        actions: tuple[Action, ...] = tuple(Action),
    ) -> None:
        self._info_factor = (
            info_factor if info_factor is not None else DEFAULT_INFO_FACTOR
        )
        self._outcome_cost = (
            outcome_cost if outcome_cost is not None else DEFAULT_OUTCOME_COST
        )
        self._actions = actions

    def evaluate(self, belief: Belief) -> list[EFEDecomposition]:
        entropy = belief.entropy_nats()
        probs = [(p, belief.fault_mode_probs.get(p, 0.0)) for p in FaultMode]
        results: list[EFEDecomposition] = []
        for action in self._actions:
            cost_table = self._outcome_cost[action]
            pragmatic = sum(prob * cost_table[p] for p, prob in probs)
            epistemic = -self._info_factor[action] * entropy
            results.append(
                EFEDecomposition(
                    action=action, pragmatic=pragmatic, epistemic=epistemic
                )
            )
        return results

    def select(self, belief: Belief) -> EFEDecomposition:
        return min(self.evaluate(belief), key=lambda d: d.total)
