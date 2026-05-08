# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Variational posterior over the factored QA phenotype.

The belief factorises as ``q(phenotype) * q(errors | phenotype)``.
The discrete factor is a categorical distribution over
:class:`~pymedphys._experimental.qa_agent.phenotypes.Phenotype`. The
continuous factor is intentionally not committed to here — a Gaussian
per phenotype is the recommended starting point.

Updates implement a single mean-field step against the observation
likelihood from `ObservationModel`. This is sufficient for
slowly-evolving QA state across fractions; richer schemes
(particle filter, structured VI) can be slotted in behind the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .observation_model import Observation, ObservationModel
from .phenotypes import Phenotype


def _uniform_prior() -> dict[Phenotype, float]:
    n = len(Phenotype)
    return {p: 1.0 / n for p in Phenotype}


@dataclass
class Belief:
    """Factored posterior over phenotype state.

    `phenotype_probs` is a categorical over discrete phenotypes.
    `error_params` holds per-phenotype continuous-error sufficient
    statistics; its concrete shape is left to the implementation
    that fills in the variational form.
    """

    phenotype_probs: dict[Phenotype, float] = field(
        default_factory=_uniform_prior
    )
    error_params: dict[Phenotype, object] = field(default_factory=dict)

    def map_phenotype(self) -> Phenotype:
        return max(self.phenotype_probs, key=self.phenotype_probs.get)

    def entropy_nats(self) -> float:
        """Shannon entropy of the discrete phenotype factor (nats)."""

        from math import log

        total = 0.0
        for p in self.phenotype_probs.values():
            if p > 0.0:
                total -= p * log(p)
        return total


class BeliefUpdater:
    """Single-step Bayesian update of `Belief` from an `Observation`.

    Stubbed. The reference implementation should:
      1. For each phenotype, compute ``log p(o | s) + log p(s)``
         using `ObservationModel.log_likelihood`.
      2. Normalise to obtain the posterior over phenotypes.
      3. Update per-phenotype continuous-error sufficient statistics
         (e.g. Kalman update for a Gaussian factor).
    """

    def __init__(self, observation_model: ObservationModel) -> None:
        self._observation_model = observation_model

    def update(self, prior: Belief, observation: Observation) -> Belief:
        raise NotImplementedError("BeliefUpdater.update is stubbed")
