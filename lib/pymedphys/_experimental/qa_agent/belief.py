# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Variational posterior over the factored QA phenotype.

The belief factorises as ``q(phenotype) * q(errors | phenotype)``.
The discrete factor is a categorical over `Phenotype`; the
continuous factor is left for a follow-up step (a Gaussian per
phenotype is the recommended starting form).

`BeliefUpdater` currently performs the closed-form Bayesian update
of the discrete factor only, using log-sum-exp for numerical
stability. The continuous-error sufficient statistics are passed
through unchanged. Callers that need richer schemes (particle
filter, structured VI) can subclass without changing the agent
plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

from .observation_model import Observation, ObservationModel
from .phenotypes import Phenotype, PhenotypeState

_LOG_FLOOR = -1e9


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

    phenotype_probs: dict[Phenotype, float] = field(default_factory=_uniform_prior)
    error_params: dict[Phenotype, object] = field(default_factory=dict)

    def map_phenotype(self) -> Phenotype:
        return max(self.phenotype_probs, key=self.phenotype_probs.get)

    def entropy_nats(self) -> float:
        """Shannon entropy of the discrete phenotype factor (nats)."""

        total = 0.0
        for p in self.phenotype_probs.values():
            if p > 0.0:
                total -= p * log(p)
        return total


class BeliefUpdater:
    """Single-step Bayesian update of `Belief` from an `Observation`.

    Implements the discrete-factor update only. For each phenotype:

        log_post(p) = log_prior(p) + log p(o | p)

    then normalised via log-sum-exp.
    """

    def __init__(self, observation_model: ObservationModel) -> None:
        self._observation_model = observation_model

    def update(self, prior: Belief, observation: Observation) -> Belief:
        log_post: dict[Phenotype, float] = {}
        for phenotype in Phenotype:
            prior_p = prior.phenotype_probs.get(phenotype, 0.0)
            log_prior = log(prior_p) if prior_p > 0.0 else _LOG_FLOOR
            log_lik = self._observation_model.log_likelihood(
                observation, PhenotypeState(phenotype=phenotype)
            )
            log_post[phenotype] = log_prior + log_lik

        max_log = max(log_post.values())
        unnormalised = {p: exp(lp - max_log) for p, lp in log_post.items()}
        z = sum(unnormalised.values())
        if z <= 0.0:
            # Degenerate case: fall back to prior to avoid NaNs.
            return Belief(
                phenotype_probs=dict(prior.phenotype_probs),
                error_params=dict(prior.error_params),
            )

        posterior = {p: v / z for p, v in unnormalised.items()}
        return Belief(
            phenotype_probs=posterior,
            error_params=dict(prior.error_params),
        )
