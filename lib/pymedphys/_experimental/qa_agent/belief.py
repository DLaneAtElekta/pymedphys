# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Variational posterior over the factored QA latent state.

The belief factorises as ``q(fault_mode) * q(errors | fault_mode)``.
The discrete factor is a categorical over `FaultMode`; the
continuous factor is left for a follow-up step (a Gaussian per
fault mode is the recommended starting form).

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

from pymedphys._imports import scipy

from .fault_modes import FaultMode, LatentState
from .observation_model import Observation, ObservationModel

_LOG_FLOOR = -1e9


def _uniform_prior() -> dict[FaultMode, float]:
    n = len(FaultMode)
    return {p: 1.0 / n for p in FaultMode}


@dataclass
class Belief:
    """Factored posterior over latent state.

    `fault_mode_probs` is a categorical over discrete fault modes.
    `error_params` holds per-fault-mode continuous-error sufficient
    statistics; its concrete shape is left to the implementation
    that fills in the variational form.
    """

    fault_mode_probs: dict[FaultMode, float] = field(default_factory=_uniform_prior)
    error_params: dict[FaultMode, object] = field(default_factory=dict)

    def map_fault_mode(self) -> FaultMode:
        return max(self.fault_mode_probs, key=lambda k: self.fault_mode_probs[k])

    def entropy_nats(self) -> float:
        """Shannon entropy of the discrete fault-mode factor (nats)."""

        total = 0.0
        for p in self.fault_mode_probs.values():
            if p > 0.0:
                total -= p * log(p)
        return total


class BeliefUpdater:
    """Single-step Bayesian update of `Belief` from an `Observation`.

    Implements the discrete-factor update only. For each fault mode:

        log_post(p) = log_prior(p) + log p(o | p)

    then normalised via log-sum-exp.
    """

    def __init__(self, observation_model: ObservationModel) -> None:
        self._observation_model = observation_model

    def update(self, prior: Belief, observation: Observation) -> Belief:
        log_post: dict[FaultMode, float] = {}
        for fault_mode in FaultMode:
            prior_p = prior.fault_mode_probs.get(fault_mode, 0.0)
            log_prior = log(prior_p) if prior_p > 0.0 else _LOG_FLOOR
            log_lik = self._observation_model.log_likelihood(
                observation, LatentState(fault_mode=fault_mode)
            )
            log_post[fault_mode] = log_prior + log_lik

        log_z = float(scipy.special.logsumexp(list(log_post.values())))  # pylint: disable=no-member
        posterior = {p: exp(lp - log_z) for p, lp in log_post.items()}
        return Belief(
            fault_mode_probs=posterior,
            error_params=dict(prior.error_params),
        )
