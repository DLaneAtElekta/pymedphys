# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Observation model for the QA agent.

`Observation` is the per-fraction summary that downstream inference
consumes. `ObservationModel` is responsible for two things:

1. *Encoding* — turn raw inputs (DICOM, TRF/iCOM, gamma results,
   IGRT residuals) into a compact `Observation`.
2. *Likelihood* — score `log p(observation | phenotype state)` so the
   belief updater can do Bayesian inference.

Both responsibilities are stubbed. The expectation is that callers
plug in pymedphys' existing `_gamma`, `_trf`, `_icom`, `_dicom`
modules to build the encoder, and a coarse parametric likelihood
(e.g. Gaussian over gamma pass-rate, log-normal over MLC residuals)
to start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .phenotypes import PhenotypeState


@dataclass(frozen=True)
class Observation:
    """Compact per-fraction observation summary.

    Fields are optional so partial QA workflows (e.g. log-file-only)
    are first-class. None means "channel not available this fraction"
    and the likelihood marginalises it out.
    """

    gamma_pass_rate: float | None = None
    mean_mlc_residual_mm: float | None = None
    output_ratio: float | None = None
    setup_residual_mm: float | None = None
    gating_dropouts: int | None = None
    plan_hash_ok: bool | None = None


class ObservationModel:
    """Encodes raw inputs into `Observation` and scores likelihoods.

    Stubbed: replace `encode` and `log_likelihood` with real
    implementations backed by pymedphys' existing analysis modules.
    """

    def encode(self, raw: dict[str, Any]) -> Observation:
        """Convert a heterogeneous raw-input dict to an `Observation`.

        Expected keys (all optional):
            gamma_result, trf_path, icom_stream, rt_record, igrt,
            plan_hash. Implementations should be permissive about
            missing keys.
        """

        raise NotImplementedError("ObservationModel.encode is stubbed")

    def log_likelihood(
        self, observation: Observation, state: PhenotypeState
    ) -> float:
        """Return ``log p(observation | state)``.

        The intended starting form is a sum of independent per-channel
        log-likelihoods conditional on `state`, with channels that are
        ``None`` contributing zero.
        """

        raise NotImplementedError(
            "ObservationModel.log_likelihood is stubbed"
        )
