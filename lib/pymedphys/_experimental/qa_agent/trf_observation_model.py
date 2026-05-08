# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""TRF-backed observation encoder for the QA agent.

Reads an Elekta Agility TRF log and fills the
``mean_mlc_residual_mm`` channel from the per-leaf, per-timestep
``Positional Error (mm)`` columns the linac itself records.

A TRF row contains both the expected and actual leaf positions; the
``Positional Error`` columns are already (actual - expected), so the
encoder simply averages absolute values across all leaves and all
timesteps. This sidesteps the harder problem of aligning a planned
reference to a delivered trajectory and is a faithful proxy for
"how well did the MLC track the plan during this fraction".

Other channels are passed through from the caller-provided ``raw``
dict so the agent can mix TRF-derived evidence with externally-
measured channels (gamma, plan hash) and Mosaiq-derived ones
(setup residual, output ratio).

Imports are kept opt-in: callers that don't need a TRF path should
not import this module.

The constructor accepts an injected ``read_trf_fn`` so the encoder
is unit-testable without a real TRF file.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Tuple

from .observation_model import LikelihoodParams, Observation, ObservationModel

ReadTrfFn = Callable[[Any], Tuple[Any, Any]]

_LEAF_ERROR_RE = re.compile(r"^Y[12] Leaf \d+/Positional Error \(mm\)$")


def _default_read_trf_fn(trf: Any) -> Tuple[Any, Any]:
    from pymedphys._trf.decode.trf2pandas import trf2pandas

    return trf2pandas(trf)


class TrfObservationModel(ObservationModel):
    """Observation encoder backed by Elekta TRF logs.

    `encode` expects ``raw`` to contain ``trf_path`` (a filesystem
    path or file-like object accepted by
    :func:`pymedphys._trf.decode.trf2pandas.trf2pandas`). Any of the
    standard `Observation` channel keys passed in ``raw`` win over
    the TRF-derived value.
    """

    def __init__(
        self,
        params: LikelihoodParams | None = None,
        read_trf_fn: ReadTrfFn | None = None,
    ) -> None:
        super().__init__(params=params)
        self._read_trf_fn = read_trf_fn or _default_read_trf_fn

    def encode(self, raw: dict[str, Any]) -> Observation:
        base = super().encode(raw)

        mean_mlc_residual_mm = base.mean_mlc_residual_mm
        trf_path = raw.get("trf_path")
        if mean_mlc_residual_mm is None and trf_path is not None:
            mean_mlc_residual_mm = self._compute_mlc_residual(trf_path)

        return Observation(
            gamma_pass_rate=base.gamma_pass_rate,
            mean_mlc_residual_mm=mean_mlc_residual_mm,
            output_ratio=base.output_ratio,
            setup_residual_mm=base.setup_residual_mm,
            gating_dropouts=base.gating_dropouts,
            plan_hash_ok=base.plan_hash_ok,
        )

    def _compute_mlc_residual(self, trf_path: Any) -> float | None:
        _, table = self._read_trf_fn(trf_path)
        leaf_columns = [c for c in table.columns if _LEAF_ERROR_RE.match(str(c))]
        if not leaf_columns:
            return None
        residuals = table[leaf_columns].abs()
        mean = float(residuals.to_numpy().mean())
        return mean
