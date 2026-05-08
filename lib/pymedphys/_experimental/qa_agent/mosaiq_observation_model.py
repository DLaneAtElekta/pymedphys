# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Mosaiq-backed observation encoder for the QA agent.

Subclasses `ObservationModel` to fill the `setup_residual_mm`
channel from the IGRT offset associated with the requested
fraction. Other channels (gamma pass rate, MLC residual, output
ratio, gating dropouts, plan hash) are passed through from the
caller-provided ``raw`` dict so the agent can mix Mosaiq-derived
evidence with externally-measured channels.

Imports are kept opt-in: callers that don't need a Mosaiq path
should not import this module, so qa_agent's lightweight default
pulls no SQL dependencies.

The constructor accepts injected `sessions_fn` / `offsets_fn` to
keep the encoder unit-testable without an MSSQL connection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable, Tuple

from .observation_model import LikelihoodParams, Observation, ObservationModel

SessionsFn = Callable[[Any, int], Iterable[Tuple[int, datetime, datetime]]]
OffsetsFn = Callable[[Any, int], Iterable[Tuple[int, Any]]]


class MosaiqObservationModel(ObservationModel):
    """Observation encoder backed by Mosaiq Dose_Hst + Offset records.

    `encode` expects ``raw`` to contain ``sit_set_id`` (int) and
    optionally ``fraction`` (1-indexed; defaults to 1). Any of the
    standard `Observation` channel keys passed in ``raw`` win over
    the Mosaiq-derived value, which lets a pre-treatment gamma
    result, plan hash, etc. be supplied alongside the per-fraction
    Mosaiq pull.
    """

    def __init__(
        self,
        connection: Any,
        params: LikelihoodParams | None = None,
        sessions_fn: SessionsFn | None = None,
        offsets_fn: OffsetsFn | None = None,
    ) -> None:
        super().__init__(params=params)
        self._connection = connection

        if sessions_fn is None or offsets_fn is None:
            from pymedphys._mosaiq.sessions import (
                session_offsets_for_site,
                sessions_for_site,
            )

            self._sessions_fn = sessions_fn or sessions_for_site
            self._offsets_fn = offsets_fn or session_offsets_for_site
        else:
            self._sessions_fn = sessions_fn
            self._offsets_fn = offsets_fn

    def encode(self, raw: dict[str, Any]) -> Observation:
        base = super().encode(raw)

        sit_set_id = int(raw["sit_set_id"])
        fraction = int(raw.get("fraction", 1))

        setup_residual_mm = base.setup_residual_mm
        if setup_residual_mm is None:
            setup_residual_mm = self._lookup_setup_residual(sit_set_id, fraction)

        return Observation(
            gamma_pass_rate=base.gamma_pass_rate,
            mean_mlc_residual_mm=base.mean_mlc_residual_mm,
            output_ratio=base.output_ratio,
            setup_residual_mm=setup_residual_mm,
            gating_dropouts=base.gating_dropouts,
            plan_hash_ok=base.plan_hash_ok,
        )

    def _lookup_setup_residual(self, sit_set_id: int, fraction: int) -> float | None:
        for session_num, offset in self._offsets_fn(self._connection, sit_set_id):
            if session_num != fraction:
                continue
            if offset is None:
                return None
            sup, ant, lat = offset[0], offset[1], offset[2]
            return float((sup * sup + ant * ant + lat * lat) ** 0.5)
        return None
