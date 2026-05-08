# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Mosaiq-backed observation encoder for the QA agent.

Subclasses `ObservationModel` to fill QA channels from Mosaiq:

* ``setup_residual_mm`` — Euclidean magnitude of the IGRT offset
  vector associated with the requested fraction (from ``Offset``
  records via :func:`session_offsets_for_site`).
* ``output_ratio`` — sum of ``Dose_Hst.Dose_Tx_Act`` over the
  fraction's session window divided by the prescribed per-fraction
  dose ``Site.Dose_Tx``.

Other channels (gamma pass rate, MLC residual, gating dropouts,
plan hash) are passed through from the caller-provided ``raw``
dict so the agent can mix Mosaiq-derived evidence with
externally-measured channels.

Imports are kept opt-in: callers that don't need a Mosaiq path
should not import this module, so qa_agent's lightweight default
pulls no SQL dependencies.

The constructor accepts injected callables for every database
lookup so the encoder is unit-testable without an MSSQL
connection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Iterable, Tuple

from .observation_model import LikelihoodParams, Observation, ObservationModel

SessionsFn = Callable[[Any, int], Iterable[Tuple[int, datetime, datetime]]]
OffsetsFn = Callable[[Any, int], Iterable[Tuple[int, Any]]]
DeliveredDoseFn = Callable[[Any, int, datetime, datetime], float]
PlannedDoseFn = Callable[[Any, int], float]


def _default_delivered_dose_fn(
    connection: Any, sit_set_id: int, start: datetime, end: datetime
) -> float:
    """Sum ``Dose_Hst.Dose_Tx_Act`` for the given session window."""

    from pymedphys._mosaiq import api

    rows = api.execute(
        connection,
        """
        SELECT
            COALESCE(SUM(Dose_Hst.Dose_Tx_Act), 0)
        FROM Dose_Hst
        INNER JOIN
            Site ON Site.SIT_ID = Dose_Hst.SIT_ID
        WHERE
            Site.SIT_SET_ID = %(sit_set_id)s
            AND Dose_Hst.Tx_DtTm BETWEEN %(start)s AND %(end)s
        """,
        {
            "sit_set_id": sit_set_id,
            "start": start,
            "end": end,
        },
    )
    return float(rows[0][0]) if rows else 0.0


def _default_planned_dose_fn(connection: Any, sit_set_id: int) -> float:
    """Read the prescribed per-fraction dose ``Site.Dose_Tx`` (cGy)."""

    from pymedphys._mosaiq import api

    rows = api.execute(
        connection,
        """
        SELECT
            Dose_Tx
        FROM Site
        WHERE
            SIT_SET_ID = %(sit_set_id)s
        """,
        {"sit_set_id": sit_set_id},
    )
    if not rows:
        return 0.0
    return float(rows[0][0])


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
        delivered_dose_fn: DeliveredDoseFn | None = None,
        planned_dose_fn: PlannedDoseFn | None = None,
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

        self._delivered_dose_fn = delivered_dose_fn or _default_delivered_dose_fn
        self._planned_dose_fn = planned_dose_fn or _default_planned_dose_fn

    def encode(self, raw: dict[str, Any]) -> Observation:
        base = super().encode(raw)

        sit_set_id = int(raw["sit_set_id"])
        fraction = int(raw.get("fraction", 1))

        setup_residual_mm = base.setup_residual_mm
        if setup_residual_mm is None:
            setup_residual_mm = self._lookup_setup_residual(sit_set_id, fraction)

        output_ratio = base.output_ratio
        if output_ratio is None:
            output_ratio = self._lookup_output_ratio(sit_set_id, fraction)

        return Observation(
            gamma_pass_rate=base.gamma_pass_rate,
            mean_mlc_residual_mm=base.mean_mlc_residual_mm,
            output_ratio=output_ratio,
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

    def _lookup_output_ratio(self, sit_set_id: int, fraction: int) -> float | None:
        window: tuple[datetime, datetime] | None = None
        for session_num, start, end in self._sessions_fn(self._connection, sit_set_id):
            if session_num == fraction:
                window = (start, end)
                break
        if window is None:
            return None

        planned = self._planned_dose_fn(self._connection, sit_set_id)
        if planned <= 0.0:
            return None

        delivered = self._delivered_dose_fn(
            self._connection, sit_set_id, window[0], window[1]
        )
        return delivered / planned
