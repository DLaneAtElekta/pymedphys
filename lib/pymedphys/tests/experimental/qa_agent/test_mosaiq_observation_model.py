# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the Mosaiq-backed observation encoder.

The unit tests use injected fakes for every database lookup so
they run without an MSSQL connection. The integration test
(under the ``mosaiqdb`` marker) exercises the real query path
against the mock Mosaiq DB and is skipped unless the marker is
enabled.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pymedphys._experimental.qa_agent import (
    Action,
    BeliefUpdater,
    Phenotype,
    Policy,
    QAAgent,
)
from pymedphys._experimental.qa_agent.mosaiq_observation_model import (
    MosaiqObservationModel,
)


# ---------------------------------------------------------------------------
# Fakes (no DB)
# ---------------------------------------------------------------------------


def _fake_sessions(_connection, _sit_set_id):
    base = datetime(2026, 5, 1, 9, 0)
    yield (1, base, base + timedelta(minutes=20))
    yield (2, base + timedelta(days=1), base + timedelta(days=1, minutes=20))
    yield (3, base + timedelta(days=2), base + timedelta(days=2, minutes=20))


def _fake_offsets_with_setup_error(_connection, _sit_set_id):
    yield (1, [0.4, -0.2, 0.3])  # ~0.54 mm
    yield (2, [3.0, 4.0, 0.0])  # 5.0 mm — setup error
    yield (3, None)


_PLANNED_PER_FRACTION = 200.0
_DELIVERED_BY_FRACTION = {
    1: 200.0,  # ratio = 1.000
    2: 200.0,  # ratio = 1.000 even though setup is bad
    3: 204.0,  # ratio = 1.020 — output drift
}


def _fake_planned_dose(_connection, _sit_set_id):
    return _PLANNED_PER_FRACTION


def _fake_delivered_dose(_connection, _sit_set_id, start, end):
    for fraction, s, e in _fake_sessions(None, None):
        if s == start and e == end:
            return _DELIVERED_BY_FRACTION.get(fraction, 0.0)
    return 0.0


def _make_encoder(**overrides):
    kwargs = dict(
        connection=object(),
        sessions_fn=_fake_sessions,
        offsets_fn=_fake_offsets_with_setup_error,
        delivered_dose_fn=_fake_delivered_dose,
        planned_dose_fn=_fake_planned_dose,
    )
    kwargs.update(overrides)
    return MosaiqObservationModel(**kwargs)


# ---------------------------------------------------------------------------
# setup_residual_mm
# ---------------------------------------------------------------------------


def test_mosaiq_encoder_pulls_setup_residual_for_requested_fraction():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 2})
    assert obs.setup_residual_mm == pytest.approx(5.0, rel=1e-6)


def test_mosaiq_encoder_returns_none_when_offset_missing_for_fraction():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 3})
    assert obs.setup_residual_mm is None


def test_mosaiq_encoder_passes_through_external_channels():
    obs = _make_encoder().encode(
        {
            "sit_set_id": 1,
            "fraction": 1,
            "gamma_pass_rate": 99.0,
            "plan_hash_ok": True,
        }
    )
    assert obs.gamma_pass_rate == 99.0
    assert obs.plan_hash_ok is True
    assert obs.setup_residual_mm == pytest.approx(0.5385, rel=1e-3)


def test_caller_supplied_setup_residual_overrides_mosaiq_lookup():
    obs = _make_encoder().encode(
        {"sit_set_id": 1, "fraction": 2, "setup_residual_mm": 0.1}
    )
    assert obs.setup_residual_mm == 0.1


def test_unknown_fraction_returns_no_setup_residual():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 99})
    assert obs.setup_residual_mm is None


# ---------------------------------------------------------------------------
# output_ratio (delivered / planned per fraction)
# ---------------------------------------------------------------------------


def test_output_ratio_is_one_for_nominal_fraction():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 1})
    assert obs.output_ratio == pytest.approx(1.0, rel=1e-6)


def test_output_ratio_picks_up_drift_in_specific_fraction():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 3})
    assert obs.output_ratio == pytest.approx(1.02, rel=1e-6)


def test_output_ratio_is_none_when_planned_dose_missing():
    obs = _make_encoder(planned_dose_fn=lambda *_: 0.0).encode(
        {"sit_set_id": 1, "fraction": 1}
    )
    assert obs.output_ratio is None


def test_output_ratio_is_none_for_unknown_fraction():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 99})
    assert obs.output_ratio is None


def test_caller_supplied_output_ratio_overrides_mosaiq_lookup():
    obs = _make_encoder().encode({"sit_set_id": 1, "fraction": 3, "output_ratio": 1.0})
    assert obs.output_ratio == 1.0


# ---------------------------------------------------------------------------
# End-to-end with the agent
# ---------------------------------------------------------------------------


def test_agent_with_mosaiq_encoder_flags_setup_error_fraction():
    om = _make_encoder()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step(
        {
            "sit_set_id": 1,
            "fraction": 2,
            "gamma_pass_rate": 85.0,  # consistent with setup error
        }
    )
    assert result.posterior.map_phenotype() is Phenotype.SETUP_ERROR
    assert result.recommendation.action is not Action.APPROVE_FRACTION


def test_agent_with_dose_drift_flags_output_drift():
    om = _make_encoder()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step({"sit_set_id": 1, "fraction": 3})
    assert result.posterior.map_phenotype() is Phenotype.OUTPUT_DRIFT


# ---------------------------------------------------------------------------
# Integration against the mock Mosaiq DB
# ---------------------------------------------------------------------------


@pytest.mark.mosaiqdb
def test_mosaiq_encoder_against_mock_db():
    """Mirrors `tests/mosaiq/test_session_logic.py` setup."""

    from pymedphys._mosaiq.mock import generate, utilities

    generate.create_test_db()
    connection = utilities.connect()

    mock_patient_ident_df = generate.create_mock_patients()
    mock_site_df = generate.create_mock_treatment_sites(mock_patient_ident_df)
    mock_txfield_df = generate.create_mock_treatment_fields(mock_site_df)
    generate.create_mock_treatment_sessions(mock_site_df, mock_txfield_df)

    om = MosaiqObservationModel(connection=connection)
    obs = om.encode({"sit_set_id": 1, "fraction": 1})

    # Mock data writes a fixed offset of (-1, 0, 1) for every session.
    expected_magnitude = (1.0 + 0.0 + 1.0) ** 0.5
    assert obs.setup_residual_mm == pytest.approx(expected_magnitude, rel=1e-6)

    # The mock generator splits Site.Dose_Tx evenly across the site's fields
    # so a nominal fraction's delivered dose sums back to the prescription.
    assert obs.output_ratio == pytest.approx(1.0, rel=1e-6)
