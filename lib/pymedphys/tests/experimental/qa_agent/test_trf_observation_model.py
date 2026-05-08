# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the TRF-backed observation encoder.

The encoder is exercised against synthetic TRF table DataFrames
injected via ``read_trf_fn`` so the tests run without a real TRF
file or any linac-specific fixtures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymedphys._experimental.qa_agent import (
    Action,
    BeliefUpdater,
    FaultMode,
    Policy,
    QAAgent,
)
from pymedphys._experimental.qa_agent.trf_observation_model import (
    TrfObservationModel,
)


def _make_table(per_leaf_errors: dict[str, list[float]]) -> pd.DataFrame:
    """Synthesize a TRF-table-shaped DataFrame.

    ``per_leaf_errors`` maps "Y1 Leaf 1" → [error_t0, error_t1, ...].
    Adds a non-MLC column to verify the column filter rejects it.
    """

    cols = {
        f"{name}/Positional Error (mm)": values
        for name, values in per_leaf_errors.items()
    }
    cols["Step Dose/Actual Value (Mu)"] = [1.0] * len(next(iter(cols.values())))
    return pd.DataFrame(cols)


def _read_trf_returning(table: pd.DataFrame):
    def _fn(_path):
        return (pd.DataFrame(), table)

    return _fn


def test_mlc_residual_is_mean_of_abs_positional_error():
    table = _make_table(
        {
            "Y1 Leaf 1": [0.2, -0.2, 0.3],
            "Y2 Leaf 1": [-0.1, 0.1, 0.0],
        }
    )
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    obs = om.encode({"trf_path": "ignored"})
    expected = (0.2 + 0.2 + 0.3 + 0.1 + 0.1 + 0.0) / 6.0
    assert obs.mean_mlc_residual_mm == pytest.approx(expected, rel=1e-6)


def test_mlc_residual_is_none_when_no_leaf_columns():
    table = pd.DataFrame({"Step Dose/Actual Value (Mu)": [1.0, 2.0]})
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    obs = om.encode({"trf_path": "ignored"})
    assert obs.mean_mlc_residual_mm is None


def test_mlc_residual_skipped_when_no_trf_path():
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(pd.DataFrame()))
    obs = om.encode({})
    assert obs.mean_mlc_residual_mm is None


def test_caller_supplied_residual_overrides_trf():
    table = _make_table({"Y1 Leaf 1": [10.0, 10.0]})
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    obs = om.encode({"trf_path": "ignored", "mean_mlc_residual_mm": 0.1})
    assert obs.mean_mlc_residual_mm == 0.1


def test_other_channels_pass_through():
    table = _make_table({"Y1 Leaf 1": [0.5, -0.5]})
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    obs = om.encode(
        {
            "trf_path": "ignored",
            "gamma_pass_rate": 99.0,
            "plan_hash_ok": True,
        }
    )
    assert obs.gamma_pass_rate == 99.0
    assert obs.plan_hash_ok is True


def test_only_positional_error_columns_contribute():
    # An "Actual Tolerance" column must NOT enter the residual mean.
    table = pd.DataFrame(
        {
            "Y1 Leaf 1/Positional Error (mm)": [0.4, -0.4],
            "Y1 Leaf 1/Actual Tolerance (mm)": [2.0, 2.0],
            "Y1 Leaf 1/Scaled Actual (mm)": [50.0, 50.0],
        }
    )
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    obs = om.encode({"trf_path": "ignored"})
    assert obs.mean_mlc_residual_mm == pytest.approx(0.4, rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end with the agent
# ---------------------------------------------------------------------------


def test_agent_with_trf_residual_flags_mlc_degraded():
    table = _make_table(
        {
            "Y1 Leaf 1": [1.6, -1.4, 1.5],
            "Y2 Leaf 1": [1.5, -1.5, 1.5],
        }
    )
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step({"trf_path": "ignored"})
    assert result.posterior.map_fault_mode() is FaultMode.MLC_DEGRADED
    assert result.recommendation.action is not Action.APPROVE_FRACTION


def test_agent_with_clean_trf_residual_recommends_approve():
    table = _make_table(
        {
            "Y1 Leaf 1": [0.1, -0.05, 0.02, -0.01, 0.0, 0.03],
            "Y2 Leaf 1": [0.0, 0.05, -0.02, 0.01, -0.03, 0.0],
        }
    )
    om = TrfObservationModel(read_trf_fn=_read_trf_returning(table))
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step(
        {
            "trf_path": "ignored",
            "gamma_pass_rate": 98.5,
            "output_ratio": 1.000,
            "setup_residual_mm": 0.4,
            "plan_hash_ok": True,
        }
    )
    assert result.posterior.map_fault_mode() is FaultMode.NOMINAL
    assert result.recommendation.action is Action.APPROVE_FRACTION
