# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke tests for the QA-agent scaffold.

These tests pin the public surface and the parts of the design that
are not numerics: factored state, uniform prior, action enum,
NotImplementedError boundaries. Numerical behaviour will be tested
once the stubs are filled in.
"""

from __future__ import annotations

from math import isclose, log

import pytest

from pymedphys._experimental.qa_agent import (
    Action,
    Belief,
    BeliefUpdater,
    Observation,
    ObservationModel,
    Phenotype,
    PhenotypeState,
    Policy,
    QAAgent,
)


def test_phenotype_state_defaults_to_nominal():
    state = PhenotypeState()
    assert state.phenotype is Phenotype.NOMINAL
    assert state.errors.mlc_leaf_bias_mm == 0.0
    assert state.errors.output_drift_pct == 0.0


def test_belief_uniform_prior_sums_to_one():
    belief = Belief()
    assert isclose(sum(belief.phenotype_probs.values()), 1.0)
    assert set(belief.phenotype_probs) == set(Phenotype)


def test_belief_entropy_is_log_n_at_uniform_prior():
    belief = Belief()
    assert isclose(belief.entropy_nats(), log(len(Phenotype)))


def test_action_enum_covers_minimum_clinical_decisions():
    required = {
        "approve_fraction",
        "hold_for_review",
        "request_remeasurement",
        "escalate",
    }
    assert required.issubset({a.value for a in Action})


def test_observation_allows_missing_channels():
    obs = Observation()
    assert obs.gamma_pass_rate is None
    assert obs.plan_hash_ok is None


def test_stubbed_components_raise_not_implemented():
    om = ObservationModel()
    with pytest.raises(NotImplementedError):
        om.encode({})
    with pytest.raises(NotImplementedError):
        om.log_likelihood(Observation(), PhenotypeState())

    bu = BeliefUpdater(om)
    with pytest.raises(NotImplementedError):
        bu.update(Belief(), Observation())

    policy = Policy()
    with pytest.raises(NotImplementedError):
        policy.evaluate(Belief())


def test_qaagent_step_propagates_stub_errors():
    agent = QAAgent(
        observation_model=ObservationModel(),
        belief_updater=BeliefUpdater(ObservationModel()),
        policy=Policy(),
    )
    with pytest.raises(NotImplementedError):
        agent.step({})
