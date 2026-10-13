# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the QA-agent scaffold.

Covers the public surface (data classes, enums) plus behavioural
checks on the first numerical pass: per-channel likelihoods,
discrete-factor Bayesian update, and EFE-based policy.
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
    FaultMode,
    LatentState,
    Policy,
    QAAgent,
)


# ---------------------------------------------------------------------------
# Surface: data classes and enums
# ---------------------------------------------------------------------------


def test_latent_state_defaults_to_nominal():
    state = LatentState()
    assert state.fault_mode is FaultMode.NOMINAL
    assert state.errors.mlc_leaf_bias_mm == 0.0


def test_belief_uniform_prior_sums_to_one():
    belief = Belief()
    assert isclose(sum(belief.fault_mode_probs.values()), 1.0)
    assert set(belief.fault_mode_probs) == set(FaultMode)


def test_belief_entropy_is_log_n_at_uniform_prior():
    assert isclose(Belief().entropy_nats(), log(len(FaultMode)))


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


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def test_log_likelihood_zero_for_no_channels():
    om = ObservationModel()
    assert om.log_likelihood(Observation(), LatentState()) == 0.0


def test_log_likelihood_higher_for_matching_fault_mode():
    om = ObservationModel()
    obs = Observation(gamma_pass_rate=98.0)
    nominal = om.log_likelihood(obs, LatentState(fault_mode=FaultMode.NOMINAL))
    corrupt = om.log_likelihood(obs, LatentState(fault_mode=FaultMode.PLAN_CORRUPTION))
    assert nominal > corrupt


def test_plan_hash_failure_strongly_implicates_corruption():
    om = ObservationModel()
    obs = Observation(plan_hash_ok=False)
    nominal = om.log_likelihood(obs, LatentState(fault_mode=FaultMode.NOMINAL))
    corrupt = om.log_likelihood(obs, LatentState(fault_mode=FaultMode.PLAN_CORRUPTION))
    assert corrupt > nominal


def test_encode_is_permissive_about_unknown_keys():
    om = ObservationModel()
    obs = om.encode({"gamma_pass_rate": 97.5, "irrelevant_field": "ignored"})
    assert obs.gamma_pass_rate == 97.5
    assert obs.output_ratio is None


# ---------------------------------------------------------------------------
# Belief update
# ---------------------------------------------------------------------------


def test_belief_update_normalises_to_one():
    om = ObservationModel()
    bu = BeliefUpdater(om)
    posterior = bu.update(Belief(), Observation(gamma_pass_rate=98.0))
    assert isclose(sum(posterior.fault_mode_probs.values()), 1.0)


def test_high_gamma_drives_posterior_to_nominal():
    om = ObservationModel()
    bu = BeliefUpdater(om)
    posterior = bu.update(Belief(), Observation(gamma_pass_rate=98.0))
    assert posterior.map_fault_mode() is FaultMode.NOMINAL


def test_low_gamma_with_large_setup_residual_implicates_setup_error():
    om = ObservationModel()
    bu = BeliefUpdater(om)
    obs = Observation(gamma_pass_rate=85.0, setup_residual_mm=5.0)
    posterior = bu.update(Belief(), obs)
    assert posterior.map_fault_mode() is FaultMode.SETUP_ERROR


def test_plan_hash_false_drives_posterior_to_corruption():
    om = ObservationModel()
    bu = BeliefUpdater(om)
    posterior = bu.update(Belief(), Observation(plan_hash_ok=False))
    assert posterior.map_fault_mode() is FaultMode.PLAN_CORRUPTION


def test_observation_reduces_belief_entropy():
    om = ObservationModel()
    bu = BeliefUpdater(om)
    prior = Belief()
    posterior = bu.update(prior, Observation(gamma_pass_rate=98.0))
    assert posterior.entropy_nats() < prior.entropy_nats()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_recommends_approve_when_confidently_nominal():
    confident_nominal = Belief(
        fault_mode_probs={
            FaultMode.NOMINAL: 0.97,
            FaultMode.MLC_DEGRADED: 0.005,
            FaultMode.OUTPUT_DRIFT: 0.005,
            FaultMode.SETUP_ERROR: 0.005,
            FaultMode.GATING_FAULT: 0.005,
            FaultMode.COLLISION_RISK: 0.005,
            FaultMode.PLAN_CORRUPTION: 0.005,
        }
    )
    assert Policy().select(confident_nominal).action is Action.APPROVE_FRACTION


def test_policy_avoids_approve_when_corruption_is_likely():
    likely_corrupt = Belief(
        fault_mode_probs={
            FaultMode.NOMINAL: 0.05,
            FaultMode.MLC_DEGRADED: 0.05,
            FaultMode.OUTPUT_DRIFT: 0.05,
            FaultMode.SETUP_ERROR: 0.05,
            FaultMode.GATING_FAULT: 0.05,
            FaultMode.COLLISION_RISK: 0.05,
            FaultMode.PLAN_CORRUPTION: 0.70,
        }
    )
    assert Policy().select(likely_corrupt).action is not Action.APPROVE_FRACTION


def test_policy_prefers_information_gathering_when_belief_is_ambiguous():
    info_actions = {
        Action.HOLD_FOR_REVIEW,
        Action.REQUEST_REMEASUREMENT,
        Action.TRIGGER_RECALIBRATION,
        Action.ESCALATE,
    }
    assert Policy().select(Belief()).action in info_actions


def test_efe_decomposition_exposes_pragmatic_and_epistemic_terms():
    decompositions = Policy().evaluate(Belief())
    by_action = {d.action: d for d in decompositions}
    approve = by_action[Action.APPROVE_FRACTION]
    remeasure = by_action[Action.REQUEST_REMEASUREMENT]
    # Approve has zero epistemic term (no new info); remeasure has a strictly
    # negative epistemic term under any non-degenerate belief.
    assert approve.epistemic == 0.0
    assert remeasure.epistemic < 0.0
    assert isclose(approve.total, approve.pragmatic + approve.epistemic)


# ---------------------------------------------------------------------------
# End-to-end agent step
# ---------------------------------------------------------------------------


def test_agent_step_returns_recommendation_for_nominal_observation():
    om = ObservationModel()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step(
        {
            "gamma_pass_rate": 98.5,
            "mean_mlc_residual_mm": 0.2,
            "output_ratio": 1.000,
            "setup_residual_mm": 0.4,
            "gating_dropouts": 0,
            "plan_hash_ok": True,
        }
    )
    assert result.posterior.map_fault_mode() is FaultMode.NOMINAL
    assert result.recommendation.action is Action.APPROVE_FRACTION
    # Alternatives include all actions for auditability.
    assert {d.action for d in result.alternatives} == set(Action)


def test_agent_step_does_not_approve_on_plan_hash_failure():
    om = ObservationModel()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step({"plan_hash_ok": False, "gamma_pass_rate": 97.0})
    assert result.recommendation.action is not Action.APPROVE_FRACTION


def test_agent_belief_persists_across_steps():
    om = ObservationModel()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    first = agent.step({"gamma_pass_rate": 85.0, "setup_residual_mm": 5.0})
    second_prior_entropy = agent.belief.entropy_nats()
    # Entropy after one informative observation must be below the uniform prior.
    assert second_prior_entropy < log(len(FaultMode))
    assert first.posterior is agent.belief


def test_unknown_keys_in_raw_inputs_are_ignored():
    pytest.importorskip("pytest")  # ensures we're using real pytest
    om = ObservationModel()
    agent = QAAgent(
        observation_model=om,
        belief_updater=BeliefUpdater(om),
        policy=Policy(),
    )
    result = agent.step({"some_future_channel": 0.0, "gamma_pass_rate": 98.0})
    assert result.observation.gamma_pass_rate == 98.0
