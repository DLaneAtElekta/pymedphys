# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

"""CLI entry for the experimental patient-QA agent.

The subcommand is intentionally thin while the underlying numerics
are stubbed; it exists so the agent can be wired into pipelines and
smoke-tested. It will raise ``NotImplementedError`` until the
observation/belief/policy modules have real implementations.
"""

from __future__ import annotations


def qa_agent_cli(subparsers):
    parser = subparsers.add_parser(
        "qa-agent",
        help="Experimental active-inference patient-QA agent (decision-support).",
    )
    parser.add_argument(
        "inputs_json",
        type=str,
        help=(
            "Path to a JSON file describing one fraction's raw QA "
            "inputs (gamma_result, trf_path, rt_record, igrt, ...). "
            "Schema is permissive; missing channels are tolerated."
        ),
    )
    parser.set_defaults(func=_run)
    return parser


def _run(args) -> None:
    import json

    from pymedphys._experimental.qa_agent import (
        BeliefUpdater,
        ObservationModel,
        Policy,
        QAAgent,
        QAAgentConfig,
    )

    with open(args.inputs_json, encoding="utf-8") as f:
        raw = json.load(f)

    agent = QAAgent(
        observation_model=ObservationModel(),
        belief_updater=BeliefUpdater(ObservationModel()),
        policy=Policy(),
        config=QAAgentConfig(),
    )
    result = agent.step(raw)
    print(
        f"recommendation: {result.recommendation.action.value} "
        f"(EFE={result.recommendation.total:.3f})"
    )
