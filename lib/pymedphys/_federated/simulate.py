# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An in-process federation, for testing the contract without a server.

This is the loop that Stage 3 of the roadmap runs before Flower's simulation
backend and long before gRPC. No networking, no IT ticket, full debugger.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

from pymedphys._imports import numpy as np

from .protocol import EvalResult, FitResult, SiteManifest, SiteTrainer


class ManifestMismatch(ValueError):
    """Raised before round 1 when sites do not agree on the representation."""


@dataclasses.dataclass(frozen=True)
class RoundHistory:
    """What happened in a single round, kept per site rather than averaged.

    Per-site evaluation is recorded separately on purpose. A federated model
    is a compromise that none of the participants would have chosen for
    themselves, and per-site losses that diverge while the global parameter
    converges are the honest picture of that -- worth showing, not smoothing.
    """

    round_number: int
    fit_metrics: dict[str, dict[str, float]]
    eval_losses: dict[str, float]
    eval_metrics: dict[str, dict[str, float]]
    num_examples: dict[str, int]


@dataclasses.dataclass
class FederationHistory:
    """The result of :func:`run_federation`."""

    compatibility_key: str
    rounds: list[RoundHistory] = dataclasses.field(default_factory=list)
    weights: Sequence["np.ndarray"] = ()

    def eval_loss_series(self, site_id: str) -> list[float]:
        """Per-round evaluation loss for one site."""

        return [
            round_history.eval_losses[site_id]
            for round_history in self.rounds
            if site_id in round_history.eval_losses
        ]


def check_manifest_compatibility(trainers: Sequence[SiteTrainer]) -> str:
    """Compare manifests before round 1 and return the shared key.

    Raises
    ------
    ManifestMismatch
        If the sites disagree on grid, spacing, canonical structures, the
        shared vocabulary hash, or the set of shared parameters. The message
        names the disagreeing sites, because "training failed" is not an
        actionable error at 2am in a hospital.
    """

    if not trainers:
        raise ManifestMismatch("A federation needs at least one site.")

    manifests = [trainer.manifest() for trainer in trainers]

    site_ids = [manifest.site_id for manifest in manifests]
    if len(set(site_ids)) != len(site_ids):
        raise ManifestMismatch(f"Site ids must be unique, got {site_ids}.")

    by_key: dict[str, list[str]] = {}
    for manifest in manifests:
        by_key.setdefault(manifest.compatibility_key, []).append(manifest.site_id)

    if len(by_key) > 1:
        raise ManifestMismatch(
            "Sites do not agree on the data representation. Groups by "
            f"compatibility key: {_describe_groups(by_key)}. "
            f"Differing fields: {_describe_differences(manifests)}."
        )

    shared_keys = {tuple(trainer.shared_keys()) for trainer in trainers}
    if len(shared_keys) > 1:
        by_shared: dict[tuple[str, ...], list[str]] = {}
        for trainer, manifest in zip(trainers, manifests):
            by_shared.setdefault(tuple(trainer.shared_keys()), []).append(
                manifest.site_id
            )

        groups = [
            {"shared_keys": list(key), "sites": sites}
            for key, sites in by_shared.items()
        ]

        raise ManifestMismatch(
            f"Sites do not agree on which parameters are shared. Groups: {groups}."
        )

    return manifests[0].compatibility_key


def federated_average(
    weight_sets: Sequence[Sequence["np.ndarray"]], num_examples: Sequence[int]
) -> list["np.ndarray"]:
    """FedAvg: a per-parameter mean weighted by local cohort size."""

    if not weight_sets:
        raise ValueError("Cannot aggregate an empty set of updates.")

    if len(weight_sets) != len(num_examples):
        raise ValueError(
            f"Got {len(weight_sets)} weight sets for {len(num_examples)} example counts."
        )

    lengths = {len(weights) for weights in weight_sets}
    if len(lengths) > 1:
        raise ValueError(
            f"Sites returned differing numbers of arrays: {sorted(lengths)}."
        )

    total = float(sum(num_examples))
    if total <= 0:
        raise ValueError("Total example count across sites must be positive.")

    aggregated = []
    for index in range(next(iter(lengths))):
        shapes = {np.shape(weights[index]) for weights in weight_sets}
        if len(shapes) > 1:
            raise ValueError(
                f"Sites disagree on the shape of array {index}: {sorted(shapes)}."
            )

        stacked = sum(
            np.asarray(weights[index], dtype=float) * (count / total)
            for weights, count in zip(weight_sets, num_examples)
        )
        aggregated.append(stacked)

    return aggregated


def run_federation(
    trainers: Sequence[SiteTrainer],
    rounds: int,
    config_fn: Callable[[int], Mapping[str, Any]] | None = None,
    evaluate: bool = True,
) -> FederationHistory:
    """Run FedAvg in process, checking manifests before the first round.

    Parameters
    ----------
    trainers
        The participating sites.
    rounds
        Number of federated rounds.
    config_fn
        Maps a round number to the config handed to every site. This is how
        the server keeps sites in step on schedules such as a KL warm-up --
        ``beta`` is decided centrally, not per site.
    evaluate
        Whether to evaluate after each round. Evaluation goes through each
        site's aperture like anything else.
    """

    compatibility_key = check_manifest_compatibility(trainers)
    history = FederationHistory(compatibility_key=compatibility_key)

    weights = list(trainers[0].get_weights())

    for round_number in range(1, rounds + 1):
        config = dict(config_fn(round_number)) if config_fn is not None else {}
        config.setdefault("round", round_number)

        fit_results: list[FitResult] = []
        for trainer in trainers:
            trainer.set_weights(weights)
            fit_results.append(trainer.fit(config))

        weights = federated_average(
            [result.weights for result in fit_results],
            [result.num_examples for result in fit_results],
        )

        eval_losses: dict[str, float] = {}
        eval_metrics: dict[str, dict[str, float]] = {}
        if evaluate:
            for trainer in trainers:
                trainer.set_weights(weights)
                result: EvalResult = trainer.evaluate(config)
                site_id = trainer.manifest().site_id
                eval_losses[site_id] = result.loss
                eval_metrics[site_id] = dict(result.metrics)

        history.rounds.append(
            RoundHistory(
                round_number=round_number,
                fit_metrics={
                    trainer.manifest().site_id: dict(result.metrics)
                    for trainer, result in zip(trainers, fit_results)
                },
                eval_losses=eval_losses,
                eval_metrics=eval_metrics,
                num_examples={
                    trainer.manifest().site_id: result.num_examples
                    for trainer, result in zip(trainers, fit_results)
                },
            )
        )

    for trainer in trainers:
        trainer.set_weights(weights)

    history.weights = tuple(weights)

    return history


def _describe_groups(by_key: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {key[:12]: list(sites) for key, sites in by_key.items()}


def _describe_differences(manifests: Sequence[SiteManifest]) -> list[str]:
    differing = []
    for field in manifests[0].compatibility_fields:
        values = {repr(manifest.compatibility_fields[field]) for manifest in manifests}
        if len(values) > 1:
            differing.append(field)

    return differing
