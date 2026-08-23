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

"""A runnable Stage 0 demonstration.

Three simulated non-IID sites, in-process FedAvg, converging to the pooled
mean; the manifest gate rejecting a site with a different vocabulary hash; the
aperture rejecting both a voxel shaped array and a metric key that is not on
the whitelist; and an audit line per emission::

    python -m pymedphys._federated.demo

NumPy is the only requirement. There is no server, no framework, and no data.
"""

from __future__ import annotations

import pathlib
import tempfile

from pymedphys._imports import numpy as np

from . import simulate, toy
from .aperture import Aperture, ApertureViolation, policy_from_manifest, read_audit_log
from .protocol import SiteManifest

GRID = (32, 32, 32)
VOCABULARY_HASH = "9" * 64

ROUNDS = 8

# A partial local step, so that the round-by-round trajectory is visible.
LEARNING_RATE = 0.5

# Non-IID on purpose: differing cohort sizes, anatomy centres and spreads, as
# three clinics would be.
SITES = [
    {"site_id": "north-general", "num_examples": 40, "centre": 0.0, "spread": 1.0},
    {"site_id": "coastal-cancer", "num_examples": 120, "centre": 5.0, "spread": 0.2},
    {"site_id": "valley-regional", "num_examples": 25, "centre": -3.0, "spread": 4.0},
]


def build_site(
    site_id: str,
    num_examples: int,
    centre: float,
    spread: float,
    seed: int,
    audit_dir: pathlib.Path,
    vocabulary_sha256: str = VOCABULARY_HASH,
    **trainer_overrides,
) -> toy.MeanVectorTrainer:
    """Assemble one clinic: a manifest, an aperture, and a local trainer."""

    manifest = SiteManifest(
        site_id=site_id,
        grid_shape=GRID,
        voxel_spacing_mm=(2.0, 2.0, 2.0),
        structure_keys=("Brainstem", "Parotid_L", "Parotid_R", "SpinalCord"),
        structure_vocabulary_sha256=vocabulary_sha256,
    )

    aperture = Aperture(
        policy=policy_from_manifest(
            manifest,
            max_bytes_per_round=64_000,
            allowed_metric_keys=["train_loss"],
            min_examples=10,
        ),
        site_id=site_id,
        audit_log_path=audit_dir / f"{site_id}.jsonl",
    )

    return toy.MeanVectorTrainer(
        data=site_data(num_examples, centre, spread, seed),
        manifest=manifest,
        aperture=aperture,
        **trainer_overrides,
    )


def site_data(
    num_examples: int, centre: float, spread: float, seed: int
) -> "np.ndarray":
    """One clinic's local data. In a real deployment this never leaves."""

    generator = np.random.default_rng(seed)

    return centre + spread * generator.standard_normal((num_examples, 3))


def build_federation(audit_dir: pathlib.Path, **overrides):
    """The three demonstration sites."""

    return [
        build_site(seed=seed, audit_dir=audit_dir, **spec, **overrides)
        for seed, spec in enumerate(SITES)
    ]


def pooled_mean() -> "np.ndarray":
    """What a single centralised trainer would have found."""

    return np.mean(
        np.concatenate(
            [
                site_data(spec["num_examples"], spec["centre"], spec["spread"], seed)
                for seed, spec in enumerate(SITES)
            ]
        ),
        axis=0,
    )


def main(audit_dir: str | pathlib.Path | None = None) -> int:
    """Run the demonstration, printing what happened. Returns an exit code."""

    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(audit_dir) if audit_dir is not None else None
        directory = directory if directory is not None else pathlib.Path(temporary)
        directory.mkdir(parents=True, exist_ok=True)

        _federate(directory)
        _reject_mismatched_site(directory)
        _reject_voxel_shaped_array(directory)
        _reject_undeclared_metric(directory)
        _show_audit_log(directory)

    return 0


def _federate(audit_dir: pathlib.Path):
    print(f"1. Three non-IID sites, {ROUNDS} rounds of FedAvg")

    trainers = build_federation(audit_dir / "federation")
    history = simulate.run_federation(
        trainers,
        rounds=ROUNDS,
        config_fn=lambda round_number: {"learning_rate": LEARNING_RATE},
    )

    (federated,) = history.weights
    pooled = pooled_mean()

    print(f"   compatibility key : {history.compatibility_key[:16]}...")
    print(f"   federated mean    : {np.round(federated, 6)}")
    print(f"   pooled mean       : {np.round(pooled, 6)}")
    print(f"   max difference    : {np.max(np.abs(federated - pooled)):.2e}")
    print(
        f"   (a partial local step of {LEARNING_RATE} converges on the pooled mean; a\n"
        "   full local step reaches it exactly in one round)"
    )

    print("\n   per-site evaluation loss, by round:")
    for seed, spec in enumerate(SITES):
        series = history.eval_loss_series(spec["site_id"])
        alone = float(
            np.var(
                site_data(spec["num_examples"], spec["centre"], spec["spread"], seed),
                axis=0,
            ).mean()
        )
        formatted = ", ".join(f"{loss:8.3f}" for loss in series)
        print(f"     {spec['site_id']:>16}: {formatted}   (alone: {alone:.3f})")

    print(
        "\n   The site losses do not all improve, and none reaches what that site\n"
        "   could have managed alone. The global model is a compromise none of\n"
        "   the participants would have chosen for themselves. That is the honest\n"
        "   picture of federating heterogeneous sites, and worth showing.\n"
    )


def _reject_mismatched_site(audit_dir: pathlib.Path):
    print("2. A site whose structure vocabulary disagrees")

    trainers = build_federation(audit_dir / "mismatch")
    trainers[2] = build_site(
        seed=2,
        audit_dir=audit_dir / "mismatch",
        vocabulary_sha256="0" * 64,
        **SITES[2],
    )

    try:
        simulate.run_federation(trainers, rounds=1)
    except simulate.ManifestMismatch as mismatch:
        print(f"   rejected before round 1: {mismatch}\n")
    else:
        raise AssertionError("The manifest gate failed to reject a mismatched site.")


def _reject_voxel_shaped_array(audit_dir: pathlib.Path):
    print("3. A debugging tensor shaped like a patient")

    trainers = build_federation(audit_dir / "leak-volume")
    trainers[0] = build_site(
        seed=0,
        audit_dir=audit_dir / "leak-volume",
        leak_debug_volume=True,
        **SITES[0],
    )

    try:
        simulate.run_federation(trainers, rounds=1)
    except ApertureViolation as violation:
        print(f"   stopped at the aperture: {violation}\n")
    else:
        raise AssertionError("The aperture failed to reject a voxel shaped array.")


def _reject_undeclared_metric(audit_dir: pathlib.Path):
    print("4. An identifier travelling as a metric")

    trainers = build_federation(audit_dir / "leak-metric")
    trainers[0] = build_site(
        seed=0,
        audit_dir=audit_dir / "leak-metric",
        leak_metric_key="patient_mrn",
        **SITES[0],
    )

    try:
        simulate.run_federation(trainers, rounds=1)
    except ApertureViolation as violation:
        print(f"   stopped at the aperture: {violation}\n")
    else:
        raise AssertionError("The aperture failed to reject an undeclared metric.")


def _show_audit_log(audit_dir: pathlib.Path):
    print("5. What left the clinic, per the audit log")

    path = audit_dir / "federation" / f"{SITES[0]['site_id']}.jsonl"
    records = read_audit_log(path)

    header = (
        f"   {'round':>5}  {'kind':<9}{'status':<10}{'arrays':>7}{'bytes':>7}{'n':>5}"
    )
    print(header)
    for record in records:
        print(
            f"   {record['round']:>5}  {record['kind']:<9}{record['status']:<10}"
            f"{record['array_count']:>7}{record['byte_count']:>7}"
            f"{record['example_count']:>5}"
        )

    total = sum(record["byte_count"] for record in records)
    print(f"\n   {len(records)} emissions, {total} bytes total, from {path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
