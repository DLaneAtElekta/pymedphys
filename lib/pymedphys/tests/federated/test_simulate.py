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

"""Stage 0 of the federated roadmap: the plumbing, with no model and no data.

Exit criterion for the stage, asserted below: a federated mean over three
non-IID sites matches the pooled mean; the manifest gate rejects a
mis-configured site before round 1; both aperture guards raise on cue; and an
audit log is written.
"""

import numpy as np
import pytest

from pymedphys._federated import aperture as aperture_module
from pymedphys._federated import protocol, simulate, toy

GRID = (8, 8, 8)
VOCABULARY_HASH = "a" * 64

# Non-IID on purpose: different cohort sizes, different means, different
# spreads, as three clinics would be.
SITE_SPECS = [
    ("site-a", 40, 0.0, 1.0),
    ("site-b", 120, 5.0, 0.2),
    ("site-c", 25, -3.0, 4.0),
]


def a_manifest(site_id, **overrides):
    fields = {
        "site_id": site_id,
        "grid_shape": GRID,
        "voxel_spacing_mm": (2.0, 2.0, 2.0),
        "structure_keys": ("Brainstem", "Parotid_L"),
        "structure_vocabulary_sha256": VOCABULARY_HASH,
    }
    fields.update(overrides)

    return protocol.SiteManifest(**fields)


def a_trainer(site_id, data, audit_dir=None, manifest=None, **trainer_overrides):
    manifest = manifest if manifest is not None else a_manifest(site_id)
    policy = aperture_module.policy_from_manifest(
        manifest,
        max_bytes_per_round=100_000,
        allowed_metric_keys=["train_loss"],
        min_examples=10,
    )
    aperture = aperture_module.Aperture(
        policy=policy,
        site_id=site_id,
        audit_log_path=(audit_dir / f"{site_id}.jsonl") if audit_dir else None,
    )

    return toy.MeanVectorTrainer(
        data=data, manifest=manifest, aperture=aperture, **trainer_overrides
    )


def site_data(seed, num_examples, centre, spread, num_features=3):
    generator = np.random.default_rng(seed)

    return centre + spread * generator.standard_normal((num_examples, num_features))


def three_sites(audit_dir=None):
    trainers = []
    for seed, (site_id, num_examples, centre, spread) in enumerate(SITE_SPECS):
        data = site_data(seed, num_examples, centre, spread)
        trainers.append(a_trainer(site_id, data, audit_dir=audit_dir))

    return trainers


def pooled_mean():
    return np.mean(
        np.concatenate(
            [
                site_data(seed, num_examples, centre, spread)
                for seed, (_, num_examples, centre, spread) in enumerate(SITE_SPECS)
            ]
        ),
        axis=0,
    )


def test_the_federated_mean_matches_the_pooled_mean(tmp_path):
    history = simulate.run_federation(three_sites(audit_dir=tmp_path), rounds=3)

    (federated,) = history.weights

    assert np.allclose(federated, pooled_mean())


def test_the_site_local_parameter_never_moves(tmp_path):
    trainers = three_sites(audit_dir=tmp_path)
    before = [trainer.local_scale for trainer in trainers]

    simulate.run_federation(trainers, rounds=3)

    for trainer, original in zip(trainers, before):
        # `shared_keys` excludes the scale, so aggregation cannot touch it.
        # This is FedBN in miniature.
        assert np.array_equal(trainer.local_scale, original)


def test_per_site_losses_are_recorded_separately(tmp_path):
    history = simulate.run_federation(three_sites(audit_dir=tmp_path), rounds=3)

    # The honest picture of federating heterogeneous sites: the global
    # parameter converges while the site that is furthest from it does worse
    # than it would alone. Recorded per site rather than averaged away.
    tight_site = history.eval_loss_series("site-b")
    assert len(tight_site) == 3

    local_only = np.var(site_data(1, 120, 5.0, 0.2), axis=0).mean()
    assert tight_site[-1] > local_only


def test_an_audit_line_is_written_for_every_emission(tmp_path):
    simulate.run_federation(three_sites(audit_dir=tmp_path), rounds=3)

    for site_id, _, _, _ in SITE_SPECS:
        records = aperture_module.read_audit_log(tmp_path / f"{site_id}.jsonl")

        assert [record["kind"] for record in records] == ["fit", "evaluate"] * 3
        assert all(record["status"] == "emitted" for record in records)
        assert [record["round"] for record in records] == [1, 1, 2, 2, 3, 3]


def test_a_site_with_a_different_vocabulary_is_rejected_before_round_one(tmp_path):
    trainers = three_sites()
    trainers[2] = a_trainer(
        "site-c",
        site_data(2, 25, -3.0, 4.0),
        manifest=a_manifest("site-c", structure_vocabulary_sha256="b" * 64),
    )

    with pytest.raises(simulate.ManifestMismatch) as error:
        simulate.run_federation(trainers, rounds=3)

    message = str(error.value)
    assert "site-c" in message
    assert "structure_vocabulary_sha256" in message


def test_a_site_that_resampled_to_a_different_grid_is_rejected(tmp_path):
    trainers = three_sites()
    trainers[1] = a_trainer(
        "site-b",
        site_data(1, 120, 5.0, 0.2),
        manifest=a_manifest("site-b", voxel_spacing_mm=(3.0, 2.0, 2.0)),
    )

    with pytest.raises(simulate.ManifestMismatch, match="voxel_spacing_mm"):
        simulate.run_federation(trainers, rounds=3)


def test_sites_must_agree_on_which_parameters_are_shared():
    trainers = three_sites()

    class SharesMore(type(trainers[0])):
        def shared_keys(self):
            return ["mean", "log_scale"]

    trainers[0].__class__ = SharesMore

    with pytest.raises(simulate.ManifestMismatch, match="shared"):
        simulate.check_manifest_compatibility(trainers)


def test_duplicate_site_ids_are_rejected():
    trainers = three_sites()
    trainers[1] = a_trainer("site-a", site_data(1, 120, 5.0, 0.2))

    with pytest.raises(simulate.ManifestMismatch, match="unique"):
        simulate.check_manifest_compatibility(trainers)


def test_a_leaked_debug_volume_is_stopped_by_the_aperture(tmp_path):
    trainers = three_sites(audit_dir=tmp_path)
    trainers[0] = a_trainer(
        "site-a", site_data(0, 40, 0.0, 1.0), audit_dir=tmp_path, leak_debug_volume=True
    )

    with pytest.raises(aperture_module.ApertureViolation, match="forbidden shape"):
        simulate.run_federation(trainers, rounds=1)

    (record,) = aperture_module.read_audit_log(tmp_path / "site-a.jsonl")
    assert record["status"] == "rejected"


def test_a_leaked_identifier_is_stopped_by_the_aperture(tmp_path):
    trainers = three_sites(audit_dir=tmp_path)
    trainers[0] = a_trainer(
        "site-a",
        site_data(0, 40, 0.0, 1.0),
        audit_dir=tmp_path,
        leak_metric_key="patient_mrn",
    )

    with pytest.raises(aperture_module.ApertureViolation, match="whitelist"):
        simulate.run_federation(trainers, rounds=1)


def test_federated_average_weights_by_cohort_size():
    aggregated = simulate.federated_average(
        [[np.array([0.0])], [np.array([10.0])]], [10, 30]
    )

    assert np.allclose(aggregated[0], 7.5)


@pytest.mark.parametrize(
    "weight_sets, counts",
    [
        ([], []),
        ([[np.ones(2)]], [0]),
        ([[np.ones(2)], [np.ones(2), np.ones(2)]], [1, 1]),
        ([[np.ones(2)], [np.ones(3)]], [1, 1]),
    ],
)
def test_federated_average_rejects_incoherent_updates(weight_sets, counts):
    with pytest.raises(ValueError):
        simulate.federated_average(weight_sets, counts)


def test_the_server_sets_the_schedule_every_site_follows(tmp_path):
    seen = []

    trainers = three_sites()

    class RecordsConfig(type(trainers[0])):
        def fit(self, config):
            seen.append(dict(config))

            return super().fit(config)

    for trainer in trainers:
        trainer.__class__ = RecordsConfig

    simulate.run_federation(
        trainers, rounds=2, config_fn=lambda round_number: {"beta": 0.1 * round_number}
    )

    # A KL warm-up has to be decided centrally or the sites drift out of step.
    assert [config["beta"] for config in seen] == [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    assert [config["round"] for config in seen] == [1, 1, 1, 2, 2, 2]


def test_a_federation_needs_at_least_one_site():
    with pytest.raises(simulate.ManifestMismatch):
        simulate.check_manifest_compatibility([])


def test_the_toy_trainer_satisfies_the_protocol():
    trainer = three_sites()[0]

    assert isinstance(trainer, protocol.SiteTrainer)
