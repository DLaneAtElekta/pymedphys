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
non-IID clinics matches the pooled mean; the manifest gate rejects a
mis-configured clinic before round 1; both aperture guards raise on cue; and an
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
CLINIC_SPECS = [
    ("clinic-a", 40, 0.0, 1.0),
    ("clinic-b", 120, 5.0, 0.2),
    ("clinic-c", 25, -3.0, 4.0),
]


def a_manifest(clinic_id, **overrides):
    fields = {
        "clinic_id": clinic_id,
        "grid_shape": GRID,
        "voxel_spacing_mm": (2.0, 2.0, 2.0),
        "structure_keys": ("Brainstem", "Parotid_L"),
        "structure_vocabulary_sha256": VOCABULARY_HASH,
    }
    fields.update(overrides)

    return protocol.ClinicManifest(**fields)


def a_trainer(clinic_id, data, audit_dir=None, manifest=None, **trainer_overrides):
    manifest = manifest if manifest is not None else a_manifest(clinic_id)
    policy = aperture_module.policy_from_manifest(
        manifest,
        max_bytes_per_round=100_000,
        allowed_metric_keys=["train_loss"],
        min_examples=10,
    )
    aperture = aperture_module.Aperture(
        policy=policy,
        clinic_id=clinic_id,
        audit_log_path=(audit_dir / f"{clinic_id}.jsonl") if audit_dir else None,
    )

    return toy.MeanVectorTrainer(
        data=data, manifest=manifest, aperture=aperture, **trainer_overrides
    )


def clinic_data(seed, num_examples, centre, spread, num_features=3):
    generator = np.random.default_rng(seed)

    return centre + spread * generator.standard_normal((num_examples, num_features))


def three_clinics(audit_dir=None):
    trainers = []
    for seed, (clinic_id, num_examples, centre, spread) in enumerate(CLINIC_SPECS):
        data = clinic_data(seed, num_examples, centre, spread)
        trainers.append(a_trainer(clinic_id, data, audit_dir=audit_dir))

    return trainers


def pooled_mean():
    return np.mean(
        np.concatenate(
            [
                clinic_data(seed, num_examples, centre, spread)
                for seed, (_, num_examples, centre, spread) in enumerate(CLINIC_SPECS)
            ]
        ),
        axis=0,
    )


def test_the_federated_mean_matches_the_pooled_mean(tmp_path):
    history = simulate.run_federation(three_clinics(audit_dir=tmp_path), rounds=3)

    (federated,) = history.weights

    assert np.allclose(federated, pooled_mean())


def test_the_clinic_local_parameter_never_moves(tmp_path):
    trainers = three_clinics(audit_dir=tmp_path)
    before = [trainer.local_scale for trainer in trainers]

    simulate.run_federation(trainers, rounds=3)

    for trainer, original in zip(trainers, before):
        # `shared_keys` excludes the scale, so aggregation cannot touch it.
        # This is FedBN in miniature.
        assert np.array_equal(trainer.local_scale, original)


def test_per_clinic_losses_are_recorded_separately(tmp_path):
    history = simulate.run_federation(three_clinics(audit_dir=tmp_path), rounds=3)

    # The honest picture of federating heterogeneous clinics: the global
    # parameter converges while the clinic that is furthest from it does worse
    # than it would alone. Recorded per clinic rather than averaged away.
    tight_clinic = history.eval_loss_series("clinic-b")
    assert len(tight_clinic) == 3

    local_only = np.var(clinic_data(1, 120, 5.0, 0.2), axis=0).mean()
    assert tight_clinic[-1] > local_only


def test_an_audit_line_is_written_for_every_emission(tmp_path):
    simulate.run_federation(three_clinics(audit_dir=tmp_path), rounds=3)

    for clinic_id, _, _, _ in CLINIC_SPECS:
        records = aperture_module.read_audit_log(tmp_path / f"{clinic_id}.jsonl")

        assert [record["kind"] for record in records] == ["fit", "evaluate"] * 3
        assert all(record["status"] == "emitted" for record in records)
        assert [record["round"] for record in records] == [1, 1, 2, 2, 3, 3]


def test_a_clinic_with_a_different_vocabulary_is_rejected_before_round_one(tmp_path):
    trainers = three_clinics()
    trainers[2] = a_trainer(
        "clinic-c",
        clinic_data(2, 25, -3.0, 4.0),
        manifest=a_manifest("clinic-c", structure_vocabulary_sha256="b" * 64),
    )

    with pytest.raises(simulate.ManifestMismatch) as error:
        simulate.run_federation(trainers, rounds=3)

    message = str(error.value)
    assert "clinic-c" in message
    assert "structure_vocabulary_sha256" in message


def test_a_clinic_that_resampled_to_a_different_grid_is_rejected(tmp_path):
    trainers = three_clinics()
    trainers[1] = a_trainer(
        "clinic-b",
        clinic_data(1, 120, 5.0, 0.2),
        manifest=a_manifest("clinic-b", voxel_spacing_mm=(3.0, 2.0, 2.0)),
    )

    with pytest.raises(simulate.ManifestMismatch, match="voxel_spacing_mm"):
        simulate.run_federation(trainers, rounds=3)


def test_clinics_must_agree_on_which_parameters_are_shared():
    trainers = three_clinics()

    class SharesMore(type(trainers[0])):
        def shared_keys(self):
            return ["mean", "log_scale"]

    trainers[0].__class__ = SharesMore

    with pytest.raises(simulate.ManifestMismatch, match="shared"):
        simulate.check_manifest_compatibility(trainers)


def test_duplicate_clinic_ids_are_rejected():
    trainers = three_clinics()
    trainers[1] = a_trainer("clinic-a", clinic_data(1, 120, 5.0, 0.2))

    with pytest.raises(simulate.ManifestMismatch, match="unique"):
        simulate.check_manifest_compatibility(trainers)


def test_a_leaked_debug_volume_is_stopped_by_the_aperture(tmp_path):
    trainers = three_clinics(audit_dir=tmp_path)
    trainers[0] = a_trainer(
        "clinic-a",
        clinic_data(0, 40, 0.0, 1.0),
        audit_dir=tmp_path,
        leak_debug_volume=True,
    )

    with pytest.raises(aperture_module.ApertureViolation, match="forbidden shape"):
        simulate.run_federation(trainers, rounds=1)

    (record,) = aperture_module.read_audit_log(tmp_path / "clinic-a.jsonl")
    assert record["status"] == "rejected"


def test_a_leaked_identifier_is_stopped_by_the_aperture(tmp_path):
    trainers = three_clinics(audit_dir=tmp_path)
    trainers[0] = a_trainer(
        "clinic-a",
        clinic_data(0, 40, 0.0, 1.0),
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


def test_the_server_sets_the_schedule_every_clinic_follows(tmp_path):
    seen = []

    trainers = three_clinics()

    class RecordsConfig(type(trainers[0])):
        def fit(self, config):
            seen.append(dict(config))

            return super().fit(config)

    for trainer in trainers:
        trainer.__class__ = RecordsConfig

    simulate.run_federation(
        trainers, rounds=2, config_fn=lambda round_number: {"beta": 0.1 * round_number}
    )

    # A KL warm-up has to be decided centrally or the clinics drift out of step.
    assert [config["beta"] for config in seen] == [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    assert [config["round"] for config in seen] == [1, 1, 1, 2, 2, 2]


def test_a_federation_needs_at_least_one_clinic():
    with pytest.raises(simulate.ManifestMismatch):
        simulate.check_manifest_compatibility([])


def test_the_toy_trainer_satisfies_the_protocol():
    trainer = three_clinics()[0]

    assert isinstance(trainer, protocol.ClinicTrainer)
