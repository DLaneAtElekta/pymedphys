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

import datetime

import numpy as np
import pytest

from pymedphys._federated import aperture as aperture_module
from pymedphys._federated import protocol

GRID = (8, 8, 8)


def a_manifest():
    return protocol.ClinicManifest(
        clinic_id="clinic-a",
        grid_shape=GRID,
        voxel_spacing_mm=(2.0, 2.0, 2.0),
        structure_keys=("Brainstem", "Parotid_L"),
        structure_vocabulary_sha256="a" * 64,
    )


def a_policy(**overrides):
    fields = {
        "max_bytes_per_round": 1_000_000,
        "forbidden_shapes": (GRID,),
        "allowed_metric_keys": frozenset({"train_loss"}),
        "min_examples": 5,
    }
    fields.update(overrides)

    return aperture_module.AperturePolicy(**fields)


def an_aperture(policy=None, path=None):
    return aperture_module.Aperture(
        policy=policy if policy is not None else a_policy(),
        clinic_id="clinic-a",
        audit_log_path=path,
        clock=lambda: datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def test_a_conforming_payload_passes_through():
    aperture = an_aperture()

    result = aperture.emit(
        weights=[np.ones(4)], num_examples=20, metrics={"train_loss": 0.5}
    )

    assert result.num_examples == 20
    assert result.metrics == {"train_loss": 0.5}
    assert aperture.records[0]["status"] == "emitted"


def test_a_voxel_shaped_array_is_rejected():
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="forbidden shape"):
        aperture.emit(weights=[np.ones(4), np.zeros(GRID)], num_examples=20)


def test_a_batch_of_voxel_shaped_arrays_is_rejected():
    # The trailing axes are what give a debugging tensor away, whatever it is
    # stacked into.
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="forbidden shape"):
        aperture.emit(weights=[np.zeros((3,) + GRID)], num_examples=20)


def test_a_metric_key_that_is_not_on_the_whitelist_is_rejected():
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="whitelist"):
        aperture.emit(
            weights=[np.ones(4)],
            num_examples=20,
            metrics={"train_loss": 0.5, "patient_mrn": 123},
        )


def test_a_whitelisted_key_may_not_smuggle_a_non_scalar():
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="only real scalars"):
        aperture.emit(
            weights=[np.ones(4)], num_examples=20, metrics={"train_loss": "SMITH^JOHN"}
        )


def test_a_statistic_over_too_few_patients_is_rejected():
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="at least 5"):
        aperture.emit(weights=[np.ones(4)], num_examples=1)


def test_the_size_cap_is_hard():
    aperture = an_aperture(policy=a_policy(max_bytes_per_round=100))

    with pytest.raises(aperture_module.ApertureViolation, match="caps a round"):
        aperture.emit(weights=[np.ones(1000)], num_examples=20)


def test_the_array_count_cap_is_enforced():
    aperture = an_aperture(policy=a_policy(max_arrays=2))

    with pytest.raises(aperture_module.ApertureViolation, match="at most 2"):
        aperture.emit(weights=[np.ones(2)] * 3, num_examples=20)


def test_non_finite_values_do_not_leave_by_default():
    aperture = an_aperture()

    with pytest.raises(aperture_module.ApertureViolation, match="non-finite"):
        aperture.emit(weights=[np.array([np.nan, 1.0])], num_examples=20)

    with pytest.raises(aperture_module.ApertureViolation, match="not finite"):
        aperture.emit(
            weights=[np.ones(2)], num_examples=20, metrics={"train_loss": np.inf}
        )


def test_evaluation_goes_through_the_same_aperture():
    aperture = an_aperture()

    result = aperture.emit_evaluation(loss=0.25, num_examples=20)

    assert result.loss == 0.25
    assert aperture.records[0]["kind"] == "evaluate"

    with pytest.raises(aperture_module.ApertureViolation, match="at least 5"):
        aperture.emit_evaluation(loss=0.25, num_examples=2)


def test_a_rejection_is_recorded_rather_than_silently_dropped(tmp_path):
    path = tmp_path / "audit" / "clinic-a.jsonl"
    aperture = an_aperture(path=path)

    with pytest.raises(aperture_module.ApertureViolation):
        aperture.emit(weights=[np.zeros(GRID)], num_examples=20, round_number=1)

    aperture.emit(weights=[np.ones(4)], num_examples=20, round_number=2)

    records = aperture_module.read_audit_log(path)

    assert [record["status"] for record in records] == ["rejected", "emitted"]
    assert "forbidden shape" in records[0]["reason"]


def test_the_audit_log_answers_what_left_and_how_big(tmp_path):
    path = tmp_path / "clinic-a.jsonl"
    aperture = an_aperture(path=path)

    aperture.emit(
        weights=[np.ones(4)],
        num_examples=20,
        metrics={"train_loss": 0.5},
        round_number=3,
    )

    (record,) = aperture_module.read_audit_log(path)

    assert record["clinic_id"] == "clinic-a"
    assert record["round"] == 3
    assert record["array_count"] == 1
    assert record["example_count"] == 20
    assert record["metric_keys"] == ["train_loss"]
    assert record["byte_count"] == 4 * 8 + len('{"train_loss":0.5}')
    assert record["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert len(record["payload_sha256"]) == 64


def test_the_audit_log_is_appended_to_not_rewritten(tmp_path):
    path = tmp_path / "clinic-a.jsonl"

    for round_number in range(1, 4):
        aperture = an_aperture(path=path)
        aperture.emit(weights=[np.ones(4)], num_examples=20, round_number=round_number)

    assert [record["round"] for record in aperture_module.read_audit_log(path)] == [
        1,
        2,
        3,
    ]


def test_the_payload_digest_covers_the_payload_and_nothing_else():
    weights = [np.ones(4)]
    metrics = {"train_loss": 0.5}

    baseline = aperture_module.payload_digest(weights, metrics, 20)

    assert baseline == aperture_module.payload_digest([np.ones(4)], dict(metrics), 20)
    assert baseline != aperture_module.payload_digest(weights, metrics, 21)
    assert baseline != aperture_module.payload_digest(weights, {"train_loss": 0.6}, 20)
    assert baseline != aperture_module.payload_digest([np.ones(5)], metrics, 20)
    assert baseline != aperture_module.payload_digest(
        [np.ones(4, dtype=np.float32)], metrics, 20
    )


def test_a_policy_derived_from_a_manifest_forbids_that_clinic_s_grid():
    policy = aperture_module.policy_from_manifest(
        a_manifest(), max_bytes_per_round=1000, allowed_metric_keys=["train_loss"]
    )

    assert GRID in policy.forbidden_shapes
    # A one-hot stack of the clinic's own structures is voxel data too.
    assert (2,) + GRID in policy.forbidden_shapes


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_bytes_per_round": 0},
        {"min_examples": 0},
        {"max_arrays": 0},
    ],
)
def test_an_incoherent_policy_is_rejected(overrides):
    with pytest.raises(ValueError):
        a_policy(**overrides)
