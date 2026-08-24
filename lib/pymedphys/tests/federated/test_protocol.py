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

import pytest

from pymedphys._federated import protocol

VOCABULARY_HASH = "a" * 64
OTHER_HASH = "b" * 64


def a_manifest(**overrides):
    fields = {
        "clinic_id": "clinic-a",
        "grid_shape": (64, 64, 64),
        "voxel_spacing_mm": (2.0, 2.0, 2.0),
        "structure_keys": ("Brainstem", "Parotid_L", "Parotid_R"),
        "structure_vocabulary_sha256": VOCABULARY_HASH,
    }
    fields.update(overrides)

    return protocol.ClinicManifest(**fields)


def test_clinic_id_and_notes_do_not_affect_the_compatibility_key():
    # Clinics are meant to differ here; they are not meant to differ anywhere
    # else in the manifest.
    first = a_manifest(clinic_id="clinic-a", notes="Scanner A")
    second = a_manifest(clinic_id="clinic-b", notes="Scanner B")

    assert first.compatibility_key == second.compatibility_key


def test_the_local_alias_hash_does_not_affect_the_compatibility_key():
    first = a_manifest(alias_table_sha256=VOCABULARY_HASH)
    second = a_manifest(alias_table_sha256=OTHER_HASH)

    assert first.compatibility_key == second.compatibility_key


@pytest.mark.parametrize(
    "overrides",
    [
        {"grid_shape": (128, 128, 128)},
        {"voxel_spacing_mm": (3.0, 2.0, 2.0)},
        {"structure_keys": ("Brainstem", "Parotid_L")},
        {"structure_vocabulary_sha256": OTHER_HASH},
    ],
)
def test_every_declared_representation_field_changes_the_key(overrides):
    assert a_manifest().compatibility_key != a_manifest(**overrides).compatibility_key


def test_structure_keys_must_be_sorted():
    with pytest.raises(protocol.ManifestError):
        a_manifest(structure_keys=("Parotid_L", "Brainstem"))


def test_structure_keys_must_be_unique():
    with pytest.raises(protocol.ManifestError):
        a_manifest(structure_keys=("Brainstem", "Brainstem"))


def test_spacing_must_match_the_grid():
    with pytest.raises(protocol.ManifestError):
        a_manifest(voxel_spacing_mm=(2.0, 2.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"grid_shape": (64, 0, 64)},
        {"voxel_spacing_mm": (2.0, -1.0, 2.0)},
        {"clinic_id": ""},
        {"structure_vocabulary_sha256": "not-a-hash"},
        {"structure_vocabulary_sha256": "z" * 64},
    ],
)
def test_an_incoherent_manifest_is_rejected(overrides):
    with pytest.raises(protocol.ManifestError):
        a_manifest(**overrides)


def test_to_dict_is_json_friendly_and_carries_the_key():
    record = a_manifest().to_dict()

    assert record["grid_shape"] == [64, 64, 64]
    assert record["compatibility_key"] == a_manifest().compatibility_key


def test_results_normalise_their_containers():
    fit = protocol.FitResult(weights=[1, 2], num_examples=10)
    evaluation = protocol.EvalResult(loss=1, num_examples=10)

    assert isinstance(fit.weights, tuple)
    assert fit.metrics == {}
    assert isinstance(evaluation.loss, float)
