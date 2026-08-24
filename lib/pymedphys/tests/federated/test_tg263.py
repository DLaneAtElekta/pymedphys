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

from pymedphys._dicom.structure import tg263

VOCABULARY = """
[vocabulary]
version = "2026-01"
structures = ["Brainstem", "Parotid_L", "Parotid_R", "SpinalCord"]
"""

CLINIC_A_ALIASES = """
[aliases]
Parotid_L = ["Lt_Parotid_gland"]
SpinalCord = ["Cord"]
"""

CLINIC_B_ALIASES = """
[aliases]
Parotid_R = ["RT PAROTID GLAND"]
Brainstem = ["Brain Stem"]
"""


def _write(tmp_path, name, contents):
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")

    return path


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Parotid_L", "PAROTID_L"),
        ("L Parotid", "PAROTID_L"),
        ("PAROTID LT", "PAROTID_L"),
        ("parotid-left", "PAROTID_L"),
        ("Rt. Parotid", "PAROTID_R"),
        ("Brainstem", "BRAINSTEM"),
        ("Brain Stem", "BRAINSTEM"),
        ("Spinal_Cord", "SPINALCORD"),
    ],
)
def test_normalise_resolves_case_punctuation_and_laterality(name, expected):
    assert tg263.normalise(name) == expected


def test_normalise_rejects_a_name_with_no_content():
    with pytest.raises(tg263.MappingError):
        tg263.normalise("___")


def test_laterality_only_names_are_left_alone():
    # A structure genuinely called "Left" should not become an empty stem.
    assert tg263.normalise("Left") == "LEFT"


def test_vocabulary_hash_is_insensitive_to_declaration_order():
    first = tg263.StructureVocabulary(version="1", structures=("A", "B", "C"))
    second = tg263.StructureVocabulary(version="1", structures=("C", "A", "B"))

    assert first.sha256 == second.sha256


def test_vocabulary_hash_changes_with_version():
    first = tg263.StructureVocabulary(version="1", structures=("A", "B"))
    second = tg263.StructureVocabulary(version="2", structures=("A", "B"))

    assert first.sha256 != second.sha256


def test_vocabulary_rejects_names_that_collide_when_normalised():
    with pytest.raises(tg263.MappingError):
        tg263.StructureVocabulary(version="1", structures=("Parotid_L", "L Parotid"))


def test_two_clinics_share_a_vocabulary_hash_but_not_an_alias_hash(tmp_path):
    clinic_a = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )
    clinic_b = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "b.toml", VOCABULARY + CLINIC_B_ALIASES)
    )

    # This is the whole point of splitting the file: local names differ, the
    # shared target list does not.
    assert clinic_a.vocabulary_sha256 == clinic_b.vocabulary_sha256
    assert clinic_a.alias_table_sha256 != clinic_b.alias_table_sha256


def test_vocabulary_hash_is_insensitive_to_file_formatting(tmp_path):
    reformatted = VOCABULARY.replace(", ", ",\n    ") + "\n\n" + CLINIC_A_ALIASES

    original = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )
    formatted = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "b.toml", reformatted)
    )

    assert original.vocabulary_sha256 == formatted.vocabulary_sha256
    assert original.alias_table_sha256 == formatted.alias_table_sha256


def test_canonicalise_uses_the_vocabulary_then_the_aliases(tmp_path):
    mapping = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )

    assert mapping.canonicalise("PAROTID LT") == "Parotid_L"
    assert mapping.canonicalise("Lt_Parotid_gland") == "Parotid_L"
    assert mapping.canonicalise("cord") == "SpinalCord"


def test_an_unmapped_structure_is_an_error_not_a_silent_drop(tmp_path):
    mapping = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )

    with pytest.raises(tg263.UnmappedStructure):
        mapping.canonicalise("Optic Chiasm")


def test_canonicalise_all_reports_every_failure_at_once(tmp_path):
    mapping = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )

    with pytest.raises(tg263.UnmappedStructure) as error:
        mapping.canonicalise_all(["Brainstem", "Optic Chiasm", "Mandible"])

    assert "Mandible" in str(error.value)
    assert "Optic Chiasm" in str(error.value)


def test_an_alias_may_not_target_two_canonical_names():
    vocabulary = tg263.StructureVocabulary(
        version="1", structures=("Parotid_L", "Parotid_R")
    )

    with pytest.raises(tg263.MappingError):
        tg263.StructureMapping(
            vocabulary=vocabulary,
            aliases={"Parotid_L": ("Gland",), "Parotid_R": ("gland",)},
        )


def test_an_alias_must_target_the_vocabulary():
    vocabulary = tg263.StructureVocabulary(version="1", structures=("Brainstem",))

    with pytest.raises(tg263.MappingError):
        tg263.StructureMapping(vocabulary=vocabulary, aliases={"Mandible": ("Jaw",)})


def test_manifest_fields_are_ready_for_a_clinic_manifest(tmp_path):
    mapping = tg263.StructureMapping.from_toml_file(
        _write(tmp_path, "a.toml", VOCABULARY + CLINIC_A_ALIASES)
    )

    fields = mapping.manifest_fields()

    assert fields["structure_keys"] == (
        "Brainstem",
        "Parotid_L",
        "Parotid_R",
        "SpinalCord",
    )
    assert len(fields["structure_vocabulary_sha256"]) == 64
