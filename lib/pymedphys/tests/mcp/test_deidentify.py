# Copyright (C) 2025 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MCP de-identification utilities."""

import json

import pytest

from pymedphys._mcp.deidentify import (
    PHI_FIELDS,
    REMOVE_FIELDS,
    create_pseudonym,
    deidentify_dict,
    deidentify_json_string,
    deidentify_list,
    deidentify_value,
)


class TestCreatePseudonym:
    """Tests for create_pseudonym function."""

    def test_empty_value_returns_empty(self):
        """Empty string should return empty string."""
        assert create_pseudonym("") == ""
        assert create_pseudonym(None) is None

    def test_creates_consistent_pseudonym(self):
        """Same value should produce same pseudonym."""
        value = "PATIENT123"
        result1 = create_pseudonym(value)
        result2 = create_pseudonym(value)
        assert result1 == result2

    def test_different_values_produce_different_pseudonyms(self):
        """Different values should produce different pseudonyms."""
        result1 = create_pseudonym("PATIENT123")
        result2 = create_pseudonym("PATIENT456")
        assert result1 != result2

    def test_pseudonym_format(self):
        """Pseudonym should have expected format."""
        result = create_pseudonym("PATIENT123")
        assert result.startswith("PSEUDO_")
        assert len(result) == len("PSEUDO_") + 8  # 8 hex characters
        # Verify it's uppercase hex
        hex_part = result[7:]
        assert hex_part.isupper()
        assert all(c in "0123456789ABCDEF" for c in hex_part)

    def test_salt_changes_pseudonym(self):
        """Adding salt should change the pseudonym."""
        value = "PATIENT123"
        result_no_salt = create_pseudonym(value)
        result_with_salt = create_pseudonym(value, salt="secret_salt")
        assert result_no_salt != result_with_salt

    def test_same_salt_produces_consistent_result(self):
        """Same value and salt should produce consistent result."""
        value = "PATIENT123"
        salt = "secret_salt"
        result1 = create_pseudonym(value, salt)
        result2 = create_pseudonym(value, salt)
        assert result1 == result2


class TestDeidentifyValue:
    """Tests for deidentify_value function."""

    def test_none_value_returns_none(self):
        """None value should return None."""
        assert deidentify_value("patient_id", None) is None

    def test_remove_fields_replaced_with_removed(self):
        """Fields in REMOVE_FIELDS should be replaced with [REMOVED]."""
        for field in REMOVE_FIELDS:
            result = deidentify_value(field, "secret_value")
            assert result == "[REMOVED]"

    def test_patient_id_pseudonymized(self):
        """Patient ID should be pseudonymized."""
        result = deidentify_value("patient_id", "12345")
        assert result.startswith("PSEUDO_")

    def test_patient_name_pseudonymized(self):
        """Patient name should be pseudonymized."""
        result = deidentify_value("patient_name", "John Smith")
        assert result.startswith("PSEUDO_")

    def test_birth_date_keeps_year_only(self):
        """Birth date should keep year only."""
        result = deidentify_value("birth_date", "1985-03-15")
        assert result == "1985-XX-XX"

        result2 = deidentify_value("dob", "19850315")
        assert result2 == "1985-XX-XX"

    def test_birth_date_without_year_removed(self):
        """Birth date without recognizable year should be removed."""
        result = deidentify_value("birth_date", "invalid-date")
        assert result == "[DATE REMOVED]"

    def test_non_phi_field_unchanged(self):
        """Non-PHI fields should remain unchanged."""
        result = deidentify_value("gantry_angle", 45.0)
        assert result == 45.0

        result2 = deidentify_value("machine_name", "TrueBeam1")
        assert result2 == "TrueBeam1"

    def test_numeric_phi_pseudonymized(self):
        """Numeric PHI values should be pseudonymized."""
        result = deidentify_value("mrn", 12345)
        assert result.startswith("PSEUDO_")

    def test_case_insensitive_field_matching(self):
        """Field matching should be case insensitive."""
        result1 = deidentify_value("PATIENT_ID", "12345")
        result2 = deidentify_value("patient_id", "12345")
        assert result1.startswith("PSEUDO_")
        assert result1 == result2


class TestDeidentifyDict:
    """Tests for deidentify_dict function."""

    def test_empty_dict(self):
        """Empty dict should return empty dict."""
        assert deidentify_dict({}) == {}

    def test_phi_fields_deidentified(self):
        """PHI fields in dict should be de-identified."""
        data = {
            "patient_id": "12345",
            "patient_name": "John Smith",
            "gantry_angle": 45.0,
        }
        result = deidentify_dict(data)

        assert result["patient_id"].startswith("PSEUDO_")
        assert result["patient_name"].startswith("PSEUDO_")
        assert result["gantry_angle"] == 45.0

    def test_nested_dict_deidentified(self):
        """Nested dicts should be de-identified."""
        data = {
            "patient": {"patient_id": "12345", "first_name": "John"},
            "machine": {"machine_name": "TrueBeam1"},
        }
        result = deidentify_dict(data)

        assert result["patient"]["patient_id"].startswith("PSEUDO_")
        assert result["patient"]["first_name"].startswith("PSEUDO_")
        assert result["machine"]["machine_name"] == "TrueBeam1"

    def test_list_in_dict_deidentified(self):
        """Lists within dict should be de-identified."""
        data = {
            "patients": [{"patient_id": "123"}, {"patient_id": "456"}],
            "values": [1, 2, 3],
        }
        result = deidentify_dict(data)

        assert result["patients"][0]["patient_id"].startswith("PSEUDO_")
        assert result["patients"][1]["patient_id"].startswith("PSEUDO_")
        assert result["values"] == [1, 2, 3]

    def test_salt_propagates(self):
        """Salt should propagate to nested de-identification."""
        data = {"patient_id": "12345"}
        result_no_salt = deidentify_dict(data)
        result_with_salt = deidentify_dict(data, salt="secret")

        assert result_no_salt["patient_id"] != result_with_salt["patient_id"]


class TestDeidentifyList:
    """Tests for deidentify_list function."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert deidentify_list([]) == []

    def test_list_of_dicts(self):
        """List of dicts should have each dict de-identified."""
        data = [{"patient_id": "123"}, {"patient_id": "456"}]
        result = deidentify_list(data)

        assert len(result) == 2
        assert result[0]["patient_id"].startswith("PSEUDO_")
        assert result[1]["patient_id"].startswith("PSEUDO_")
        # Different IDs should produce different pseudonyms
        assert result[0]["patient_id"] != result[1]["patient_id"]

    def test_nested_lists(self):
        """Nested lists should be handled."""
        data = [[{"patient_id": "123"}]]
        result = deidentify_list(data)

        assert result[0][0]["patient_id"].startswith("PSEUDO_")

    def test_primitive_list_with_parent_key(self):
        """Primitive list items use parent key for de-identification."""
        # If parent key is a PHI field, items should be de-identified
        data = ["12345", "67890"]
        result = deidentify_list(data, parent_key="patient_id")

        assert result[0].startswith("PSEUDO_")
        assert result[1].startswith("PSEUDO_")


class TestDeidentifyJsonString:
    """Tests for deidentify_json_string function."""

    def test_valid_json_object(self):
        """Valid JSON object should be de-identified."""
        data = {"patient_id": "12345", "gantry_angle": 45.0}
        json_str = json.dumps(data)
        result_str = deidentify_json_string(json_str)
        result = json.loads(result_str)

        assert result["patient_id"].startswith("PSEUDO_")
        assert result["gantry_angle"] == 45.0

    def test_valid_json_array(self):
        """Valid JSON array should be de-identified."""
        data = [{"patient_id": "123"}, {"patient_id": "456"}]
        json_str = json.dumps(data)
        result_str = deidentify_json_string(json_str)
        result = json.loads(result_str)

        assert len(result) == 2
        assert result[0]["patient_id"].startswith("PSEUDO_")

    def test_invalid_json_returns_original(self):
        """Invalid JSON should return original string."""
        invalid_json = "not valid json {"
        result = deidentify_json_string(invalid_json)
        assert result == invalid_json

    def test_primitive_json_returns_original(self):
        """Primitive JSON values should return original string."""
        result = deidentify_json_string('"just a string"')
        assert result == '"just a string"'

    def test_salt_applied(self):
        """Salt should be applied to de-identification."""
        data = {"patient_id": "12345"}
        json_str = json.dumps(data)

        result_no_salt = json.loads(deidentify_json_string(json_str))
        result_with_salt = json.loads(deidentify_json_string(json_str, salt="secret"))

        assert result_no_salt["patient_id"] != result_with_salt["patient_id"]


class TestPhiFieldsConfiguration:
    """Tests for PHI fields configuration."""

    def test_expected_phi_fields_present(self):
        """Expected PHI fields should be in the configuration."""
        expected_fields = {
            "patient_id",
            "patient_name",
            "birth_date",
            "last_name",
            "first_name",
            "mrn",
        }
        assert expected_fields.issubset(PHI_FIELDS)

    def test_expected_remove_fields_present(self):
        """Expected remove fields should be in the configuration."""
        expected_remove = {"ssn", "address", "phone", "email"}
        assert expected_remove.issubset(REMOVE_FIELDS)

    def test_remove_fields_subset_of_phi_fields(self):
        """Remove fields should be a subset of PHI fields."""
        assert REMOVE_FIELDS.issubset(PHI_FIELDS)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_deeply_nested_structure(self):
        """Deeply nested structures should be handled."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "patient_id": "12345",
                    }
                }
            }
        }
        result = deidentify_dict(data)
        assert result["level1"]["level2"]["level3"]["patient_id"].startswith("PSEUDO_")

    def test_mixed_types_in_list(self):
        """Lists with mixed types should be handled."""
        data = [
            {"patient_id": "123"},
            "string_value",
            42,
            None,
            [{"patient_id": "456"}],
        ]
        result = deidentify_list(data)

        assert result[0]["patient_id"].startswith("PSEUDO_")
        assert result[1] == "string_value"
        assert result[2] == 42
        assert result[3] is None
        assert result[4][0]["patient_id"].startswith("PSEUDO_")

    def test_unicode_values(self):
        """Unicode values should be handled."""
        data = {"patient_name": "José García"}
        result = deidentify_dict(data)
        assert result["patient_name"].startswith("PSEUDO_")

    def test_special_characters_in_values(self):
        """Special characters in values should be handled."""
        data = {"patient_id": "ID-123/456\\789"}
        result = deidentify_dict(data)
        assert result["patient_id"].startswith("PSEUDO_")

    def test_large_numeric_id(self):
        """Large numeric IDs should be handled."""
        data = {"mrn": 999999999999999}
        result = deidentify_dict(data)
        assert result["mrn"].startswith("PSEUDO_")
