# Copyright (C) 2024 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""De-identification utilities for MCP server to protect PHI."""

import hashlib
import re
from typing import Any


# PHI fields that should be de-identified
PHI_FIELDS = {
    # Patient identifiers
    "patient_id",
    "pat_id",
    "pat_id1",
    "mrn",
    "medical_record_number",
    # Names
    "patient_name",
    "last_name",
    "first_name",
    "middle_name",
    "full_name",
    "patient_last_name",
    "patient_first_name",
    "completed_by",
    "completed_by_last",
    "completed_by_first",
    # Dates (except treatment dates which are clinically relevant)
    "birth_date",
    "dob",
    "date_of_birth",
    "ssn",
    "ssn_last4",
    "social_security",
    # Contact info
    "address",
    "phone",
    "email",
    # Other identifiers
    "study_uid",
}

# Fields that should be completely removed (not even hashed)
REMOVE_FIELDS = {
    "ssn",
    "ssn_last4",
    "social_security",
    "address",
    "phone",
    "email",
}


def create_pseudonym(value: str, salt: str = "") -> str:
    """Create a consistent pseudonym for a value using hashing.

    Parameters
    ----------
    value : str
        The value to pseudonymize
    salt : str
        Optional salt for additional security

    Returns
    -------
    str
        A pseudonymized identifier (first 8 chars of SHA-256 hash)
    """
    if not value:
        return value

    # Create hash of value + salt
    hash_input = f"{value}{salt}".encode("utf-8")
    hash_value = hashlib.sha256(hash_input).hexdigest()

    # Return first 8 characters as pseudonym
    return f"PSEUDO_{hash_value[:8].upper()}"


def deidentify_value(key: str, value: Any, salt: str = "") -> Any:
    """De-identify a single value based on its field name.

    Parameters
    ----------
    key : str
        The field name (used to determine de-identification strategy)
    value : Any
        The value to de-identify
    salt : str
        Optional salt for pseudonymization

    Returns
    -------
    Any
        De-identified value
    """
    if value is None:
        return None

    key_lower = key.lower()

    # Check if field should be completely removed
    if key_lower in REMOVE_FIELDS:
        return "[REMOVED]"

    # Check if field should be pseudonymized
    if key_lower in PHI_FIELDS:
        if isinstance(value, str):
            # Check for name patterns
            if "name" in key_lower:
                return create_pseudonym(value, salt)
            # Check for ID patterns
            elif "id" in key_lower or key_lower == "mrn":
                return create_pseudonym(value, salt)
            # Birth dates - keep year only
            elif "birth" in key_lower or "dob" in key_lower:
                # Try to extract year from date string
                year_match = re.search(r"\b(19|20)\d{2}\b", str(value))
                if year_match:
                    return f"{year_match.group()}-XX-XX"
                return "[DATE REMOVED]"
            else:
                return create_pseudonym(value, salt)
        elif isinstance(value, (int, float)):
            return create_pseudonym(str(value), salt)

    return value


def deidentify_dict(data: dict[str, Any], salt: str = "") -> dict[str, Any]:
    """De-identify all PHI fields in a dictionary.

    Parameters
    ----------
    data : dict
        Dictionary potentially containing PHI
    salt : str
        Optional salt for pseudonymization

    Returns
    -------
    dict
        De-identified dictionary
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = deidentify_dict(value, salt)
        elif isinstance(value, list):
            result[key] = deidentify_list(value, key, salt)
        else:
            result[key] = deidentify_value(key, value, salt)
    return result


def deidentify_list(data: list, parent_key: str = "", salt: str = "") -> list:
    """De-identify all PHI in a list.

    Parameters
    ----------
    data : list
        List potentially containing PHI
    parent_key : str
        Parent key name for context
    salt : str
        Optional salt for pseudonymization

    Returns
    -------
    list
        De-identified list
    """
    result = []
    for item in data:
        if isinstance(item, dict):
            result.append(deidentify_dict(item, salt))
        elif isinstance(item, list):
            result.append(deidentify_list(item, parent_key, salt))
        else:
            result.append(deidentify_value(parent_key, item, salt))
    return result


def deidentify_json_string(json_str: str, salt: str = "") -> str:
    """De-identify PHI in a JSON string.

    Parameters
    ----------
    json_str : str
        JSON string potentially containing PHI
    salt : str
        Optional salt for pseudonymization

    Returns
    -------
    str
        De-identified JSON string
    """
    import json

    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            deidentified = deidentify_dict(data, salt)
        elif isinstance(data, list):
            deidentified = deidentify_list(data, "", salt)
        else:
            return json_str
        return json.dumps(deidentified)
    except json.JSONDecodeError:
        return json_str
