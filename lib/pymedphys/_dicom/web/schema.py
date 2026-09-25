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

"""Table and column names used to read DICOM objects out of Mosaiq.

The defaults below describe the assumed layout of Mosaiq's DICOM
tables. They have not been verified against every Mosaiq version, so
every name can be overridden, either in Python or from a TOML file
passed to ``pymedphys dicom web --schema``.
"""

import dataclasses
import re

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclasses.dataclass(frozen=True)
class MosaiqDicomSchema:
    """Where the DICOM study / series / instance hierarchy lives in Mosaiq.

    Patient demographics are read from the standard Mosaiq ``Ident``
    (``IDA`` is the patient ID / MRN) and ``Patient`` tables, joined on
    ``Pat_ID1``.
    """

    # Study level
    study_table: str = "DCMStudy"
    study_key: str = "DCMStudy_ID"
    study_patient_key: str = "Pat_ID1"
    study_instance_uid: str = "Study_Instance_UID"
    study_date: str = "Study_DtTm"
    study_description: str = "Study_Description"
    accession_number: str = "Accession_Number"

    # Series level
    series_table: str = "DCMSeries"
    series_key: str = "DCMSeries_ID"
    series_study_key: str = "DCMStudy_ID"
    series_instance_uid: str = "Series_Instance_UID"
    modality: str = "Modality"
    series_number: str = "Series_Number"
    series_description: str = "Series_Description"

    # Instance level
    instance_table: str = "DCMInstance"
    instance_series_key: str = "DCMSeries_ID"
    sop_instance_uid: str = "SOP_Instance_UID"
    sop_class_uid: str = "SOP_Class_UID"
    instance_number: str = "Instance_Number"
    instance_data: str = "DICOM_Object"
    """Column holding the raw DICOM Part 10 file bytes."""

    # Patient demographics (standard Mosaiq tables)
    ident_table: str = "Ident"
    patient_table: str = "Patient"
    patient_key: str = "Pat_ID1"
    patient_id: str = "IDA"
    patient_last_name: str = "Last_Name"
    patient_first_name: str = "First_Name"

    def __post_init__(self):
        # Table and column names cannot be passed as SQL parameters, so
        # they are validated here before being placed within queries.
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not _IDENTIFIER.match(value):
                raise ValueError(f"Invalid SQL identifier for {field.name}: {value!r}")

    @classmethod
    def from_mapping(cls, mapping: dict) -> "MosaiqDicomSchema":
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown schema keys: {sorted(unknown)}")

        return cls(**mapping)

    @classmethod
    def from_toml(cls, path) -> "MosaiqDicomSchema":
        from pymedphys._imports import toml

        with open(path, encoding="utf-8") as f:
            return cls.from_mapping(toml.load(f))
