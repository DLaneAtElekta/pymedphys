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

"""Unit tests for the Mosaiq -> DICOM JSON mapping.

These tests exercise the pure translation layer and require neither a
Mosaiq database nor the ``flask`` optional dependency.
"""

from pymedphys._dicomweb import mapping


def test_derive_uid_is_deterministic():
    first = mapping.derive_uid("study", 1234, 5678)
    second = mapping.derive_uid("study", 1234, 5678)
    different = mapping.derive_uid("study", 1234, 9999)

    assert first == second
    assert first != different
    assert first.startswith(mapping.UID_ROOT)
    # DICOM UIDs must be 64 characters or fewer.
    assert len(first) <= 64


def test_to_dicom_date_from_mosaiq_datetime_string():
    assert mapping._to_dicom_date("1970-08-15 00:00:00") == "19700815"
    assert mapping._to_dicom_date("1970-08-15 00:00:00.123456") == "19700815"
    assert mapping._to_dicom_date("1970-08-15") == "19700815"


def test_to_dicom_date_handles_missing():
    assert mapping._to_dicom_date(None) is None
    assert mapping._to_dicom_date("") is None
    assert mapping._to_dicom_date("not-a-date") is None


def test_study_dataset_maps_to_dicom_json():
    row = {
        "patient_id": "MR8002",
        "last_name": "HOWARD",
        "first_name": "Moe",
        "birth_date": "1970-08-15 00:00:00",
        "pat_id1": 42,
        "sit_set_id": 7,
        "study_date": "2026-01-02 09:30:00",
        "study_description": "Pelvis",
        "modality": "RTPLAN",
    }

    dataset = mapping.study_dataset(row)
    json_dict = dataset.to_json_dict()

    # PatientID (0010,0020)
    assert json_dict["00100020"]["Value"] == ["MR8002"]
    # PatientName (0010,0010) is encoded as a PersonName component group.
    assert json_dict["00100010"]["Value"][0]["Alphabetic"] == "HOWARD^Moe"
    # PatientBirthDate (0010,0030)
    assert json_dict["00100030"]["Value"] == ["19700815"]
    # StudyDate (0008,0020)
    assert json_dict["00080020"]["Value"] == ["20260102"]
    # StudyDescription (0008,1030)
    assert json_dict["00081030"]["Value"] == ["Pelvis"]
    # ModalitiesInStudy (0008,0061)
    assert json_dict["00080061"]["Value"] == ["RTPLAN"]
    # StudyInstanceUID (0020,000D) is deterministic and rooted.
    assert json_dict["0020000D"]["Value"][0].startswith(mapping.UID_ROOT)


def test_study_dataset_stable_uid_across_calls():
    row = {"pat_id1": 42, "sit_set_id": 7}
    first = mapping.study_dataset(row).StudyInstanceUID
    second = mapping.study_dataset(row).StudyInstanceUID
    assert first == second
