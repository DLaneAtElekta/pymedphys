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

"""Unit tests for the QIDO-RS -> SQL translation (no database required)."""

from pymedphys._dicomweb import qido


def test_wildcard_detection():
    assert qido._matches_wildcard("HOW*")
    assert qido._matches_wildcard("HOW?RD")
    assert not qido._matches_wildcard("HOWARD")


def test_qido_wildcard_to_sql_like():
    assert qido._qido_wildcard_to_sql_like("HOW*") == "HOW%"
    assert qido._qido_wildcard_to_sql_like("HOW?RD") == "HOW_RD"
    # Literal SQL wildcards are escaped so they match literally.
    assert qido._qido_wildcard_to_sql_like("50%*") == "50\\%%"


def test_build_study_where_exact_patient_id():
    clauses, parameters = qido._build_study_where({"PatientID": "MR8002"})
    assert clauses == ["Ident.IDA = %(patient_id)s"]
    assert parameters == {"patient_id": "MR8002"}


def test_build_study_where_wildcard_patient_id():
    clauses, parameters = qido._build_study_where({"PatientID": "MR*"})
    assert "LIKE" in clauses[0]
    assert parameters["patient_id"] == "MR%"


def test_build_study_where_patient_name_uses_last_name_component():
    clauses, parameters = qido._build_study_where({"PatientName": "HOWARD*^Moe"})
    assert "Patient.Last_Name LIKE" in clauses[0]
    assert parameters["patient_name"] == "HOWARD%"


def test_parse_single_date_range():
    start, end = qido._parse_date_range("20260102")
    assert start == "2026-01-02 00:00:00"
    assert end == "2026-01-02 23:59:59"


def test_parse_open_ended_date_range():
    start, end = qido._parse_date_range("20260101-")
    assert start == "2026-01-01 00:00:00"
    assert end is None

    start, end = qido._parse_date_range("-20260101")
    assert start is None
    assert end == "2026-01-01 23:59:59"


def test_no_filters_produce_no_clauses():
    clauses, parameters = qido._build_study_where({})
    assert clauses == []
    assert parameters == {}
