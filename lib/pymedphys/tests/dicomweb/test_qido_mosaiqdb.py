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

"""QIDO-RS study search against the mock Mosaiq database.

These tests require the Mosaiq mock SQL Server instance and so carry the
``mosaiqdb`` marker (they are excluded from the default test run).
"""

from pymedphys._imports import pytest

import pymedphys
from pymedphys._dicomweb import qido
from pymedphys._mosaiq.mock import generate, utilities


@pytest.fixture(name="connection")
def fixture_check_create_test_db() -> "pymedphys.mosaiq.Connection":
    generate.create_test_db()
    return utilities.connect()


@pytest.mark.mosaiqdb
def test_search_for_studies_returns_patient_studies(connection):
    mock_patient_ident_df = generate.create_mock_patients()
    generate.create_mock_treatment_sites(mock_patient_ident_df)

    results = qido.search_for_studies(connection)

    # Every result is a DICOM JSON object carrying a PatientID (0010,0020).
    assert results
    for study in results:
        assert "00100020" in study
        assert "0020000D" in study  # StudyInstanceUID


@pytest.mark.mosaiqdb
def test_search_for_studies_filters_by_patient_id(connection):
    mock_patient_ident_df = generate.create_mock_patients()
    generate.create_mock_treatment_sites(mock_patient_ident_df)

    results = qido.search_for_studies(connection, {"PatientID": "MR8002"})

    assert results
    for study in results:
        assert study["00100020"]["Value"] == ["MR8002"]
