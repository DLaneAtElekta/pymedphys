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

"""Tests for the DICOMweb Flask routing layer.

The Mosaiq database is stubbed out by monkeypatching the QIDO-RS query
functions, so these tests exercise only the HTTP surface. They are skipped
when the optional ``flask`` dependency is unavailable.
"""

from pymedphys._imports import pytest

flask = pytest.importorskip("flask")

from pymedphys._dicomweb import qido, server  # noqa: E402


def _client(monkeypatch, studies=None):
    monkeypatch.setattr(qido, "search_for_studies", lambda *a, **k: list(studies or []))
    app = server.create_app(get_connection=lambda: object())
    app.testing = True
    return app.test_client()


def test_capabilities_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["dicomweb"]["qido-rs"]["studies"] == "implemented"


def test_search_for_studies_empty_is_204(monkeypatch):
    client = _client(monkeypatch, studies=[])
    response = client.get("/studies")
    assert response.status_code == 204


def test_search_for_studies_returns_dicom_json(monkeypatch):
    study = {"00100020": {"vr": "LO", "Value": ["MR8002"]}}
    client = _client(monkeypatch, studies=[study])
    response = client.get("/studies?PatientID=MR8002")

    assert response.status_code == 200
    assert response.content_type == server.DICOM_JSON_CONTENT_TYPE
    assert response.json == [study]


def test_series_scaffold_returns_204(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/studies/1.2.3/series")
    assert response.status_code == 204


def test_wado_retrieve_is_501(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/studies/1.2.3")
    assert response.status_code == 501


def test_stow_store_is_501(monkeypatch):
    client = _client(monkeypatch)
    response = client.post("/studies", data=b"")
    assert response.status_code == 501
