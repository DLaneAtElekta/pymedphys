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

# pylint: disable=redefined-outer-name

import datetime
import email.parser
import io
import json
import re
import sqlite3
import threading
import urllib.request
from wsgiref.simple_server import make_server
from wsgiref.util import setup_testing_defaults

from pymedphys._imports import pydicom, pytest

from pymedphys._dicom.web.repository import MosaiqDicomRepository
from pymedphys._dicom.web.schema import MosaiqDicomSchema
from pymedphys._dicom.web.server import create_app

STUDY_UID = "1.2.826.0.1.3680043.8.498.1"
SERIES_UID = "1.2.826.0.1.3680043.8.498.1.1"
SOP_UID = "1.2.826.0.1.3680043.8.498.1.1.1"
RT_PLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"


def _dicom_bytes() -> bytes:
    ds = pydicom.Dataset()
    ds.PatientID = "000123"
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.SOPInstanceUID = SOP_UID
    ds.SOPClassUID = RT_PLAN_STORAGE
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = RT_PLAN_STORAGE
    ds.file_meta.MediaStorageSOPInstanceUID = SOP_UID

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, enforce_file_format=True)

    return buffer.getvalue()


@pytest.fixture
def repository():
    """A SQLite stand in for the Mosaiq tables described by the default schema."""
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(
        """
        CREATE TABLE Ident (Pat_ID1 INTEGER, IDA TEXT);
        CREATE TABLE Patient (Pat_ID1 INTEGER, Last_Name TEXT, First_Name TEXT);
        CREATE TABLE DCMStudy (
            DCMStudy_ID INTEGER, Pat_ID1 INTEGER, Study_Instance_UID TEXT,
            Study_DtTm TEXT, Study_Description TEXT, Accession_Number TEXT
        );
        CREATE TABLE DCMSeries (
            DCMSeries_ID INTEGER, DCMStudy_ID INTEGER, Series_Instance_UID TEXT,
            Modality TEXT, Series_Number INTEGER, Series_Description TEXT
        );
        CREATE TABLE DCMInstance (
            DCMSeries_ID INTEGER, SOP_Instance_UID TEXT, SOP_Class_UID TEXT,
            Instance_Number INTEGER, DICOM_Object BLOB
        );

        INSERT INTO Ident VALUES (1, '000123'), (2, '000456');
        INSERT INTO Patient VALUES (1, 'PHANTOM', 'PELVIS'), (2, 'PHANTOM', 'RW3');
        INSERT INTO DCMStudy VALUES
            (10, 1, '1.2.826.0.1.3680043.8.498.1',
             '2026-03-04 13:30:00', 'Pelvis plan', 'ACC1'),
            (20, 2, '1.2.826.0.1.3680043.8.498.2',
             '2026-05-06 09:00:00', NULL, NULL);
        INSERT INTO DCMSeries VALUES
            (100, 10, '1.2.826.0.1.3680043.8.498.1.1', 'RTPLAN', 1, 'Plan');
        """
    )
    connection.execute(
        "INSERT INTO DCMInstance VALUES (100, ?, ?, 1, ?)",
        (SOP_UID, RT_PLAN_STORAGE, _dicom_bytes()),
    )

    def execute(query: str, parameters: dict) -> list[tuple]:
        # Convert pymssql style parameters to their SQLite equivalent
        query = re.sub(r"%\((\w+)\)s", r":\1", query)
        parameters = {
            key: value.isoformat(" ") if isinstance(value, datetime.datetime) else value
            for key, value in parameters.items()
        }
        return connection.execute(query, parameters).fetchall()

    yield MosaiqDicomRepository(execute)

    connection.close()


def _get(app, path, query_string=""):
    environ = {"PATH_INFO": path, "QUERY_STRING": query_string}
    setup_testing_defaults(environ)

    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    body = b"".join(app(environ, start_response))

    return int(captured["status"].split()[0]), captured["headers"], body


def _values(results, tag):
    return [result[tag].get("Value", [None])[0] for result in results]


def test_search_all_studies(repository):
    status, headers, body = _get(create_app(repository), "/studies")

    assert status == 200
    assert headers["Content-Type"] == "application/dicom+json"

    studies = json.loads(body)
    # Most recent first
    assert _values(studies, "0020000D") == [
        "1.2.826.0.1.3680043.8.498.2",
        STUDY_UID,
    ]

    study = studies[1]
    assert study["00100020"] == {"vr": "LO", "Value": ["000123"]}
    assert study["00100010"] == {
        "vr": "PN",
        "Value": [{"Alphabetic": "PHANTOM^PELVIS"}],
    }
    assert study["00080020"] == {"vr": "DA", "Value": ["20260304"]}
    assert study["00080050"] == {"vr": "SH", "Value": ["ACC1"]}

    # Missing values are present but empty
    assert studies[0]["00080050"] == {"vr": "SH"}


@pytest.mark.parametrize(
    "query_string, expected",
    [
        ("PatientID=000123", [STUDY_UID]),
        ("PatientID=000*", ["1.2.826.0.1.3680043.8.498.2", STUDY_UID]),
        ("PatientName=PHAN*", ["1.2.826.0.1.3680043.8.498.2", STUDY_UID]),
        ("StudyInstanceUID=" + STUDY_UID, [STUDY_UID]),
        ("AccessionNumber=ACC1", [STUDY_UID]),
        ("StudyDate=20260304", [STUDY_UID]),
        ("StudyDate=20260401-", ["1.2.826.0.1.3680043.8.498.2"]),
        ("StudyDate=-20260401", [STUDY_UID]),
        ("PatientID=000*&limit=1&offset=1", [STUDY_UID]),
    ],
)
def test_study_filters(repository, query_string, expected):
    _, _, body = _get(create_app(repository), "/studies", query_string)

    assert _values(json.loads(body), "0020000D") == expected


def test_no_matches_is_no_content(repository):
    status, _, body = _get(create_app(repository), "/studies", "PatientID=nobody")

    assert status == 204
    assert body == b""


def test_wildcard_escapes_like_characters(repository):
    status, _, _ = _get(create_app(repository), "/studies", "PatientID=%25*")

    assert status == 204


def test_search_series_and_instances(repository):
    app = create_app(repository)

    _, _, body = _get(app, f"/studies/{STUDY_UID}/series")
    series = json.loads(body)
    assert _values(series, "0020000E") == [SERIES_UID]
    assert _values(series, "00080060") == ["RTPLAN"]

    _, _, body = _get(app, f"/studies/{STUDY_UID}/series/{SERIES_UID}/instances")
    instances = json.loads(body)
    assert _values(instances, "00080018") == [SOP_UID]
    assert _values(instances, "00080016") == [RT_PLAN_STORAGE]


def test_retrieve_instance(repository):
    status, headers, body = _get(
        create_app(repository),
        f"/studies/{STUDY_UID}/series/{SERIES_UID}/instances/{SOP_UID}",
    )

    assert status == 200
    assert headers["Content-Type"].startswith(
        'multipart/related; type="application/dicom"'
    )

    message = email.parser.BytesParser().parsebytes(
        b"Content-Type: " + headers["Content-Type"].encode() + b"\r\n\r\n" + body
    )
    (part,) = message.get_payload()
    assert part.get_content_type() == "application/dicom"

    ds = pydicom.dcmread(io.BytesIO(part.get_payload(decode=True)))
    assert ds.SOPInstanceUID == SOP_UID
    assert ds.PatientID == "000123"


@pytest.mark.parametrize(
    "path, query_string, expected_status",
    [
        (f"/studies/{STUDY_UID}/series/{SERIES_UID}/instances/1.2.3", "", 404),
        ("/unknown", "", 404),
        ("/studies", "limit=abc", 400),
        ("/studies", "offset=-1", 400),
    ],
)
def test_errors(repository, path, query_string, expected_status):
    status, _, _ = _get(create_app(repository), path, query_string)

    assert status == expected_status


def test_only_get_is_allowed(repository):
    environ = {"PATH_INFO": "/studies", "REQUEST_METHOD": "POST"}
    setup_testing_defaults(environ)
    statuses = []

    create_app(repository)(environ, lambda status, headers: statuses.append(status))

    assert statuses == ["405 Method Not Allowed"]


def test_served_over_http(repository):
    with make_server("127.0.0.1", 0, create_app(repository)) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}/studies?PatientID=000123"
            with urllib.request.urlopen(url) as response:
                studies = json.loads(response.read())
        finally:
            httpd.shutdown()

    assert _values(studies, "0020000D") == [STUDY_UID]


def test_schema_rejects_unsafe_identifiers():
    with pytest.raises(ValueError):
        MosaiqDicomSchema(study_table="DCMStudy; DROP TABLE Patient")

    with pytest.raises(ValueError):
        MosaiqDicomSchema.from_mapping({"not_a_field": "x"})


def test_schema_from_toml(tmp_path):
    path = tmp_path / "schema.toml"
    path.write_text('study_table = "Study"\ninstance_data = "Object_Blob"\n')

    schema = MosaiqDicomSchema.from_toml(path)

    assert schema.study_table == "Study"
    assert schema.instance_data == "Object_Blob"
    assert schema.series_table == "DCMSeries"
