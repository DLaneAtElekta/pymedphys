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

"""Flask application exposing Mosaiq over DICOMweb.

The application is created with :func:`create_app`, which is given a
``get_connection`` callable returning an open Mosaiq connection (see
:func:`pymedphys.mosaiq.connect`). Routes implement the DICOMweb URI
templates from PS3.18:

======================================  =========  ==================================
Route                                   Service    Status
======================================  =========  ==================================
``GET  /studies``                       QIDO-RS    Implemented (Mosaiq study search)
``GET  /studies/<uid>/series``          QIDO-RS    Scaffold (returns no matches)
``GET  /studies/<uid>/instances``       QIDO-RS    Scaffold (returns no matches)
``GET  /studies/<uid>/series/<uid>/instances``  QIDO-RS  Scaffold
``GET  /studies/<uid>``                 WADO-RS    Scaffold (501 Not Implemented)
``POST /studies``                       STOW-RS    Scaffold (501 Not Implemented)
======================================  =========  ==================================
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from pymedphys._imports import flask

from pymedphys._mosaiq import api as _mosaiq_api

from . import qido, stow, wado

DICOM_JSON_CONTENT_TYPE = "application/dicom+json"

# Query parameters that control the QIDO-RS request rather than matching
# against DICOM attributes.
_CONTROL_PARAMETERS = {
    "limit",
    "offset",
    "includefield",
    "fuzzymatching",
}


def _split_query_parameters(
    args: "Mapping[str, str]",
) -> tuple[dict[str, str], dict[str, str]]:
    """Separate DICOM attribute matching parameters from control parameters."""
    filters: dict[str, str] = {}
    control: dict[str, str] = {}

    for key in args:
        if key.lower() in _CONTROL_PARAMETERS:
            control[key.lower()] = args.get(key)
        else:
            filters[key] = args.get(key)

    return filters, control


def _dicom_json_response(results: list[dict[str, Any]]):
    """Build a DICOMweb JSON response, honouring the 204 empty-match rule."""
    if not results:
        # PS3.18: an empty QIDO-RS result set is 204 No Content.
        return flask.Response(status=204)

    return flask.Response(
        response=flask.json.dumps(results),
        status=200,
        content_type=DICOM_JSON_CONTENT_TYPE,
    )


def create_app(
    get_connection: Callable[[], "_mosaiq_api.Connection"],
) -> "flask.Flask":
    """Create the Mosaiq-backed DICOMweb Flask application.

    Parameters
    ----------
    get_connection
        A zero-argument callable returning an open Mosaiq connection. It is
        invoked once per request that needs database access, allowing the
        caller to manage pooling / reconnection.

    Returns
    -------
    flask.Flask
        The configured application. Run it with ``app.run(...)`` for
        development or hand it to a WSGI server for production.
    """
    app = flask.Flask("pymedphys.dicomweb")
    app.config["GET_MOSAIQ_CONNECTION"] = get_connection

    @app.get("/")
    def capabilities():
        """A minimal service description / health check."""
        return flask.jsonify(
            {
                "service": "pymedphys mosaiq dicomweb",
                "dicomweb": {
                    "qido-rs": {"studies": "implemented"},
                    "wado-rs": "not-implemented",
                    "stow-rs": "not-implemented",
                },
            }
        )

    @app.get("/studies")
    def search_for_studies():
        filters, control = _split_query_parameters(flask.request.args)
        connection = app.config["GET_MOSAIQ_CONNECTION"]()

        limit = int(control["limit"]) if "limit" in control else None
        offset = int(control.get("offset", 0))

        results = qido.search_for_studies(
            connection, filters, limit=limit, offset=offset
        )
        return _dicom_json_response(results)

    @app.get("/studies/<study_instance_uid>/series")
    def search_for_series(study_instance_uid: str):
        filters, _ = _split_query_parameters(flask.request.args)
        connection = app.config["GET_MOSAIQ_CONNECTION"]()
        results = qido.search_for_series(connection, study_instance_uid, filters)
        return _dicom_json_response(results)

    @app.get("/studies/<study_instance_uid>/instances")
    def search_for_study_instances(study_instance_uid: str):
        filters, _ = _split_query_parameters(flask.request.args)
        connection = app.config["GET_MOSAIQ_CONNECTION"]()
        results = qido.search_for_instances(
            connection, study_instance_uid, None, filters
        )
        return _dicom_json_response(results)

    @app.get("/studies/<study_instance_uid>/series/<series_instance_uid>/instances")
    def search_for_instances(study_instance_uid: str, series_instance_uid: str):
        filters, _ = _split_query_parameters(flask.request.args)
        connection = app.config["GET_MOSAIQ_CONNECTION"]()
        results = qido.search_for_instances(
            connection, study_instance_uid, series_instance_uid, filters
        )
        return _dicom_json_response(results)

    @app.get("/studies/<study_instance_uid>")
    def retrieve_study(study_instance_uid: str):
        try:
            wado.retrieve_study(study_instance_uid)
        except wado.WadoNotImplemented as exc:
            return _not_implemented(str(exc))

    @app.post("/studies")
    def store_instances():
        try:
            stow.store_instances(None, flask.request.data)
        except stow.StowNotImplemented as exc:
            return _not_implemented(str(exc))

    return app


def _not_implemented(message: str):
    """Return a 501 response for scaffolded services."""
    return flask.Response(
        response=flask.json.dumps({"error": message}),
        status=501,
        content_type="application/json",
    )
