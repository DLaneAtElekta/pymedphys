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

"""A minimal DICOMweb (QIDO-RS search, WADO-RS retrieve) server.

Routes
------
``GET /studies``
    Search for studies. See ``MosaiqDicomRepository.search_studies``.
``GET /studies/{study}/series``
    List the series within a study.
``GET /studies/{study}/series/{series}/instances``
    List the instances within a series.
``GET /studies/{study}/series/{series}/instances/{instance}``
    Retrieve an instance as ``multipart/related; type="application/dicom"``.

Only the Python standard library is used, and requests are served one
at a time, which suits a single shared Mosaiq database connection.
"""

import functools
import json
import logging
import urllib.parse
import uuid
from wsgiref.simple_server import make_server

from pymedphys._mosaiq import api as mosaiq

from .repository import MosaiqDicomRepository
from .schema import MosaiqDicomSchema

DEFAULT_LIMIT = 100

_REASONS = {
    200: "OK",
    204: "No Content",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    500: "Internal Server Error",
}


class _HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _status(code: int) -> str:
    return f"{code} {_REASONS[code]}"


def _query_filters(query_string: str) -> dict[str, str]:
    # A leading "00100020"-style tag or a keyword are both valid QIDO
    # match keys. Only keywords are supported for now.
    return {
        key: values[-1] for key, values in urllib.parse.parse_qs(query_string).items()
    }


def _int_param(filters: dict[str, str], key: str, default: int) -> int:
    value = filters.pop(key, None)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as e:
        raise _HttpError(400, f"{key} must be an integer") from e
    if parsed < 0:
        raise _HttpError(400, f"{key} must not be negative")

    return parsed


def create_app(repository: MosaiqDicomRepository):
    """Create a WSGI application serving ``repository`` over DICOMweb."""

    def json_response(start_response, results: list):
        if not results:
            start_response(_status(204), [])
            return [b""]

        body = json.dumps(results).encode("utf-8")
        start_response(
            _status(200),
            [
                ("Content-Type", "application/dicom+json"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    def instance_response(start_response, data: bytes):
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                b"Content-Type: application/dicom\r\n\r\n",
                data,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        start_response(
            _status(200),
            [
                (
                    "Content-Type",
                    f'multipart/related; type="application/dicom"; boundary={boundary}',
                ),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    def route(environ, start_response):
        if environ["REQUEST_METHOD"] != "GET":
            raise _HttpError(405, "Only GET is supported")

        parts = [
            urllib.parse.unquote(part)
            for part in environ.get("PATH_INFO", "").split("/")
            if part
        ]

        match parts:
            case ["studies"]:
                filters = _query_filters(environ.get("QUERY_STRING", ""))
                limit = _int_param(filters, "limit", DEFAULT_LIMIT)
                offset = _int_param(filters, "offset", 0)
                return json_response(
                    start_response,
                    repository.search_studies(filters, limit=limit, offset=offset),
                )
            case ["studies", study, "series"]:
                return json_response(start_response, repository.search_series(study))
            case ["studies", study, "series", series, "instances"]:
                return json_response(
                    start_response, repository.search_instances(study, series)
                )
            case ["studies", study, "series", series, "instances", instance]:
                data = repository.retrieve_instance(study, series, instance)
                if data is None:
                    raise _HttpError(404, "Instance not found")
                return instance_response(start_response, data)

        raise _HttpError(404, "Unknown resource")

    def app(environ, start_response):
        try:
            return route(environ, start_response)
        except _HttpError as e:
            body = e.message.encode("utf-8")
            start_response(
                _status(e.status),
                [("Content-Type", "text/plain; charset=utf-8")],
            )
            return [body]
        except Exception:  # pylint: disable = broad-except
            logging.exception("Error while handling %s", environ.get("PATH_INFO"))
            start_response(
                _status(500), [("Content-Type", "text/plain; charset=utf-8")]
            )
            return [b"Internal server error"]

    return app


def serve(repository: MosaiqDicomRepository, host: str, port: int):
    """Serve ``repository`` over DICOMweb until interrupted."""
    with make_server(host, port, create_app(repository)) as httpd:
        logging.info("DICOMweb server listening on http://%s:%s", host, port)
        httpd.serve_forever()


def web_cli(args):
    """Start the DICOMweb server from the command line interface"""
    schema = MosaiqDicomSchema.from_toml(args.schema) if args.schema else None

    with mosaiq.connect(
        args.mosaiq_hostname,
        port=args.mosaiq_port,
        database=args.mosaiq_database,
    ) as connection:
        repository = MosaiqDicomRepository(
            functools.partial(mosaiq.execute, connection), schema
        )
        serve(repository, host=args.host, port=args.port)
