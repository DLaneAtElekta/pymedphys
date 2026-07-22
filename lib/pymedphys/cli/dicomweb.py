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

"""CLI for the Mosaiq-backed DICOMweb service."""


def dicomweb_cli(subparsers):
    dicomweb_parser = subparsers.add_parser(
        "dicomweb",
        help="Serve Elekta Mosaiq data over DICOMweb (QIDO-RS/WADO-RS/STOW-RS).",
    )
    dicomweb_subparsers = dicomweb_parser.add_subparsers(dest="dicomweb")

    _serve(dicomweb_subparsers)

    return dicomweb_parser


def _serve(dicomweb_subparsers):
    parser = dicomweb_subparsers.add_parser(
        "serve",
        help=(
            "Start a development DICOMweb server that translates QIDO-RS "
            "queries into Mosaiq SQL queries."
        ),
    )

    parser.add_argument(
        "hostname", help="The IP address or hostname of the Mosaiq SQL server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1433,
        help="The port of the Mosaiq SQL server (default: 1433).",
    )
    parser.add_argument(
        "--database",
        default="MOSAIQ",
        help="The Mosaiq MSSQL database name (default: MOSAIQ).",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="A human readable alias for the Mosaiq server (used at the "
        "credential prompt).",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="The interface for the DICOMweb server to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8008,
        help="The port for the DICOMweb server to listen on (default: 8008).",
    )

    parser.set_defaults(func=serve_cli)


def serve_cli(args):
    # Imported lazily so that ``pymedphys`` (and its top-level CLI) does not
    # require the optional ``dicomweb`` dependencies to be installed.
    import pymedphys.mosaiq
    from pymedphys._dicomweb import server

    connection = pymedphys.mosaiq.connect(
        args.hostname,
        port=args.port,
        database=args.database,
        alias=args.alias,
    )

    app = server.create_app(get_connection=lambda: connection)
    app.run(host=args.bind, port=args.http_port)
