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

"""CLI commands for PyMedPhys MCP server."""

import argparse


def mcp_cli(subparsers):
    """Add MCP server CLI commands."""
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="MCP (Model Context Protocol) server for AI integration",
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    # Serve command
    serve_parser = mcp_subparsers.add_parser(
        "serve",
        help="Start the MCP server (stdio transport)",
    )
    serve_parser.add_argument(
        "--dicom-dir",
        action="append",
        dest="dicom_dirs",
        help="Directory containing DICOM files (can be specified multiple times)",
    )
    serve_parser.add_argument(
        "--trf-dir",
        action="append",
        dest="trf_dirs",
        help="Directory containing TRF files (can be specified multiple times)",
    )
    serve_parser.add_argument(
        "--working-dir",
        dest="working_dir",
        help="Working directory for file operations",
    )
    serve_parser.add_argument(
        "--mosaiq-server",
        dest="mosaiq_server",
        help="Mosaiq SQL server hostname",
    )
    serve_parser.add_argument(
        "--mosaiq-database",
        dest="mosaiq_database",
        default="MOSAIQ",
        help="Mosaiq database name (default: MOSAIQ)",
    )
    serve_parser.set_defaults(func=_mcp_serve)

    # Info command
    info_parser = mcp_subparsers.add_parser(
        "info",
        help="Display MCP server information and available tools",
    )
    info_parser.set_defaults(func=_mcp_info)

    # Config command
    config_parser = mcp_subparsers.add_parser(
        "config",
        help="Generate configuration for Claude Desktop or other MCP clients",
    )
    config_parser.add_argument(
        "--format",
        choices=["claude-desktop", "json"],
        default="claude-desktop",
        help="Output format (default: claude-desktop)",
    )
    config_parser.add_argument(
        "--dicom-dir",
        action="append",
        dest="dicom_dirs",
        help="Directory containing DICOM files",
    )
    config_parser.add_argument(
        "--trf-dir",
        action="append",
        dest="trf_dirs",
        help="Directory containing TRF files",
    )
    config_parser.set_defaults(func=_mcp_config)

    mcp_parser.set_defaults(func=lambda args: mcp_parser.print_help())


def _mcp_serve(args):
    """Start the MCP server."""
    import asyncio

    from pymedphys._mcp import run_server

    # Setup Mosaiq connection if server specified
    mosaiq_connection = None
    if args.mosaiq_server:
        try:
            import pymedphys

            mosaiq_connection = pymedphys.mosaiq.connect(
                args.mosaiq_server,
                database=args.mosaiq_database,
            )
            print(f"Connected to Mosaiq: {args.mosaiq_server}/{args.mosaiq_database}")
        except Exception as e:
            print(f"Warning: Failed to connect to Mosaiq: {e}")
            print("Continuing without Mosaiq integration...")

    # Run the server
    asyncio.run(
        run_server(
            mosaiq_connection=mosaiq_connection,
            dicom_directories=args.dicom_dirs or [],
            trf_directories=args.trf_dirs or [],
            working_directory=args.working_dir,
        )
    )


def _mcp_info(args):
    """Display MCP server information."""
    info = """
PyMedPhys MCP Server
====================

The PyMedPhys MCP server provides an interface for AI assistants (like Claude)
to interact with medical physics data and tools.

RESOURCES
---------
The server exposes the following resource types:

  mosaiq://     - Patient data from Mosaiq OIS
  dicom://      - DICOM files (RT Plan, RT Dose, RT Struct, CT)
  trf://        - Elekta TRF treatment log files

TOOLS
-----
Available tools:

  Analysis:
    - gamma_analysis        Compare dose distributions using gamma analysis
    - calculate_metersetmap Calculate fluence maps from delivery data

  DICOM:
    - read_dicom           Read and parse DICOM files
    - anonymize_dicom      Anonymize DICOM files
    - create_rt_plan       Create RT Plan DICOM from delivery data
    - create_rtpconnect    Create prescription files for Mosaiq import

  TRF:
    - read_trf             Parse Elekta TRF log files
    - compare_trf_to_plan  Compare delivery logs to planned parameters

  Mosaiq (when connected):
    - query_mosaiq         Query the Mosaiq database
    - get_patient_chart    Retrieve comprehensive patient data

PROMPTS
-------
Pre-defined prompt templates:

    - patient_qa_review      Review patient QA data
    - treatment_comparison   Compare planned vs delivered treatment
    - create_prescription    Guide for creating prescription files

USAGE
-----
To start the server:

    pymedphys mcp serve --dicom-dir /path/to/dicom --trf-dir /path/to/trf

With Mosaiq:

    pymedphys mcp serve --mosaiq-server sql-server.hospital.org

To generate Claude Desktop configuration:

    pymedphys mcp config --format claude-desktop

"""
    print(info)


def _mcp_config(args):
    """Generate MCP client configuration."""
    import json
    import sys

    # Build the command with arguments
    command_args = ["pymedphys", "mcp", "serve"]

    if args.dicom_dirs:
        for d in args.dicom_dirs:
            command_args.extend(["--dicom-dir", d])

    if args.trf_dirs:
        for d in args.trf_dirs:
            command_args.extend(["--trf-dir", d])

    if args.format == "claude-desktop":
        config = {
            "mcpServers": {
                "pymedphys": {
                    "command": sys.executable,
                    "args": ["-m", "pymedphys", "mcp", "serve"]
                    + command_args[3:],  # Skip 'pymedphys mcp serve'
                }
            }
        }

        print("Add the following to your Claude Desktop configuration:")
        print("(Located at ~/Library/Application Support/Claude/claude_desktop_config.json on macOS)")
        print()
        print(json.dumps(config, indent=2))
        print()
        print("Or add just the server entry to an existing config:")
        print()
        print(json.dumps(config["mcpServers"], indent=2))

    elif args.format == "json":
        config = {
            "name": "pymedphys",
            "command": sys.executable,
            "args": ["-m", "pymedphys", "mcp", "serve"] + command_args[3:],
            "description": "PyMedPhys MCP server for medical physics tools and data",
        }
        print(json.dumps(config, indent=2))
