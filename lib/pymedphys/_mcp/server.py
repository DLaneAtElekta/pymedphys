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

"""Main MCP server implementation for PyMedPhys."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

logger = logging.getLogger(__name__)

# Global configuration for the server
_server_config: dict[str, Any] = {
    "mosaiq_connection": None,
    "dicom_directories": [],
    "trf_directories": [],
    "working_directory": None,
    "deidentify": False,
    "deidentify_salt": "",
}


def create_server(
    mosaiq_connection: Any | None = None,
    dicom_directories: list[str | Path] | None = None,
    trf_directories: list[str | Path] | None = None,
    working_directory: str | Path | None = None,
    deidentify: bool = False,
    deidentify_salt: str = "",
) -> Server:
    """Create and configure the PyMedPhys MCP server.

    Parameters
    ----------
    mosaiq_connection : optional
        An active Mosaiq database connection for patient data queries.
    dicom_directories : list of str or Path, optional
        Directories containing DICOM files to expose as resources.
    trf_directories : list of str or Path, optional
        Directories containing TRF log files to expose as resources.
    working_directory : str or Path, optional
        Working directory for file operations and output.
    deidentify : bool
        If True, de-identify PHI (Protected Health Information) before
        returning data. Patient IDs and names will be pseudonymized,
        and sensitive fields (SSN, contact info) will be removed.
        Default is False for backwards compatibility, but SHOULD be
        enabled when connecting to AI assistants.
    deidentify_salt : str
        Optional salt for pseudonymization. Using the same salt ensures
        consistent pseudonyms across sessions, allowing correlation of
        de-identified data.

    Returns
    -------
    Server
        Configured MCP server instance.
    """
    server = Server("pymedphys")

    # Store configuration
    _server_config["mosaiq_connection"] = mosaiq_connection
    _server_config["dicom_directories"] = [Path(d) for d in (dicom_directories or [])]
    _server_config["trf_directories"] = [Path(d) for d in (trf_directories or [])]
    _server_config["working_directory"] = (
        Path(working_directory) if working_directory else Path.cwd()
    )
    _server_config["deidentify"] = deidentify
    _server_config["deidentify_salt"] = deidentify_salt

    # Log de-identification status
    if deidentify:
        logger.info("De-identification ENABLED - PHI will be pseudonymized")
    else:
        logger.warning(
            "De-identification DISABLED - PHI will be sent to AI assistant. "
            "Consider enabling with --deidentify flag for HIPAA compliance."
        )

    # Register handlers
    _register_resources(server)
    _register_tools(server)
    _register_prompts(server)

    return server


def _register_resources(server: Server):
    """Register resource handlers for patient data."""

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List all available patient data resources."""
        resources = []

        # Add Mosaiq resources if connection is available
        if _server_config["mosaiq_connection"] is not None:
            resources.append(
                Resource(
                    uri="mosaiq://patients",
                    name="Mosaiq Patient List",
                    description="List of patients in the Mosaiq OIS database",
                    mimeType="application/json",
                )
            )
            resources.append(
                Resource(
                    uri="mosaiq://machines",
                    name="Mosaiq Machine List",
                    description="List of treatment machines in Mosaiq",
                    mimeType="application/json",
                )
            )
            resources.append(
                Resource(
                    uri="mosaiq://sites",
                    name="Mosaiq Site List",
                    description="List of treatment sites in Mosaiq (includes RT Plan and TX field status)",
                    mimeType="application/json",
                )
            )

        # Add DICOM file resources
        for dicom_dir in _server_config["dicom_directories"]:
            if dicom_dir.exists():
                for dcm_file in dicom_dir.rglob("*.dcm"):
                    rel_path = dcm_file.relative_to(dicom_dir)
                    resources.append(
                        Resource(
                            uri=f"dicom://{dicom_dir.name}/{rel_path}",
                            name=f"DICOM: {dcm_file.stem}",
                            description=f"DICOM file at {dcm_file}",
                            mimeType="application/dicom",
                        )
                    )

        # Add TRF file resources
        for trf_dir in _server_config["trf_directories"]:
            if trf_dir.exists():
                for trf_file in trf_dir.rglob("*.trf"):
                    rel_path = trf_file.relative_to(trf_dir)
                    resources.append(
                        Resource(
                            uri=f"trf://{trf_dir.name}/{rel_path}",
                            name=f"TRF: {trf_file.stem}",
                            description=f"Elekta TRF log file at {trf_file}",
                            mimeType="application/octet-stream",
                        )
                    )

        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a specific patient data resource."""
        from . import deidentify
        from .resources import dicom as dicom_resources
        from .resources import mosaiq as mosaiq_resources
        from .resources import trf as trf_resources

        if uri.startswith("mosaiq://"):
            result = await mosaiq_resources.read_resource(
                uri, _server_config["mosaiq_connection"]
            )
        elif uri.startswith("dicom://"):
            result = await dicom_resources.read_resource(
                uri, _server_config["dicom_directories"]
            )
        elif uri.startswith("trf://"):
            result = await trf_resources.read_resource(
                uri, _server_config["trf_directories"]
            )
        else:
            raise ValueError(f"Unknown resource URI scheme: {uri}")

        # Apply de-identification if enabled
        if _server_config["deidentify"]:
            result = deidentify.deidentify_json_string(
                result, _server_config["deidentify_salt"]
            )

        return result


def _register_tools(server: Server):
    """Register tool handlers for analysis and operations."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools."""
        tools = [
            # Analysis tools
            Tool(
                name="gamma_analysis",
                description="Perform gamma analysis comparing two dose distributions. "
                "Returns gamma index values and pass rates for quality assurance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reference_dose_path": {
                            "type": "string",
                            "description": "Path to reference dose file (DICOM or numpy)",
                        },
                        "evaluation_dose_path": {
                            "type": "string",
                            "description": "Path to evaluation dose file (DICOM or numpy)",
                        },
                        "dose_threshold_percent": {
                            "type": "number",
                            "description": "Dose difference threshold in percent (e.g., 3 for 3%)",
                            "default": 3,
                        },
                        "distance_threshold_mm": {
                            "type": "number",
                            "description": "Distance-to-agreement threshold in mm (e.g., 3 for 3mm)",
                            "default": 3,
                        },
                        "lower_dose_cutoff_percent": {
                            "type": "number",
                            "description": "Lower dose cutoff as percent of max (e.g., 20 for 20%)",
                            "default": 20,
                        },
                        "local_gamma": {
                            "type": "boolean",
                            "description": "Use local gamma (True) or global gamma (False)",
                            "default": False,
                        },
                    },
                    "required": ["reference_dose_path", "evaluation_dose_path"],
                },
            ),
            Tool(
                name="calculate_metersetmap",
                description="Calculate a MetersetMap (fluence map) from MLC and jaw positions. "
                "Useful for visualizing and comparing treatment delivery.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source type: 'trf', 'dicom', or 'mosaiq'",
                        },
                        "source_path": {
                            "type": "string",
                            "description": "Path to source file (for trf/dicom) or field identifier (for mosaiq)",
                        },
                        "grid_resolution": {
                            "type": "number",
                            "description": "Grid resolution in mm",
                            "default": 1.0,
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Output format: 'json', 'numpy', or 'image'",
                            "default": "json",
                        },
                    },
                    "required": ["source", "source_path"],
                },
            ),
            Tool(
                name="check_metersetmap_status",
                description="Check if MetersetMap QA has been completed for a patient's treatment. "
                "Scans the output directory for existing MetersetMap reports, checks Mosaiq QCL "
                "(Quality Checklist) status, and compares against treatment delivery history to "
                "determine if QA is complete, pending, or overdue.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "string",
                            "description": "Patient ID to check",
                        },
                        "output_directory": {
                            "type": "string",
                            "description": "Directory where MetersetMap PDF/PNG results are stored "
                            "(e.g., ~/pymedphys-gui-metersetmap)",
                        },
                        "site_id": {
                            "type": "string",
                            "description": "Optional: Specific site ID to check",
                        },
                        "field_identifier": {
                            "type": "string",
                            "description": "Optional: Specific field identifier to check",
                        },
                        "qcl_task_description": {
                            "type": "string",
                            "description": "Optional: QCL task description to search for in Mosaiq "
                            "(e.g., 'MetersetMap Check', 'Physics Check', 'IMRT QA'). "
                            "If provided, also checks Mosaiq QCL for completion status.",
                        },
                    },
                    "required": ["patient_id", "output_directory"],
                },
            ),
            Tool(
                name="find_pending_metersetmap_checks",
                description="Find patients with RT Plans that need MetersetMap QA checks. "
                "Scans Mosaiq for sites with RT Plans imported in the last N days that don't have "
                "corresponding MetersetMap check files. Returns prioritized list by urgency.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_directory": {
                            "type": "string",
                            "description": "Directory where MetersetMap results are stored",
                        },
                        "days_threshold": {
                            "type": "integer",
                            "description": "Look for RT Plans imported within this many days",
                            "default": 7,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 50,
                        },
                    },
                    "required": ["output_directory"],
                },
            ),
            # DICOM tools
            Tool(
                name="read_dicom",
                description="Read and parse a DICOM file, returning key metadata and content. "
                "Supports RT Plan, RT Dose, RT Structure, and CT files.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the DICOM file",
                        },
                        "include_pixel_data": {
                            "type": "boolean",
                            "description": "Include pixel/dose data in response",
                            "default": False,
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="anonymize_dicom",
                description="Anonymize a DICOM file by removing or replacing patient-identifying information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Path to input DICOM file",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path for anonymized output file",
                        },
                        "replacement_id": {
                            "type": "string",
                            "description": "Replacement patient ID (optional)",
                        },
                        "delete_private_tags": {
                            "type": "boolean",
                            "description": "Delete private DICOM tags",
                            "default": True,
                        },
                    },
                    "required": ["input_path", "output_path"],
                },
            ),
            Tool(
                name="create_rt_plan",
                description="Create an RT Plan DICOM file from treatment delivery data. "
                "Can be used to generate plans for import into treatment management systems.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "description": "Source data type: 'trf', 'delivery', or 'parameters'",
                        },
                        "source_data": {
                            "type": "object",
                            "description": "Source data (file path or delivery parameters)",
                        },
                        "template_path": {
                            "type": "string",
                            "description": "Path to template DICOM RT Plan (required for some sources)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output path for generated RT Plan",
                        },
                        "patient_info": {
                            "type": "object",
                            "description": "Patient information (name, ID, etc.)",
                            "properties": {
                                "patient_id": {"type": "string"},
                                "patient_name": {"type": "string"},
                                "birth_date": {"type": "string"},
                            },
                        },
                    },
                    "required": ["source_type", "output_path"],
                },
            ),
            Tool(
                name="create_rtpconnect",
                description="Create an RT ION Plan Connect (RTPCONNECT) file for prescription import. "
                "This file can be imported into Mosaiq to create a treatment prescription.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prescription": {
                            "type": "object",
                            "description": "Prescription details",
                            "properties": {
                                "site_name": {
                                    "type": "string",
                                    "description": "Treatment site name",
                                },
                                "fractions": {
                                    "type": "integer",
                                    "description": "Number of fractions",
                                },
                                "dose_per_fraction": {
                                    "type": "number",
                                    "description": "Dose per fraction in cGy",
                                },
                                "total_dose": {
                                    "type": "number",
                                    "description": "Total prescribed dose in cGy",
                                },
                                "technique": {
                                    "type": "string",
                                    "description": "Treatment technique (e.g., VMAT, IMRT, 3DCRT)",
                                },
                            },
                            "required": ["site_name", "fractions", "dose_per_fraction"],
                        },
                        "patient_info": {
                            "type": "object",
                            "description": "Patient information for DICOM header",
                            "properties": {
                                "patient_id": {"type": "string"},
                                "patient_name": {"type": "string"},
                                "birth_date": {"type": "string"},
                            },
                            "required": ["patient_id", "patient_name"],
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output path for RTPCONNECT file",
                        },
                        "instructions": {
                            "type": "boolean",
                            "description": "Include import instructions in response",
                            "default": True,
                        },
                    },
                    "required": ["prescription", "patient_info", "output_path"],
                },
            ),
            # TRF tools
            Tool(
                name="read_trf",
                description="Read and parse an Elekta TRF (treatment record file) log file. "
                "Returns delivery parameters including MLC positions, gantry angles, and monitor units.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the TRF file",
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Output format: 'summary', 'detailed', or 'dataframe'",
                            "default": "summary",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="compare_trf_to_plan",
                description="Compare TRF log file delivery to planned parameters from DICOM or Mosaiq.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trf_path": {
                            "type": "string",
                            "description": "Path to TRF file",
                        },
                        "plan_source": {
                            "type": "string",
                            "description": "Plan source: 'dicom' or 'mosaiq'",
                        },
                        "plan_path_or_id": {
                            "type": "string",
                            "description": "DICOM file path or Mosaiq field ID",
                        },
                    },
                    "required": ["trf_path", "plan_source", "plan_path_or_id"],
                },
            ),
        ]

        # Add Mosaiq-specific tools if connection is available
        if _server_config["mosaiq_connection"] is not None:
            tools.extend(
                [
                    Tool(
                        name="query_mosaiq",
                        description="Execute a query against the Mosaiq database. "
                        "Returns patient, treatment, or machine data.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query_type": {
                                    "type": "string",
                                    "description": "Query type: 'patient', 'field', 'treatment', 'machine', or 'custom'",
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Query parameters (patient_id, field_id, date_range, etc.)",
                                },
                                "sql": {
                                    "type": "string",
                                    "description": "Custom SQL query (only for query_type='custom')",
                                },
                            },
                            "required": ["query_type"],
                        },
                    ),
                    Tool(
                        name="get_patient_chart",
                        description="Retrieve comprehensive patient chart data from Mosaiq, "
                        "including demographics, prescriptions, treatment history, and QA records.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "patient_id": {
                                    "type": "string",
                                    "description": "Patient ID in Mosaiq",
                                },
                                "include_sections": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Sections to include: 'demographics', 'prescriptions', "
                                    "'treatments', 'qa', 'documents'",
                                    "default": [
                                        "demographics",
                                        "prescriptions",
                                        "treatments",
                                    ],
                                },
                            },
                            "required": ["patient_id"],
                        },
                    ),
                    Tool(
                        name="get_site_details",
                        description="Get detailed information about a Mosaiq treatment site, "
                        "including prescriptions, RT plans, and treatment fields. "
                        "Identifies sites that need review (RT Plan exists but no TX fields).",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "site_id": {
                                    "type": "string",
                                    "description": "Mosaiq Site ID (SIT_ID)",
                                },
                            },
                            "required": ["site_id"],
                        },
                    ),
                    Tool(
                        name="find_sites_needing_review",
                        description="Find all treatment sites that have an imported RT Plan "
                        "but no treatment fields defined. These sites need review using the RT Viewer.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "patient_id": {
                                    "type": "string",
                                    "description": "Optional: Filter by patient ID",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of sites to return",
                                    "default": 50,
                                },
                            },
                        },
                    ),
                    Tool(
                        name="launch_rt_viewer",
                        description="Launch the PyMedPhys RT Viewer (Streamlit app) to review "
                        "DICOM RT data including CT, RT Structure, RT Dose, and RT Plan. "
                        "Use this to visually review sites with imported RT Plans.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "dicom_directory": {
                                    "type": "string",
                                    "description": "Directory containing DICOM files (CT series)",
                                },
                                "rtstruct_path": {
                                    "type": "string",
                                    "description": "Path to RT Structure file",
                                },
                                "rtdose_path": {
                                    "type": "string",
                                    "description": "Optional: Path to RT Dose file",
                                },
                                "port": {
                                    "type": "integer",
                                    "description": "Port to run Streamlit server on",
                                    "default": 8501,
                                },
                            },
                            "required": ["dicom_directory", "rtstruct_path"],
                        },
                    ),
                ]
            )

        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool with the given arguments."""
        from .tools import analysis, dicom, mosaiq, trf

        try:
            if name == "gamma_analysis":
                result = await analysis.gamma_analysis(**arguments)
            elif name == "calculate_metersetmap":
                result = await analysis.calculate_metersetmap(
                    **arguments,
                    mosaiq_connection=_server_config["mosaiq_connection"],
                )
            elif name == "check_metersetmap_status":
                result = await analysis.check_metersetmap_status(
                    **arguments,
                    mosaiq_connection=_server_config["mosaiq_connection"],
                )
            elif name == "find_pending_metersetmap_checks":
                result = await analysis.find_pending_metersetmap_checks(
                    **arguments,
                    mosaiq_connection=_server_config["mosaiq_connection"],
                )
            elif name == "read_dicom":
                result = await dicom.read_dicom(**arguments)
            elif name == "anonymize_dicom":
                result = await dicom.anonymize_dicom(**arguments)
            elif name == "create_rt_plan":
                result = await dicom.create_rt_plan(**arguments)
            elif name == "create_rtpconnect":
                result = await dicom.create_rtpconnect(**arguments)
            elif name == "read_trf":
                result = await trf.read_trf(**arguments)
            elif name == "compare_trf_to_plan":
                result = await trf.compare_trf_to_plan(
                    **arguments,
                    mosaiq_connection=_server_config["mosaiq_connection"],
                )
            elif name == "query_mosaiq":
                result = await mosaiq.query_mosaiq(
                    **arguments,
                    connection=_server_config["mosaiq_connection"],
                )
            elif name == "get_patient_chart":
                result = await mosaiq.get_patient_chart(
                    **arguments,
                    connection=_server_config["mosaiq_connection"],
                )
            elif name == "get_site_details":
                result = await mosaiq.get_site_details(
                    **arguments,
                    connection=_server_config["mosaiq_connection"],
                )
            elif name == "find_sites_needing_review":
                result = await mosaiq.find_sites_needing_review(
                    **arguments,
                    connection=_server_config["mosaiq_connection"],
                )
            elif name == "launch_rt_viewer":
                result = await mosaiq.launch_rt_viewer(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}

            # Apply de-identification if enabled
            if _server_config["deidentify"]:
                from . import deidentify

                result = deidentify.deidentify_dict(
                    result, _server_config["deidentify_salt"]
                )

            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "tool": name}, indent=2),
                )
            ]


def _register_prompts(server: Server):
    """Register prompt templates for common workflows."""

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """List available prompt templates."""
        return [
            Prompt(
                name="patient_qa_review",
                description="Review patient QA data and provide analysis summary",
                arguments=[
                    PromptArgument(
                        name="patient_id",
                        description="Patient ID to review",
                        required=True,
                    ),
                ],
            ),
            Prompt(
                name="treatment_comparison",
                description="Compare planned vs delivered treatment parameters",
                arguments=[
                    PromptArgument(
                        name="trf_path",
                        description="Path to TRF log file",
                        required=True,
                    ),
                    PromptArgument(
                        name="plan_path",
                        description="Path to DICOM RT Plan",
                        required=True,
                    ),
                ],
            ),
            Prompt(
                name="create_prescription",
                description="Guide for creating a prescription RTPCONNECT file for Mosaiq import",
                arguments=[
                    PromptArgument(
                        name="site",
                        description="Treatment site (e.g., 'Brain', 'Lung')",
                        required=True,
                    ),
                    PromptArgument(
                        name="dose",
                        description="Total dose in cGy",
                        required=True,
                    ),
                    PromptArgument(
                        name="fractions",
                        description="Number of fractions",
                        required=True,
                    ),
                ],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None
    ) -> GetPromptResult:
        """Get a specific prompt template with arguments."""
        arguments = arguments or {}

        if name == "patient_qa_review":
            patient_id = arguments.get("patient_id", "UNKNOWN")
            return GetPromptResult(
                description=f"QA Review for patient {patient_id}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""Please review the QA data for patient {patient_id}.

Use the following tools to gather information:
1. get_patient_chart - to retrieve patient treatment history
2. read_trf - to analyze any treatment log files
3. gamma_analysis - to compare dose distributions if available

Provide a summary including:
- Treatment history overview
- Any QA concerns or deviations
- Recommendations for follow-up""",
                        ),
                    )
                ],
            )

        elif name == "treatment_comparison":
            trf_path = arguments.get("trf_path", "")
            plan_path = arguments.get("plan_path", "")
            return GetPromptResult(
                description="Treatment comparison analysis",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""Compare the delivered treatment to the plan.

TRF Log File: {trf_path}
DICOM Plan: {plan_path}

Use these tools:
1. read_trf - to parse the delivery log
2. read_dicom - to parse the RT Plan
3. compare_trf_to_plan - for detailed comparison
4. calculate_metersetmap - to visualize fluence differences

Report on:
- MU agreement
- Gantry angle accuracy
- MLC position deviations
- Overall delivery quality""",
                        ),
                    )
                ],
            )

        elif name == "create_prescription":
            site = arguments.get("site", "")
            dose = arguments.get("dose", "")
            fractions = arguments.get("fractions", "")
            return GetPromptResult(
                description="Create prescription RTPCONNECT file",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""Create an RTPCONNECT prescription file for Mosaiq import.

Prescription Details:
- Treatment Site: {site}
- Total Dose: {dose} cGy
- Fractions: {fractions}

Use the create_rtpconnect tool with appropriate patient information.

After creating the file, provide instructions for importing into Mosaiq:
1. File location
2. Mosaiq import steps
3. Verification checklist""",
                        ),
                    )
                ],
            )

        raise ValueError(f"Unknown prompt: {name}")


async def run_server(
    mosaiq_connection: Any | None = None,
    dicom_directories: list[str | Path] | None = None,
    trf_directories: list[str | Path] | None = None,
    working_directory: str | Path | None = None,
    deidentify: bool = False,
    deidentify_salt: str = "",
):
    """Run the PyMedPhys MCP server.

    This function starts the MCP server using stdio transport,
    suitable for integration with AI assistants like Claude.

    Parameters
    ----------
    mosaiq_connection : optional
        An active Mosaiq database connection.
    dicom_directories : list of str or Path, optional
        Directories containing DICOM files.
    trf_directories : list of str or Path, optional
        Directories containing TRF files.
    working_directory : str or Path, optional
        Working directory for file operations.
    deidentify : bool
        If True, de-identify PHI before sending to AI assistant.
        RECOMMENDED for HIPAA compliance.
    deidentify_salt : str
        Optional salt for consistent pseudonymization.
    """
    server = create_server(
        mosaiq_connection=mosaiq_connection,
        dicom_directories=dicom_directories,
        trf_directories=trf_directories,
        working_directory=working_directory,
        deidentify=deidentify,
        deidentify_salt=deidentify_salt,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
