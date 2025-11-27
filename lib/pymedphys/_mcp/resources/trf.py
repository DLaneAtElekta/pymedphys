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

"""TRF (Treatment Record File) resource handlers for MCP server."""

import json
from pathlib import Path
from urllib.parse import unquote, urlparse


async def read_resource(uri: str, trf_directories: list[Path]) -> str:
    """Read a TRF resource by URI.

    Parameters
    ----------
    uri : str
        Resource URI in format trf://directory_name/path/to/file.trf
    trf_directories : list of Path
        List of configured TRF directories

    Returns
    -------
    str
        JSON-encoded TRF summary data
    """
    parsed = urlparse(uri)
    dir_name = parsed.netloc
    file_path = unquote(parsed.path.lstrip("/"))

    # Find the matching directory
    target_dir = None
    for trf_dir in trf_directories:
        if trf_dir.name == dir_name:
            target_dir = trf_dir
            break

    if target_dir is None:
        return json.dumps({"error": f"TRF directory not found: {dir_name}"})

    full_path = target_dir / file_path

    if not full_path.exists():
        return json.dumps({"error": f"TRF file not found: {full_path}"})

    try:
        return await _read_trf_file(full_path)
    except Exception as e:
        return json.dumps({"error": f"Failed to read TRF file: {str(e)}"})


async def _read_trf_file(file_path: Path) -> str:
    """Read and parse a TRF file, returning summary data."""
    import pymedphys

    # Read TRF file using pymedphys
    header, table = pymedphys.trf.read(str(file_path))

    # Extract header information
    header_data = {}
    if hasattr(header, "to_dict"):
        header_data = header.to_dict()
    elif isinstance(header, dict):
        header_data = header
    else:
        # Try to extract common header fields
        for attr in ["machine", "date", "time", "patient_id", "field_label"]:
            if hasattr(header, attr):
                header_data[attr] = getattr(header, attr)

    # Get table summary (not full data to avoid large payloads)
    table_summary = {
        "num_rows": len(table) if hasattr(table, "__len__") else None,
        "columns": list(table.columns) if hasattr(table, "columns") else [],
    }

    # Extract key delivery parameters summary
    delivery_summary = {}
    if hasattr(table, "columns"):
        if "Step_Gantry/Actual" in table.columns:
            delivery_summary["gantry_range"] = [
                float(table["Step_Gantry/Actual"].min()),
                float(table["Step_Gantry/Actual"].max()),
            ]
        if "Step_Dose/Actual" in table.columns or "Y1_Dose/Actual" in table.columns:
            dose_col = (
                "Step_Dose/Actual"
                if "Step_Dose/Actual" in table.columns
                else "Y1_Dose/Actual"
            )
            delivery_summary["total_mu"] = float(table[dose_col].max())

    result = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "header": header_data,
        "table_summary": table_summary,
        "delivery_summary": delivery_summary,
    }

    return json.dumps(result, indent=2, default=str)
