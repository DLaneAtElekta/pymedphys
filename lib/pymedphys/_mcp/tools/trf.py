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

"""TRF file tools for MCP server."""

from pathlib import Path
from typing import Any

import numpy as np


async def read_trf(
    file_path: str,
    output_format: str = "summary",
) -> dict[str, Any]:
    """Read and parse an Elekta TRF log file.

    Parameters
    ----------
    file_path : str
        Path to the TRF file
    output_format : str
        Output format: 'summary', 'detailed', or 'dataframe'

    Returns
    -------
    dict
        TRF file contents in requested format
    """
    import pymedphys

    path = Path(file_path)
    if not path.exists():
        return {"error": f"TRF file not found: {file_path}"}

    # Read TRF file
    header, table = pymedphys.trf.read(str(path))

    # Build header dictionary
    header_dict = {}
    if hasattr(header, "to_dict"):
        header_dict = header.to_dict()
    elif isinstance(header, dict):
        header_dict = header
    else:
        for attr in dir(header):
            if not attr.startswith("_"):
                try:
                    val = getattr(header, attr)
                    if not callable(val):
                        header_dict[attr] = val
                except Exception:
                    pass

    result = {
        "file_path": str(path),
        "file_name": path.name,
        "header": header_dict,
    }

    if output_format == "summary":
        # Provide summary statistics
        result["summary"] = _get_trf_summary(table)

    elif output_format == "detailed":
        # Include more detailed information
        result["summary"] = _get_trf_summary(table)
        result["columns"] = list(table.columns) if hasattr(table, "columns") else []

        # Include first and last few rows
        if hasattr(table, "head"):
            result["first_rows"] = table.head(5).to_dict("records")
            result["last_rows"] = table.tail(5).to_dict("records")

    elif output_format == "dataframe":
        # Include full dataframe as records (warning: can be large)
        if hasattr(table, "to_dict"):
            result["data"] = table.to_dict("records")
            result["num_rows"] = len(table)
        else:
            result["error"] = "Unable to convert to dataframe format"

    return result


def _get_trf_summary(table) -> dict[str, Any]:
    """Extract summary statistics from TRF table data."""
    summary = {
        "num_samples": len(table) if hasattr(table, "__len__") else 0,
    }

    if not hasattr(table, "columns"):
        return summary

    columns = table.columns

    # Gantry angle range
    gantry_cols = [c for c in columns if "Gantry" in c and "Actual" in c]
    if gantry_cols:
        col = gantry_cols[0]
        summary["gantry_angle"] = {
            "min": float(table[col].min()),
            "max": float(table[col].max()),
            "column": col,
        }

    # Collimator angle
    coll_cols = [c for c in columns if "Coll" in c and "Actual" in c and "Angle" in c]
    if coll_cols:
        col = coll_cols[0]
        summary["collimator_angle"] = {
            "min": float(table[col].min()),
            "max": float(table[col].max()),
            "column": col,
        }

    # Monitor units / dose
    dose_cols = [c for c in columns if "Dose" in c and "Actual" in c]
    if dose_cols:
        col = dose_cols[0]
        summary["monitor_units"] = {
            "total": float(table[col].max()),
            "column": col,
        }

    # Jaw positions
    for jaw in ["Y1", "Y2", "X1", "X2"]:
        jaw_cols = [c for c in columns if jaw in c and "Actual" in c]
        if jaw_cols:
            col = jaw_cols[0]
            summary[f"jaw_{jaw.lower()}"] = {
                "min": float(table[col].min()),
                "max": float(table[col].max()),
                "column": col,
            }

    # MLC summary
    mlc_cols = [c for c in columns if c.startswith("A") or c.startswith("B")]
    mlc_cols = [c for c in mlc_cols if "Actual" in c]
    if mlc_cols:
        summary["mlc"] = {
            "num_leaf_columns": len(mlc_cols),
            "columns_sample": mlc_cols[:5],
        }

    return summary


async def compare_trf_to_plan(
    trf_path: str,
    plan_source: str,
    plan_path_or_id: str,
    mosaiq_connection: Any | None = None,
) -> dict[str, Any]:
    """Compare TRF delivery to planned parameters.

    Parameters
    ----------
    trf_path : str
        Path to TRF file
    plan_source : str
        Plan source: 'dicom' or 'mosaiq'
    plan_path_or_id : str
        DICOM file path or Mosaiq field ID
    mosaiq_connection : optional
        Mosaiq connection for 'mosaiq' source

    Returns
    -------
    dict
        Comparison results including deviations
    """
    import pymedphys

    # Load TRF delivery
    trf_delivery = pymedphys.Delivery.from_trf(trf_path)

    # Load plan delivery
    if plan_source == "dicom":
        import pydicom

        ds = pydicom.dcmread(plan_path_or_id)
        plan_delivery = pymedphys.Delivery.from_dicom(ds)
    elif plan_source == "mosaiq":
        if mosaiq_connection is None:
            return {"error": "Mosaiq connection required"}
        plan_delivery = pymedphys.Delivery.from_mosaiq(
            mosaiq_connection, plan_path_or_id
        )
    else:
        return {"error": f"Unknown plan source: {plan_source}"}

    # Compare deliveries
    comparison = {
        "trf_file": trf_path,
        "plan_source": plan_source,
        "plan_identifier": plan_path_or_id,
    }

    # Monitor unit comparison
    trf_mu_total = float(np.sum(trf_delivery.monitor_units))
    plan_mu_total = float(np.sum(plan_delivery.monitor_units))
    mu_diff = trf_mu_total - plan_mu_total
    mu_diff_percent = (mu_diff / plan_mu_total * 100) if plan_mu_total > 0 else 0

    comparison["monitor_units"] = {
        "trf_total": trf_mu_total,
        "plan_total": plan_mu_total,
        "difference": mu_diff,
        "difference_percent": mu_diff_percent,
    }

    # Gantry angle comparison
    if len(trf_delivery.gantry) > 0 and len(plan_delivery.gantry) > 0:
        comparison["gantry"] = {
            "trf_range": [
                float(np.min(trf_delivery.gantry)),
                float(np.max(trf_delivery.gantry)),
            ],
            "plan_range": [
                float(np.min(plan_delivery.gantry)),
                float(np.max(plan_delivery.gantry)),
            ],
        }

    # Collimator comparison
    if len(trf_delivery.collimator) > 0 and len(plan_delivery.collimator) > 0:
        comparison["collimator"] = {
            "trf_range": [
                float(np.min(trf_delivery.collimator)),
                float(np.max(trf_delivery.collimator)),
            ],
            "plan_range": [
                float(np.min(plan_delivery.collimator)),
                float(np.max(plan_delivery.collimator)),
            ],
        }

    # MLC comparison (simplified)
    if trf_delivery.mlc is not None and plan_delivery.mlc is not None:
        trf_mlc = np.array(trf_delivery.mlc)
        plan_mlc = np.array(plan_delivery.mlc)

        if trf_mlc.shape == plan_mlc.shape:
            mlc_diff = trf_mlc - plan_mlc
            comparison["mlc"] = {
                "max_deviation_mm": float(np.max(np.abs(mlc_diff))),
                "mean_deviation_mm": float(np.mean(np.abs(mlc_diff))),
                "shape": list(trf_mlc.shape),
            }
        else:
            comparison["mlc"] = {
                "note": "MLC arrays have different shapes",
                "trf_shape": list(trf_mlc.shape),
                "plan_shape": list(plan_mlc.shape),
            }

    # Overall assessment
    issues = []
    if abs(mu_diff_percent) > 2:
        issues.append(f"MU difference of {mu_diff_percent:.2f}% exceeds 2% threshold")

    comparison["assessment"] = {
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "REVIEW",
    }

    return comparison
