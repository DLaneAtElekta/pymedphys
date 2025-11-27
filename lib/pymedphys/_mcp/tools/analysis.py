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

"""Analysis tools for MCP server (gamma, metersetmap, etc.)."""

from pathlib import Path
from typing import Any

import numpy as np


async def gamma_analysis(
    reference_dose_path: str,
    evaluation_dose_path: str,
    dose_threshold_percent: float = 3.0,
    distance_threshold_mm: float = 3.0,
    lower_dose_cutoff_percent: float = 20.0,
    local_gamma: bool = False,
) -> dict[str, Any]:
    """Perform gamma analysis comparing two dose distributions.

    Parameters
    ----------
    reference_dose_path : str
        Path to reference dose file (DICOM RTDose or numpy .npy)
    evaluation_dose_path : str
        Path to evaluation dose file
    dose_threshold_percent : float
        Dose difference threshold in percent
    distance_threshold_mm : float
        Distance-to-agreement threshold in mm
    lower_dose_cutoff_percent : float
        Lower dose cutoff as percent of max dose
    local_gamma : bool
        Use local gamma (True) or global gamma (False)

    Returns
    -------
    dict
        Gamma analysis results including pass rate and statistics
    """
    import pymedphys

    ref_path = Path(reference_dose_path)
    eval_path = Path(evaluation_dose_path)

    # Load dose data based on file type
    ref_axes, ref_dose = _load_dose_data(ref_path)
    eval_axes, eval_dose = _load_dose_data(eval_path)

    # Perform gamma analysis
    gamma = pymedphys.gamma(
        ref_axes,
        ref_dose,
        eval_axes,
        eval_dose,
        dose_threshold_percent,
        distance_threshold_mm,
        lower_percent_dose_cutoff=lower_dose_cutoff_percent,
        local_gamma=local_gamma,
    )

    # Calculate statistics
    valid_gamma = gamma[~np.isnan(gamma)]
    pass_rate = (
        np.sum(valid_gamma <= 1) / len(valid_gamma) * 100 if len(valid_gamma) > 0 else 0
    )

    results = {
        "pass_rate_percent": float(pass_rate),
        "criteria": {
            "dose_threshold_percent": dose_threshold_percent,
            "distance_threshold_mm": distance_threshold_mm,
            "lower_dose_cutoff_percent": lower_dose_cutoff_percent,
            "local_gamma": local_gamma,
        },
        "statistics": {
            "mean_gamma": float(np.mean(valid_gamma)) if len(valid_gamma) > 0 else None,
            "max_gamma": float(np.max(valid_gamma)) if len(valid_gamma) > 0 else None,
            "min_gamma": float(np.min(valid_gamma)) if len(valid_gamma) > 0 else None,
            "std_gamma": float(np.std(valid_gamma)) if len(valid_gamma) > 0 else None,
            "num_evaluated_points": int(len(valid_gamma)),
        },
        "reference_file": str(ref_path),
        "evaluation_file": str(eval_path),
    }

    return results


def _load_dose_data(file_path: Path) -> tuple:
    """Load dose data from file.

    Returns
    -------
    tuple
        (axes, dose) where axes is coordinate tuple and dose is numpy array
    """
    if file_path.suffix.lower() == ".dcm":
        import pydicom

        import pymedphys

        ds = pydicom.dcmread(str(file_path))
        axes, dose = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
        return axes, dose

    elif file_path.suffix.lower() == ".npy":
        # Assume numpy file contains dict with 'axes' and 'dose' keys
        # or just the dose array
        data = np.load(str(file_path), allow_pickle=True)
        if isinstance(data, dict):
            return data["axes"], data["dose"]
        else:
            # Generate default axes based on array shape
            shape = data.shape
            axes = tuple(np.arange(s) for s in shape)
            return axes, data

    else:
        raise ValueError(f"Unsupported dose file format: {file_path.suffix}")


async def calculate_metersetmap(
    source: str,
    source_path: str,
    grid_resolution: float = 1.0,
    output_format: str = "json",
    mosaiq_connection: Any | None = None,
) -> dict[str, Any]:
    """Calculate a MetersetMap (fluence map) from treatment delivery data.

    Parameters
    ----------
    source : str
        Source type: 'trf', 'dicom', or 'mosaiq'
    source_path : str
        Path to source file or Mosaiq field identifier
    grid_resolution : float
        Grid resolution in mm
    output_format : str
        Output format: 'json', 'numpy', or 'image'
    mosaiq_connection : optional
        Mosaiq connection for 'mosaiq' source type

    Returns
    -------
    dict
        MetersetMap data and metadata
    """
    import pymedphys

    # Load delivery data based on source
    if source == "trf":
        delivery = pymedphys.Delivery.from_trf(source_path)
    elif source == "dicom":
        import pydicom

        ds = pydicom.dcmread(source_path)
        delivery = pymedphys.Delivery.from_dicom(ds)
    elif source == "mosaiq":
        if mosaiq_connection is None:
            return {"error": "Mosaiq connection required for mosaiq source"}
        delivery = pymedphys.Delivery.from_mosaiq(mosaiq_connection, source_path)
    else:
        return {"error": f"Unknown source type: {source}"}

    # Calculate metersetmap
    mu = delivery.monitor_units
    mlc = delivery.mlc
    jaw = delivery.jaw

    grid = pymedphys.metersetmap.grid(
        max_leaf_gap=400,
        grid_resolution=grid_resolution,
    )

    metersetmap = pymedphys.metersetmap.calculate(
        mu=mu,
        mlc=mlc,
        jaw=jaw,
        grid_resolution=grid_resolution,
    )

    # Format output
    result = {
        "source": source,
        "source_path": source_path,
        "grid_resolution_mm": grid_resolution,
        "shape": list(metersetmap.shape),
        "total_mu": float(np.sum(mu)),
        "max_fluence": float(np.max(metersetmap)),
    }

    if output_format == "json":
        # Include grid coordinates and flattened data for smaller maps
        if metersetmap.size < 10000:
            result["grid_x"] = grid["x"].tolist()
            result["grid_y"] = grid["y"].tolist()
            result["data"] = metersetmap.tolist()
        else:
            result["note"] = "Data too large for JSON, use 'numpy' format"
    elif output_format == "numpy":
        # Save to temporary file and return path
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False, mode="wb") as f:
            np.save(f, {"grid": grid, "metersetmap": metersetmap})
            result["numpy_file"] = f.name
    elif output_format == "image":
        # Generate image representation
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pcolormesh(grid["x"], grid["y"], metersetmap, shading="auto")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("MetersetMap")
        ax.set_aspect("equal")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        result["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    return result


async def check_metersetmap_status(
    patient_id: str,
    output_directory: str,
    mosaiq_connection: Any | None = None,
    site_id: str | None = None,
    field_identifier: str | None = None,
    qcl_task_description: str | None = None,
) -> dict[str, Any]:
    """Check if MetersetMap QA has been completed for a patient's treatment.

    This tool scans the configured output directory for existing MetersetMap
    reports, checks Mosaiq QCL (Quality Checklist) status, and compares against
    treatment delivery history to determine if QA is complete, pending, or overdue.

    Parameters
    ----------
    patient_id : str
        Patient ID to check
    output_directory : str
        Directory where MetersetMap PDF/PNG results are stored
        (e.g., ~/pymedphys-gui-metersetmap)
    mosaiq_connection : optional
        Mosaiq connection for checking treatment history and QCL status
    site_id : str, optional
        Specific site ID to check (if not provided, checks all sites)
    field_identifier : str, optional
        Specific field identifier to check
    qcl_task_description : str, optional
        QCL task description to search for (e.g., "MetersetMap Check",
        "Physics Check", "IMRT QA"). If provided, checks Mosaiq QCL for
        completion status.

    Returns
    -------
    dict
        Status information including:
        - has_completed_check: bool - Whether a map check exists (file or QCL)
        - check_files: list - Paths to existing check files
        - qcl_status: dict - Mosaiq QCL checklist status (if connected)
        - treatment_status: dict - Treatment delivery info from Mosaiq
        - needs_check: bool - Whether a check is needed
        - reason: str - Explanation of the status
    """
    import os
    from datetime import datetime
    from pathlib import Path

    output_path = Path(os.path.expanduser(output_directory)).resolve()

    result = {
        "patient_id": patient_id,
        "output_directory": str(output_path),
        "has_completed_check": False,
        "has_file_check": False,
        "has_qcl_check": False,
        "check_files": [],
        "qcl_status": None,
        "treatment_status": None,
        "needs_check": False,
        "reason": "",
    }

    # Search for existing map check files matching patient ID
    if output_path.exists():
        # Look for PDF files with patient ID in name
        pdf_files = list(output_path.glob(f"{patient_id}*.pdf"))
        png_dirs = [
            d
            for d in output_path.iterdir()
            if d.is_dir() and d.name.startswith(patient_id)
        ]

        check_files = []
        for pdf in pdf_files:
            stat = pdf.stat()
            check_files.append(
                {
                    "path": str(pdf),
                    "type": "pdf",
                    "filename": pdf.name,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        for png_dir in png_dirs:
            report_png = png_dir / "report.png"
            if report_png.exists():
                stat = report_png.stat()
                check_files.append(
                    {
                        "path": str(png_dir),
                        "type": "png_directory",
                        "filename": png_dir.name,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        result["check_files"] = check_files
        result["has_file_check"] = len(check_files) > 0

    # If Mosaiq connection available, check QCL status and treatment status
    if mosaiq_connection is not None:
        try:
            treatment_status = await _get_treatment_status(
                mosaiq_connection, patient_id, site_id
            )
            result["treatment_status"] = treatment_status

            # Check QCL (Quality Checklist) status if task description provided
            if qcl_task_description:
                qcl_status = await _get_qcl_status(
                    mosaiq_connection, patient_id, qcl_task_description
                )
                result["qcl_status"] = qcl_status
                result["has_qcl_check"] = qcl_status.get("has_completed_qcl", False)

            # Determine overall completion status (file OR QCL)
            result["has_completed_check"] = (
                result["has_file_check"] or result["has_qcl_check"]
            )

            # Build status message including QCL info
            qcl_info = ""
            if result["qcl_status"]:
                if result["has_qcl_check"]:
                    qcl_info = (
                        f" QCL completed: {result['qcl_status'].get('completed_date')}."
                    )
                else:
                    pending = result["qcl_status"].get("pending_count", 0)
                    if pending > 0:
                        qcl_info = f" QCL pending: {pending} item(s)."

            # Determine if check is needed
            if treatment_status.get("has_rt_plan") and not treatment_status.get(
                "has_treatments"
            ):
                # RT Plan imported but no treatments yet
                if not result["has_completed_check"]:
                    result["needs_check"] = True
                    result["reason"] = (
                        "RT Plan imported but no MetersetMap check found. "
                        f"Check should be completed before first treatment.{qcl_info}"
                    )
                else:
                    result["needs_check"] = False
                    result["reason"] = (
                        f"MetersetMap check completed, ready for treatment.{qcl_info}"
                    )

            elif treatment_status.get("has_treatments"):
                # Treatment has started
                if not result["has_completed_check"]:
                    result["needs_check"] = True
                    result["reason"] = (
                        "URGENT: Treatment has started but no MetersetMap check found! "
                        f"First treatment: {treatment_status.get('first_treatment_date')}.{qcl_info}"
                    )
                else:
                    # Check if check was done before first treatment
                    first_tx_date = treatment_status.get("first_treatment_date")
                    check_date = None

                    # Use QCL date if available, otherwise use file date
                    if result["has_qcl_check"] and result["qcl_status"].get(
                        "completed_date"
                    ):
                        check_date = result["qcl_status"]["completed_date"]
                    elif result["check_files"]:
                        latest_check = max(
                            result["check_files"], key=lambda x: x["modified"]
                        )
                        check_date = latest_check["modified"]

                    if check_date and first_tx_date and check_date < first_tx_date:
                        result["needs_check"] = False
                        result["reason"] = (
                            f"MetersetMap check completed before treatment started.{qcl_info}"
                        )
                    elif check_date:
                        result["reason"] = (
                            f"WARNING: MetersetMap check ({check_date}) was done "
                            f"after first treatment ({first_tx_date}). "
                            f"Review institutional policy.{qcl_info}"
                        )
                    else:
                        result["reason"] = (
                            f"MetersetMap check completed (no timestamp available).{qcl_info}"
                        )
            else:
                result["reason"] = (
                    f"No RT Plan or treatments found for this patient.{qcl_info}"
                )

        except Exception as e:
            result["treatment_status"] = {"error": str(e)}
    else:
        # No Mosaiq connection - can only check files
        result["has_completed_check"] = result["has_file_check"]
        if result["has_completed_check"]:
            result["reason"] = (
                f"Found {len(result['check_files'])} MetersetMap check(s). "
                "Connect to Mosaiq to verify against treatment schedule and QCL status."
            )
        else:
            result["reason"] = (
                "No MetersetMap checks found. "
                "Connect to Mosaiq to determine if check is needed and check QCL status."
            )

    return result


async def _get_qcl_status(
    connection: Any, patient_id: str, task_description: str
) -> dict[str, Any]:
    """Get QCL (Quality Checklist) status from Mosaiq for a patient.

    Parameters
    ----------
    connection : Any
        Active Mosaiq database connection
    patient_id : str
        Patient ID to check
    task_description : str
        QCL task description to search for (e.g., "MetersetMap Check")

    Returns
    -------
    dict
        QCL status including:
        - has_completed_qcl: bool - Whether QCL is completed
        - completed_date: str - Date QCL was completed (if applicable)
        - pending_count: int - Number of pending QCL items
        - qcl_items: list - Details of matching QCL items
    """
    import pymedphys

    result = {
        "has_completed_qcl": False,
        "completed_date": None,
        "pending_count": 0,
        "qcl_items": [],
        "task_searched": task_description,
    }

    # Query for QCL items matching the task description for this patient
    query = """
    SELECT
        Chklist.Chk_id as checklist_id,
        QCLTask.Description as task_description,
        Chklist.Complete as is_complete,
        Chklist.Due_DtTm as due_date,
        Chklist.Act_DtTm as completed_date,
        Chklist.Notes as notes,
        Com_Staff.Last_Name as completed_by_last,
        Com_Staff.First_Name as completed_by_first
    FROM Chklist
    INNER JOIN Patient Pat ON Chklist.Pat_ID1 = Pat.Pat_ID1
    INNER JOIN QCLTask ON QCLTask.TSK_ID = Chklist.TSK_ID
    LEFT JOIN Staff Com_Staff ON Com_Staff.Staff_ID = Chklist.Com_Staff_ID
    WHERE Pat.Pat_ID1 = %s
      AND QCLTask.Description LIKE %s
    ORDER BY Chklist.Act_DtTm DESC, Chklist.Due_DtTm DESC
    """

    # Use wildcard matching for task description
    task_pattern = f"%{task_description}%"

    try:
        results = pymedphys.mosaiq.execute(
            connection, query, [patient_id, task_pattern]
        )

        for row in results:
            qcl_item = {
                "checklist_id": row[0],
                "task_description": row[1],
                "is_complete": bool(row[2]) if row[2] is not None else False,
                "due_date": str(row[3]) if row[3] else None,
                "completed_date": str(row[4]) if row[4] else None,
                "notes": row[5],
                "completed_by": (f"{row[7]} {row[6]}" if row[6] and row[7] else None),
            }
            result["qcl_items"].append(qcl_item)

            if qcl_item["is_complete"]:
                result["has_completed_qcl"] = True
                if qcl_item["completed_date"] and not result["completed_date"]:
                    result["completed_date"] = qcl_item["completed_date"]
            else:
                result["pending_count"] += 1

    except Exception as e:
        result["error"] = str(e)

    return result


async def _get_treatment_status(
    connection: Any, patient_id: str, site_id: str | None = None
) -> dict[str, Any]:
    """Get treatment status from Mosaiq for a patient.

    Returns information about RT Plans, treatment sites, and delivery history.
    """
    import pymedphys

    result = {
        "has_rt_plan": False,
        "has_treatments": False,
        "first_treatment_date": None,
        "last_treatment_date": None,
        "total_fractions_delivered": 0,
        "sites": [],
    }

    # Query for patient's sites and RT Plans
    site_query = """
    SELECT
        Site.SIT_ID,
        Site.Site_Name,
        Site.Version,
        (SELECT COUNT(*) FROM Study
         WHERE Study.Pat_ID1 = Pat.Pat_ID1
         AND Study.Modality = 'RTPLAN') as rt_plan_count
    FROM Site
    INNER JOIN Patient Pat ON Site.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    """

    if site_id:
        site_query += " AND Site.SIT_ID = %s"
        params = [patient_id, site_id]
    else:
        params = [patient_id]

    site_results = pymedphys.mosaiq.execute(connection, site_query, params)

    for row in site_results:
        site_info = {
            "site_id": row[0],
            "site_name": row[1],
            "version": row[2],
            "rt_plan_count": row[3],
        }
        result["sites"].append(site_info)
        if row[3] > 0:
            result["has_rt_plan"] = True

    # Query for treatment history (Dose_Hst)
    tx_query = """
    SELECT
        MIN(Dose_Hst.Tx_DtTm) as first_treatment,
        MAX(Dose_Hst.Tx_DtTm) as last_treatment,
        COUNT(*) as fraction_count
    FROM Dose_Hst
    INNER JOIN Site ON Dose_Hst.SIT_ID = Site.SIT_ID
    INNER JOIN Patient Pat ON Site.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    """

    if site_id:
        tx_query += " AND Site.SIT_ID = %s"

    tx_results = pymedphys.mosaiq.execute(connection, tx_query, params)

    if tx_results and tx_results[0][0]:
        result["has_treatments"] = True
        result["first_treatment_date"] = str(tx_results[0][0])
        result["last_treatment_date"] = str(tx_results[0][1])
        result["total_fractions_delivered"] = tx_results[0][2]

    return result


async def find_pending_metersetmap_checks(
    output_directory: str,
    mosaiq_connection: Any,
    days_threshold: int = 7,
    limit: int = 50,
) -> dict[str, Any]:
    """Find patients with RT Plans that need MetersetMap QA checks.

    Scans Mosaiq for sites with RT Plans imported in the last N days
    that don't have corresponding MetersetMap check files.

    Parameters
    ----------
    output_directory : str
        Directory where MetersetMap results are stored
    mosaiq_connection : Any
        Active Mosaiq database connection
    days_threshold : int
        Look for RT Plans imported within this many days
    limit : int
        Maximum number of results to return

    Returns
    -------
    dict
        List of patients/sites needing MetersetMap checks
    """
    import os
    from pathlib import Path

    import pymedphys

    output_path = Path(os.path.expanduser(output_directory)).resolve()

    # Find recent RT Plan imports
    query = """
    SELECT TOP %s
        Pat.Pat_ID1 as patient_id,
        Pat.Last_Name as last_name,
        Pat.First_Name as first_name,
        Site.SIT_ID as site_id,
        Site.Site_Name as site_name,
        Study.Study_DtTm as import_date,
        (SELECT COUNT(*) FROM TxField WHERE TxField.SIT_ID = Site.SIT_ID) as field_count,
        (SELECT COUNT(*) FROM Dose_Hst WHERE Dose_Hst.SIT_ID = Site.SIT_ID) as treatment_count
    FROM Study
    INNER JOIN Patient Pat ON Study.Pat_ID1 = Pat.Pat_ID1
    LEFT JOIN Site ON Site.Pat_ID1 = Pat.Pat_ID1
    WHERE Study.Modality = 'RTPLAN'
      AND Study.Study_DtTm >= DATEADD(day, -%s, GETDATE())
    ORDER BY Study.Study_DtTm DESC
    """

    results = pymedphys.mosaiq.execute(
        mosaiq_connection, query, [limit, days_threshold]
    )

    pending_checks = []
    completed_checks = []

    for row in results:
        patient_id = str(row[0])

        # Check if MetersetMap exists for this patient
        has_check = False
        if output_path.exists():
            pdf_files = list(output_path.glob(f"{patient_id}*.pdf"))
            has_check = len(pdf_files) > 0

        patient_info = {
            "patient_id": patient_id,
            "patient_name": f"{row[2]} {row[1]}",
            "site_id": row[3],
            "site_name": row[4],
            "import_date": str(row[5]) if row[5] else None,
            "field_count": row[6],
            "treatment_count": row[7],
            "has_metersetmap_check": has_check,
        }

        if has_check:
            completed_checks.append(patient_info)
        else:
            # Prioritize by urgency
            if row[7] > 0:  # Has treatments
                patient_info["urgency"] = "CRITICAL"
                patient_info["urgency_reason"] = "Treatment started without check"
            elif row[6] > 0:  # Has TX fields
                patient_info["urgency"] = "HIGH"
                patient_info["urgency_reason"] = "Ready for treatment, needs check"
            else:
                patient_info["urgency"] = "NORMAL"
                patient_info["urgency_reason"] = "RT Plan imported, awaiting setup"

            pending_checks.append(patient_info)

    # Sort by urgency
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2}
    pending_checks.sort(key=lambda x: urgency_order.get(x["urgency"], 3))

    return {
        "pending_checks": pending_checks,
        "completed_checks": completed_checks,
        "summary": {
            "total_pending": len(pending_checks),
            "total_completed": len(completed_checks),
            "critical_count": len(
                [p for p in pending_checks if p["urgency"] == "CRITICAL"]
            ),
            "high_priority_count": len(
                [p for p in pending_checks if p["urgency"] == "HIGH"]
            ),
        },
        "output_directory": str(output_path),
        "days_threshold": days_threshold,
    }
