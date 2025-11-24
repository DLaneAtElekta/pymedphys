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

"""Mosaiq resource handlers for MCP server."""

import json
from typing import Any
from urllib.parse import urlparse


async def read_resource(uri: str, connection: Any) -> str:
    """Read a Mosaiq resource by URI.

    Parameters
    ----------
    uri : str
        Resource URI in format mosaiq://resource_type[/parameters]
    connection : Any
        Active Mosaiq database connection

    Returns
    -------
    str
        JSON-encoded resource data
    """
    if connection is None:
        return json.dumps({"error": "No Mosaiq connection available"})

    parsed = urlparse(uri)
    resource_path = parsed.netloc + parsed.path

    try:
        if resource_path == "patients":
            return await _get_patient_list(connection)
        elif resource_path == "machines":
            return await _get_machine_list(connection)
        elif resource_path.startswith("patient/"):
            patient_id = resource_path.split("/")[1]
            return await _get_patient_data(connection, patient_id)
        elif resource_path.startswith("field/"):
            field_id = resource_path.split("/")[1]
            return await _get_field_data(connection, field_id)
        else:
            return json.dumps({"error": f"Unknown Mosaiq resource: {resource_path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_patient_list(connection: Any) -> str:
    """Get list of patients from Mosaiq."""
    import pymedphys

    # Query for recent patients with activity
    query = """
    SELECT TOP 100
        Pat.Pat_ID1 as patient_id,
        Pat.Last_Name as last_name,
        Pat.First_Name as first_name,
        Pat.Pat_Name as full_name,
        Pat.Birth_DtTm as birth_date
    FROM Patient Pat
    WHERE Pat.Pat_ID1 IS NOT NULL
    ORDER BY Pat.Pat_ID1 DESC
    """

    try:
        results = pymedphys.mosaiq.execute(connection, query)
        patients = [
            {
                "patient_id": row[0],
                "last_name": row[1],
                "first_name": row[2],
                "full_name": row[3],
                "birth_date": str(row[4]) if row[4] else None,
            }
            for row in results
        ]
        return json.dumps({"patients": patients, "count": len(patients)})
    except Exception as e:
        return json.dumps({"error": f"Failed to query patients: {str(e)}"})


async def _get_machine_list(connection: Any) -> str:
    """Get list of treatment machines from Mosaiq."""
    import pymedphys

    query = """
    SELECT
        Machine.Machine_ID as machine_id,
        Machine.Machine_Name as machine_name,
        Machine.Machine_Type as machine_type,
        Machine.Active as is_active
    FROM Machine
    WHERE Machine.Machine_Name IS NOT NULL
    ORDER BY Machine.Machine_Name
    """

    try:
        results = pymedphys.mosaiq.execute(connection, query)
        machines = [
            {
                "machine_id": row[0],
                "machine_name": row[1],
                "machine_type": row[2],
                "is_active": bool(row[3]) if row[3] is not None else None,
            }
            for row in results
        ]
        return json.dumps({"machines": machines, "count": len(machines)})
    except Exception as e:
        return json.dumps({"error": f"Failed to query machines: {str(e)}"})


async def _get_patient_data(connection: Any, patient_id: str) -> str:
    """Get detailed patient data from Mosaiq."""
    import pymedphys

    # Get patient demographics
    demo_query = """
    SELECT
        Pat.Pat_ID1 as patient_id,
        Pat.Last_Name as last_name,
        Pat.First_Name as first_name,
        Pat.Middle_Name as middle_name,
        Pat.Birth_DtTm as birth_date,
        Pat.Gender as gender,
        Pat.SSN as ssn_last4
    FROM Patient Pat
    WHERE Pat.Pat_ID1 = %s
    """

    # Get prescriptions
    rx_query = """
    SELECT
        PCP.PCP_ID as prescription_id,
        PCP.Label as label,
        PCP.Total_Dose as total_dose,
        PCP.Daily_Dose as daily_dose,
        PCP.Fractions as fractions,
        PCP.Pri_Diag as diagnosis,
        Site.Site_Name as site_name
    FROM PCP
    LEFT JOIN Site ON PCP.SIT_ID = Site.SIT_ID
    INNER JOIN Patient Pat ON PCP.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    ORDER BY PCP.PCP_ID DESC
    """

    try:
        # Get demographics
        demo_results = pymedphys.mosaiq.execute(connection, demo_query, [patient_id])
        if not demo_results:
            return json.dumps({"error": f"Patient not found: {patient_id}"})

        demo = demo_results[0]
        patient_data = {
            "patient_id": demo[0],
            "last_name": demo[1],
            "first_name": demo[2],
            "middle_name": demo[3],
            "birth_date": str(demo[4]) if demo[4] else None,
            "gender": demo[5],
        }

        # Get prescriptions
        rx_results = pymedphys.mosaiq.execute(connection, rx_query, [patient_id])
        prescriptions = [
            {
                "prescription_id": row[0],
                "label": row[1],
                "total_dose": float(row[2]) if row[2] else None,
                "daily_dose": float(row[3]) if row[3] else None,
                "fractions": int(row[4]) if row[4] else None,
                "diagnosis": row[5],
                "site_name": row[6],
            }
            for row in rx_results
        ]

        return json.dumps(
            {
                "patient": patient_data,
                "prescriptions": prescriptions,
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to get patient data: {str(e)}"})


async def _get_field_data(connection: Any, field_id: str) -> str:
    """Get treatment field data from Mosaiq."""
    import pymedphys

    query = """
    SELECT
        TxField.FLD_ID as field_id,
        TxField.Field_Label as field_label,
        TxField.Field_Name as field_name,
        TxField.Meterset as meterset,
        TxField.Gantry_Ang as gantry_angle,
        TxField.Coll_Ang as collimator_angle,
        TxField.Couch_Ang as couch_angle,
        TxField.Energy as energy,
        TxField.Technique as technique,
        Machine.Machine_Name as machine_name
    FROM TxField
    LEFT JOIN Machine ON TxField.Machine_ID = Machine.Machine_ID
    WHERE TxField.FLD_ID = %s
    """

    try:
        results = pymedphys.mosaiq.execute(connection, query, [field_id])
        if not results:
            return json.dumps({"error": f"Field not found: {field_id}"})

        row = results[0]
        field_data = {
            "field_id": row[0],
            "field_label": row[1],
            "field_name": row[2],
            "meterset": float(row[3]) if row[3] else None,
            "gantry_angle": float(row[4]) if row[4] else None,
            "collimator_angle": float(row[5]) if row[5] else None,
            "couch_angle": float(row[6]) if row[6] else None,
            "energy": row[7],
            "technique": row[8],
            "machine_name": row[9],
        }

        return json.dumps({"field": field_data})
    except Exception as e:
        return json.dumps({"error": f"Failed to get field data: {str(e)}"})
