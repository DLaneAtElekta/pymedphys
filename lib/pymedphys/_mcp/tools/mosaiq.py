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

"""Mosaiq database tools for MCP server."""

from typing import Any


async def query_mosaiq(
    query_type: str,
    parameters: dict[str, Any] | None = None,
    sql: str | None = None,
    connection: Any | None = None,
) -> dict[str, Any]:
    """Execute a query against the Mosaiq database.

    Parameters
    ----------
    query_type : str
        Query type: 'patient', 'field', 'treatment', 'machine', or 'custom'
    parameters : dict, optional
        Query parameters
    sql : str, optional
        Custom SQL query (only for query_type='custom')
    connection : optional
        Mosaiq database connection

    Returns
    -------
    dict
        Query results
    """
    import pymedphys

    if connection is None:
        return {"error": "No Mosaiq connection available"}

    parameters = parameters or {}

    try:
        if query_type == "patient":
            return await _query_patient(connection, parameters)
        elif query_type == "field":
            return await _query_field(connection, parameters)
        elif query_type == "treatment":
            return await _query_treatment(connection, parameters)
        elif query_type == "machine":
            return await _query_machine(connection, parameters)
        elif query_type == "custom":
            if not sql:
                return {"error": "SQL query required for 'custom' query_type"}
            return await _query_custom(connection, sql, parameters)
        else:
            return {"error": f"Unknown query_type: {query_type}"}
    except Exception as e:
        return {"error": str(e)}


async def _query_patient(connection: Any, parameters: dict) -> dict:
    """Query patient information."""
    import pymedphys

    patient_id = parameters.get("patient_id")
    search_name = parameters.get("search_name")
    limit = parameters.get("limit", 100)

    if patient_id:
        query = """
        SELECT
            Pat.Pat_ID1, Pat.Last_Name, Pat.First_Name,
            Pat.Middle_Name, Pat.Birth_DtTm, Pat.Gender
        FROM Patient Pat
        WHERE Pat.Pat_ID1 = %s
        """
        results = pymedphys.mosaiq.execute(connection, query, [patient_id])
    elif search_name:
        query = """
        SELECT TOP %s
            Pat.Pat_ID1, Pat.Last_Name, Pat.First_Name,
            Pat.Middle_Name, Pat.Birth_DtTm, Pat.Gender
        FROM Patient Pat
        WHERE Pat.Last_Name LIKE %s OR Pat.First_Name LIKE %s
        ORDER BY Pat.Last_Name, Pat.First_Name
        """
        search_pattern = f"%{search_name}%"
        results = pymedphys.mosaiq.execute(
            connection, query, [limit, search_pattern, search_pattern]
        )
    else:
        query = """
        SELECT TOP %s
            Pat.Pat_ID1, Pat.Last_Name, Pat.First_Name,
            Pat.Middle_Name, Pat.Birth_DtTm, Pat.Gender
        FROM Patient Pat
        ORDER BY Pat.Pat_ID1 DESC
        """
        results = pymedphys.mosaiq.execute(connection, query, [limit])

    patients = [
        {
            "patient_id": row[0],
            "last_name": row[1],
            "first_name": row[2],
            "middle_name": row[3],
            "birth_date": str(row[4]) if row[4] else None,
            "gender": row[5],
        }
        for row in results
    ]

    return {"patients": patients, "count": len(patients)}


async def _query_field(connection: Any, parameters: dict) -> dict:
    """Query treatment field information."""
    import pymedphys

    field_id = parameters.get("field_id")
    patient_id = parameters.get("patient_id")

    if field_id:
        query = """
        SELECT
            TxField.FLD_ID, TxField.Field_Label, TxField.Field_Name,
            TxField.Meterset, TxField.Gantry_Ang, TxField.Coll_Ang,
            TxField.Couch_Ang, TxField.Energy, TxField.Technique,
            Machine.Machine_Name
        FROM TxField
        LEFT JOIN Machine ON TxField.Machine_ID = Machine.Machine_ID
        WHERE TxField.FLD_ID = %s
        """
        results = pymedphys.mosaiq.execute(connection, query, [field_id])
    elif patient_id:
        query = """
        SELECT
            TxField.FLD_ID, TxField.Field_Label, TxField.Field_Name,
            TxField.Meterset, TxField.Gantry_Ang, TxField.Coll_Ang,
            TxField.Couch_Ang, TxField.Energy, TxField.Technique,
            Machine.Machine_Name
        FROM TxField
        LEFT JOIN Machine ON TxField.Machine_ID = Machine.Machine_ID
        INNER JOIN Site ON TxField.SIT_ID = Site.SIT_ID
        INNER JOIN Patient Pat ON Site.Pat_ID1 = Pat.Pat_ID1
        WHERE Pat.Pat_ID1 = %s
        ORDER BY TxField.FLD_ID
        """
        results = pymedphys.mosaiq.execute(connection, query, [patient_id])
    else:
        return {"error": "Either field_id or patient_id required"}

    fields = [
        {
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
        for row in results
    ]

    return {"fields": fields, "count": len(fields)}


async def _query_treatment(connection: Any, parameters: dict) -> dict:
    """Query treatment delivery records."""
    import pymedphys

    patient_id = parameters.get("patient_id")
    field_id = parameters.get("field_id")
    date_from = parameters.get("date_from")
    date_to = parameters.get("date_to")
    limit = parameters.get("limit", 100)

    base_query = """
    SELECT TOP %s
        TxFieldPoint.TFP_ID,
        TxFieldPoint.FLD_ID,
        TxFieldPoint.Tx_DtTm,
        TxFieldPoint.Meterset,
        TxFieldPoint.Actual_Meterset,
        TxField.Field_Label,
        Machine.Machine_Name,
        Pat.Pat_ID1
    FROM TxFieldPoint
    INNER JOIN TxField ON TxFieldPoint.FLD_ID = TxField.FLD_ID
    LEFT JOIN Machine ON TxField.Machine_ID = Machine.Machine_ID
    INNER JOIN Site ON TxField.SIT_ID = Site.SIT_ID
    INNER JOIN Patient Pat ON Site.Pat_ID1 = Pat.Pat_ID1
    WHERE 1=1
    """

    params = [limit]
    conditions = []

    if patient_id:
        conditions.append("AND Pat.Pat_ID1 = %s")
        params.append(patient_id)

    if field_id:
        conditions.append("AND TxFieldPoint.FLD_ID = %s")
        params.append(field_id)

    if date_from:
        conditions.append("AND TxFieldPoint.Tx_DtTm >= %s")
        params.append(date_from)

    if date_to:
        conditions.append("AND TxFieldPoint.Tx_DtTm <= %s")
        params.append(date_to)

    query = base_query + " ".join(conditions) + " ORDER BY TxFieldPoint.Tx_DtTm DESC"

    results = pymedphys.mosaiq.execute(connection, query, params)

    treatments = [
        {
            "treatment_id": row[0],
            "field_id": row[1],
            "treatment_datetime": str(row[2]) if row[2] else None,
            "planned_meterset": float(row[3]) if row[3] else None,
            "actual_meterset": float(row[4]) if row[4] else None,
            "field_label": row[5],
            "machine_name": row[6],
            "patient_id": row[7],
        }
        for row in results
    ]

    return {"treatments": treatments, "count": len(treatments)}


async def _query_machine(connection: Any, parameters: dict) -> dict:
    """Query machine information."""
    import pymedphys

    machine_name = parameters.get("machine_name")

    if machine_name:
        query = """
        SELECT
            Machine.Machine_ID, Machine.Machine_Name,
            Machine.Machine_Type, Machine.Active
        FROM Machine
        WHERE Machine.Machine_Name = %s
        """
        results = pymedphys.mosaiq.execute(connection, query, [machine_name])
    else:
        query = """
        SELECT
            Machine.Machine_ID, Machine.Machine_Name,
            Machine.Machine_Type, Machine.Active
        FROM Machine
        ORDER BY Machine.Machine_Name
        """
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

    return {"machines": machines, "count": len(machines)}


async def _query_custom(connection: Any, sql: str, parameters: dict) -> dict:
    """Execute a custom SQL query."""
    import pymedphys

    # Basic SQL injection prevention
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    sql_upper = sql.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return {"error": f"Dangerous SQL keyword not allowed: {keyword}"}

    # Execute query
    query_params = parameters.get("query_params", [])
    results = pymedphys.mosaiq.execute(connection, sql, query_params)

    # Convert to list of dicts if possible
    if results:
        # Return as list of tuples with column indices
        return {
            "results": [list(row) for row in results],
            "count": len(results),
            "note": "Column names not available for custom queries",
        }
    else:
        return {"results": [], "count": 0}


async def get_patient_chart(
    patient_id: str,
    include_sections: list[str] | None = None,
    connection: Any | None = None,
) -> dict[str, Any]:
    """Retrieve comprehensive patient chart data from Mosaiq.

    Parameters
    ----------
    patient_id : str
        Patient ID in Mosaiq
    include_sections : list of str, optional
        Sections to include: 'demographics', 'prescriptions', 'treatments', 'qa', 'documents'
    connection : optional
        Mosaiq database connection

    Returns
    -------
    dict
        Patient chart data organized by section
    """
    import pymedphys

    if connection is None:
        return {"error": "No Mosaiq connection available"}

    include_sections = include_sections or ["demographics", "prescriptions", "treatments"]

    chart = {
        "patient_id": patient_id,
        "sections": {},
    }

    try:
        if "demographics" in include_sections:
            chart["sections"]["demographics"] = await _get_demographics(
                connection, patient_id
            )

        if "prescriptions" in include_sections:
            chart["sections"]["prescriptions"] = await _get_prescriptions(
                connection, patient_id
            )

        if "treatments" in include_sections:
            treatment_result = await _query_treatment(
                connection, {"patient_id": patient_id, "limit": 50}
            )
            chart["sections"]["treatments"] = treatment_result

        if "qa" in include_sections:
            chart["sections"]["qa"] = await _get_qa_records(connection, patient_id)

        if "documents" in include_sections:
            chart["sections"]["documents"] = await _get_documents(
                connection, patient_id
            )

    except Exception as e:
        chart["error"] = str(e)

    return chart


async def _get_demographics(connection: Any, patient_id: str) -> dict:
    """Get patient demographics."""
    import pymedphys

    query = """
    SELECT
        Pat.Pat_ID1, Pat.Last_Name, Pat.First_Name, Pat.Middle_Name,
        Pat.Birth_DtTm, Pat.Gender, Pat.SSN,
        Pat.Adr1, Pat.Adr2, Pat.City, Pat.State_Province, Pat.Postal
    FROM Patient Pat
    WHERE Pat.Pat_ID1 = %s
    """

    results = pymedphys.mosaiq.execute(connection, query, [patient_id])
    if not results:
        return {"error": "Patient not found"}

    row = results[0]
    return {
        "patient_id": row[0],
        "last_name": row[1],
        "first_name": row[2],
        "middle_name": row[3],
        "birth_date": str(row[4]) if row[4] else None,
        "gender": row[5],
        "address": {
            "line1": row[7],
            "line2": row[8],
            "city": row[9],
            "state": row[10],
            "postal": row[11],
        },
    }


async def _get_prescriptions(connection: Any, patient_id: str) -> dict:
    """Get patient prescriptions."""
    import pymedphys

    query = """
    SELECT
        PCP.PCP_ID, PCP.Label, PCP.Total_Dose, PCP.Daily_Dose,
        PCP.Fractions, PCP.Pri_Diag, PCP.Note,
        Site.Site_Name, Site.Target_Vol
    FROM PCP
    LEFT JOIN Site ON PCP.SIT_ID = Site.SIT_ID
    INNER JOIN Patient Pat ON PCP.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    ORDER BY PCP.PCP_ID
    """

    results = pymedphys.mosaiq.execute(connection, query, [patient_id])

    prescriptions = [
        {
            "prescription_id": row[0],
            "label": row[1],
            "total_dose_cGy": float(row[2]) if row[2] else None,
            "daily_dose_cGy": float(row[3]) if row[3] else None,
            "fractions": int(row[4]) if row[4] else None,
            "diagnosis": row[5],
            "notes": row[6],
            "site_name": row[7],
            "target_volume": row[8],
        }
        for row in results
    ]

    return {"prescriptions": prescriptions, "count": len(prescriptions)}


async def _get_qa_records(connection: Any, patient_id: str) -> dict:
    """Get patient QA records."""
    import pymedphys

    # This query structure may vary based on Mosaiq version
    query = """
    SELECT TOP 50
        QA.QA_ID, QA.QA_Type, QA.QA_DtTm,
        QA.Status, QA.Note,
        Staff.Last_Name as reviewer_name
    FROM QA
    LEFT JOIN Staff ON QA.Staff_ID = Staff.Staff_ID
    INNER JOIN Patient Pat ON QA.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    ORDER BY QA.QA_DtTm DESC
    """

    try:
        results = pymedphys.mosaiq.execute(connection, query, [patient_id])
        qa_records = [
            {
                "qa_id": row[0],
                "qa_type": row[1],
                "qa_datetime": str(row[2]) if row[2] else None,
                "status": row[3],
                "notes": row[4],
                "reviewer": row[5],
            }
            for row in results
        ]
        return {"qa_records": qa_records, "count": len(qa_records)}
    except Exception:
        # QA table structure varies between Mosaiq versions
        return {"qa_records": [], "count": 0, "note": "QA query not available"}


async def _get_documents(connection: Any, patient_id: str) -> dict:
    """Get patient documents list."""
    import pymedphys

    query = """
    SELECT TOP 50
        Doc.Doc_ID, Doc.Doc_Type, Doc.Doc_DtTm,
        Doc.Description, Doc.File_Name
    FROM Document Doc
    INNER JOIN Patient Pat ON Doc.Pat_ID1 = Pat.Pat_ID1
    WHERE Pat.Pat_ID1 = %s
    ORDER BY Doc.Doc_DtTm DESC
    """

    try:
        results = pymedphys.mosaiq.execute(connection, query, [patient_id])
        documents = [
            {
                "document_id": row[0],
                "document_type": row[1],
                "document_datetime": str(row[2]) if row[2] else None,
                "description": row[3],
                "file_name": row[4],
            }
            for row in results
        ]
        return {"documents": documents, "count": len(documents)}
    except Exception:
        return {"documents": [], "count": 0, "note": "Document query not available"}
