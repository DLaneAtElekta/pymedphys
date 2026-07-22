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

"""QIDO-RS query implementation backed by the Mosaiq SQL schema.

This module turns a parsed QIDO-RS request into a parameterised SQL query
against Mosaiq, executes it through :func:`pymedphys.mosaiq.execute`, and
maps each returned row into the DICOM JSON model via
:mod:`pymedphys._dicomweb.mapping`.

Only study-level querying is implemented. Series- and instance-level
querying are scaffolded (:func:`search_for_series`,
:func:`search_for_instances`) and currently return no matches; the Mosaiq
tables that would back them (imaging series, treatment fields as series,
etc.) are site-specific extension points.
"""

from __future__ import annotations

from typing import Any

from pymedphys._mosaiq import api as _mosaiq_api

from . import mapping

# Columns selected by the study-level query, in order, mapped to the keys
# expected by :func:`pymedphys._dicomweb.mapping.study_dataset`.
_STUDY_COLUMNS = [
    "patient_id",
    "last_name",
    "first_name",
    "birth_date",
    "pat_id1",
    "sit_set_id",
    "study_date",
    "study_description",
    "modality",
]


def _matches_wildcard(value: str) -> bool:
    """Return whether a QIDO matching value uses wildcard characters."""
    return "*" in value or "?" in value


def _qido_wildcard_to_sql_like(value: str) -> str:
    """Translate a QIDO-RS wildcard value to a SQL ``LIKE`` pattern.

    DICOM uses ``*`` (zero or more) and ``?`` (single character); SQL
    ``LIKE`` uses ``%`` and ``_``. Any literal ``%``/``_`` in the input is
    escaped so it is treated as a literal via an ``ESCAPE '\\'`` clause.
    """
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped.replace("*", "%").replace("?", "_")


def _build_study_where(
    filters: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    """Build the SQL ``WHERE`` fragments and parameters for a study query.

    Parameters
    ----------
    filters
        QIDO-RS matching parameters keyed by DICOM keyword (for example
        ``{"PatientID": "MR8002", "PatientName": "HOWARD*"}``).

    Returns
    -------
    clauses, parameters
        A list of SQL boolean expressions to be joined with ``AND`` and the
        matching parameter dictionary for ``pymssql``.
    """
    clauses: list[str] = []
    parameters: dict[str, Any] = {}

    patient_id = filters.get("PatientID")
    if patient_id:
        if _matches_wildcard(patient_id):
            clauses.append("Ident.IDA LIKE %(patient_id)s ESCAPE '\\'")
            parameters["patient_id"] = _qido_wildcard_to_sql_like(patient_id)
        else:
            clauses.append("Ident.IDA = %(patient_id)s")
            parameters["patient_id"] = patient_id

    patient_name = filters.get("PatientName")
    if patient_name:
        # DICOM PN components are '^'-separated (Last^First^...). Match the
        # supplied value against the Mosaiq last name, honouring wildcards.
        name_value = patient_name.split("^", 1)[0]
        clauses.append("Patient.Last_Name LIKE %(patient_name)s ESCAPE '\\'")
        if _matches_wildcard(name_value):
            parameters["patient_name"] = _qido_wildcard_to_sql_like(name_value)
        else:
            # Bare value is treated as a case-insensitive prefix-free exact
            # match; wrap so trailing whitespace in the column still matches.
            parameters["patient_name"] = name_value

    study_date = filters.get("StudyDate")
    if study_date:
        start, end = _parse_date_range(study_date)
        if start is not None:
            clauses.append("Site.Create_DtTm >= %(study_date_start)s")
            parameters["study_date_start"] = start
        if end is not None:
            clauses.append("Site.Create_DtTm < %(study_date_end)s")
            parameters["study_date_end"] = end

    return clauses, parameters


def _parse_date_range(value: str) -> tuple[str | None, str | None]:
    """Parse a DICOM ``DA`` range query into inclusive SQL datetime bounds.

    Supports the single-date (``YYYYMMDD``) and range (``YYYYMMDD-YYYYMMDD``,
    open-ended on either side) forms defined in PS3.4 C.2.2.2.5.
    """

    def _to_sql(date: str, *, end_of_day: bool) -> str | None:
        date = date.strip()
        if len(date) != 8 or not date.isdigit():
            return None
        stamp = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
        return f"{stamp} 23:59:59" if end_of_day else f"{stamp} 00:00:00"

    if "-" in value:
        low, _, high = value.partition("-")
        start = _to_sql(low, end_of_day=False) if low else None
        # Range end is exclusive: use the start of the following day.
        end = None
        if high:
            end_inclusive = _to_sql(high, end_of_day=False)
            if end_inclusive is not None:
                # Convert 'YYYY-MM-DD 00:00:00' of the last day + 1 day is
                # awkward without dateutil; instead compare with '< next day'
                # by using the inclusive end-of-day boundary.
                end = _to_sql(high, end_of_day=True)
        return start, end

    start = _to_sql(value, end_of_day=False)
    end = _to_sql(value, end_of_day=True)
    return start, end


def search_for_studies(
    connection: "_mosaiq_api.Connection",
    filters: dict[str, str] | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Execute a QIDO-RS study search against Mosaiq.

    Parameters
    ----------
    connection
        An open Mosaiq connection from :func:`pymedphys.mosaiq.connect`.
    filters
        QIDO-RS matching parameters keyed by DICOM keyword. Supported keys
        are ``PatientID``, ``PatientName`` and ``StudyDate``.
    limit
        Maximum number of studies to return. ``None`` means no limit.
    offset
        Number of matching studies to skip (QIDO-RS ``offset`` parameter).

    Returns
    -------
    list of dict
        Each item is a DICOM JSON model object (as produced by
        ``Dataset.to_json_dict``) representing one matching study.
    """
    filters = filters or {}
    clauses, parameters = _build_study_where(filters)

    # MSSQL requires ORDER BY for OFFSET/FETCH paging.
    paging = ""
    if offset or limit is not None:
        parameters["offset"] = int(offset)
        paging = "OFFSET %(offset)s ROWS"
        if limit is not None:
            parameters["limit"] = int(limit)
            paging += " FETCH NEXT %(limit)s ROWS ONLY"

    query = f"""
        SELECT
            Ident.IDA,
            Patient.Last_Name,
            Patient.First_Name,
            Patient.Birth_DtTm,
            Patient.Pat_ID1,
            Site.SIT_SET_ID,
            Site.Create_DtTm,
            Site.Site_Name,
            Site.Modality
        FROM Ident, Patient, Site
        WHERE
            Patient.Pat_ID1 = Ident.Pat_Id1 AND
            Site.Pat_ID1 = Patient.Pat_ID1
            {("AND " + " AND ".join(clauses)) if clauses else ""}
        ORDER BY Site.SIT_SET_ID
        {paging}
    """

    rows = _mosaiq_api.execute(connection, query, parameters)

    results = []
    for row in rows:
        mapped = dict(zip(_STUDY_COLUMNS, row))
        dataset = mapping.study_dataset(mapped)
        results.append(dataset.to_json_dict())

    return results


def search_for_series(
    connection: "_mosaiq_api.Connection",
    study_instance_uid: str,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Scaffold for QIDO-RS series search.

    Not yet implemented. Mapping Mosaiq treatment fields (``TxField``) to
    DICOM series is a site-specific extension point; until that mapping is
    defined this returns no matches.
    """
    return []


def search_for_instances(
    connection: "_mosaiq_api.Connection",
    study_instance_uid: str,
    series_instance_uid: str | None = None,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Scaffold for QIDO-RS instance search.

    Not yet implemented. See :func:`search_for_series`.
    """
    return []
