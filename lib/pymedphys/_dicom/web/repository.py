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

"""Query the Mosaiq DICOM tables and shape the results as DICOM JSON."""

import datetime
from collections.abc import Callable, Mapping
from typing import Any

from .schema import MosaiqDicomSchema

Execute = Callable[[str, dict], list[tuple]]
"""Runs a pymssql style query (``%(name)s`` parameters) and returns rows."""

DicomJson = dict[str, dict[str, Any]]


def _q(name: str) -> str:
    return f"[{name}]"


def _col(table: str, column: str) -> str:
    return f"{_q(table)}.{_q(column)}"


def _element(vr: str, value) -> dict[str, Any]:
    if value is None or value == "":
        return {"vr": vr}

    return {"vr": vr, "Value": [value]}


def _format_date(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value)

    return value.strftime("%Y%m%d")


def _person_name(last, first) -> str | None:
    if not last and not first:
        return None

    return f"{last or ''}^{first or ''}".rstrip("^")


def _parse_da(value: str) -> datetime.datetime:
    return datetime.datetime.strptime(value, "%Y%m%d")


def _study_date_range(
    value: str,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Parse a QIDO-RS date match: ``YYYYMMDD``, ``A-B``, ``A-`` or ``-B``.

    Returns a half open ``[start, end)`` range so that datetime columns
    with a time component still match on the whole day.
    """
    one_day = datetime.timedelta(days=1)
    if "-" not in value:
        start = _parse_da(value)
        return start, start + one_day

    lower, upper = value.split("-", 1)
    start = _parse_da(lower) if lower else None
    end = _parse_da(upper) + one_day if upper else None

    return start, end


def _wildcard_to_like(value: str) -> str:
    escaped = value.replace("[", "[[]").replace("%", "[%]").replace("_", "[_]")

    return escaped.replace("*", "%").replace("?", "_")


class MosaiqDicomRepository:
    """Read-only access to DICOM objects stored within Mosaiq.

    Parameters
    ----------
    execute : Callable[[str, dict], list[tuple]]
        Runs a query against the Mosaiq database. For a live Mosaiq
        connection use
        ``functools.partial(pymedphys.mosaiq.execute, connection)``.
    schema : MosaiqDicomSchema, optional
        The table and column names to query.
    """

    def __init__(self, execute: Execute, schema: MosaiqDicomSchema | None = None):
        self._execute = execute
        self.schema = schema or MosaiqDicomSchema()

    def search_studies(
        self, filters: Mapping[str, str], limit: int = 100, offset: int = 0
    ) -> list[DicomJson]:
        """QIDO-RS study search.

        Supported match keys are ``PatientID``, ``PatientName``,
        ``StudyInstanceUID``, ``AccessionNumber`` and ``StudyDate``.
        ``PatientID`` and ``PatientName`` accept ``*`` and ``?``
        wildcards. ``PatientName`` matches against the family name.
        """
        s = self.schema
        study, ident, patient = s.study_table, s.ident_table, s.patient_table

        conditions = []
        parameters: dict[str, Any] = {}

        def match(column: str, key: str, wildcards: bool = False):
            value = filters.get(key)
            if not value:
                return
            if wildcards and ("*" in value or "?" in value):
                conditions.append(f"{column} LIKE %({key})s")
                parameters[key] = _wildcard_to_like(value)
            else:
                conditions.append(f"{column} = %({key})s")
                parameters[key] = value

        match(_col(ident, s.patient_id), "PatientID", wildcards=True)
        match(_col(patient, s.patient_last_name), "PatientName", wildcards=True)
        match(_col(study, s.study_instance_uid), "StudyInstanceUID")
        match(_col(study, s.accession_number), "AccessionNumber")

        if filters.get("StudyDate"):
            start, end = _study_date_range(filters["StudyDate"])
            if start is not None:
                conditions.append(f"{_col(study, s.study_date)} >= %(date_start)s")
                parameters["date_start"] = start
            if end is not None:
                conditions.append(f"{_col(study, s.study_date)} < %(date_end)s")
                parameters["date_end"] = end

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT
                {_col(study, s.study_instance_uid)},
                {_col(study, s.study_date)},
                {_col(study, s.accession_number)},
                {_col(study, s.study_description)},
                {_col(ident, s.patient_id)},
                {_col(patient, s.patient_last_name)},
                {_col(patient, s.patient_first_name)}
            FROM {_q(study)}
            JOIN {_q(ident)}
                ON {_col(ident, s.patient_key)} = {_col(study, s.study_patient_key)}
            JOIN {_q(patient)}
                ON {_col(patient, s.patient_key)} = {_col(study, s.study_patient_key)}
            {where}
            ORDER BY {_col(study, s.study_date)} DESC
        """

        # Paging is done here rather than in SQL to keep the query
        # portable. Prefer narrow filters on large databases.
        rows = self._execute(query, parameters)[offset : offset + limit]

        return [
            {
                "00080020": _element("DA", _format_date(date)),
                "00080050": _element("SH", accession),
                "00081030": _element("LO", description),
                "00100010": _element(
                    "PN",
                    {"Alphabetic": name}
                    if (name := _person_name(last, first))
                    else None,
                ),
                "00100020": _element("LO", patient_id),
                "0020000D": _element("UI", study_uid),
            }
            for (
                study_uid,
                date,
                accession,
                description,
                patient_id,
                last,
                first,
            ) in rows
        ]

    def search_series(self, study_uid: str) -> list[DicomJson]:
        """QIDO-RS series search within a single study."""
        s = self.schema
        study, series = s.study_table, s.series_table

        query = f"""
            SELECT
                {_col(series, s.series_instance_uid)},
                {_col(series, s.modality)},
                {_col(series, s.series_number)},
                {_col(series, s.series_description)}
            FROM {_q(series)}
            JOIN {_q(study)}
                ON {_col(study, s.study_key)} = {_col(series, s.series_study_key)}
            WHERE {_col(study, s.study_instance_uid)} = %(study_uid)s
            ORDER BY {_col(series, s.series_number)}
        """
        rows = self._execute(query, {"study_uid": study_uid})

        return [
            {
                "00080060": _element("CS", modality),
                "0008103E": _element("LO", description),
                "0020000D": _element("UI", study_uid),
                "0020000E": _element("UI", series_uid),
                "00200011": _element("IS", number),
            }
            for series_uid, modality, number, description in rows
        ]

    def search_instances(self, study_uid: str, series_uid: str) -> list[DicomJson]:
        """QIDO-RS instance search within a single series."""
        s = self.schema
        study, series, instance = s.study_table, s.series_table, s.instance_table

        query = f"""
            SELECT
                {_col(instance, s.sop_class_uid)},
                {_col(instance, s.sop_instance_uid)},
                {_col(instance, s.instance_number)}
            FROM {_q(instance)}
            {self._instance_joins()}
            WHERE
                {_col(study, s.study_instance_uid)} = %(study_uid)s AND
                {_col(series, s.series_instance_uid)} = %(series_uid)s
            ORDER BY {_col(instance, s.instance_number)}
        """
        rows = self._execute(query, {"study_uid": study_uid, "series_uid": series_uid})

        return [
            {
                "00080016": _element("UI", sop_class_uid),
                "00080018": _element("UI", sop_instance_uid),
                "0020000D": _element("UI", study_uid),
                "0020000E": _element("UI", series_uid),
                "00200013": _element("IS", number),
            }
            for sop_class_uid, sop_instance_uid, number in rows
        ]

    def retrieve_instance(
        self, study_uid: str, series_uid: str, sop_instance_uid: str
    ) -> bytes | None:
        """WADO-RS instance retrieve. Returns the DICOM Part 10 bytes."""
        s = self.schema
        study, series, instance = s.study_table, s.series_table, s.instance_table

        query = f"""
            SELECT {_col(instance, s.instance_data)}
            FROM {_q(instance)}
            {self._instance_joins()}
            WHERE
                {_col(study, s.study_instance_uid)} = %(study_uid)s AND
                {_col(series, s.series_instance_uid)} = %(series_uid)s AND
                {_col(instance, s.sop_instance_uid)} = %(sop_instance_uid)s
        """
        rows = self._execute(
            query,
            {
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_instance_uid": sop_instance_uid,
            },
        )
        if not rows or rows[0][0] is None:
            return None

        return bytes(rows[0][0])

    def _instance_joins(self) -> str:
        s = self.schema
        study, series, instance = s.study_table, s.series_table, s.instance_table

        return f"""
            JOIN {_q(series)}
                ON {_col(series, s.series_key)} = {_col(instance, s.instance_series_key)}
            JOIN {_q(study)}
                ON {_col(study, s.study_key)} = {_col(series, s.series_study_key)}
        """
