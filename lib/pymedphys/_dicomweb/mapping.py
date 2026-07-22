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

"""Translation between Mosaiq SQL rows and the DICOM JSON model.

The functions in this module are deliberately free of any database or
web-framework dependency so that the Mosaiq-to-DICOM mapping can be
unit tested in isolation. They take plain Python values (as returned by
:func:`pymedphys.mosaiq.execute`) and return :class:`pydicom.dataset.Dataset`
objects, which the QIDO-RS layer serialises to the DICOM JSON model via
``Dataset.to_json_dict``.
"""

from __future__ import annotations

import datetime
from typing import Any

from pymedphys._imports import pydicom

# A UID root registered for deterministically-derived UIDs. Until a site
# populates real DICOM UIDs into Mosaiq, study/series/instance UIDs are
# derived deterministically from the Mosaiq primary keys so that repeat
# queries return stable identifiers. Replace ``UID_ROOT`` with an
# organisation-specific root when deploying.
UID_ROOT = "1.2.826.0.1.3680043.8.498."


def derive_uid(*keys: Any) -> str:
    """Deterministically derive a DICOM UID from Mosaiq primary keys.

    Mosaiq does not always store DICOM Study/Series/SOP Instance UIDs for
    the treatment data it holds. Where a real UID is unavailable this
    helper produces a stable UID from the supplied Mosaiq keys, so that
    the same row always maps to the same UID across queries.

    Parameters
    ----------
    *keys
        The Mosaiq identifiers that uniquely determine the object (for
        example ``("study", pat_id1, sit_set_id)``). ``None`` values are
        permitted and are coerced to empty strings.

    Returns
    -------
    str
        A DICOM UID rooted at :data:`UID_ROOT`.
    """
    entropy = [str(key) if key is not None else "" for key in keys]
    return pydicom.uid.generate_uid(prefix=UID_ROOT, entropy_srcs=entropy)


def _to_dicom_date(value: Any) -> str | None:
    """Convert a Mosaiq datetime value to a DICOM ``DA`` string.

    Returns ``None`` when the value is missing so that the caller can omit
    the attribute rather than emit an empty one.
    """
    if value is None or value == "":
        return None

    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.strftime("%Y%m%d")

    text = str(value).strip()
    # Mosaiq typically returns "YYYY-MM-DD HH:MM:SS[.ffffff]".
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).strftime("%Y%m%d")
        except ValueError:
            continue

    return None


def _format_patient_name(last_name: Any, first_name: Any) -> str:
    """Build a DICOM ``PN`` component group from split name parts."""
    last = (last_name or "").strip()
    first = (first_name or "").strip()
    return f"{last}^{first}".rstrip("^")


def study_dataset(row: dict[str, Any]) -> "pydicom.dataset.Dataset":
    """Map a Mosaiq study-level row to a QIDO-RS study :class:`Dataset`.

    Parameters
    ----------
    row
        A mapping of Mosaiq column values with (at minimum) the keys
        ``patient_id``, ``last_name``, ``first_name``, ``birth_date``,
        ``pat_id1``, ``sit_set_id``, ``study_date``, ``study_description``
        and ``modality``.

    Returns
    -------
    pydicom.dataset.Dataset
        A dataset populated with the QIDO-RS study-level matching and
        return attributes described in PS3.18 Section 10.6.
    """
    ds = pydicom.dataset.Dataset()

    ds.SpecificCharacterSet = "ISO_IR 192"

    ds.PatientID = str(row.get("patient_id") or "")
    ds.PatientName = _format_patient_name(row.get("last_name"), row.get("first_name"))

    birth_date = _to_dicom_date(row.get("birth_date"))
    ds.PatientBirthDate = birth_date or ""

    ds.StudyInstanceUID = derive_uid("study", row.get("pat_id1"), row.get("sit_set_id"))

    study_date = _to_dicom_date(row.get("study_date"))
    ds.StudyDate = study_date or ""
    ds.StudyTime = ""
    ds.StudyDescription = str(row.get("study_description") or "")
    ds.AccessionNumber = str(row.get("accession_number") or "")
    ds.StudyID = str(row.get("sit_set_id") or "")

    modality = row.get("modality")
    ds.ModalitiesInStudy = [str(modality)] if modality else []

    # Counts are scaffolded to zero. A full implementation would populate
    # these from the corresponding series/instance queries.
    ds.NumberOfStudyRelatedSeries = 0
    ds.NumberOfStudyRelatedInstances = 0

    return ds
