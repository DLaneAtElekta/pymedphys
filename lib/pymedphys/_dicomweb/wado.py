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

"""WADO-RS (retrieve) scaffold for the Mosaiq-backed DICOMweb service.

WADO-RS returns composite DICOM SOP instances (including pixel data).
Mosaiq is an oncology information system, not a PACS: it does not itself
store the composite instances a WADO-RS request expects. This module is
therefore a scaffold. A full implementation would either

* synthesise DICOM objects (for example RT Plan / RT Beams Treatment
  Record) from Mosaiq treatment data, or
* proxy the retrieve to the archive that does hold the pixel data.
"""

from __future__ import annotations


class WadoNotImplemented(NotImplementedError):
    """Raised when a WADO-RS retrieve is requested against Mosaiq."""


def retrieve_study(study_instance_uid: str):
    """Scaffold for WADO-RS study retrieval.

    Raises
    ------
    WadoNotImplemented
        Always, until a Mosaiq-to-SOP-instance synthesis (or proxy) is
        implemented.
    """
    raise WadoNotImplemented(
        "WADO-RS retrieval is not implemented for the Mosaiq backend."
    )
