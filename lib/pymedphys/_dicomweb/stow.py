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

"""STOW-RS (store) scaffold for the Mosaiq-backed DICOMweb service.

STOW-RS accepts DICOM objects and stores them. A Mosaiq-backed
implementation would decompose incoming objects and write the relevant
fields back into the Mosaiq schema (for example importing an RT Plan via
the RTPCONNECT pathway). This module is a scaffold that defines the
extension point without performing any writes.
"""

from __future__ import annotations


class StowNotImplemented(NotImplementedError):
    """Raised when a STOW-RS store is requested against Mosaiq."""


def store_instances(study_instance_uid: str | None, datasets):
    """Scaffold for STOW-RS store.

    Raises
    ------
    StowNotImplemented
        Always, until a Mosaiq write-back pathway is implemented.
    """
    raise StowNotImplemented(
        "STOW-RS storage is not implemented for the Mosaiq backend."
    )
