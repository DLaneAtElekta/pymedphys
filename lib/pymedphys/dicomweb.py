"""A Mosaiq-backed DICOMweb service.

Exposes an Elekta Mosaiq oncology information system over the DICOMweb
QIDO-RS / WADO-RS / STOW-RS services. See :mod:`pymedphys._dicomweb` for
details of what is implemented versus scaffolded.
"""

# pylint: disable = unused-import
# ruff: noqa: F401

from ._dicomweb.qido import (
    search_for_instances,
    search_for_series,
    search_for_studies,
)
from ._dicomweb.server import create_app
