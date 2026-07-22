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

"""A Mosaiq-backed DICOMweb service.

This package exposes the treatment data held within an Elekta Mosaiq
oncology information system (OIS) over the DICOMweb_ set of RESTful
services. The Mosaiq SQL schema is translated, on the fly, into the
DICOM JSON model (PS3.18 Annex F) so that DICOMweb clients can query
Mosaiq as though it were a conventional DICOM archive.

Three DICOMweb service classes make up the surface:

* **QIDO-RS** (*Query based on ID for DICOM Objects*) -- search for
  studies, series and instances. This is the primary capability of a
  Mosaiq-backed service because Mosaiq is fundamentally a queryable
  relational database of patient and treatment metadata.
* **WADO-RS** (*Web Access to DICOM Objects*) -- retrieve DICOM objects.
  Scaffolded but not implemented: Mosaiq does not itself store the pixel
  data / composite SOP instances that WADO-RS returns.
* **STOW-RS** (*Store Over the Web*) -- store DICOM objects. Scaffolded
  but not implemented.

.. _DICOMweb: https://www.dicomstandard.org/using/dicomweb
"""
