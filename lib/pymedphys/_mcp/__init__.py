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

"""PyMedPhys MCP (Model Context Protocol) Server.

This module provides an MCP interface for PyMedPhys, enabling AI models
to interact with medical physics data and tools.

Resources:
    - Patient chart data from Mosaiq OIS
    - DICOM files (RT Plan, RT Dose, RT Struct, CT)
    - TRF/log file data from Elekta linacs

Tools:
    - Gamma analysis for dose comparison
    - MetersetMap calculation
    - DICOM anonymization
    - DICOM file generation (RT Plan, RTPCONNECT)
    - Mosaiq database queries
    - TRF file parsing
"""

from .server import create_server, run_server

__all__ = ["create_server", "run_server"]
