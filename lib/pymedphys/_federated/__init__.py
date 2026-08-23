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

"""A site-side boundary for federated learning.

PyMedPhys ships the client half of a federation: the aperture that decides
what may leave a clinic, the contract a local trainer implements, and an
in-process loop to exercise both without a server. Hosting an aggregator is
deliberately out of scope -- sites federate with whoever they choose.
"""
