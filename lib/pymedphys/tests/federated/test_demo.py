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

import subprocess
import sys
import textwrap

import numpy as np

from pymedphys._federated import demo, simulate


def test_the_demo_runs_and_reports_each_guard(tmp_path, capsys):
    assert demo.main(audit_dir=tmp_path / "audit") == 0

    printed = capsys.readouterr().out

    assert "rejected before round 1" in printed
    assert printed.count("stopped at the aperture") == 2
    assert "emissions" in printed


def test_the_demo_federation_converges_on_the_pooled_mean(tmp_path):
    history = simulate.run_federation(
        demo.build_federation(tmp_path), rounds=3, config_fn=lambda _: {}
    )

    # With a full local step, FedAvg weighted by cohort size is the pooled
    # mean exactly. Anything else is a plumbing failure rather than an
    # optimiser one.
    (federated,) = history.weights

    assert np.allclose(federated, demo.pooled_mean())


def test_the_public_surface_imports_without_numpy_or_a_framework():
    # The dependency policy: `pymedphys.beta.federated` must be importable,
    # documented and testable in an environment with neither torch nor flwr,
    # and must not drag heavyweight imports in at module import time.
    script = textwrap.dedent(
        """
        import sys

        import pymedphys.beta.federated as federated

        assert federated.ClinicManifest is not None
        assert federated.Aperture is not None

        eager = sorted(
            name
            for name in ("numpy", "pydicom", "torch", "flwr", "pandas")
            if name in sys.modules
        )
        assert not eager, f"eagerly imported {eager}"
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
