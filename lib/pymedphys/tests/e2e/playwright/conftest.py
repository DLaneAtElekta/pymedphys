# Copyright (C) 2024 Cancer Care Associates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest fixtures for Playwright E2E tests of Streamlit apps."""

import os
import subprocess
import time

import pytest

import pymedphys._utilities.test as pmp_test_utils

# Default port for the Streamlit app
DEFAULT_PORT = 8501
DEFAULT_TIMEOUT = 30  # seconds to wait for the server to start


@pytest.fixture(scope="session")
def streamlit_port():
    """Return the port for the Streamlit server."""
    return int(os.environ.get("PYMEDPHYS_GUI_PORT", DEFAULT_PORT))


@pytest.fixture(scope="session")
def streamlit_base_url(streamlit_port):
    """Return the base URL for the Streamlit server."""
    return os.environ.get("PYMEDPHYS_GUI_URL", f"http://localhost:{streamlit_port}")


@pytest.fixture(scope="session")
def streamlit_server(streamlit_port):
    """Start and manage the Streamlit server for the test session.

    This fixture starts the PyMedPhys GUI before running tests and
    ensures it is properly cleaned up afterward.
    """
    gui_command = [
        pmp_test_utils.get_executable_even_when_embedded(),
        "-m",
        "pymedphys",
        "gui",
        "--port",
        str(streamlit_port),
    ]

    # Start the server process
    with pmp_test_utils.process(gui_command) as proc:
        # Wait for the server to be ready
        _wait_for_server(f"http://localhost:{streamlit_port}", timeout=DEFAULT_TIMEOUT)
        yield proc


def _wait_for_server(url, timeout=DEFAULT_TIMEOUT):
    """Wait for the Streamlit server to be ready.

    Parameters
    ----------
    url : str
        The URL to check for server readiness.
    timeout : int
        Maximum time to wait in seconds.

    Raises
    ------
    TimeoutError
        If the server doesn't become ready within the timeout.
    """
    import urllib.error
    import urllib.request

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.5)

    raise TimeoutError(f"Streamlit server did not start within {timeout} seconds")


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure the browser context for Playwright tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 1024},
        "ignore_https_errors": True,
    }
