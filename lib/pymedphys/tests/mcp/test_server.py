# Copyright (C) 2025 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MCP server creation and configuration."""

import tempfile
from pathlib import Path

import pytest

from pymedphys._mcp.server import _server_config, create_server


class TestCreateServer:
    """Tests for create_server function."""

    def test_create_server_basic(self):
        """Server should be created with default configuration."""
        server = create_server()
        assert server is not None
        assert server.name == "pymedphys"

    def test_create_server_with_dicom_directories(self, tmp_path):
        """Server should accept DICOM directories configuration."""
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        server = create_server(dicom_directories=[str(dicom_dir)])
        assert server is not None
        assert len(_server_config["dicom_directories"]) == 1
        assert _server_config["dicom_directories"][0] == dicom_dir

    def test_create_server_with_trf_directories(self, tmp_path):
        """Server should accept TRF directories configuration."""
        trf_dir = tmp_path / "trf"
        trf_dir.mkdir()

        server = create_server(trf_directories=[str(trf_dir)])
        assert server is not None
        assert len(_server_config["trf_directories"]) == 1
        assert _server_config["trf_directories"][0] == trf_dir

    def test_create_server_with_working_directory(self, tmp_path):
        """Server should accept working directory configuration."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        server = create_server(working_directory=str(work_dir))
        assert server is not None
        assert _server_config["working_directory"] == work_dir

    def test_create_server_default_working_directory(self):
        """Server should use cwd as default working directory."""
        server = create_server(working_directory=None)
        assert server is not None
        assert _server_config["working_directory"] == Path.cwd()

    def test_create_server_with_deidentify_enabled(self):
        """Server should accept deidentify configuration."""
        server = create_server(deidentify=True, deidentify_salt="test_salt")
        assert server is not None
        assert _server_config["deidentify"] is True
        assert _server_config["deidentify_salt"] == "test_salt"

    def test_create_server_with_deidentify_disabled(self):
        """Server should default to deidentify disabled."""
        server = create_server()
        assert server is not None
        assert _server_config["deidentify"] is False

    def test_create_server_stores_mosaiq_connection(self):
        """Server should store mosaiq connection in config."""
        mock_connection = "mock_connection"
        server = create_server(mosaiq_connection=mock_connection)
        assert server is not None
        assert _server_config["mosaiq_connection"] == mock_connection

    def test_create_server_with_path_objects(self, tmp_path):
        """Server should accept Path objects for directories."""
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        server = create_server(dicom_directories=[dicom_dir])
        assert server is not None
        assert _server_config["dicom_directories"][0] == dicom_dir

    def test_create_server_with_multiple_directories(self, tmp_path):
        """Server should accept multiple DICOM/TRF directories."""
        dicom_dir1 = tmp_path / "dicom1"
        dicom_dir1.mkdir()
        dicom_dir2 = tmp_path / "dicom2"
        dicom_dir2.mkdir()

        server = create_server(dicom_directories=[str(dicom_dir1), str(dicom_dir2)])
        assert server is not None
        assert len(_server_config["dicom_directories"]) == 2

    def test_create_server_empty_directories(self):
        """Server should handle empty directory lists."""
        server = create_server(dicom_directories=[], trf_directories=[])
        assert server is not None
        assert _server_config["dicom_directories"] == []
        assert _server_config["trf_directories"] == []


class TestServerConfiguration:
    """Tests for server configuration state."""

    def test_config_isolation(self):
        """Creating new server should update global config."""
        # First server
        create_server(deidentify=True)
        assert _server_config["deidentify"] is True

        # Second server with different config
        create_server(deidentify=False)
        assert _server_config["deidentify"] is False

    def test_config_keys_present(self):
        """All expected config keys should be present."""
        create_server()
        expected_keys = {
            "mosaiq_connection",
            "dicom_directories",
            "trf_directories",
            "working_directory",
            "deidentify",
            "deidentify_salt",
        }
        assert expected_keys.issubset(set(_server_config.keys()))
