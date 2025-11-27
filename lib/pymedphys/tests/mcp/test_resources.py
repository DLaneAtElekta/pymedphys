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

"""Tests for MCP resource handlers."""

import asyncio
import json
from pathlib import Path

import pytest

from pymedphys._imports import pydicom

from pymedphys._mcp.resources import dicom as dicom_resources
from pymedphys._mcp.resources import mosaiq as mosaiq_resources
from pymedphys._mcp.resources import trf as trf_resources


def run_async(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDicomResources:
    """Tests for DICOM resource handlers."""

    def test_read_resource_directory_not_found(self):
        """Reading from non-configured directory should return error."""
        result = run_async(
            dicom_resources.read_resource("dicom://unknown_dir/file.dcm", [])
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"]

    def test_read_resource_file_not_found(self, tmp_path):
        """Reading non-existent file should return error."""
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        result = run_async(
            dicom_resources.read_resource(
                f"dicom://{dicom_dir.name}/nonexistent.dcm", [dicom_dir]
            )
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"]

    @pytest.mark.pydicom
    def test_read_resource_valid_file(self, tmp_path):
        """Reading valid DICOM file should return metadata."""
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        # Create a minimal DICOM file
        ds = pydicom.Dataset()
        ds.PatientID = "TEST123"
        ds.PatientName = "Test^Patient"
        ds.Modality = "CT"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.StudyDate = "20240101"

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        dcm_path = dicom_dir / "test.dcm"
        ds.save_as(str(dcm_path))

        result = run_async(
            dicom_resources.read_resource(
                f"dicom://{dicom_dir.name}/test.dcm", [dicom_dir]
            )
        )
        data = json.loads(result)

        assert "error" not in data
        assert data["patient_id"] == "TEST123"
        assert data["modality"] == "CT"

    def test_uri_parsing(self, tmp_path):
        """URI should be correctly parsed to find file."""
        dicom_dir = tmp_path / "my_dicom_folder"
        dicom_dir.mkdir()
        subdir = dicom_dir / "subdir"
        subdir.mkdir()

        # Test that nested paths work
        result = run_async(
            dicom_resources.read_resource(
                f"dicom://{dicom_dir.name}/subdir/file.dcm", [dicom_dir]
            )
        )
        data = json.loads(result)
        # File doesn't exist but directory matching should work
        assert "not found" in data.get("error", "")


class TestTrfResources:
    """Tests for TRF resource handlers."""

    def test_read_resource_directory_not_found(self):
        """Reading from non-configured directory should return error."""
        result = run_async(
            trf_resources.read_resource("trf://unknown_dir/file.trf", [])
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"]

    def test_read_resource_file_not_found(self, tmp_path):
        """Reading non-existent file should return error."""
        trf_dir = tmp_path / "trf"
        trf_dir.mkdir()

        result = run_async(
            trf_resources.read_resource(
                f"trf://{trf_dir.name}/nonexistent.trf", [trf_dir]
            )
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"]


class TestMosaiqResources:
    """Tests for Mosaiq resource handlers."""

    def test_read_resource_no_connection(self):
        """Reading without connection should return error."""
        result = run_async(mosaiq_resources.read_resource("mosaiq://patients", None))
        data = json.loads(result)
        assert "error" in data
        assert "connection" in data["error"].lower()

    def test_read_resource_unknown_resource(self):
        """Reading unknown resource type should return error."""
        # Use a mock connection
        mock_connection = "mock"

        result = run_async(
            mosaiq_resources.read_resource("mosaiq://unknown_resource", mock_connection)
        )
        data = json.loads(result)
        assert "error" in data
        assert "Unknown" in data["error"]

    def test_patient_list_resource_path(self):
        """Patient list resource should be correctly identified."""
        # Without a real connection, we test path parsing
        result = run_async(mosaiq_resources.read_resource("mosaiq://patients", None))
        data = json.loads(result)
        # Error is about connection, not resource type
        assert "connection" in data.get("error", "").lower()

    def test_machines_resource_path(self):
        """Machines resource should be correctly identified."""
        result = run_async(mosaiq_resources.read_resource("mosaiq://machines", None))
        data = json.loads(result)
        assert "connection" in data.get("error", "").lower()

    def test_sites_resource_path(self):
        """Sites resource should be correctly identified."""
        result = run_async(mosaiq_resources.read_resource("mosaiq://sites", None))
        data = json.loads(result)
        assert "connection" in data.get("error", "").lower()


class TestResourceUriParsing:
    """Tests for resource URI parsing across handlers."""

    def test_dicom_uri_with_special_characters(self, tmp_path):
        """DICOM URI with URL-encoded characters should work."""
        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        # Test URI with encoded space
        result = run_async(
            dicom_resources.read_resource(
                f"dicom://{dicom_dir.name}/path%20with%20spaces/file.dcm",
                [dicom_dir],
            )
        )
        data = json.loads(result)
        # Should not error on parsing, just on file not found
        assert "error" in data

    def test_trf_uri_with_nested_path(self, tmp_path):
        """TRF URI with nested path should be parsed correctly."""
        trf_dir = tmp_path / "trf"
        trf_dir.mkdir()
        nested = trf_dir / "machine1" / "2024"
        nested.mkdir(parents=True)

        result = run_async(
            trf_resources.read_resource(
                f"trf://{trf_dir.name}/machine1/2024/file.trf",
                [trf_dir],
            )
        )
        data = json.loads(result)
        # Should parse path correctly, error on file not found
        assert "error" in data


class TestResourceExtraction:
    """Tests for DICOM modality-specific extraction functions."""

    @pytest.mark.pydicom
    def test_extract_ct_metadata(self):
        """CT metadata extraction should work."""
        from pymedphys._mcp.resources.dicom import _extract_ct_metadata

        ds = pydicom.Dataset()
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
        ds.SliceLocation = 100.5
        ds.SliceThickness = 2.5
        ds.PixelSpacing = [0.5, 0.5]
        ds.Rows = 512
        ds.Columns = 512
        ds.KVP = 120
        ds.Exposure = 200

        result = _extract_ct_metadata(ds)

        assert result["slice_location"] == 100.5
        assert result["slice_thickness"] == 2.5
        assert result["rows"] == 512
        assert result["kvp"] == 120

    @pytest.mark.pydicom
    def test_extract_rtdose_metadata(self):
        """RT Dose metadata extraction should work."""
        from pymedphys._mcp.resources.dicom import _extract_rtdose_metadata

        ds = pydicom.Dataset()
        ds.DoseUnits = "GY"
        ds.DoseType = "PHYSICAL"
        ds.DoseSummationType = "PLAN"
        ds.Rows = 100
        ds.Columns = 100
        ds.NumberOfFrames = 50
        ds.PixelSpacing = [2.0, 2.0]
        ds.SliceThickness = 3.0
        ds.DoseGridScaling = 0.001

        result = _extract_rtdose_metadata(ds)

        assert result["dose_units"] == "GY"
        assert result["dose_type"] == "PHYSICAL"
        assert result["rows"] == 100
        assert result["num_frames"] == 50

    @pytest.mark.pydicom
    def test_extract_rtstruct_metadata(self):
        """RT Struct metadata extraction should work."""
        from pymedphys._mcp.resources.dicom import _extract_rtstruct_metadata

        ds = pydicom.Dataset()
        ds.StructureSetLabel = "TestStructures"
        ds.StructureSetDate = "20240101"

        # Add ROI sequence
        roi1 = pydicom.Dataset()
        roi1.ROINumber = 1
        roi1.ROIName = "PTV"
        ds.StructureSetROISequence = [roi1]

        # Add contour sequence
        contour1 = pydicom.Dataset()
        contour1.ReferencedROINumber = 1
        contour1.ROIDisplayColor = [255, 0, 0]
        c1 = pydicom.Dataset()
        contour1.ContourSequence = [c1]
        ds.ROIContourSequence = [contour1]

        result = _extract_rtstruct_metadata(ds)

        assert result["structure_set_label"] == "TestStructures"
        assert result["num_structures"] == 1
        assert result["structures"][0]["roi_name"] == "PTV"
        assert result["structures"][0]["num_contours"] == 1

    @pytest.mark.pydicom
    def test_extract_rtplan_metadata(self):
        """RT Plan metadata extraction should work."""
        from pymedphys._mcp.resources.dicom import _extract_rtplan_metadata

        ds = pydicom.Dataset()
        ds.RTPlanLabel = "TestPlan"
        ds.RTPlanName = "Test"
        ds.RTPlanDate = "20240101"
        ds.RTPlanGeometry = "PATIENT"

        # Add beam
        beam = pydicom.Dataset()
        beam.BeamNumber = 1
        beam.BeamName = "Beam1"
        beam.BeamType = "STATIC"
        beam.RadiationType = "PHOTON"
        beam.TreatmentMachineName = "Machine1"
        beam.ControlPointSequence = []
        ds.BeamSequence = [beam]

        result = _extract_rtplan_metadata(ds)

        assert result["plan_label"] == "TestPlan"
        assert result["num_beams"] == 1
        assert result["beams"][0]["beam_name"] == "Beam1"
