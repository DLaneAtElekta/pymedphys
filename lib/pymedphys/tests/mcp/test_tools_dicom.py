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

"""Tests for MCP DICOM tools."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from pymedphys._imports import pydicom

from pymedphys._mcp.tools.dicom import (
    _extract_ct_data,
    _extract_rtdose_data,
    _extract_rtplan_data,
    _extract_rtstruct_data,
    anonymize_dicom,
    create_rt_plan,
    create_rtpconnect,
    read_dicom,
)


def run_async(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def sample_rtplan_dataset():
    """Create a minimal RT Plan dataset for testing."""
    ds = pydicom.Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.5"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "RTPLAN"
    ds.PatientID = "TEST123"
    ds.PatientName = "Test^Patient"
    ds.StudyDate = "20240101"
    ds.StudyTime = "120000"
    ds.StudyDescription = "Test Study"
    ds.SeriesDescription = "Test RT Plan"
    ds.Manufacturer = "Test Manufacturer"
    ds.StationName = "TestStation"
    ds.RTPlanLabel = "TestPlan"
    ds.RTPlanName = "Test Plan Name"
    ds.RTPlanDate = "20240101"
    ds.RTPlanGeometry = "PATIENT"

    # Add beam sequence
    beam = pydicom.Dataset()
    beam.BeamNumber = 1
    beam.BeamName = "Beam1"
    beam.BeamType = "STATIC"
    beam.RadiationType = "PHOTON"
    beam.TreatmentMachineName = "TestMachine"

    # Add control point sequence
    cp = pydicom.Dataset()
    cp.GantryAngle = 0.0
    cp.BeamLimitingDeviceAngle = 0.0
    beam.ControlPointSequence = [cp]

    ds.BeamSequence = [beam]

    # Add fraction group
    fg = pydicom.Dataset()
    fg.FractionGroupNumber = 1
    fg.NumberOfFractionsPlanned = 30
    fg.NumberOfBeams = 1

    ref_beam = pydicom.Dataset()
    ref_beam.ReferencedBeamNumber = 1
    ref_beam.BeamMeterset = 100.0
    fg.ReferencedBeamSequence = [ref_beam]

    ds.FractionGroupSequence = [fg]

    return ds


@pytest.fixture
def sample_rtplan_file(sample_rtplan_dataset, tmp_path):
    """Create a temporary RT Plan DICOM file."""
    file_path = tmp_path / "test_rtplan.dcm"

    # Set up file meta
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = sample_rtplan_dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = sample_rtplan_dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    sample_rtplan_dataset.file_meta = file_meta
    sample_rtplan_dataset.is_little_endian = True
    sample_rtplan_dataset.is_implicit_VR = True

    sample_rtplan_dataset.save_as(str(file_path))
    return file_path


@pytest.mark.pydicom
class TestReadDicom:
    """Tests for read_dicom function."""

    def test_read_nonexistent_file(self):
        """Reading nonexistent file should return error."""
        result = run_async(read_dicom("/nonexistent/path/file.dcm"))
        assert "error" in result
        assert "not found" in result["error"]

    def test_read_rtplan_file(self, sample_rtplan_file):
        """Reading RT Plan file should extract metadata."""
        result = run_async(read_dicom(str(sample_rtplan_file)))

        assert "error" not in result
        assert result["modality"] == "RTPLAN"
        assert result["patient_id"] == "TEST123"
        assert "rt_plan" in result
        assert result["rt_plan"]["plan_label"] == "TestPlan"
        assert len(result["rt_plan"]["beams"]) == 1

    def test_read_dicom_includes_common_metadata(self, sample_rtplan_file):
        """Reading DICOM should include common metadata fields."""
        result = run_async(read_dicom(str(sample_rtplan_file)))

        assert "file_path" in result
        assert "sop_class_uid" in result
        assert "sop_instance_uid" in result
        assert "modality" in result
        assert "patient_id" in result
        assert "patient_name" in result
        assert "study_date" in result


@pytest.mark.pydicom
class TestExtractRtplanData:
    """Tests for _extract_rtplan_data function."""

    def test_extract_basic_plan_data(self, sample_rtplan_dataset):
        """Should extract basic RT Plan data."""
        result = _extract_rtplan_data(sample_rtplan_dataset)

        assert result["plan_label"] == "TestPlan"
        assert result["plan_name"] == "Test Plan Name"
        assert result["plan_date"] == "20240101"
        assert result["plan_geometry"] == "PATIENT"

    def test_extract_beam_data(self, sample_rtplan_dataset):
        """Should extract beam sequence data."""
        result = _extract_rtplan_data(sample_rtplan_dataset)

        assert len(result["beams"]) == 1
        beam = result["beams"][0]
        assert beam["beam_number"] == 1
        assert beam["beam_name"] == "Beam1"
        assert beam["beam_type"] == "STATIC"
        assert beam["radiation_type"] == "PHOTON"
        assert beam["machine_name"] == "TestMachine"
        assert beam["num_control_points"] == 1
        assert beam["start_gantry_angle"] == 0.0

    def test_extract_fraction_group_data(self, sample_rtplan_dataset):
        """Should extract fraction group data."""
        result = _extract_rtplan_data(sample_rtplan_dataset)

        assert len(result["fraction_groups"]) == 1
        fg = result["fraction_groups"][0]
        assert fg["fraction_group_number"] == 1
        assert fg["num_fractions"] == 30
        assert fg["num_beams"] == 1
        assert len(fg["beam_metersets"]) == 1
        assert fg["beam_metersets"][0]["meterset"] == 100.0

    def test_extract_plan_without_beams(self):
        """Should handle plan without beam sequence."""
        ds = pydicom.Dataset()
        ds.RTPlanLabel = "NoPlan"
        result = _extract_rtplan_data(ds)

        assert result["plan_label"] == "NoPlan"
        assert result["beams"] == []


@pytest.mark.pydicom
class TestAnonymizeDicom:
    """Tests for anonymize_dicom function."""

    def test_anonymize_nonexistent_file(self, tmp_path):
        """Anonymizing nonexistent file should return error."""
        result = run_async(
            anonymize_dicom("/nonexistent/file.dcm", str(tmp_path / "output.dcm"))
        )
        assert "error" in result
        assert "not found" in result["error"]

    def test_anonymize_rtplan(self, sample_rtplan_file, tmp_path):
        """Anonymizing RT Plan should create anonymized file."""
        output_path = tmp_path / "anon_rtplan.dcm"
        result = run_async(anonymize_dicom(str(sample_rtplan_file), str(output_path)))

        assert result.get("success") is True
        assert output_path.exists()
        assert result["original_patient_id"] == "TEST123"
        assert result["original_patient_name"] == "Test^Patient"

    def test_anonymize_with_replacement_id(self, sample_rtplan_file, tmp_path):
        """Anonymizing with replacement ID should use the new ID."""
        output_path = tmp_path / "anon_rtplan.dcm"
        result = run_async(
            anonymize_dicom(
                str(sample_rtplan_file), str(output_path), replacement_id="ANON001"
            )
        )

        assert result.get("success") is True
        assert result["new_patient_id"] == "ANON001"

        # Verify the file has the new ID
        ds = pydicom.dcmread(str(output_path))
        assert ds.PatientID == "ANON001"


@pytest.mark.pydicom
class TestCreateRtpconnect:
    """Tests for create_rtpconnect function."""

    def test_create_rtpconnect_missing_required_rx_fields(self, tmp_path):
        """Missing required prescription fields should return error."""
        result = run_async(
            create_rtpconnect(
                prescription={"site_name": "Brain"},
                patient_info={"patient_id": "123", "patient_name": "Test"},
                output_path=str(tmp_path / "test.dcm"),
            )
        )
        assert "error" in result
        assert "fractions" in result["error"] or "dose_per_fraction" in result["error"]

    def test_create_rtpconnect_missing_patient_info(self, tmp_path):
        """Missing required patient info should return error."""
        result = run_async(
            create_rtpconnect(
                prescription={
                    "site_name": "Brain",
                    "fractions": 30,
                    "dose_per_fraction": 200,
                },
                patient_info={},
                output_path=str(tmp_path / "test.dcm"),
            )
        )
        assert "error" in result
        assert "patient_id" in result["error"] or "patient_name" in result["error"]

    def test_create_rtpconnect_success(self, tmp_path):
        """Creating RTPCONNECT with valid data should succeed."""
        output_path = tmp_path / "test_rtpconnect.dcm"
        result = run_async(
            create_rtpconnect(
                prescription={
                    "site_name": "Brain",
                    "fractions": 30,
                    "dose_per_fraction": 200,
                    "technique": "VMAT",
                },
                patient_info={"patient_id": "123", "patient_name": "Test Patient"},
                output_path=str(output_path),
            )
        )

        assert result.get("success") is True
        assert output_path.exists()
        assert result["prescription_summary"]["site_name"] == "Brain"
        assert result["prescription_summary"]["fractions"] == 30
        assert result["prescription_summary"]["dose_per_fraction_cGy"] == 200
        assert result["prescription_summary"]["total_dose_cGy"] == 6000

    def test_create_rtpconnect_calculates_total_dose(self, tmp_path):
        """Total dose should be calculated from fractions * dose_per_fraction."""
        output_path = tmp_path / "test_rtpconnect.dcm"
        result = run_async(
            create_rtpconnect(
                prescription={
                    "site_name": "Lung",
                    "fractions": 5,
                    "dose_per_fraction": 1000,
                },
                patient_info={"patient_id": "123", "patient_name": "Test"},
                output_path=str(output_path),
            )
        )

        assert result.get("success") is True
        assert result["prescription_summary"]["total_dose_cGy"] == 5000

    def test_create_rtpconnect_includes_instructions(self, tmp_path):
        """RTPCONNECT should include import instructions by default."""
        output_path = tmp_path / "test_rtpconnect.dcm"
        result = run_async(
            create_rtpconnect(
                prescription={
                    "site_name": "Brain",
                    "fractions": 30,
                    "dose_per_fraction": 200,
                },
                patient_info={"patient_id": "123", "patient_name": "Test"},
                output_path=str(output_path),
                instructions=True,
            )
        )

        assert "import_instructions" in result
        assert "mosaiq_import_steps" in result["import_instructions"]
        assert "verification_checklist" in result["import_instructions"]

    def test_create_rtpconnect_without_instructions(self, tmp_path):
        """RTPCONNECT can exclude import instructions."""
        output_path = tmp_path / "test_rtpconnect.dcm"
        result = run_async(
            create_rtpconnect(
                prescription={
                    "site_name": "Brain",
                    "fractions": 30,
                    "dose_per_fraction": 200,
                },
                patient_info={"patient_id": "123", "patient_name": "Test"},
                output_path=str(output_path),
                instructions=False,
            )
        )

        assert result.get("success") is True
        assert "import_instructions" not in result


@pytest.mark.pydicom
class TestCreateRtPlan:
    """Tests for create_rt_plan function."""

    def test_create_rt_plan_invalid_source_type(self, tmp_path):
        """Invalid source type should return error."""
        result = run_async(
            create_rt_plan(
                source_type="invalid",
                output_path=str(tmp_path / "test.dcm"),
            )
        )
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_create_rt_plan_trf_without_path(self, tmp_path):
        """TRF source without file path should return error."""
        result = run_async(
            create_rt_plan(
                source_type="trf",
                output_path=str(tmp_path / "test.dcm"),
                source_data={},
            )
        )
        assert "error" in result
        assert "TRF file path" in result["error"]

    def test_create_rt_plan_trf_without_template(self, tmp_path):
        """TRF source without template should return error."""
        result = run_async(
            create_rt_plan(
                source_type="trf",
                output_path=str(tmp_path / "test.dcm"),
                source_data={"file_path": "/some/trf.trf"},
            )
        )
        assert "error" in result
        assert "Template" in result["error"]
