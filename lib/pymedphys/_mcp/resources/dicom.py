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

"""DICOM resource handlers for MCP server."""

import json
from pathlib import Path
from urllib.parse import unquote, urlparse


async def read_resource(uri: str, dicom_directories: list[Path]) -> str:
    """Read a DICOM resource by URI.

    Parameters
    ----------
    uri : str
        Resource URI in format dicom://directory_name/path/to/file.dcm
    dicom_directories : list of Path
        List of configured DICOM directories

    Returns
    -------
    str
        JSON-encoded DICOM metadata
    """
    parsed = urlparse(uri)
    dir_name = parsed.netloc
    file_path = unquote(parsed.path.lstrip("/"))

    # Find the matching directory
    target_dir = None
    for dicom_dir in dicom_directories:
        if dicom_dir.name == dir_name:
            target_dir = dicom_dir
            break

    if target_dir is None:
        return json.dumps({"error": f"DICOM directory not found: {dir_name}"})

    full_path = target_dir / file_path

    if not full_path.exists():
        return json.dumps({"error": f"DICOM file not found: {full_path}"})

    try:
        return await _read_dicom_file(full_path)
    except Exception as e:
        return json.dumps({"error": f"Failed to read DICOM file: {str(e)}"})


async def _read_dicom_file(file_path: Path) -> str:
    """Read and parse a DICOM file, returning key metadata."""
    import pydicom

    ds = pydicom.dcmread(str(file_path), force=True)

    # Extract common metadata
    metadata = {
        "file_path": str(file_path),
        "sop_class_uid": str(getattr(ds, "SOPClassUID", "Unknown")),
        "modality": getattr(ds, "Modality", "Unknown"),
        "patient_id": getattr(ds, "PatientID", "Unknown"),
        "patient_name": str(getattr(ds, "PatientName", "Unknown")),
        "study_date": getattr(ds, "StudyDate", "Unknown"),
        "study_description": getattr(ds, "StudyDescription", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
    }

    # Add modality-specific metadata
    modality = metadata["modality"]

    if modality == "RTPLAN":
        metadata["rt_plan"] = _extract_rtplan_metadata(ds)
    elif modality == "RTDOSE":
        metadata["rt_dose"] = _extract_rtdose_metadata(ds)
    elif modality == "RTSTRUCT":
        metadata["rt_struct"] = _extract_rtstruct_metadata(ds)
    elif modality == "CT":
        metadata["ct"] = _extract_ct_metadata(ds)

    return json.dumps(metadata, indent=2, default=str)


def _extract_rtplan_metadata(ds) -> dict:
    """Extract RT Plan specific metadata."""
    plan_data = {
        "plan_label": getattr(ds, "RTPlanLabel", "Unknown"),
        "plan_name": getattr(ds, "RTPlanName", ""),
        "plan_date": getattr(ds, "RTPlanDate", ""),
        "plan_geometry": getattr(ds, "RTPlanGeometry", ""),
    }

    # Extract beam information
    beams = []
    if hasattr(ds, "BeamSequence"):
        for beam in ds.BeamSequence:
            beam_info = {
                "beam_number": getattr(beam, "BeamNumber", None),
                "beam_name": getattr(beam, "BeamName", ""),
                "beam_type": getattr(beam, "BeamType", ""),
                "radiation_type": getattr(beam, "RadiationType", ""),
                "machine_name": getattr(beam, "TreatmentMachineName", ""),
                "num_control_points": len(getattr(beam, "ControlPointSequence", [])),
            }
            beams.append(beam_info)

    plan_data["beams"] = beams
    plan_data["num_beams"] = len(beams)

    # Extract fraction group information
    if hasattr(ds, "FractionGroupSequence"):
        fg = ds.FractionGroupSequence[0]
        plan_data["fractions"] = getattr(fg, "NumberOfFractionsPlanned", None)
        plan_data["num_beams_in_fraction"] = getattr(
            fg, "NumberOfBeams", None
        )

    return plan_data


def _extract_rtdose_metadata(ds) -> dict:
    """Extract RT Dose specific metadata."""
    dose_data = {
        "dose_units": getattr(ds, "DoseUnits", "Unknown"),
        "dose_type": getattr(ds, "DoseType", "Unknown"),
        "dose_summation_type": getattr(ds, "DoseSummationType", ""),
    }

    # Get dose grid information
    if hasattr(ds, "PixelData"):
        dose_data["rows"] = getattr(ds, "Rows", None)
        dose_data["columns"] = getattr(ds, "Columns", None)
        dose_data["num_frames"] = getattr(ds, "NumberOfFrames", 1)
        dose_data["pixel_spacing"] = list(getattr(ds, "PixelSpacing", []))
        dose_data["slice_thickness"] = getattr(ds, "SliceThickness", None)
        dose_data["dose_grid_scaling"] = getattr(ds, "DoseGridScaling", None)

    return dose_data


def _extract_rtstruct_metadata(ds) -> dict:
    """Extract RT Structure Set specific metadata."""
    struct_data = {
        "structure_set_label": getattr(ds, "StructureSetLabel", ""),
        "structure_set_date": getattr(ds, "StructureSetDate", ""),
    }

    # Extract structure information
    structures = []
    if hasattr(ds, "StructureSetROISequence"):
        roi_dict = {
            roi.ROINumber: roi.ROIName
            for roi in ds.StructureSetROISequence
        }

        if hasattr(ds, "ROIContourSequence"):
            for contour in ds.ROIContourSequence:
                roi_num = getattr(contour, "ReferencedROINumber", None)
                roi_name = roi_dict.get(roi_num, "Unknown")
                num_contours = len(getattr(contour, "ContourSequence", []))

                structures.append({
                    "roi_number": roi_num,
                    "roi_name": roi_name,
                    "num_contours": num_contours,
                    "color": list(getattr(contour, "ROIDisplayColor", [])),
                })

    struct_data["structures"] = structures
    struct_data["num_structures"] = len(structures)

    return struct_data


def _extract_ct_metadata(ds) -> dict:
    """Extract CT specific metadata."""
    ct_data = {
        "image_type": list(getattr(ds, "ImageType", [])),
        "slice_location": getattr(ds, "SliceLocation", None),
        "slice_thickness": getattr(ds, "SliceThickness", None),
        "pixel_spacing": list(getattr(ds, "PixelSpacing", [])),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "kvp": getattr(ds, "KVP", None),
        "exposure": getattr(ds, "Exposure", None),
    }

    return ct_data
