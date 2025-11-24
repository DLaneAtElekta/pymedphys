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

"""DICOM tools for MCP server."""

import datetime
import uuid
from pathlib import Path
from typing import Any


async def read_dicom(
    file_path: str,
    include_pixel_data: bool = False,
) -> dict[str, Any]:
    """Read and parse a DICOM file.

    Parameters
    ----------
    file_path : str
        Path to the DICOM file
    include_pixel_data : bool
        Include pixel/dose data in response

    Returns
    -------
    dict
        DICOM file metadata and optionally pixel data
    """
    import pydicom

    path = Path(file_path)
    if not path.exists():
        return {"error": f"DICOM file not found: {file_path}"}

    try:
        ds = pydicom.dcmread(str(path), force=True)
    except Exception as e:
        return {"error": f"Failed to read DICOM file: {str(e)}"}

    # Build metadata dictionary
    metadata = {
        "file_path": str(path),
        "sop_class_uid": str(getattr(ds, "SOPClassUID", "Unknown")),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "modality": getattr(ds, "Modality", "Unknown"),
        "patient_id": getattr(ds, "PatientID", ""),
        "patient_name": str(getattr(ds, "PatientName", "")),
        "study_date": getattr(ds, "StudyDate", ""),
        "study_time": getattr(ds, "StudyTime", ""),
        "study_description": getattr(ds, "StudyDescription", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "manufacturer": getattr(ds, "Manufacturer", ""),
        "station_name": getattr(ds, "StationName", ""),
    }

    modality = metadata["modality"]

    # Add modality-specific data
    if modality == "RTPLAN":
        metadata["rt_plan"] = _extract_rtplan_data(ds)
    elif modality == "RTDOSE":
        metadata["rt_dose"] = _extract_rtdose_data(ds, include_pixel_data)
    elif modality == "RTSTRUCT":
        metadata["rt_struct"] = _extract_rtstruct_data(ds)
    elif modality == "CT":
        metadata["ct"] = _extract_ct_data(ds, include_pixel_data)

    return metadata


def _extract_rtplan_data(ds) -> dict:
    """Extract RT Plan data."""
    data = {
        "plan_label": getattr(ds, "RTPlanLabel", ""),
        "plan_name": getattr(ds, "RTPlanName", ""),
        "plan_date": getattr(ds, "RTPlanDate", ""),
        "plan_geometry": getattr(ds, "RTPlanGeometry", ""),
        "beams": [],
        "fraction_groups": [],
    }

    # Extract beams
    if hasattr(ds, "BeamSequence"):
        for beam in ds.BeamSequence:
            beam_data = {
                "beam_number": getattr(beam, "BeamNumber", None),
                "beam_name": getattr(beam, "BeamName", ""),
                "beam_type": getattr(beam, "BeamType", ""),
                "radiation_type": getattr(beam, "RadiationType", ""),
                "machine_name": getattr(beam, "TreatmentMachineName", ""),
                "num_control_points": len(getattr(beam, "ControlPointSequence", [])),
            }

            # Get first/last control point info
            if (
                hasattr(beam, "ControlPointSequence")
                and len(beam.ControlPointSequence) > 0
            ):
                first_cp = beam.ControlPointSequence[0]
                beam_data["start_gantry_angle"] = getattr(first_cp, "GantryAngle", None)
                beam_data["start_collimator_angle"] = getattr(
                    first_cp, "BeamLimitingDeviceAngle", None
                )

            data["beams"].append(beam_data)

    # Extract fraction groups
    if hasattr(ds, "FractionGroupSequence"):
        for fg in ds.FractionGroupSequence:
            fg_data = {
                "fraction_group_number": getattr(fg, "FractionGroupNumber", None),
                "num_fractions": getattr(fg, "NumberOfFractionsPlanned", None),
                "num_beams": getattr(fg, "NumberOfBeams", None),
            }

            # Extract beam references
            if hasattr(fg, "ReferencedBeamSequence"):
                fg_data["beam_metersets"] = [
                    {
                        "beam_number": getattr(ref, "ReferencedBeamNumber", None),
                        "meterset": getattr(ref, "BeamMeterset", None),
                    }
                    for ref in fg.ReferencedBeamSequence
                ]

            data["fraction_groups"].append(fg_data)

    return data


def _extract_rtdose_data(ds, include_pixel_data: bool) -> dict:
    """Extract RT Dose data."""
    data = {
        "dose_units": getattr(ds, "DoseUnits", ""),
        "dose_type": getattr(ds, "DoseType", ""),
        "dose_summation_type": getattr(ds, "DoseSummationType", ""),
        "dose_grid_scaling": getattr(ds, "DoseGridScaling", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "num_frames": getattr(ds, "NumberOfFrames", 1),
    }

    if hasattr(ds, "PixelSpacing"):
        data["pixel_spacing"] = list(ds.PixelSpacing)

    if include_pixel_data and hasattr(ds, "PixelData"):
        import numpy as np

        import pymedphys

        try:
            axes, dose = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
            data["dose_statistics"] = {
                "max": float(np.max(dose)),
                "min": float(np.min(dose)),
                "mean": float(np.mean(dose)),
            }
        except Exception:
            pass

    return data


def _extract_rtstruct_data(ds) -> dict:
    """Extract RT Structure Set data."""
    data = {
        "structure_set_label": getattr(ds, "StructureSetLabel", ""),
        "structure_set_date": getattr(ds, "StructureSetDate", ""),
        "structures": [],
    }

    if hasattr(ds, "StructureSetROISequence"):
        roi_dict = {roi.ROINumber: roi.ROIName for roi in ds.StructureSetROISequence}

        if hasattr(ds, "ROIContourSequence"):
            for contour in ds.ROIContourSequence:
                roi_num = getattr(contour, "ReferencedROINumber", None)
                data["structures"].append(
                    {
                        "roi_number": roi_num,
                        "roi_name": roi_dict.get(roi_num, "Unknown"),
                        "num_contours": len(getattr(contour, "ContourSequence", [])),
                        "color": list(getattr(contour, "ROIDisplayColor", [])),
                    }
                )

    return data


def _extract_ct_data(ds, include_pixel_data: bool) -> dict:
    """Extract CT data."""
    data = {
        "slice_location": getattr(ds, "SliceLocation", None),
        "slice_thickness": getattr(ds, "SliceThickness", None),
        "pixel_spacing": list(getattr(ds, "PixelSpacing", [])),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "kvp": getattr(ds, "KVP", None),
    }

    if include_pixel_data and hasattr(ds, "PixelData"):
        import numpy as np

        try:
            pixel_array = ds.pixel_array
            data["pixel_statistics"] = {
                "max": int(np.max(pixel_array)),
                "min": int(np.min(pixel_array)),
                "mean": float(np.mean(pixel_array)),
            }
        except Exception:
            pass

    return data


async def anonymize_dicom(
    input_path: str,
    output_path: str,
    replacement_id: str | None = None,
    delete_private_tags: bool = True,
) -> dict[str, Any]:
    """Anonymize a DICOM file.

    Parameters
    ----------
    input_path : str
        Path to input DICOM file
    output_path : str
        Path for anonymized output file
    replacement_id : str, optional
        Replacement patient ID
    delete_private_tags : bool
        Delete private DICOM tags

    Returns
    -------
    dict
        Anonymization result
    """
    import pydicom

    import pymedphys

    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        return {"error": f"Input file not found: {input_path}"}

    try:
        ds = pydicom.dcmread(str(in_path))

        # Store original values for reporting
        original_patient_id = getattr(ds, "PatientID", "")
        original_patient_name = str(getattr(ds, "PatientName", ""))

        # Perform anonymization
        anonymized_ds = pymedphys.dicom.anonymise(
            ds,
            delete_private_tags=delete_private_tags,
        )

        # Apply replacement ID if provided
        if replacement_id:
            anonymized_ds.PatientID = replacement_id
            anonymized_ds.PatientName = replacement_id

        # Save anonymized file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anonymized_ds.save_as(str(out_path))

        return {
            "success": True,
            "input_file": str(in_path),
            "output_file": str(out_path),
            "original_patient_id": original_patient_id,
            "original_patient_name": original_patient_name,
            "new_patient_id": getattr(anonymized_ds, "PatientID", ""),
            "private_tags_deleted": delete_private_tags,
        }

    except Exception as e:
        return {"error": f"Anonymization failed: {str(e)}"}


async def create_rt_plan(
    source_type: str,
    output_path: str,
    source_data: dict[str, Any] | None = None,
    template_path: str | None = None,
    patient_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an RT Plan DICOM file.

    Parameters
    ----------
    source_type : str
        Source data type: 'trf', 'delivery', or 'parameters'
    output_path : str
        Output path for generated RT Plan
    source_data : dict, optional
        Source data configuration
    template_path : str, optional
        Path to template DICOM RT Plan
    patient_info : dict, optional
        Patient information

    Returns
    -------
    dict
        Creation result with file path and instructions
    """
    import pydicom

    import pymedphys

    source_data = source_data or {}
    patient_info = patient_info or {}
    out_path = Path(output_path)

    try:
        if source_type == "trf":
            trf_path = source_data.get("file_path")
            if not trf_path:
                return {"error": "TRF file path required in source_data"}

            if not template_path:
                return {"error": "Template DICOM required for TRF source"}

            # Load delivery from TRF
            delivery = pymedphys.Delivery.from_trf(trf_path)

            # Load template
            template_ds = pydicom.dcmread(template_path)

            # Convert delivery to DICOM
            new_ds = delivery.to_dicom(template_ds)

        elif source_type == "delivery":
            # Create from delivery parameters directly
            mu = source_data.get("monitor_units", [])
            mlc = source_data.get("mlc", [])
            jaw = source_data.get("jaw", [])
            gantry = source_data.get("gantry", [])
            collimator = source_data.get("collimator", [])

            delivery = pymedphys.Delivery(
                monitor_units=mu,
                gantry=gantry,
                collimator=collimator,
                mlc=mlc,
                jaw=jaw,
            )

            if not template_path:
                return {"error": "Template DICOM required for delivery source"}

            template_ds = pydicom.dcmread(template_path)
            new_ds = delivery.to_dicom(template_ds)

        elif source_type == "parameters":
            # Create minimal RT Plan from parameters
            if not template_path:
                return {"error": "Template DICOM required"}

            new_ds = pydicom.dcmread(template_path)

            # Update with provided parameters
            if "plan_label" in source_data:
                new_ds.RTPlanLabel = source_data["plan_label"]
            if "plan_name" in source_data:
                new_ds.RTPlanName = source_data["plan_name"]

        else:
            return {"error": f"Unknown source_type: {source_type}"}

        # Update patient information
        if patient_info:
            if "patient_id" in patient_info:
                new_ds.PatientID = patient_info["patient_id"]
            if "patient_name" in patient_info:
                new_ds.PatientName = patient_info["patient_name"]
            if "birth_date" in patient_info:
                new_ds.PatientBirthDate = patient_info["birth_date"]

        # Generate new UIDs
        new_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        new_ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_ds.save_as(str(out_path))

        return {
            "success": True,
            "output_file": str(out_path),
            "sop_instance_uid": str(new_ds.SOPInstanceUID),
            "patient_id": getattr(new_ds, "PatientID", ""),
            "plan_label": getattr(new_ds, "RTPlanLabel", ""),
        }

    except Exception as e:
        return {"error": f"RT Plan creation failed: {str(e)}"}


async def create_rtpconnect(
    prescription: dict[str, Any],
    patient_info: dict[str, Any],
    output_path: str,
    instructions: bool = True,
) -> dict[str, Any]:
    """Create an RT ION Plan Connect (RTPCONNECT) file for prescription import.

    This creates a DICOM file that can be imported into Mosaiq to establish
    a treatment prescription.

    Parameters
    ----------
    prescription : dict
        Prescription details including:
        - site_name: Treatment site name
        - fractions: Number of fractions
        - dose_per_fraction: Dose per fraction in cGy
        - total_dose: Total prescribed dose in cGy (optional, calculated if not provided)
        - technique: Treatment technique (e.g., VMAT, IMRT, 3DCRT)
        - notes: Additional prescription notes
    patient_info : dict
        Patient information:
        - patient_id: Patient ID
        - patient_name: Patient name
        - birth_date: Birth date (YYYYMMDD format)
    output_path : str
        Output path for RTPCONNECT file
    instructions : bool
        Include import instructions in response

    Returns
    -------
    dict
        Creation result with file path and import instructions
    """
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ImplicitVRLittleEndian

    out_path = Path(output_path)

    # Validate inputs
    required_rx = ["site_name", "fractions", "dose_per_fraction"]
    for field in required_rx:
        if field not in prescription:
            return {"error": f"Missing required prescription field: {field}"}

    required_pt = ["patient_id", "patient_name"]
    for field in required_pt:
        if field not in patient_info:
            return {"error": f"Missing required patient_info field: {field}"}

    try:
        # Calculate total dose if not provided
        total_dose = prescription.get(
            "total_dose",
            prescription["fractions"] * prescription["dose_per_fraction"],
        )

        # Create file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = (
            "1.2.840.10008.5.1.4.1.1.481.8"  # RT Ion Plan Storage
        )
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        # Create the FileDataset
        ds = FileDataset(
            str(out_path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Patient Module
        ds.PatientName = patient_info["patient_name"]
        ds.PatientID = patient_info["patient_id"]
        ds.PatientBirthDate = patient_info.get("birth_date", "")
        ds.PatientSex = patient_info.get("sex", "")

        # General Study Module
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        ds.ReferringPhysicianName = patient_info.get("referring_physician", "")
        ds.StudyID = str(uuid.uuid4())[:8].upper()
        ds.AccessionNumber = ""

        # General Series Module
        ds.Modality = "RTPLAN"
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesNumber = 1
        ds.SeriesDescription = f"Prescription - {prescription['site_name']}"

        # RT General Plan Module
        ds.RTPlanLabel = prescription["site_name"]
        ds.RTPlanName = prescription.get("plan_name", prescription["site_name"])
        ds.RTPlanDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.RTPlanTime = datetime.datetime.now().strftime("%H%M%S")
        ds.RTPlanGeometry = "PATIENT"
        ds.TreatmentProtocols = prescription.get("technique", "")

        # SOP Common Module
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.InstanceCreationDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.InstanceCreationTime = datetime.datetime.now().strftime("%H%M%S")

        # Dose Reference Sequence (Prescription)
        dose_ref = Dataset()
        dose_ref.DoseReferenceNumber = 1
        dose_ref.DoseReferenceUID = pydicom.uid.generate_uid()
        dose_ref.DoseReferenceStructureType = "SITE"
        dose_ref.DoseReferenceDescription = prescription["site_name"]
        dose_ref.DoseReferenceType = "TARGET"
        dose_ref.TargetPrescriptionDose = total_dose / 100.0  # Convert cGy to Gy
        dose_ref.TargetMinimumDose = (total_dose / 100.0) * 0.95
        dose_ref.TargetMaximumDose = (total_dose / 100.0) * 1.07
        dose_ref.DeliveryMaximumDose = (prescription["dose_per_fraction"] / 100.0) * 1.1

        ds.DoseReferenceSequence = Sequence([dose_ref])

        # Fraction Group Sequence
        fx_group = Dataset()
        fx_group.FractionGroupNumber = 1
        fx_group.FractionGroupDescription = prescription["site_name"]
        fx_group.NumberOfFractionsPlanned = prescription["fractions"]
        fx_group.NumberOfBeams = 0  # Will be filled when beams are added
        fx_group.NumberOfBrachyApplicationSetups = 0

        ds.FractionGroupSequence = Sequence([fx_group])

        # Approval Status
        ds.ApprovalStatus = "UNAPPROVED"

        # Add prescription notes as private tag or comment if provided
        if "notes" in prescription:
            ds.RTPlanDescription = prescription["notes"]

        # Save the file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(out_path))

        result = {
            "success": True,
            "output_file": str(out_path),
            "sop_instance_uid": str(ds.SOPInstanceUID),
            "prescription_summary": {
                "site_name": prescription["site_name"],
                "fractions": prescription["fractions"],
                "dose_per_fraction_cGy": prescription["dose_per_fraction"],
                "total_dose_cGy": total_dose,
                "technique": prescription.get("technique", ""),
            },
            "patient": {
                "patient_id": patient_info["patient_id"],
                "patient_name": patient_info["patient_name"],
            },
        }

        if instructions:
            result["import_instructions"] = {
                "mosaiq_import_steps": [
                    "1. Open Mosaiq and navigate to the patient's chart",
                    "2. Go to File > Import > DICOM RT Plan",
                    f"3. Select the file: {out_path.name}",
                    "4. Review the imported prescription details",
                    "5. Verify the site name, fractions, and dose match your intent",
                    "6. Approve the prescription after physician review",
                ],
                "verification_checklist": [
                    f"- Site Name: {prescription['site_name']}",
                    f"- Number of Fractions: {prescription['fractions']}",
                    f"- Dose per Fraction: {prescription['dose_per_fraction']} cGy",
                    f"- Total Dose: {total_dose} cGy",
                    f"- Patient ID: {patient_info['patient_id']}",
                    "- Verify patient identity matches Mosaiq record",
                ],
                "notes": [
                    "This RTPCONNECT file creates a prescription framework only",
                    "Treatment fields must be added separately",
                    "Physician approval required before treatment",
                    "Verify all dose values are in the expected units",
                ],
            }

        return result

    except Exception as e:
        return {"error": f"RTPCONNECT creation failed: {str(e)}"}
