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

"""Analysis tools for MCP server (gamma, metersetmap, etc.)."""

from pathlib import Path
from typing import Any

import numpy as np


async def gamma_analysis(
    reference_dose_path: str,
    evaluation_dose_path: str,
    dose_threshold_percent: float = 3.0,
    distance_threshold_mm: float = 3.0,
    lower_dose_cutoff_percent: float = 20.0,
    local_gamma: bool = False,
) -> dict[str, Any]:
    """Perform gamma analysis comparing two dose distributions.

    Parameters
    ----------
    reference_dose_path : str
        Path to reference dose file (DICOM RTDose or numpy .npy)
    evaluation_dose_path : str
        Path to evaluation dose file
    dose_threshold_percent : float
        Dose difference threshold in percent
    distance_threshold_mm : float
        Distance-to-agreement threshold in mm
    lower_dose_cutoff_percent : float
        Lower dose cutoff as percent of max dose
    local_gamma : bool
        Use local gamma (True) or global gamma (False)

    Returns
    -------
    dict
        Gamma analysis results including pass rate and statistics
    """
    import pymedphys

    ref_path = Path(reference_dose_path)
    eval_path = Path(evaluation_dose_path)

    # Load dose data based on file type
    ref_axes, ref_dose = _load_dose_data(ref_path)
    eval_axes, eval_dose = _load_dose_data(eval_path)

    # Perform gamma analysis
    gamma = pymedphys.gamma(
        ref_axes,
        ref_dose,
        eval_axes,
        eval_dose,
        dose_threshold_percent,
        distance_threshold_mm,
        lower_percent_dose_cutoff=lower_dose_cutoff_percent,
        local_gamma=local_gamma,
    )

    # Calculate statistics
    valid_gamma = gamma[~np.isnan(gamma)]
    pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma) * 100 if len(valid_gamma) > 0 else 0

    results = {
        "pass_rate_percent": float(pass_rate),
        "criteria": {
            "dose_threshold_percent": dose_threshold_percent,
            "distance_threshold_mm": distance_threshold_mm,
            "lower_dose_cutoff_percent": lower_dose_cutoff_percent,
            "local_gamma": local_gamma,
        },
        "statistics": {
            "mean_gamma": float(np.mean(valid_gamma)) if len(valid_gamma) > 0 else None,
            "max_gamma": float(np.max(valid_gamma)) if len(valid_gamma) > 0 else None,
            "min_gamma": float(np.min(valid_gamma)) if len(valid_gamma) > 0 else None,
            "std_gamma": float(np.std(valid_gamma)) if len(valid_gamma) > 0 else None,
            "num_evaluated_points": int(len(valid_gamma)),
        },
        "reference_file": str(ref_path),
        "evaluation_file": str(eval_path),
    }

    return results


def _load_dose_data(file_path: Path) -> tuple:
    """Load dose data from file.

    Returns
    -------
    tuple
        (axes, dose) where axes is coordinate tuple and dose is numpy array
    """
    if file_path.suffix.lower() == ".dcm":
        import pydicom

        import pymedphys

        ds = pydicom.dcmread(str(file_path))
        axes, dose = pymedphys.dicom.zyx_and_dose_from_dataset(ds)
        return axes, dose

    elif file_path.suffix.lower() == ".npy":
        # Assume numpy file contains dict with 'axes' and 'dose' keys
        # or just the dose array
        data = np.load(str(file_path), allow_pickle=True)
        if isinstance(data, dict):
            return data["axes"], data["dose"]
        else:
            # Generate default axes based on array shape
            shape = data.shape
            axes = tuple(np.arange(s) for s in shape)
            return axes, data

    else:
        raise ValueError(f"Unsupported dose file format: {file_path.suffix}")


async def calculate_metersetmap(
    source: str,
    source_path: str,
    grid_resolution: float = 1.0,
    output_format: str = "json",
    mosaiq_connection: Any | None = None,
) -> dict[str, Any]:
    """Calculate a MetersetMap (fluence map) from treatment delivery data.

    Parameters
    ----------
    source : str
        Source type: 'trf', 'dicom', or 'mosaiq'
    source_path : str
        Path to source file or Mosaiq field identifier
    grid_resolution : float
        Grid resolution in mm
    output_format : str
        Output format: 'json', 'numpy', or 'image'
    mosaiq_connection : optional
        Mosaiq connection for 'mosaiq' source type

    Returns
    -------
    dict
        MetersetMap data and metadata
    """
    import pymedphys

    # Load delivery data based on source
    if source == "trf":
        delivery = pymedphys.Delivery.from_trf(source_path)
    elif source == "dicom":
        import pydicom

        ds = pydicom.dcmread(source_path)
        delivery = pymedphys.Delivery.from_dicom(ds)
    elif source == "mosaiq":
        if mosaiq_connection is None:
            return {"error": "Mosaiq connection required for mosaiq source"}
        delivery = pymedphys.Delivery.from_mosaiq(mosaiq_connection, source_path)
    else:
        return {"error": f"Unknown source type: {source}"}

    # Calculate metersetmap
    mu = delivery.monitor_units
    mlc = delivery.mlc
    jaw = delivery.jaw

    grid = pymedphys.metersetmap.grid(
        max_leaf_gap=400,
        grid_resolution=grid_resolution,
    )

    metersetmap = pymedphys.metersetmap.calculate(
        mu=mu,
        mlc=mlc,
        jaw=jaw,
        grid_resolution=grid_resolution,
    )

    # Format output
    result = {
        "source": source,
        "source_path": source_path,
        "grid_resolution_mm": grid_resolution,
        "shape": list(metersetmap.shape),
        "total_mu": float(np.sum(mu)),
        "max_fluence": float(np.max(metersetmap)),
    }

    if output_format == "json":
        # Include grid coordinates and flattened data for smaller maps
        if metersetmap.size < 10000:
            result["grid_x"] = grid["x"].tolist()
            result["grid_y"] = grid["y"].tolist()
            result["data"] = metersetmap.tolist()
        else:
            result["note"] = "Data too large for JSON, use 'numpy' format"
    elif output_format == "numpy":
        # Save to temporary file and return path
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, mode="wb"
        ) as f:
            np.save(f, {"grid": grid, "metersetmap": metersetmap})
            result["numpy_file"] = f.name
    elif output_format == "image":
        # Generate image representation
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pcolormesh(grid["x"], grid["y"], metersetmap, shading="auto")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("MetersetMap")
        ax.set_aspect("equal")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        result["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    return result
