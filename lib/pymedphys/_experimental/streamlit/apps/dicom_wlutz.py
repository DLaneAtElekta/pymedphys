# Copyright (C) 2021 Cancer Care Associates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import tempfile

from pymedphys._imports import numpy as np
from pymedphys._imports import pydicom
from pymedphys._imports import streamlit as st

from pymedphys._streamlit import categories

from pymedphys._experimental.streamlit.utilities.dicom import loader as _loader
from pymedphys._experimental.vendor.pylinac_vendored._pylinac_installed import (
    pylinac as _pylinac_installed,
)
from pymedphys._experimental.wlutz import reporting as _reporting

CATEGORY = categories.PLANNING
TITLE = "DICOM WLutz"


def main():
    dicom_datasets = _loader.dicom_file_loader(
        accept_multiple_files=True, stop_before_pixels=False
    )

    progress_bar = st.progress(0)
    wl_images = []
    for i, dataset in enumerate(dicom_datasets):
        wl_image = _nasty_wrapper_around_pylinac(dataset)
        wl_images.append(wl_image)

        progress_bar.progress((i + 1) / len(dicom_datasets))

        st.write(wl_image.array.shape)
        st.write(wl_image.center)
        st.write(wl_image.field_cax)
        st.write(wl_image.bb)
        st.write(wl_image.rad_field_bounding_box[0])

    bb_diameter = st.number_input(
        "Ball bearing diameter (mm): ", min_value=0.0, max_value=None, value=8.0
    )
    penumbra = st.number_input(
        "Field penumbra size (mm): ", min_value=0.0, max_value=None, value=3.0
    )
    (
        x,
        y,
        image,
        bb_centre,
        field_centre,
        edge_lengths,
    ) = _display_parameters_from_wl_image(wl_images[0], penumbra)
    st.write(dir(wl_images[0]))

    field_rotation = 0

    fig, _ = _reporting.image_analysis_figure(
        x,
        y,
        image,
        bb_centre,
        field_centre,
        field_rotation,
        bb_diameter,
        edge_lengths,
        penumbra,
    )

    st.pyplot(fig)


def _display_parameters_from_wl_image(wl_image, penumbra):
    centre = wl_image.center
    dpmm = wl_image.dpmm
    bb = wl_image.bb
    field = wl_image.field_cax

    image = wl_image.array
    x = (np.arange(image.shape[1]) - centre.x) / dpmm
    y = (np.arange(image.shape[0]) - centre.y) / dpmm

    bb_centre = _diff_to_centre(bb, centre, dpmm)
    field_centre = _diff_to_centre(field, centre, dpmm)

    y_length = (
        wl_image.rad_field_bounding_box[1] - wl_image.rad_field_bounding_box[0]
    ) / dpmm - penumbra
    x_length = (
        wl_image.rad_field_bounding_box[3] - wl_image.rad_field_bounding_box[2]
    ) / dpmm - penumbra

    edge_lengths = (x_length, y_length)

    return x, y, image, bb_centre, field_centre, edge_lengths


def _diff_to_centre(item, centre, dpmm):
    point = (item - centre) / dpmm
    if point.z != 0:
        raise ValueError("Only 2D points supported")

    return (point.x, point.y)


@st.cache(allow_output_mutation=True, show_spinner=True)
def _nasty_wrapper_around_pylinac(dataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_file = pathlib.Path(temp_dir, "dicom_file.dcm")
        pydicom.dcmwrite(dicom_file, dataset)
        wl_image = _pylinac_installed.winston_lutz.WLImage(
            dicom_file, use_filenames=False
        )

    return wl_image
