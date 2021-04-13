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


from pymedphys._imports import numpy as np
from pymedphys._imports import pytest

import pymedphys
from pymedphys._mosaiq import helpers

from . import _connect
from .data import mimics

PATIENT_ID = 989898
FIELD_ID = 88043
A_TREATMENT_TIME = "2020-04-27 08:03:28.513"
MACHINE_ID = "2619"
FIELD_NAME = "3ABUT"
TIMEZONE = "Australia/Sydney"
FIRST_NAME = "MOCK"
LAST_NAME = "PHYSICS"
FULL_NAME = f"{LAST_NAME}, {FIRST_NAME.capitalize()}"


@pytest.fixture(name="connection")
def connection_base():
    """ will create the test database, if it does not already exist on the instance """
    mimics.create_db_with_tables()
    return _connect.connect(database=mimics.DATABASE)


@pytest.fixture(name="trf_filepath")
def trf_filepath_base():
    data_paths = pymedphys.zip_data_paths("metersetmap-gui-e2e-data.zip")
    date = A_TREATMENT_TIME.split(" ")[0]

    filtered_paths = [
        path
        for path in data_paths
        if date in str(path) and str(FIELD_ID) in str(path) and path.suffix == ".trf"
    ]

    return filtered_paths[0]


@pytest.fixture(name="dicom_filepath")
def dicom_filepath_base():
    data_paths = pymedphys.zip_data_paths("metersetmap-gui-e2e-data.zip")

    filtered_paths = [
        path for path in data_paths if path.name == f"{PATIENT_ID}_{FIELD_NAME}.dcm"
    ]

    return filtered_paths[0]


@pytest.mark.mosaiqdb
def test_get_patient_name(connection):
    name = helpers.get_patient_name(connection, PATIENT_ID)
    assert name == FULL_NAME


@pytest.mark.mosaiqdb
def test_get_patient_fields(connection):
    tx_fields = helpers.get_patient_fields(connection, PATIENT_ID)
    field_id = tx_fields["field_id"].iloc[0]
    assert field_id == FIELD_ID


@pytest.mark.mosaiqdb
def test_get_treatment_times(connection):
    treatment_times = helpers.get_treatment_times(connection, FIELD_ID)
    assert np.datetime64(A_TREATMENT_TIME) in treatment_times["start"].tolist()


@pytest.mark.mosaiqdb
def test_get_treatments(connection):
    dt = np.timedelta64(4, "h")
    start = np.datetime64(A_TREATMENT_TIME) - dt
    end = np.datetime64(A_TREATMENT_TIME) + dt

    treatments = helpers.get_treatments(connection, start, end, MACHINE_ID)
    assert np.datetime64(A_TREATMENT_TIME) in treatments["start"].tolist()


@pytest.mark.mosaiqdb
def test_delivery_from_mosaiq(connection, trf_filepath, dicom_filepath):
    trf_delivery = pymedphys.Delivery.from_trf(trf_filepath)
    dicom_delivery = pymedphys.Delivery.from_dicom(dicom_filepath)
    mosaiq_delivery = pymedphys.Delivery.from_mosaiq(connection, FIELD_ID)

    assert np.allclose(dicom_delivery.mu, mosaiq_delivery.mu, atol=1)
    assert np.allclose(dicom_delivery.mlc, mosaiq_delivery.mlc, atol=0.1)
    assert np.allclose(dicom_delivery.jaw, mosaiq_delivery.jaw, atol=0.1)
    assert np.allclose(dicom_delivery.gantry, mosaiq_delivery.gantry, atol=0.1)
    assert np.allclose(dicom_delivery.collimator, mosaiq_delivery.collimator, atol=0.1)

    assert np.abs(trf_delivery.mu[-1] - mosaiq_delivery.mu[-1]) < 0.2
    trf_metersetmap = trf_delivery.metersetmap(grid_resolution=5)
    mosaiq_metersetmap = mosaiq_delivery.metersetmap(grid_resolution=5)

    max_deviation = np.max(np.abs(trf_metersetmap - mosaiq_metersetmap))
    assert max_deviation < 3


@pytest.mark.mosaiqdb
def test_trf_identification(connection: pymedphys.mosaiq.Connection, trf_filepath):
    delivery_details = pymedphys.trf.identify(connection, trf_filepath, TIMEZONE)
    assert delivery_details.field_id == FIELD_ID
    assert delivery_details.first_name == FIRST_NAME
    assert delivery_details.last_name == LAST_NAME
    assert delivery_details.patient_id == str(PATIENT_ID)
