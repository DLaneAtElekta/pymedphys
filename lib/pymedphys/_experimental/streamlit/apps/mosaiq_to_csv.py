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

import base64
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from pymedphys._imports import numpy as np
from pymedphys._imports import pandas as pd
from pymedphys._imports import streamlit as st
from pymedphys._imports import toml

import pymedphys
from pymedphys._config import get_config_dir
from pymedphys._mosaiq import connect as _connect
from pymedphys._mosaiq import credentials as _credentials
from pymedphys._streamlit import categories
from pymedphys._streamlit.utilities import config as st_config
from pymedphys._streamlit.utilities import mosaiq as _mosaiq

CATEGORY = categories.DRAFT
TITLE = "Mosaiq to CSV"

DEFAULT_PORT = 1433
DEFAULT_DATABASE = "MOSAIQ"

LIB_ROOT = pathlib.Path(__file__).parents[3]

# pylint: disable=no-member
TEST_DATA_DIR = LIB_ROOT.joinpath("tests", "mosaiq", "data")
# pylint: enable=no-member

PASSWORD_REPLACE = b"\x00" * 15
FIRST_NAME_USERNAME_MAP = {
    "Simon": "dummyusername",
}

# Given dynamic SQL queries are created in the functions below the SQL
# query is sanitised by only allowing table names and column names to
# pull from the below.
ALLOWLIST_TABLE_NAMES = [
    "Ident",
    "Patient",
    "TxField",
    "TxFieldPoint",
    "Site",
    "TrackTreatment",
    "Staff",
    "Chklist",
    "QCLTask",
]

ALLOWLIST_COLUMN_NAMES = [
    "IDA",
    "Pat_ID1",
    "SIT_Set_ID",
    "FLD_ID",
    "Staff_ID",
    "TSK_ID",
]


def _get_valid_mosaiq_sites_from_config(
    config: Dict,
) -> Dict[str, Dict[str, Any]]:
    """Extract valid Mosaiq site configurations from the config.

    Parameters
    ----------
    config : Dict
        The PyMedPhys configuration dictionary.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping site names to their Mosaiq configuration
        (hostname, port, alias).
    """
    valid_sites = {}

    for site_config in config.get("site", []):
        try:
            site_name = site_config["name"]
            mosaiq_config = site_config["mosaiq"]
            hostname = mosaiq_config["hostname"]
        except KeyError:
            continue

        port = mosaiq_config.get("port", DEFAULT_PORT)
        alias = mosaiq_config.get("alias")

        valid_sites[site_name] = {
            "hostname": hostname,
            "port": port,
            "alias": alias,
        }

    return valid_sites


def _prompt_for_mosaiq_connection() -> Tuple[str, int, str, str]:
    """Prompt the user to enter Mosaiq connection details via Streamlit UI.

    Returns
    -------
    Tuple[str, int, str, str]
        A tuple of (hostname, port, database, site_name).
    """
    st.write("## Mosaiq Database Connection Setup")
    st.write(
        "No Mosaiq database configuration found. "
        "Please enter your connection details below."
    )

    site_name = st.text_input(
        "Site Name",
        value="my-site",
        help="A friendly name for this site configuration.",
    )

    hostname = st.text_input(
        "SQL Server Hostname",
        help="The IP address or hostname of the Mosaiq SQL server.",
    )

    port = st.number_input(
        "SQL Server Port",
        value=DEFAULT_PORT,
        min_value=1,
        max_value=65535,
        help="The port number for the SQL server (default: 1433).",
    )

    database = st.text_input(
        "Database Name",
        value=DEFAULT_DATABASE,
        help="The name of the Mosaiq database (default: MOSAIQ).",
    )

    return hostname, int(port), database, site_name


def _save_mosaiq_config_to_file(
    site_name: str,
    hostname: str,
    port: int = DEFAULT_PORT,
    database: str = DEFAULT_DATABASE,
) -> pathlib.Path:
    """Save Mosaiq connection configuration to the config.toml file.

    Parameters
    ----------
    site_name : str
        The friendly name for this site.
    hostname : str
        The SQL server hostname.
    port : int, optional
        The SQL server port, by default 1433.
    database : str, optional
        The database name, by default "MOSAIQ".

    Returns
    -------
    pathlib.Path
        The path to the config file that was updated.
    """
    config_dir = get_config_dir()
    config_path = config_dir / "config.toml"

    # Load existing config or create new one
    if config_path.exists():
        with open(config_path) as f:
            config = toml.load(f)
    else:
        config = {"version": 0}

    # Ensure 'site' key exists as a list
    if "site" not in config:
        config["site"] = []

    # Check if site already exists and update it, or add new site
    site_exists = False
    for site_config in config["site"]:
        if site_config.get("name") == site_name:
            # Update existing site
            if "mosaiq" not in site_config:
                site_config["mosaiq"] = {}
            site_config["mosaiq"]["hostname"] = hostname
            site_config["mosaiq"]["port"] = port
            site_config["mosaiq"]["alias"] = f"{site_name} Mosaiq SQL Server"
            site_exists = True
            break

    if not site_exists:
        # Add new site configuration
        new_site = {
            "name": site_name,
            "mosaiq": {
                "hostname": hostname,
                "port": port,
                "alias": f"{site_name} Mosaiq SQL Server",
            },
        }
        config["site"].append(new_site)

    # Write the updated config back to file
    with open(config_path, "w") as f:
        toml.dump(config, f)

    return config_path


def _get_mosaiq_connection() -> _connect.Connection:
    """Get a Mosaiq database connection, prompting for config if needed.

    This function checks for existing Mosaiq configuration. If none exists,
    it prompts the user to enter connection details and optionally saves
    them to the config file.

    Returns
    -------
    pymedphys.mosaiq.Connection
        A connection object to the Mosaiq database.
    """
    # Try to load existing config
    try:
        config = st_config.get_config()
    except FileNotFoundError:
        config = {}

    valid_sites = _get_valid_mosaiq_sites_from_config(config)

    if valid_sites:
        # Use existing configuration
        return _mosaiq.get_single_mosaiq_connection_with_config(config)

    # No valid config found, prompt user for connection details
    hostname, port, database, site_name = _prompt_for_mosaiq_connection()

    if not hostname:
        st.warning("Please enter a hostname to continue.")
        st.stop()

    # Option to save configuration
    col1, col2 = st.columns(2)

    with col1:
        save_config = st.checkbox(
            "Save connection details to config file",
            value=True,
            help="Save these settings to ~/.pymedphys/config.toml for future use.",
        )

    with col2:
        connect_button = st.button("Connect", type="primary")

    if not connect_button:
        st.stop()

    # Save config if requested
    if save_config:
        config_path = _save_mosaiq_config_to_file(
            site_name=site_name,
            hostname=hostname,
            port=port,
            database=database,
        )
        st.success(f"Configuration saved to {config_path}")

    # Now get the connection using the streamlit mosaiq utility
    # which will prompt for credentials if needed
    return _mosaiq.get_uncached_mosaiq_connection(
        hostname=hostname,
        port=port,
        database=database,
        alias=f"{site_name} Mosaiq SQL Server",
    )


def main():
    connection = _get_mosaiq_connection()

    comma_sep_patient_ids: str = st.text_input("Comma Separated Patient IDs")
    if comma_sep_patient_ids == "":
        st.stop()

    patient_ids = [item.strip() for item in comma_sep_patient_ids.split(",")]
    tables, types_map = _get_all_tables(connection, patient_ids)
    _apply_table_type_conversions_inplace(tables, types_map)

    _save_tables_to_tests_directory(tables, types_map)


def _get_all_tables(
    connection: pymedphys.mosaiq.Connection, patient_ids: List[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, str]]]:
    """Get Mosaiq tables that are relevant for PyMedPhys regression testing.

    Take a list of patient_ids and steps through the MSSQL Mosaiq
    database with the intent to extract the relevant rows from the
    relevant tables for PyMedPhys regression testing.


    Parameters
    ----------
    connection : pymedphys.mosaiq.Connection
        A connection object to the Mosaiq SQL server.
    patient_ids : List[str]
        The list of Patient IDs (MRNs) to use for extracting data from
        Mosaiq.

    Returns
    -------
    tables
        A dictionary of Pandas DataFrames with dictionary keys defined
        by the Mosaiq table name and the table contents being the Mosaiq
        rows that are relevant to the Patient IDs provided.

    types_map
        A dictionary of dictionaries that present the MSSQL column types
        of the tables.
    """
    tables: Dict[str, pd.DataFrame] = {}
    types_map: Dict[str, Dict[str, str]] = {}

    tables["Ident"] = get_filtered_table(
        connection, types_map, "Ident", "IDA", patient_ids
    )

    # Patient.Pat_ID1 = Ident.Pat_ID1
    pat_id1s = tables["Ident"]["Pat_Id1"].unique()
    tables["Patient"] = get_filtered_table(
        connection, types_map, "Patient", "Pat_ID1", pat_id1s
    )
    tables["TxField"] = get_filtered_table(
        connection, types_map, "TxField", "Pat_ID1", pat_id1s
    )

    # TxField.SIT_Set_ID = Site.SIT_Set_ID
    sit_set_ids = tables["TxField"]["SIT_Set_ID"].unique()
    tables["Site"] = get_filtered_table(
        connection, types_map, "Site", "SIT_Set_ID", sit_set_ids
    )

    # TrackTreatment.FLD_ID = TxField.FLD_ID
    fld_ids = tables["TxField"]["FLD_ID"].unique()
    tables["TrackTreatment"] = get_filtered_table(
        connection, types_map, "TrackTreatment", "FLD_ID", fld_ids
    )

    # Chklist.Pat_ID1 = Ident.Pat_ID1 AND
    # Patient.Pat_ID1 = Ident.Pat_ID1 AND
    # QCLTask.TSK_ID = Chklist.TSK_ID AND
    # Staff.Staff_ID = Chklist.Rsp_Staff_ID AND
    tables["Chklist"] = get_filtered_table(
        connection, types_map, "Chklist", "Pat_ID1", pat_id1s
    )

    tsk_ids = tables["Chklist"]["TSK_ID"].unique()
    tables["QCLTask"] = get_filtered_table(
        connection, types_map, "QCLTask", "TSK_ID", tsk_ids
    )

    responsible_staff_ids = tables["Chklist"]["Rsp_Staff_ID"].unique()
    completed_staff_ids = tables["Chklist"]["Com_Staff_ID"].unique()
    machine_staff_ids = tables["TrackTreatment"]["Machine_ID_Staff_ID"].unique()
    staff_ids_with_nans = (
        set(responsible_staff_ids).union(completed_staff_ids).union(machine_staff_ids)
    )
    staff_ids = np.array(list(staff_ids_with_nans))
    staff_ids = staff_ids[np.logical_not(np.isnan(staff_ids))]
    staff_ids = staff_ids.astype(int)

    # Staff.Staff_ID = TrackTreatment.Machine_ID_Staff_ID
    tables["Staff"] = get_filtered_table(
        connection, types_map, "Staff", "Staff_ID", staff_ids.tolist()
    )
    tables["Staff"]["PasswordBytes"] = tables["Staff"]["PasswordBytes"].apply(
        lambda x: PASSWORD_REPLACE
    )
    for index, row in tables["Staff"].iterrows():
        first_name = row["First_Name"]
        if first_name.strip() == "":
            continue

        new_username = FIRST_NAME_USERNAME_MAP[first_name]
        tables["Staff"].loc[index, "User_Name"] = new_username

    # TxFieldPoint.FLD_ID = %(field_id)s
    tables["TxFieldPoint"] = get_filtered_table(
        connection, types_map, "TxFieldPoint", "FLD_ID", fld_ids
    )

    return tables, types_map


def _apply_table_type_conversions_inplace(tables, types_map):
    """Convert binary types to b64 and make sure pandas defines datetime
    types even if a column has a None entry.

    Utilised for reliable saving and loading to and from a csv file.
    """
    for table_name, table in tables.items():
        column_types = types_map[table_name]
        for column_name, column_type in column_types.items():
            if column_type in ["binary", "timestamp"]:
                table[column_name] = table[column_name].apply(
                    lambda x: base64.urlsafe_b64encode(x).decode()
                )
                continue
            if column_type == "datetime":
                table[column_name] = table[column_name].apply(_convert_to_datetime)

        st.write(f"## `{table_name}` Table")
        st.write(table)


def _save_tables_to_tests_directory(tables, types_map):
    """Save the tables within the PyMedPhys testing directory."""
    if not st.button("Save tables within PyMedPhys mosaiq testing dir"):
        st.stop()

    for table_name, df in tables.items():
        filepath = TEST_DATA_DIR.joinpath(table_name).with_suffix(".csv")
        df.to_csv(filepath)

    toml_filepath = TEST_DATA_DIR.joinpath("types_map.toml")

    with open(toml_filepath, "w") as f:
        toml.dump(types_map, f)


def get_filtered_table(
    connection: pymedphys.mosaiq.Connection,
    types_map: Dict[str, Dict[str, str]],
    table: str,
    column_name: str,
    column_values: List[Any],
) -> "pd.DataFrame":
    """Step through a set of provided column values extracting these
    from the MSSQL database.

    Parameters
    ----------
    connection : pymedphys.mosaiq.Connection
    types_map : Dict[str, Dict[str, str]]
        The types_map to append to the new column schema to
    table : str
        The table name to pull data from
    column_name : str
        The column name to pull data from
    column_values : List[Any]
        The values to match against within the columns

    Returns
    -------
    df : pd.DataFrame
    """
    column_names, column_types = _get_all_columns(connection, table)
    df = pd.DataFrame(data=[], columns=column_names)
    for column_value in column_values:
        df = _append_filtered_table(connection, df, table, column_name, column_value)

    types_map[table] = dict(zip(column_names, column_types))

    return df


def _append_filtered_table(connection, df, table, column_name, column_value):
    """Append the rows from an MSSQL table where the column_value
    matches within the given column_name.
    """
    df = pd.concat(
        [df, _get_filtered_table(connection, table, column_name, column_value)],
        ignore_index=True,
    )
    return df


@st.cache_data(ttl=86400, hash_funcs={pymedphys.mosaiq.Connection: id})
def _get_all_columns(connection, table):
    """Get the column schema from an MSSQL table."""
    raw_columns = pymedphys.mosaiq.execute(
        connection,
        """
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = %(table)s
        """,
        {
            "table": table,
        },
    )

    columns = [item[0] for item in raw_columns]
    types = [item[1] for item in raw_columns]

    return columns, types


@st.cache_data(ttl=86400, hash_funcs={pymedphys.mosaiq.Connection: id})
def _get_filtered_table(connection, table, column_name, column_value):
    """Get the rows from an MSSQL table where the column_value matches
    within the given column_name."""
    if table not in ALLOWLIST_TABLE_NAMES:
        raise ValueError(f"{table} must be within the allowlist")

    if column_name not in ALLOWLIST_COLUMN_NAMES:
        raise ValueError(f"{column_name} must be within the allowlist")

    column_value = str(column_value)
    column_names, _ = _get_all_columns(connection, table)

    sql_string = f"""
        SELECT *
        FROM {table}
        WHERE {table}.{column_name} = %(column_value)s
        """

    raw_results = pymedphys.mosaiq.execute(
        connection,
        sql_string,
        {
            "table": table,
            "column_name": column_name,
            "column_value": column_value,
        },
    )

    df = pd.DataFrame(raw_results, columns=column_names)

    return df


def _convert_to_datetime(item):
    if item is not None:
        return pd.to_datetime(item)

    return item
