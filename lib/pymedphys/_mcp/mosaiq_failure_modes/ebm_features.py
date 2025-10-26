"""Feature extraction for Energy-Based Model (EBM) anomaly detection.

This module extracts QA features from Mosaiq database records to serve as inputs
for an adversarial EBM trained to detect data corruption and third-party write failures.
"""

import struct
from datetime import datetime
from typing import Any

import numpy as np
import pymssql


def extract_mlc_features(a_leaf_set: bytes, b_leaf_set: bytes) -> dict[str, float]:
    """Extract MLC-related features for anomaly detection.

    Features:
    - Byte array length consistency
    - Leaf position statistics (mean, std, min, max)
    - Leaf gap statistics
    - Out-of-range indicators
    - Leaf pair violations

    Args:
        a_leaf_set: Binary MLC A-bank data
        b_leaf_set: Binary MLC B-bank data

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: Byte length properties
    features["a_byte_length"] = len(a_leaf_set)
    features["b_byte_length"] = len(b_leaf_set)
    features["byte_length_match"] = float(len(a_leaf_set) == len(b_leaf_set))
    features["byte_length_even"] = float(len(a_leaf_set) % 2 == 0 and len(b_leaf_set) % 2 == 0)

    # Decode MLC positions (2 bytes per leaf, little-endian signed short)
    try:
        a_positions = [
            struct.unpack("<h", a_leaf_set[i : i + 2])[0] / 100.0  # Convert to cm
            for i in range(0, len(a_leaf_set) - 1, 2)
        ]
        b_positions = [
            struct.unpack("<h", b_leaf_set[i : i + 2])[0] / 100.0
            for i in range(0, len(b_leaf_set) - 1, 2)
        ]

        # Feature 2: Position statistics
        features["a_mean_pos"] = float(np.mean(a_positions))
        features["a_std_pos"] = float(np.std(a_positions))
        features["a_min_pos"] = float(np.min(a_positions))
        features["a_max_pos"] = float(np.max(a_positions))

        features["b_mean_pos"] = float(np.mean(b_positions))
        features["b_std_pos"] = float(np.std(b_positions))
        features["b_min_pos"] = float(np.min(b_positions))
        features["b_max_pos"] = float(np.max(b_positions))

        # Feature 3: Out-of-range indicators (typical limits ±20cm)
        features["a_out_of_range"] = float(
            any(abs(p) > 20.0 for p in a_positions)
        )
        features["b_out_of_range"] = float(
            any(abs(p) > 20.0 for p in b_positions)
        )

        # Feature 4: Leaf gap analysis (A should be <= B for each pair)
        if len(a_positions) == len(b_positions):
            gaps = [b - a for a, b in zip(a_positions, b_positions)]
            features["min_gap"] = float(np.min(gaps))
            features["mean_gap"] = float(np.mean(gaps))
            features["negative_gap_count"] = float(sum(1 for g in gaps if g < 0))
            features["max_gap"] = float(np.max(gaps))
        else:
            features["min_gap"] = -999.0  # Invalid indicator
            features["mean_gap"] = -999.0
            features["negative_gap_count"] = 999.0
            features["max_gap"] = -999.0

        # Feature 5: Leaf count
        features["leaf_count"] = float(len(a_positions))

    except (struct.error, ValueError):
        # Corrupted binary data - fill with invalid indicators
        features.update(
            {
                "a_mean_pos": -999.0,
                "a_std_pos": -999.0,
                "a_min_pos": -999.0,
                "a_max_pos": -999.0,
                "b_mean_pos": -999.0,
                "b_std_pos": -999.0,
                "b_min_pos": -999.0,
                "b_max_pos": -999.0,
                "a_out_of_range": 1.0,
                "b_out_of_range": 1.0,
                "min_gap": -999.0,
                "mean_gap": -999.0,
                "negative_gap_count": 999.0,
                "max_gap": -999.0,
                "leaf_count": 0.0,
            }
        )

    return features


def extract_angle_features(
    gantry_angles: list[float], collimator_angles: list[float]
) -> dict[str, float]:
    """Extract gantry and collimator angle features.

    Features:
    - Range validation (0-360°)
    - Angle transition statistics
    - Discontinuity detection
    - Speed feasibility (if timing available)

    Args:
        gantry_angles: List of gantry angles per control point
        collimator_angles: List of collimator angles per control point

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: Range violations
    features["gantry_out_of_range"] = float(
        any(a < 0 or a > 360 for a in gantry_angles)
    )
    features["coll_out_of_range"] = float(
        any(a < 0 or a > 360 for a in collimator_angles)
    )

    # Feature 2: Angle statistics
    features["gantry_mean"] = float(np.mean(gantry_angles))
    features["gantry_std"] = float(np.std(gantry_angles))
    features["gantry_range"] = float(np.max(gantry_angles) - np.min(gantry_angles))

    features["coll_mean"] = float(np.mean(collimator_angles))
    features["coll_std"] = float(np.std(collimator_angles))
    features["coll_range"] = float(np.max(collimator_angles) - np.min(collimator_angles))

    # Feature 3: Transition analysis
    if len(gantry_angles) > 1:
        gantry_deltas = [
            abs(gantry_angles[i] - gantry_angles[i - 1])
            for i in range(1, len(gantry_angles))
        ]
        # Handle wraparound (359° to 1° is 2°, not 358°)
        gantry_deltas = [min(d, 360 - d) for d in gantry_deltas]

        features["gantry_max_delta"] = float(np.max(gantry_deltas))
        features["gantry_mean_delta"] = float(np.mean(gantry_deltas))

        coll_deltas = [
            abs(collimator_angles[i] - collimator_angles[i - 1])
            for i in range(1, len(collimator_angles))
        ]
        coll_deltas = [min(d, 360 - d) for d in coll_deltas]

        features["coll_max_delta"] = float(np.max(coll_deltas))
        features["coll_mean_delta"] = float(np.mean(coll_deltas))

        # Feature 4: Suspicious jumps (large but not wraparound)
        features["gantry_suspicious_jumps"] = float(
            sum(1 for d in gantry_deltas if 90 < d < 270)
        )
        features["coll_suspicious_jumps"] = float(
            sum(1 for d in coll_deltas if 90 < d < 270)
        )
    else:
        features.update(
            {
                "gantry_max_delta": 0.0,
                "gantry_mean_delta": 0.0,
                "coll_max_delta": 0.0,
                "coll_mean_delta": 0.0,
                "gantry_suspicious_jumps": 0.0,
                "coll_suspicious_jumps": 0.0,
            }
        )

    return features


def extract_control_point_features(
    field_data: dict[str, Any]
) -> dict[str, float]:
    """Extract control point sequence features.

    Features:
    - Control point count
    - Sequential index validation
    - Missing control point detection
    - Index gap statistics

    Args:
        field_data: Dictionary with 'control_points' list containing Point values

    Returns:
        Dictionary of feature values
    """
    features = {}

    control_points = sorted(field_data.get("control_points", []))

    # Feature 1: Count
    features["cp_count"] = float(len(control_points))
    features["cp_minimum_met"] = float(len(control_points) >= 2)

    # Feature 2: Sequential validation
    if control_points:
        expected = list(range(len(control_points)))
        features["cp_sequential"] = float(control_points == expected)

        # Feature 3: Gap detection
        gaps = [
            control_points[i] - control_points[i - 1]
            for i in range(1, len(control_points))
        ]
        features["cp_max_gap"] = float(max(gaps)) if gaps else 0.0
        features["cp_gap_count"] = float(sum(1 for g in gaps if g > 1))

        # Feature 4: Starting point
        features["cp_starts_at_zero"] = float(control_points[0] == 0)
    else:
        features.update(
            {
                "cp_sequential": 0.0,
                "cp_max_gap": 0.0,
                "cp_gap_count": 0.0,
                "cp_starts_at_zero": 0.0,
            }
        )

    return features


def extract_timestamp_features(
    create_time: datetime, edit_time: datetime, dose_time: datetime | None = None
) -> dict[str, float]:
    """Extract timestamp relationship features.

    Features:
    - Edit before create flag
    - Future timestamp flag
    - Treatment duration
    - Duration z-score (if baseline provided)

    Args:
        create_time: Treatment start time
        edit_time: Treatment end time
        dose_time: Dose history timestamp (optional)

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: Temporal ordering
    features["edit_before_create"] = float(edit_time < create_time)

    # Feature 2: Future timestamp (relative to current time)
    now = datetime.now()
    features["create_in_future"] = float(create_time > now)
    features["edit_in_future"] = float(edit_time > now)

    # Feature 3: Duration
    duration_seconds = (edit_time - create_time).total_seconds()
    features["duration_seconds"] = duration_seconds
    features["duration_negative"] = float(duration_seconds < 0)
    features["duration_too_short"] = float(duration_seconds < 10)
    features["duration_too_long"] = float(duration_seconds > 1800)  # 30 minutes

    # Feature 4: Cross-reference with dose history
    if dose_time:
        dose_delta = abs((create_time - dose_time).total_seconds())
        features["dose_time_delta"] = dose_delta
        features["dose_time_mismatch"] = float(dose_delta > 300)  # >5 minutes
    else:
        features["dose_time_delta"] = 0.0
        features["dose_time_mismatch"] = 0.0

    return features


def extract_offset_features(
    superior: float, anterior: float, lateral: float, offset_type: int, offset_state: int
) -> dict[str, float]:
    """Extract patient positioning offset features.

    Features:
    - Offset magnitudes (individual and vector)
    - Extreme value flags
    - Invalid enumeration flags
    - Direction-specific outliers

    Args:
        superior: Superior offset in cm
        anterior: Anterior offset in cm
        lateral: Lateral offset in cm
        offset_type: Type enumeration (2/3/4)
        offset_state: State enumeration (1/2)

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: Individual components
    features["offset_superior"] = superior
    features["offset_anterior"] = anterior
    features["offset_lateral"] = lateral

    # Feature 2: Vector magnitude
    vector_mag = np.sqrt(superior**2 + anterior**2 + lateral**2)
    features["offset_vector_magnitude"] = float(vector_mag)

    # Feature 3: Extreme value flags (typical limits ±10cm)
    features["superior_extreme"] = float(abs(superior) > 10.0)
    features["anterior_extreme"] = float(abs(anterior) > 10.0)
    features["lateral_extreme"] = float(abs(lateral) > 10.0)
    features["vector_extreme"] = float(vector_mag > 15.0)

    # Feature 4: Critical level (±20cm)
    features["offset_critical"] = float(vector_mag > 20.0)

    # Feature 5: Enumeration validation
    features["type_valid"] = float(offset_type in [2, 3, 4])
    features["state_valid"] = float(offset_state in [1, 2])
    features["type_third_party"] = float(offset_type == 4)

    return features


def extract_meterset_features(
    planned_meterset: float,
    cp_mu_values: list[float] | None = None,
) -> dict[str, float]:
    """Extract meterset (MU) features.

    Features:
    - Negative MU flag
    - Extreme value flag
    - CP sum mismatch (if CP MU available)
    - MU statistics

    Args:
        planned_meterset: Planned MU from TxField
        cp_mu_values: Optional list of control point MU values

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: Basic validation
    features["mu_negative"] = float(planned_meterset < 0)
    features["mu_extreme_low"] = float(planned_meterset < 10)
    features["mu_extreme_high"] = float(planned_meterset > 1000)

    # Feature 2: Value
    features["mu_planned"] = planned_meterset

    # Feature 3: Control point comparison
    if cp_mu_values:
        cp_sum = sum(cp_mu_values)
        features["mu_cp_sum"] = cp_sum
        mu_diff = abs(cp_sum - planned_meterset)
        features["mu_cp_difference"] = mu_diff
        features["mu_cp_percent_diff"] = (
            (mu_diff / planned_meterset * 100) if planned_meterset > 0 else 999.0
        )
        features["mu_cp_mismatch"] = float(mu_diff > planned_meterset * 0.05)  # >5%
    else:
        features.update(
            {
                "mu_cp_sum": 0.0,
                "mu_cp_difference": 0.0,
                "mu_cp_percent_diff": 0.0,
                "mu_cp_mismatch": 0.0,
            }
        )

    return features


def extract_foreign_key_features(cursor: pymssql.Cursor, record_data: dict) -> dict[str, float]:
    """Extract foreign key integrity features.

    Features:
    - FK existence flags for each relationship
    - NULL FK count

    Args:
        cursor: Database cursor for FK validation queries
        record_data: Dictionary with FK fields (Pat_ID1, FLD_ID, SIT_ID, etc.)

    Returns:
        Dictionary of feature values
    """
    features = {}

    # Feature 1: NULL checks
    features["pat_id_null"] = float(record_data.get("Pat_ID1") is None)
    features["fld_id_null"] = float(record_data.get("FLD_ID") is None)
    features["sit_id_null"] = float(record_data.get("SIT_ID") is None)

    # Feature 2: FK existence validation
    if record_data.get("Pat_ID1"):
        cursor.execute(
            "SELECT COUNT(*) FROM Patient WHERE Pat_ID1 = %s",
            (record_data["Pat_ID1"],),
        )
        features["patient_exists"] = float(cursor.fetchone()[0] > 0)
    else:
        features["patient_exists"] = 0.0

    if record_data.get("FLD_ID"):
        cursor.execute(
            "SELECT COUNT(*) FROM TxField WHERE FLD_ID = %s",
            (record_data["FLD_ID"],),
        )
        features["field_exists"] = float(cursor.fetchone()[0] > 0)
    else:
        features["field_exists"] = 0.0

    if record_data.get("SIT_ID"):
        cursor.execute(
            "SELECT COUNT(*) FROM Site WHERE SIT_ID = %s",
            (record_data["SIT_ID"],),
        )
        features["site_exists"] = float(cursor.fetchone()[0] > 0)
    else:
        features["site_exists"] = 0.0

    return features


def extract_all_features(
    cursor: pymssql.Cursor,
    ttx_id: int,
) -> dict[str, float]:
    """Extract all features for a TrackTreatment record.

    This is the main feature extraction function that combines all feature types.

    Args:
        cursor: Database cursor
        ttx_id: TrackTreatment ID to extract features for

    Returns:
        Dictionary of all feature values for EBM input
    """
    # Query all relevant data
    cursor.execute(
        """
        SELECT
            tt.TTX_ID, tt.Pat_ID1, tt.FLD_ID, tt.SIT_ID,
            tt.Create_DtTm, tt.Edit_DtTm,
            f.Meterset, f.Field_Label,
            dh.Tx_DtTm
        FROM TrackTreatment tt
        LEFT JOIN TxField f ON tt.FLD_ID = f.FLD_ID
        LEFT JOIN Dose_Hst dh ON tt.Pat_ID1 = dh.Pat_ID1
                              AND tt.FLD_ID = dh.FLD_ID
                              AND ABS(DATEDIFF(second, tt.Create_DtTm, dh.Tx_DtTm)) < 300
        WHERE tt.TTX_ID = %s
        """,
        (ttx_id,),
    )

    row = cursor.fetchone()
    if not row:
        raise ValueError(f"TrackTreatment {ttx_id} not found")

    record_data = {
        "TTX_ID": row[0],
        "Pat_ID1": row[1],
        "FLD_ID": row[2],
        "SIT_ID": row[3],
        "Create_DtTm": row[4],
        "Edit_DtTm": row[5],
        "Meterset": row[6],
        "Field_Label": row[7],
        "Tx_DtTm": row[8],
    }

    # Initialize feature dictionary
    features = {}

    # 1. FK features
    features.update(extract_foreign_key_features(cursor, record_data))

    # 2. Timestamp features
    features.update(
        extract_timestamp_features(
            record_data["Create_DtTm"],
            record_data["Edit_DtTm"],
            record_data["Tx_DtTm"],
        )
    )

    # 3. Meterset features
    if record_data["Meterset"]:
        features.update(extract_meterset_features(record_data["Meterset"]))

    # 4. Control point features
    if record_data["FLD_ID"]:
        cursor.execute(
            """
            SELECT Point, Gantry_Ang, Coll_Ang, A_Leaf_Set, B_Leaf_Set
            FROM TxFieldPoint
            WHERE FLD_ID = %s
            ORDER BY Point
            """,
            (record_data["FLD_ID"],),
        )

        cp_rows = cursor.fetchall()
        if cp_rows:
            control_points = [row[0] for row in cp_rows]
            gantry_angles = [row[1] for row in cp_rows]
            coll_angles = [row[2] for row in cp_rows]

            features.update(
                extract_control_point_features({"control_points": control_points})
            )
            features.update(extract_angle_features(gantry_angles, coll_angles))

            # MLC features (use first control point as representative)
            if cp_rows[0][3] and cp_rows[0][4]:
                features.update(extract_mlc_features(cp_rows[0][3], cp_rows[0][4]))

    # 5. Offset features (if available)
    cursor.execute(
        """
        SELECT Superior_Offset, Anterior_Offset, Lateral_Offset,
               Offset_Type, Offset_State
        FROM Offset o
        JOIN Site s ON o.SIT_SET_ID = s.SIT_SET_ID
        WHERE s.SIT_ID = %s
        ORDER BY Study_DtTm DESC
        """,
        (record_data["SIT_ID"],),
    )

    offset_row = cursor.fetchone()
    if offset_row:
        features.update(
            extract_offset_features(
                offset_row[0], offset_row[1], offset_row[2], offset_row[3], offset_row[4]
            )
        )

    return features


def create_feature_vector(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """Convert feature dictionary to numpy array for EBM input.

    Args:
        features: Dictionary of feature values
        feature_names: Ordered list of feature names

    Returns:
        Numpy array of feature values in consistent order
    """
    return np.array([features.get(name, 0.0) for name in feature_names])


# Define standard feature set for EBM training
STANDARD_FEATURE_NAMES = [
    # FK features
    "pat_id_null",
    "fld_id_null",
    "sit_id_null",
    "patient_exists",
    "field_exists",
    "site_exists",
    # Timestamp features
    "edit_before_create",
    "create_in_future",
    "edit_in_future",
    "duration_seconds",
    "duration_negative",
    "duration_too_short",
    "duration_too_long",
    "dose_time_delta",
    "dose_time_mismatch",
    # Meterset features
    "mu_negative",
    "mu_extreme_low",
    "mu_extreme_high",
    "mu_planned",
    "mu_cp_sum",
    "mu_cp_difference",
    "mu_cp_percent_diff",
    "mu_cp_mismatch",
    # Control point features
    "cp_count",
    "cp_minimum_met",
    "cp_sequential",
    "cp_max_gap",
    "cp_gap_count",
    "cp_starts_at_zero",
    # Angle features
    "gantry_out_of_range",
    "coll_out_of_range",
    "gantry_mean",
    "gantry_std",
    "gantry_range",
    "coll_mean",
    "coll_std",
    "coll_range",
    "gantry_max_delta",
    "gantry_mean_delta",
    "coll_max_delta",
    "coll_mean_delta",
    "gantry_suspicious_jumps",
    "coll_suspicious_jumps",
    # MLC features
    "a_byte_length",
    "b_byte_length",
    "byte_length_match",
    "byte_length_even",
    "a_mean_pos",
    "a_std_pos",
    "a_min_pos",
    "a_max_pos",
    "b_mean_pos",
    "b_std_pos",
    "b_min_pos",
    "b_max_pos",
    "a_out_of_range",
    "b_out_of_range",
    "min_gap",
    "mean_gap",
    "negative_gap_count",
    "max_gap",
    "leaf_count",
    # Offset features
    "offset_superior",
    "offset_anterior",
    "offset_lateral",
    "offset_vector_magnitude",
    "superior_extreme",
    "anterior_extreme",
    "lateral_extreme",
    "vector_extreme",
    "offset_critical",
    "type_valid",
    "state_valid",
    "type_third_party",
]
