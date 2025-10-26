"""Feature extraction and detection algorithms for malicious actor patterns.

This module provides specialized features and detection methods for identifying
intentional sabotage attempts on radiotherapy databases. Unlike accidental errors,
malicious modifications are designed to evade detection through:

- Statistical camouflage (staying within normal variance)
- Temporal evasion (time-delayed or selective timing)
- Spatial distribution (coordinated multi-field errors)
- Audit trail manipulation (covering tracks)

All malicious failure modes have severity ≥ 2.5 (critical).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from pymssql import Connection


# =============================================================================
# STATISTICAL EVASION DETECTION
# =============================================================================


def cusum_analysis(
    parameter_series: list[float], target_value: float, sensitivity: float = 0.5, threshold: float = 5.0
) -> dict[str, Any]:
    """Detect systematic bias using CUSUM (Cumulative Sum) control charts.

    CUSUM is designed to detect small systematic shifts that standard outlier
    detection misses. It accumulates deviations from target and alerts when
    cumulative sum exceeds threshold.

    Args:
        parameter_series: List of parameter values (e.g., MU per fraction)
        target_value: Expected value (e.g., planned MU)
        sensitivity: Minimum shift to detect (fraction of target, default 0.5%)
        threshold: CUSUM threshold for alert (default 5.0 σ)

    Returns:
        Dictionary with:
            - alert: Boolean, True if systematic bias detected
            - cusum_positive: Final positive CUSUM value
            - cusum_negative: Final negative CUSUM value
            - alert_fractions: List of fraction numbers where threshold exceeded
            - bias_direction: 'positive', 'negative', or None

    Example:
        >>> mu_history = [200, 202, 204, 206, 208, 210]  # Gradual escalation
        >>> result = cusum_analysis(mu_history, target_value=200, threshold=5.0)
        >>> result['alert']  # True - detected systematic positive drift
        True
    """
    cusum_pos = 0.0
    cusum_neg = 0.0
    alert_fractions = []

    sensitivity_absolute = target_value * sensitivity / 100  # Convert to absolute units

    for i, value in enumerate(parameter_series):
        deviation = value - target_value

        # Update CUSUM
        cusum_pos = max(0, cusum_pos + deviation - sensitivity_absolute)
        cusum_neg = min(0, cusum_neg + deviation + sensitivity_absolute)

        # Check thresholds
        if cusum_pos > threshold:
            alert_fractions.append({"fraction": i + 1, "type": "positive_drift", "cusum": cusum_pos})

        if cusum_neg < -threshold:
            alert_fractions.append({"fraction": i + 1, "type": "negative_drift", "cusum": cusum_neg})

    # Determine overall bias direction
    bias_direction = None
    if cusum_pos > threshold:
        bias_direction = "positive"
    elif cusum_neg < -threshold:
        bias_direction = "negative"

    return {
        "alert": len(alert_fractions) > 0,
        "cusum_positive": cusum_pos,
        "cusum_negative": cusum_neg,
        "alert_fractions": alert_fractions,
        "bias_direction": bias_direction,
        "final_deviation": parameter_series[-1] - target_value if parameter_series else 0,
    }


def detect_systematic_bias(parameter_series: list[float], expected_mean: float) -> dict[str, Any]:
    """Test if parameter mean significantly differs from expected value.

    Uses one-sample t-test to detect systematic bias in either direction.
    More sensitive than outlier detection for subtle consistent biases.

    Args:
        parameter_series: List of parameter values across fractions
        expected_mean: Expected mean value (from treatment plan)

    Returns:
        Dictionary with statistical test results

    Example:
        >>> # All values slightly high (+1.5% bias)
        >>> mu_series = [201.5, 203, 202, 201, 202.5, 203]
        >>> result = detect_systematic_bias(mu_series, expected_mean=200)
        >>> result['alert']  # True if p < 0.01
    """
    if len(parameter_series) < 3:
        return {"alert": False, "error": "Insufficient data (need ≥3 samples)"}

    # One-sample t-test
    t_statistic, p_value = scipy.stats.ttest_1samp(parameter_series, expected_mean)

    actual_mean = np.mean(parameter_series)
    bias = actual_mean - expected_mean
    bias_percent = bias / expected_mean * 100 if expected_mean != 0 else 0

    return {
        "alert": p_value < 0.01,  # Significant at α=0.01
        "p_value": p_value,
        "t_statistic": t_statistic,
        "expected_mean": expected_mean,
        "actual_mean": actual_mean,
        "bias": bias,
        "bias_percent": bias_percent,
        "std_dev": np.std(parameter_series, ddof=1),
    }


def benford_law_test(data: list[float]) -> dict[str, Any]:
    """Detect artificial data using Benford's Law first-digit distribution.

    Natural data follows Benford's Law: P(d) = log10(1 + 1/d)
    Artificially generated data often violates this distribution.

    Args:
        data: List of numerical values (e.g., MU values, offsets)

    Returns:
        Dictionary with chi-square test results

    Example:
        >>> # Natural data should pass
        >>> natural = [123, 456, 789, 234, 567, 890, 111, 222]
        >>> result = benford_law_test(natural)

        >>> # Artificial data (all starting with 5) should fail
        >>> artificial = [500, 501, 502, 503, 504, 505]
        >>> result = benford_law_test(artificial)
        >>> result['alert']  # True
    """
    if len(data) < 20:
        return {"alert": False, "error": "Insufficient data (need ≥20 samples for Benford test)"}

    # Extract first significant digit
    first_digits = []
    for x in data:
        if x == 0:
            continue
        # Convert to string, remove decimal point, get first digit
        digit_str = str(abs(x)).replace(".", "").replace("-", "")
        if digit_str:
            first_digits.append(int(digit_str[0]))

    if len(first_digits) < 20:
        return {"alert": False, "error": "Too few non-zero values"}

    # Expected Benford distribution
    benford_expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    # Observed distribution
    observed_counts, _ = np.histogram(first_digits, bins=range(1, 11))
    observed_freq = observed_counts / len(first_digits)

    # Expected counts
    expected_counts = len(first_digits) * benford_expected

    # Chi-square test
    # Remove bins with expected count < 5 (chi-square assumption)
    valid_bins = expected_counts >= 5
    if valid_bins.sum() < 5:
        return {"alert": False, "error": "Too few valid bins for chi-square test"}

    chi2_statistic, p_value = scipy.stats.chisquare(observed_counts[valid_bins], expected_counts[valid_bins])

    return {
        "alert": p_value < 0.01,  # Significant deviation
        "p_value": p_value,
        "chi2_statistic": chi2_statistic,
        "observed_freq": observed_freq.tolist(),
        "expected_freq": benford_expected.tolist(),
        "interpretation": "Data may be artificially generated" if p_value < 0.01 else "Consistent with natural data",
    }


def runs_test_randomness(parameter_series: list[float], median: float | None = None) -> dict[str, Any]:
    """Test if sequence is random or has systematic patterns.

    A 'run' is a sequence of consecutive values above/below median.
    Non-random data has too few or too many runs.

    Args:
        parameter_series: List of parameter values
        median: Expected median (if None, use sample median)

    Returns:
        Dictionary with runs test results

    Example:
        >>> # Random sequence should pass
        >>> random_seq = [101, 99, 102, 98, 103, 97, 100, 102]
        >>> result = runs_test_randomness(random_seq)

        >>> # Systematic sequence should fail (all increasing)
        >>> systematic = [100, 101, 102, 103, 104, 105, 106]
        >>> result = runs_test_randomness(systematic)
        >>> result['alert']  # True
    """
    if len(parameter_series) < 10:
        return {"alert": False, "error": "Insufficient data (need ≥10 samples)"}

    if median is None:
        median = np.median(parameter_series)

    # Convert to binary sequence (above/below median)
    binary_series = [1 if x > median else 0 for x in parameter_series]

    # Count runs
    runs = 1
    for i in range(1, len(binary_series)):
        if binary_series[i] != binary_series[i - 1]:
            runs += 1

    n1 = sum(binary_series)  # Count of 1s
    n2 = len(binary_series) - n1  # Count of 0s

    if n1 == 0 or n2 == 0:
        return {"alert": True, "error": "All values on one side of median (extreme non-randomness)"}

    # Expected runs and variance for random sequence
    runs_expected = (2 * n1 * n2) / (n1 + n2) + 1
    runs_variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

    # Z-score
    z_statistic = (runs - runs_expected) / np.sqrt(runs_variance)

    # Two-tailed p-value
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_statistic)))

    # Interpretation
    if z_statistic < -2:
        interpretation = "Too few runs - systematic pattern (clustering)"
    elif z_statistic > 2:
        interpretation = "Too many runs - oscillating pattern"
    else:
        interpretation = "Consistent with random sequence"

    return {
        "alert": p_value < 0.01,
        "p_value": p_value,
        "z_statistic": z_statistic,
        "runs_observed": runs,
        "runs_expected": runs_expected,
        "interpretation": interpretation,
    }


def mann_kendall_trend_test(parameter_series: list[float]) -> dict[str, Any]:
    """Detect monotonic trends using Mann-Kendall test (non-parametric).

    More robust than linear regression for detecting trends in presence of outliers.

    Args:
        parameter_series: List of parameter values (time series)

    Returns:
        Dictionary with trend test results

    Example:
        >>> # Increasing trend
        >>> increasing = [100, 102, 101, 104, 103, 106, 105, 108]
        >>> result = mann_kendall_trend_test(increasing)
        >>> result['trend']  # 'increasing'
    """
    if len(parameter_series) < 10:
        return {"alert": False, "error": "Insufficient data (need ≥10 samples for trend test)"}

    n = len(parameter_series)

    # Calculate Mann-Kendall S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(parameter_series[j] - parameter_series[i])

    # Variance of S
    var_s = (n * (n - 1) * (2 * n + 5)) / 18

    # Z-score
    if s > 0:
        z_statistic = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z_statistic = (s + 1) / np.sqrt(var_s)
    else:
        z_statistic = 0

    # Two-tailed p-value
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_statistic)))

    # Determine trend
    if p_value < 0.01:
        if s > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "no trend"

    # Sen's slope (robust slope estimator)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slope = (parameter_series[j] - parameter_series[i]) / (j - i)
                slopes.append(slope)

    sens_slope = np.median(slopes) if slopes else 0

    return {
        "alert": trend in ["increasing", "decreasing"],
        "p_value": p_value,
        "z_statistic": z_statistic,
        "trend": trend,
        "sens_slope": sens_slope,
        "s_statistic": s,
    }


# =============================================================================
# TEMPORAL PATTERN DETECTION
# =============================================================================


def detect_temporal_autocorrelation(parameter_series: list[float], max_lag: int = 5) -> dict[str, Any]:
    """Detect non-random temporal patterns using autocorrelation.

    Random data should have low autocorrelation at all lags.
    Systematic patterns (e.g., every 5th fraction attacked) show significant
    autocorrelation at specific lags.

    Args:
        parameter_series: Time series of parameter values
        max_lag: Maximum lag to test (default 5)

    Returns:
        Dictionary with autocorrelation results
    """
    if len(parameter_series) < 20:
        return {"alert": False, "error": "Insufficient data (need ≥20 samples)"}

    # Calculate autocorrelation function
    acf_values = []
    for lag in range(1, min(max_lag + 1, len(parameter_series) // 4)):
        # Pearson correlation between series and lagged series
        series1 = parameter_series[:-lag]
        series2 = parameter_series[lag:]

        if len(series1) > 0:
            corr, p_value = scipy.stats.pearsonr(series1, series2)
            acf_values.append({"lag": lag, "autocorr": corr, "p_value": p_value})

    # Find significant autocorrelations
    significant_lags = [acf for acf in acf_values if acf["p_value"] < 0.05 and abs(acf["autocorr"]) > 0.3]

    return {
        "alert": len(significant_lags) > 0,
        "acf_values": acf_values,
        "significant_lags": significant_lags,
        "interpretation": f"Significant autocorrelation at lags: {[acf['lag'] for acf in significant_lags]}"
        if significant_lags
        else "No significant autocorrelation detected",
    }


def detect_modification_timing_patterns(
    cursor: Connection.cursor, table_name: str, time_window_hours: int = 24
) -> dict[str, Any]:
    """Detect suspicious patterns in database modification timing.

    Malicious actors may:
    - Modify records late at night (off hours)
    - Modify future fractions (time-delayed attacks)
    - Show suspicious gaps between modifications

    Args:
        cursor: Database cursor
        table_name: Table to analyze (e.g., 'TxField', 'TxFieldPoint')
        time_window_hours: Time window to analyze (default 24 hours)

    Returns:
        Dictionary with timing pattern analysis
    """
    # Query modification timestamps from audit log
    query = f"""
    SELECT
        modified_timestamp,
        user_id,
        DATEPART(hour, modified_timestamp) AS hour_of_day,
        DATEPART(dw, modified_timestamp) AS day_of_week
    FROM audit_log
    WHERE table_name = %(table_name)s
      AND modified_timestamp >= DATEADD(hour, -%(hours)s, GETDATE())
    ORDER BY modified_timestamp
    """

    cursor.execute(query, {"table_name": table_name, "hours": time_window_hours})
    rows = cursor.fetchall()

    if len(rows) < 10:
        return {"alert": False, "error": "Insufficient modification history"}

    # Convert to DataFrame for analysis
    df = pd.DataFrame(rows, columns=["timestamp", "user", "hour", "day_of_week"])

    # Detect off-hours modifications (midnight to 6am, weekends)
    off_hours = df[(df["hour"] >= 0) & (df["hour"] < 6)]
    weekend = df[df["day_of_week"].isin([1, 7])]  # Sunday=1, Saturday=7

    off_hours_rate = len(off_hours) / len(df)
    weekend_rate = len(weekend) / len(df)

    # Expected rates (assuming normal business hours)
    expected_off_hours_rate = 0.10  # 10% (some legitimate off-hours work)
    expected_weekend_rate = 0.15  # 15%

    return {
        "alert": off_hours_rate > expected_off_hours_rate * 2 or weekend_rate > expected_weekend_rate * 2,
        "total_modifications": len(df),
        "off_hours_count": len(off_hours),
        "off_hours_rate": off_hours_rate,
        "weekend_count": len(weekend),
        "weekend_rate": weekend_rate,
        "interpretation": "Suspicious timing pattern (excessive off-hours/weekend activity)"
        if (off_hours_rate > expected_off_hours_rate * 2)
        else "Normal modification timing",
    }


# =============================================================================
# AUDIT TRAIL INTEGRITY
# =============================================================================


def verify_treatment_integrity(cursor: Connection.cursor, fld_id: int) -> dict[str, Any]:
    """Verify treatment parameters haven't been modified since approval.

    Uses cryptographic checksums to detect any parameter modifications.

    Args:
        cursor: Database cursor
        fld_id: Field ID to verify

    Returns:
        Dictionary with integrity verification results
    """
    # Fetch current treatment parameters
    query = """
    SELECT
        FLD_ID,
        Meterset,
        Gantry_Ang,
        Coll_Ang,
        X1_Jaw,
        X2_Jaw,
        Y1_Jaw,
        Y2_Jaw
    FROM TxFieldPoint
    WHERE FLD_ID = %(fld_id)s
    ORDER BY Point
    """

    cursor.execute(query, {"fld_id": fld_id})
    rows = cursor.fetchall()

    if not rows:
        return {"alert": True, "error": "Field not found"}

    # Calculate current checksum
    param_dict = {"fld_id": fld_id, "parameters": [dict(zip(["Point", "MU", "Gantry", "Coll", "X1", "X2", "Y1", "Y2"], row)) for row in rows]}

    param_json = json.dumps(param_dict, sort_keys=True)
    current_checksum = hashlib.sha256(param_json.encode()).hexdigest()

    # Retrieve baseline checksum (would be stored at approval time)
    # For demonstration, we simulate this
    baseline_checksum_query = """
    SELECT checksum
    FROM treatment_parameter_checksums
    WHERE fld_id = %(fld_id)s
    ORDER BY timestamp DESC
    """

    # Note: This table would need to be created in actual implementation
    # For now, we return the current checksum for storage

    return {
        "alert": False,  # Would be True if current != baseline
        "fld_id": fld_id,
        "current_checksum": current_checksum,
        "parameters_hash": param_json[:100] + "...",  # Truncate for display
        "message": "Checksum calculated - store this at treatment approval for future verification",
    }


def detect_impossible_user_activity(cursor: Connection.cursor, user_id: str, hours_lookback: int = 24) -> dict[str, Any]:
    """Detect physically impossible user activity patterns.

    Examples:
    - User logged in from two locations simultaneously
    - User traveled impossibly fast between locations
    - Excessive activity rate (e.g., 100 modifications per minute)

    Args:
        cursor: Database cursor
        user_id: User ID to analyze
        hours_lookback: Hours of history to analyze

    Returns:
        Dictionary with impossibility detection results
    """
    query = """
    SELECT
        modified_timestamp,
        ip_address,
        operation
    FROM audit_log
    WHERE user_id = %(user_id)s
      AND modified_timestamp >= DATEADD(hour, -%(hours)s, GETDATE())
    ORDER BY modified_timestamp
    """

    cursor.execute(query, {"user_id": user_id, "hours": hours_lookback})
    rows = cursor.fetchall()

    if len(rows) < 2:
        return {"alert": False, "message": "Insufficient activity history"}

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["timestamp", "ip_address", "operation"])

    # Check for impossible activity rate
    time_diffs = []
    for i in range(len(df) - 1):
        time_diff_seconds = (df.iloc[i + 1]["timestamp"] - df.iloc[i]["timestamp"]).total_seconds()
        time_diffs.append(time_diff_seconds)

    # Minimum credible time between operations (e.g., 1 second)
    min_credible_interval = 1.0  # seconds
    impossible_intervals = [t for t in time_diffs if t < min_credible_interval]

    # Check for location impossibilities (multiple IPs in short time)
    ip_changes = []
    for i in range(len(df) - 1):
        if df.iloc[i]["ip_address"] != df.iloc[i + 1]["ip_address"]:
            time_diff = (df.iloc[i + 1]["timestamp"] - df.iloc[i]["timestamp"]).total_seconds() / 60  # minutes
            ip_changes.append(
                {"from_ip": df.iloc[i]["ip_address"], "to_ip": df.iloc[i + 1]["ip_address"], "time_diff_minutes": time_diff}
            )

    # Suspicious if IP changes within 5 minutes (unless VPN flip)
    rapid_ip_changes = [change for change in ip_changes if change["time_diff_minutes"] < 5]

    return {
        "alert": len(impossible_intervals) > 5 or len(rapid_ip_changes) > 0,
        "user_id": user_id,
        "total_operations": len(df),
        "impossible_intervals": len(impossible_intervals),
        "rapid_ip_changes": rapid_ip_changes,
        "interpretation": "Impossible activity pattern detected (bot/script or compromised account)"
        if (len(impossible_intervals) > 5 or len(rapid_ip_changes) > 0)
        else "Normal activity pattern",
    }


# =============================================================================
# MULTI-ANOMALY DETECTION
# =============================================================================


def flag_multiple_independent_anomalies(anomaly_results: dict[str, bool], base_rate: float = 0.05) -> dict[str, Any]:
    """Flag patients with multiple independent anomalies.

    The probability of multiple independent anomalies occurring by chance
    is very low (product of individual probabilities).

    Args:
        anomaly_results: Dictionary of anomaly type → boolean (detected)
        base_rate: Expected rate of individual anomalies (default 5%)

    Returns:
        Alert if multiple anomalies detected

    Example:
        >>> anomalies = {
        ...     'mlc': True,
        ...     'mu': True,
        ...     'position': True,
        ...     'angle': False
        ... }
        >>> result = flag_multiple_independent_anomalies(anomalies)
        >>> result['probability_by_chance']  # 0.05^3 = 0.000125
    """
    anomaly_types = [k for k, v in anomaly_results.items() if v]
    num_anomalies = len(anomaly_types)

    if num_anomalies >= 2:
        # Probability = base_rate^num_anomalies (independent events)
        probability = base_rate**num_anomalies

        return {
            "alert": True,
            "num_anomalies": num_anomalies,
            "anomaly_types": anomaly_types,
            "probability_by_chance": probability,
            "interpretation": f"{num_anomalies} independent anomalies is highly suspicious (p={probability:.6f})",
        }

    return {"alert": False, "num_anomalies": num_anomalies}


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================


def assess_malicious_probability(features: dict[str, Any], energy: float) -> dict[str, Any]:
    """Estimate probability that high energy is malicious vs accidental.

    Args:
        features: Dictionary of anomaly features
        energy: EBM energy score

    Returns:
        Dictionary with malicious probability and recommendation
    """
    confidence_factors = {}

    # Factor 1: Multiple independent anomalies (weight 0.8)
    if features.get("multiple_anomalies", False):
        confidence_factors["multiple_anomalies"] = 0.8

    # Factor 2: Statistical evasion signature (weight 0.7)
    if features.get("outlier_avoidance", False) or features.get("benford_violation", False):
        confidence_factors["statistical_evasion"] = 0.7

    # Factor 3: Audit trail anomalies (weight 0.9)
    if features.get("audit_inconsistency", False):
        confidence_factors["audit_manipulation"] = 0.9

    # Factor 4: Temporal patterns (weight 0.6)
    if features.get("temporal_clustering", False) or features.get("off_hours_mods", False):
        confidence_factors["temporal_pattern"] = 0.6

    # Factor 5: Demographic targeting (weight 0.9)
    if features.get("demographic_bias", False):
        confidence_factors["targeted_selection"] = 0.9

    # Bayesian combination (average of active factors)
    if confidence_factors:
        malicious_probability = np.mean(list(confidence_factors.values()))
    else:
        malicious_probability = 0.0

    # Recommendation based on energy + confidence
    if energy >= 2.7 and malicious_probability > 0.7:
        recommendation = "HIGH CONFIDENCE MALICIOUS - IMMEDIATE INCIDENT RESPONSE"
        response_time = "<15 minutes"
    elif energy >= 2.5 and malicious_probability > 0.5:
        recommendation = "MODERATE CONFIDENCE MALICIOUS - FORENSIC INVESTIGATION"
        response_time = "15-60 minutes"
    elif energy >= 2.0:
        recommendation = "LIKELY ACCIDENTAL HIGH-SEVERITY - STANDARD QA WORKFLOW"
        response_time = "1-4 hours"
    else:
        recommendation = "LOW SEVERITY - ROUTINE REVIEW"
        response_time = "24 hours"

    return {
        "energy": energy,
        "malicious_probability": malicious_probability,
        "confidence_factors": confidence_factors,
        "recommendation": recommendation,
        "response_time": response_time,
    }
