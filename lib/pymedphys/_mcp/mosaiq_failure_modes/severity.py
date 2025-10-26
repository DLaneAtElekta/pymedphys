"""Severity scoring utilities for failure modes.

This module provides functions to retrieve severity scores for different
failure modes and variants, enabling severity-weighted EBM training.
"""

from typing import Dict

from .server import FAILURE_MODES


def get_severity(failure_mode: str, variant: str | None = None) -> float:
    """Get severity score for a failure mode and variant.

    Args:
        failure_mode: Name of the failure mode (e.g., "corrupt_mlc_data")
        variant: Specific variant (e.g., "out_of_range"), or None for base severity

    Returns:
        Severity score (0=normal, 0.5-3.0=anomaly severity)

    Raises:
        ValueError: If failure mode or variant doesn't exist
    """
    if failure_mode not in FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {failure_mode}")

    severity_info = FAILURE_MODES[failure_mode]["severity"]

    if variant is None:
        return severity_info["base"]

    if variant not in severity_info["variants"]:
        # Fallback to base severity if variant not found
        return severity_info["base"]

    return severity_info["variants"][variant]


def get_all_severities() -> Dict[str, Dict[str, float]]:
    """Get all severity scores organized by failure mode.

    Returns:
        Dictionary mapping failure_mode -> variant -> severity
    """
    severities = {}

    for failure_mode, info in FAILURE_MODES.items():
        severities[failure_mode] = {
            "base": info["severity"]["base"],
            "variants": info["severity"]["variants"].copy(),
        }

    return severities


def categorize_severity(severity: float) -> str:
    """Categorize a severity score into human-readable category.

    Args:
        severity: Numerical severity score

    Returns:
        Category name: "normal", "low", "medium", "high", or "critical"
    """
    if severity < 0.4:
        return "normal"
    elif severity < 1.0:
        return "low"
    elif severity < 1.8:
        return "medium"
    elif severity < 2.5:
        return "high"
    else:
        return "critical"


def get_severity_description(severity: float) -> str:
    """Get human-readable description of severity level.

    Args:
        severity: Numerical severity score

    Returns:
        Description of what this severity level means
    """
    category = categorize_severity(severity)

    descriptions = {
        "normal": "Normal data with no detected issues",
        "low": "Low severity - data quality issues, non-critical (e.g., parsing errors)",
        "medium": "Medium severity - data integrity or workflow issues (e.g., timestamp errors)",
        "high": "High severity - dose delivery errors or treatment integrity issues",
        "critical": "Critical severity - patient safety risk (wrong patient/position/dose)",
    }

    return descriptions[category]


def get_severity_thresholds() -> Dict[str, tuple[float, float]]:
    """Get severity threshold ranges for each category.

    Returns:
        Dictionary mapping category -> (min_severity, max_severity)
    """
    return {
        "normal": (0.0, 0.4),
        "low": (0.4, 1.0),
        "medium": (1.0, 1.8),
        "high": (1.8, 2.5),
        "critical": (2.5, 3.0),
    }


# Severity scale documentation
SEVERITY_SCALE = """
Mosaiq Failure Mode Severity Scale
===================================

0.0 - Normal Data
-----------------
Clean data with no detected issues. Target energy for normal records.

0.5-0.8 - Low Severity
----------------------
Data quality issues, non-critical. Examples:
- Invalid angles (likely data entry error, delivery has safety constraints)
- Corrupt MLC data with odd bytes (parsing issue, may not affect treatment)
- Invalid offset type/state (data quality, not patient safety)

Impact: Minimal clinical risk, primarily data quality concerns

1.0-1.5 - Medium Severity
-------------------------
Data integrity and workflow issues. Examples:
- Duplicate treatment entries (billing/record issues)
- Timestamp inconsistencies (record keeping issues)
- NULL non-critical fields (data corruption, queries may fail)

Impact: Moderate administrative/workflow impact, low clinical risk

1.8-2.3 - High Severity
-----------------------
Dose delivery errors and treatment integrity issues. Examples:
- Missing control points (incomplete dose delivery data)
- MLC data out of range (dose calculation errors)
- MLC leaf count mismatch (dose calculation errors)
- Negative MLC gaps (physical impossibility, delivery fails)

Impact: High clinical risk, potential for incorrect dose delivery

2.5-3.0 - Critical Severity
---------------------------
Patient safety risks: wrong patient, position, or dose. Examples:
- Orphaned patient records (treatment for wrong patient)
- Extreme offset values (wrong treatment position, geometric miss)
- Negative or extreme meterset (wrong dose, safety interlock)
- NULL patient ID (patient identification lost)

Impact: Critical patient safety risk, immediate intervention required

Usage in EBM Training
=====================
The EBM is trained to output energy values matching these severity scores.
This allows the model to:
1. Detect anomalies (energy > 0.4)
2. Assess severity (continuous 0-3 scale)
3. Prioritize alerts (critical > high > medium > low)
4. Reduce alert fatigue (low-severity issues may only log, not alert)
"""


def print_severity_scale():
    """Print the severity scale documentation."""
    print(SEVERITY_SCALE)


if __name__ == "__main__":
    # Print severity scale and examples when run as script
    print_severity_scale()

    print("\n" + "=" * 80)
    print("Example Severity Scores by Failure Mode")
    print("=" * 80 + "\n")

    for failure_mode, info in FAILURE_MODES.items():
        print(f"{failure_mode}:")
        print(f"  Base severity: {info['severity']['base']} ({categorize_severity(info['severity']['base'])})")

        if info['severity']['variants']:
            print("  Variants:")
            for variant, score in info['severity']['variants'].items():
                category = categorize_severity(score)
                print(f"    - {variant}: {score} ({category})")

        print()
