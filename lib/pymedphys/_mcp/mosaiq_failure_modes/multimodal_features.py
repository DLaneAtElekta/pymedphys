"""Multi-modal QA Feature Extraction for EBM Training.

This module integrates features from multiple QA data sources:
1. Mosaiq database (treatment records)
2. TRF files (machine log files from Elekta linacs)
3. Portal dosimetry (EPID-based measurements)
4. Phantom dosimetry (physical QA measurements)

Multi-modal QA provides independent verification and detects failures
that may not be apparent in any single data source.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logger.warning("pydicom not available. DICOM portal dosimetry features will not work.")


# =============================================================================
# TRF (Treatment Record File) Analysis Features
# =============================================================================

def extract_trf_features(trf_path: Path, mosaiq_delivery: dict) -> dict[str, float]:
    """Extract features from Elekta TRF file for comparison with Mosaiq.

    TRF files contain machine log data recorded during treatment delivery.
    They provide ground truth for:
    - Actual MLC positions delivered
    - Actual gantry/collimator angles
    - Actual MU delivered
    - Beam hold events
    - Interlock triggers

    Args:
        trf_path: Path to TRF file
        mosaiq_delivery: Mosaiq delivery data for comparison

    Returns:
        Dictionary of TRF-derived features including cross-validation metrics
    """
    from pymedphys._trf import delivery as trf_delivery

    features = {}

    try:
        # Read TRF file
        trf_data = trf_delivery(trf_path)

        # Feature 1: MU delivered vs planned
        trf_mu = trf_data.mu[-1]  # Final cumulative MU
        mosaiq_mu = mosaiq_delivery.get('meterset', 0)

        features['trf_mu_delivered'] = trf_mu
        features['trf_mu_mosaiq_diff'] = abs(trf_mu - mosaiq_mu)
        features['trf_mu_mosaiq_percent_diff'] = (
            abs(trf_mu - mosaiq_mu) / mosaiq_mu * 100 if mosaiq_mu > 0 else 999.0
        )
        features['trf_mu_tolerance_exceeded'] = float(
            features['trf_mu_mosaiq_percent_diff'] > 1.0  # >1% difference
        )

        # Feature 2: MLC position comparison
        # Compare TRF recorded MLC positions with Mosaiq planned positions
        if hasattr(trf_data, 'mlc') and mosaiq_delivery.get('mlc_a') is not None:
            trf_mlc_a = np.array(trf_data.mlc.leaf_pair_widths)  # Example - actual structure varies
            mosaiq_mlc_a = np.array(mosaiq_delivery['mlc_a'])

            if len(trf_mlc_a) == len(mosaiq_mlc_a):
                mlc_diff = np.abs(trf_mlc_a - mosaiq_mlc_a)
                features['trf_mlc_max_deviation'] = float(np.max(mlc_diff))
                features['trf_mlc_mean_deviation'] = float(np.mean(mlc_diff))
                features['trf_mlc_rms_deviation'] = float(np.sqrt(np.mean(mlc_diff**2)))
                features['trf_mlc_tolerance_exceeded'] = float(np.max(mlc_diff) > 0.2)  # >2mm
            else:
                features['trf_mlc_max_deviation'] = -999.0  # Mismatch indicator
                features['trf_mlc_mean_deviation'] = -999.0
                features['trf_mlc_rms_deviation'] = -999.0
                features['trf_mlc_tolerance_exceeded'] = 1.0

        # Feature 3: Gantry angle comparison
        if hasattr(trf_data, 'gantry'):
            trf_gantry = np.array(trf_data.gantry)
            mosaiq_gantry = np.array(mosaiq_delivery.get('gantry_angles', []))

            if len(trf_gantry) > 0 and len(mosaiq_gantry) > 0:
                # Compare at corresponding control points
                min_len = min(len(trf_gantry), len(mosaiq_gantry))
                gantry_diff = np.abs(trf_gantry[:min_len] - mosaiq_gantry[:min_len])
                features['trf_gantry_max_deviation'] = float(np.max(gantry_diff))
                features['trf_gantry_mean_deviation'] = float(np.mean(gantry_diff))
                features['trf_gantry_tolerance_exceeded'] = float(np.max(gantry_diff) > 1.0)  # >1°
            else:
                features['trf_gantry_max_deviation'] = 0.0
                features['trf_gantry_mean_deviation'] = 0.0
                features['trf_gantry_tolerance_exceeded'] = 0.0

        # Feature 4: Beam interruptions
        if hasattr(trf_data, 'monitor_units'):
            # Detect beam holds (flat regions in cumulative MU)
            mu_array = np.array(trf_data.monitor_units)
            mu_diff = np.diff(mu_array)
            beam_hold_count = np.sum(mu_diff < 0.001)  # Essentially no MU change
            features['trf_beam_hold_count'] = float(beam_hold_count)
            features['trf_beam_hold_detected'] = float(beam_hold_count > 0)
        else:
            features['trf_beam_hold_count'] = 0.0
            features['trf_beam_hold_detected'] = 0.0

        # Feature 5: Treatment duration
        if hasattr(trf_data, 'time'):
            trf_duration = trf_data.time[-1] - trf_data.time[0]
            features['trf_duration_seconds'] = trf_duration

            # Compare with Mosaiq
            mosaiq_duration = mosaiq_delivery.get('duration_seconds', 0)
            features['trf_duration_mosaiq_diff'] = abs(trf_duration - mosaiq_duration)
            features['trf_duration_tolerance_exceeded'] = float(
                abs(trf_duration - mosaiq_duration) > 30  # >30s difference
            )
        else:
            features['trf_duration_seconds'] = 0.0
            features['trf_duration_mosaiq_diff'] = 0.0
            features['trf_duration_tolerance_exceeded'] = 0.0

        # Feature 6: File integrity
        features['trf_file_readable'] = 1.0
        features['trf_data_complete'] = float(
            hasattr(trf_data, 'mu') and
            hasattr(trf_data, 'mlc') and
            hasattr(trf_data, 'gantry')
        )

    except Exception as e:
        logger.warning(f"Failed to extract TRF features from {trf_path}: {e}")
        # Fill with error indicators
        features = {
            'trf_file_readable': 0.0,
            'trf_data_complete': 0.0,
            'trf_mu_delivered': -999.0,
            'trf_mu_mosaiq_diff': 999.0,
            'trf_mu_mosaiq_percent_diff': 999.0,
            'trf_mu_tolerance_exceeded': 1.0,
            'trf_mlc_max_deviation': -999.0,
            'trf_mlc_mean_deviation': -999.0,
            'trf_mlc_rms_deviation': -999.0,
            'trf_mlc_tolerance_exceeded': 1.0,
            'trf_gantry_max_deviation': 0.0,
            'trf_gantry_mean_deviation': 0.0,
            'trf_gantry_tolerance_exceeded': 0.0,
            'trf_beam_hold_count': 0.0,
            'trf_beam_hold_detected': 0.0,
            'trf_duration_seconds': 0.0,
            'trf_duration_mosaiq_diff': 0.0,
            'trf_duration_tolerance_exceeded': 0.0,
        }

    return features


# =============================================================================
# Portal Dosimetry Features (EPID-based)
# =============================================================================

def extract_portal_dosimetry_features(
    portal_image_path: Path,
    reference_image_path: Path | None = None,
    gamma_criteria: tuple[float, float] = (3.0, 3.0),  # 3%/3mm
) -> dict[str, float]:
    """Extract features from portal dosimetry (EPID) images.

    Portal dosimetry provides independent dose verification by:
    - Comparing measured vs predicted portal dose
    - Gamma analysis
    - Detecting systematic delivery errors

    Args:
        portal_image_path: Path to measured EPID DICOM image
        reference_image_path: Path to predicted portal dose (optional)
        gamma_criteria: (dose_percent, distance_mm) for gamma analysis

    Returns:
        Dictionary of portal dosimetry features
    """
    if not PYDICOM_AVAILABLE:
        return _portal_features_unavailable()

    features = {}

    try:
        # Read portal image
        portal_ds = pydicom.dcmread(portal_image_path)

        # Feature 1: Image quality metrics
        portal_dose = portal_ds.pixel_array * portal_ds.DoseGridScaling
        features['portal_mean_dose'] = float(np.mean(portal_dose))
        features['portal_max_dose'] = float(np.max(portal_dose))
        features['portal_dose_range'] = float(np.max(portal_dose) - np.min(portal_dose))

        # Feature 2: Image uniformity (for open fields)
        # Central 10x10cm region
        center_y, center_x = portal_dose.shape[0] // 2, portal_dose.shape[1] // 2
        roi_size = 20  # pixels (assuming ~5mm pixel size)
        central_roi = portal_dose[
            center_y - roi_size : center_y + roi_size,
            center_x - roi_size : center_x + roi_size,
        ]
        features['portal_central_dose'] = float(np.mean(central_roi))
        features['portal_uniformity_std'] = float(np.std(central_roi))
        features['portal_uniformity_cov'] = float(
            np.std(central_roi) / np.mean(central_roi) * 100 if np.mean(central_roi) > 0 else 999.0
        )

        # Feature 3: Gamma analysis (if reference available)
        if reference_image_path and reference_image_path.exists():
            reference_ds = pydicom.dcmread(reference_image_path)
            reference_dose = reference_ds.pixel_array * reference_ds.DoseGridScaling

            # Perform gamma analysis (using pymedphys gamma function)
            from pymedphys._gamma import gamma_shell

            gamma_options = {
                'dose_percent_threshold': gamma_criteria[0],
                'distance_mm_threshold': gamma_criteria[1],
                'lower_percent_dose_cutoff': 20,
                'interp_fraction': 10,
                'max_gamma': 2.0,
            }

            try:
                gamma_result = gamma_shell(
                    reference_dose,
                    portal_dose,
                    **gamma_options
                )

                features['portal_gamma_pass_rate'] = float(
                    np.sum(gamma_result <= 1.0) / gamma_result.size * 100
                )
                features['portal_gamma_mean'] = float(np.mean(gamma_result[~np.isnan(gamma_result)]))
                features['portal_gamma_max'] = float(np.max(gamma_result[~np.isnan(gamma_result)]))
                features['portal_gamma_tolerance_exceeded'] = float(
                    features['portal_gamma_pass_rate'] < 95.0  # <95% pass rate
                )
            except Exception as e:
                logger.warning(f"Gamma analysis failed: {e}")
                features['portal_gamma_pass_rate'] = -999.0
                features['portal_gamma_mean'] = -999.0
                features['portal_gamma_max'] = -999.0
                features['portal_gamma_tolerance_exceeded'] = 1.0
        else:
            # No reference - can't do gamma
            features['portal_gamma_pass_rate'] = 0.0
            features['portal_gamma_mean'] = 0.0
            features['portal_gamma_max'] = 0.0
            features['portal_gamma_tolerance_exceeded'] = 0.0

        # Feature 4: Dose agreement with Mosaiq planned dose
        # This would require planned portal dose from TPS
        features['portal_file_readable'] = 1.0

    except Exception as e:
        logger.warning(f"Failed to extract portal features from {portal_image_path}: {e}")
        features = _portal_features_unavailable()
        features['portal_file_readable'] = 0.0

    return features


def _portal_features_unavailable() -> dict[str, float]:
    """Return default portal features when unavailable."""
    return {
        'portal_file_readable': 0.0,
        'portal_mean_dose': 0.0,
        'portal_max_dose': 0.0,
        'portal_dose_range': 0.0,
        'portal_central_dose': 0.0,
        'portal_uniformity_std': 0.0,
        'portal_uniformity_cov': 0.0,
        'portal_gamma_pass_rate': 0.0,
        'portal_gamma_mean': 0.0,
        'portal_gamma_max': 0.0,
        'portal_gamma_tolerance_exceeded': 0.0,
    }


# =============================================================================
# Phantom Dosimetry Features
# =============================================================================

def extract_phantom_dosimetry_features(
    phantom_measurements: dict[str, float],
    expected_values: dict[str, float],
) -> dict[str, float]:
    """Extract features from phantom dosimetry QA.

    Phantom dosimetry provides end-to-end system verification:
    - Output factors
    - Dose linearity
    - Small field dosimetry
    - IMRT/VMAT QA (e.g., ArcCHECK, Delta4)

    Args:
        phantom_measurements: Measured values from phantom
            Example: {'center_dose': 100.2, 'off_axis_ratio': 1.001, ...}
        expected_values: Expected values from TPS or baseline
            Example: {'center_dose': 100.0, 'off_axis_ratio': 1.000, ...}

    Returns:
        Dictionary of phantom dosimetry features
    """
    features = {}

    # Feature 1: Absolute dose agreement
    if 'center_dose' in phantom_measurements and 'center_dose' in expected_values:
        measured = phantom_measurements['center_dose']
        expected = expected_values['center_dose']

        features['phantom_dose_measured'] = measured
        features['phantom_dose_expected'] = expected
        features['phantom_dose_diff_cGy'] = measured - expected
        features['phantom_dose_diff_percent'] = (
            (measured - expected) / expected * 100 if expected > 0 else 999.0
        )
        features['phantom_dose_tolerance_exceeded'] = float(
            abs(features['phantom_dose_diff_percent']) > 3.0  # >3% difference
        )
    else:
        features['phantom_dose_measured'] = 0.0
        features['phantom_dose_expected'] = 0.0
        features['phantom_dose_diff_cGy'] = 0.0
        features['phantom_dose_diff_percent'] = 0.0
        features['phantom_dose_tolerance_exceeded'] = 0.0

    # Feature 2: IMRT/VMAT QA pass rate (if available)
    if 'gamma_pass_rate' in phantom_measurements:
        pass_rate = phantom_measurements['gamma_pass_rate']
        features['phantom_gamma_pass_rate'] = pass_rate
        features['phantom_gamma_tolerance_exceeded'] = float(pass_rate < 95.0)

        if 'gamma_mean' in phantom_measurements:
            features['phantom_gamma_mean'] = phantom_measurements['gamma_mean']
        else:
            features['phantom_gamma_mean'] = 0.0
    else:
        features['phantom_gamma_pass_rate'] = 0.0
        features['phantom_gamma_tolerance_exceeded'] = 0.0
        features['phantom_gamma_mean'] = 0.0

    # Feature 3: Output factor agreement
    if 'output_factor' in phantom_measurements and 'output_factor' in expected_values:
        measured_of = phantom_measurements['output_factor']
        expected_of = expected_values['output_factor']

        features['phantom_output_factor_measured'] = measured_of
        features['phantom_output_factor_expected'] = expected_of
        features['phantom_output_factor_diff_percent'] = (
            (measured_of - expected_of) / expected_of * 100 if expected_of > 0 else 999.0
        )
        features['phantom_output_tolerance_exceeded'] = float(
            abs(features['phantom_output_factor_diff_percent']) > 2.0  # >2%
        )
    else:
        features['phantom_output_factor_measured'] = 0.0
        features['phantom_output_factor_expected'] = 0.0
        features['phantom_output_factor_diff_percent'] = 0.0
        features['phantom_output_tolerance_exceeded'] = 0.0

    # Feature 4: Measurement completeness
    features['phantom_measurements_available'] = float(len(phantom_measurements) > 0)
    features['phantom_measurement_count'] = float(len(phantom_measurements))

    return features


# =============================================================================
# Multi-Modal Cross-Validation Features
# =============================================================================

def extract_cross_validation_features(
    mosaiq_features: dict[str, float],
    trf_features: dict[str, float],
    portal_features: dict[str, float],
    phantom_features: dict[str, float],
) -> dict[str, float]:
    """Extract cross-validation features across QA modalities.

    Independent QA strands provide verification:
    - If Mosaiq and TRF disagree → data corruption
    - If portal dosimetry fails but Mosaiq/TRF agree → delivery issue
    - If all modalities disagree → systematic problem

    Args:
        mosaiq_features: Features from Mosaiq database
        trf_features: Features from TRF analysis
        portal_features: Features from portal dosimetry
        phantom_features: Features from phantom dosimetry

    Returns:
        Dictionary of cross-validation features
    """
    features = {}

    # Feature 1: Data source availability
    features['cv_mosaiq_available'] = float(mosaiq_features.get('patient_exists', 0.0) > 0)
    features['cv_trf_available'] = float(trf_features.get('trf_file_readable', 0.0) > 0)
    features['cv_portal_available'] = float(portal_features.get('portal_file_readable', 0.0) > 0)
    features['cv_phantom_available'] = float(phantom_features.get('phantom_measurements_available', 0.0) > 0)

    features['cv_data_sources_count'] = (
        features['cv_mosaiq_available'] +
        features['cv_trf_available'] +
        features['cv_portal_available'] +
        features['cv_phantom_available']
    )

    # Feature 2: Cross-validation consistency flags
    # MU agreement across modalities
    features['cv_mu_consistent'] = 1.0  # Default: assume consistent

    if features['cv_mosaiq_available'] and features['cv_trf_available']:
        trf_mu_diff = trf_features.get('trf_mu_mosaiq_percent_diff', 0.0)
        if trf_mu_diff > 1.0:  # >1% disagreement
            features['cv_mu_consistent'] = 0.0

    # Feature 3: Gamma analysis consistency
    # Portal and phantom gamma should both pass if delivery correct
    features['cv_gamma_consistent'] = 1.0

    portal_gamma_exceeded = portal_features.get('portal_gamma_tolerance_exceeded', 0.0)
    phantom_gamma_exceeded = phantom_features.get('phantom_gamma_tolerance_exceeded', 0.0)

    if portal_gamma_exceeded > 0 or phantom_gamma_exceeded > 0:
        features['cv_gamma_consistent'] = 0.0

    # Feature 4: Red flag: All modalities show issues
    # Very high severity if multiple independent QA strands fail
    features['cv_multi_modal_failure'] = float(
        (trf_features.get('trf_mu_tolerance_exceeded', 0.0) > 0) +
        (portal_features.get('portal_gamma_tolerance_exceeded', 0.0) > 0) +
        (phantom_features.get('phantom_gamma_tolerance_exceeded', 0.0) > 0)
        >= 2  # At least 2 modalities failing
    )

    # Feature 5: Confidence score
    # Higher when multiple modalities available and agree
    if features['cv_data_sources_count'] >= 3:
        if features['cv_mu_consistent'] and features['cv_gamma_consistent']:
            features['cv_confidence_score'] = 1.0  # High confidence
        else:
            features['cv_confidence_score'] = 0.3  # Disagreement
    elif features['cv_data_sources_count'] == 2:
        features['cv_confidence_score'] = 0.7  # Medium confidence
    elif features['cv_data_sources_count'] == 1:
        features['cv_confidence_score'] = 0.4  # Low confidence
    else:
        features['cv_confidence_score'] = 0.0  # No data

    return features


# =============================================================================
# Combined Multi-Modal Feature Extraction
# =============================================================================

def extract_all_multimodal_features(
    ttx_id: int,
    cursor,  # Mosaiq database cursor
    trf_path: Path | None = None,
    portal_image_path: Path | None = None,
    portal_reference_path: Path | None = None,
    phantom_measurements: dict[str, float] | None = None,
    phantom_expected: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract all features from multiple QA modalities.

    This is the main entry point for multi-modal feature extraction.

    Args:
        ttx_id: TrackTreatment ID
        cursor: Mosaiq database cursor
        trf_path: Path to TRF file (optional)
        portal_image_path: Path to portal image (optional)
        portal_reference_path: Path to reference portal dose (optional)
        phantom_measurements: Phantom measurements dict (optional)
        phantom_expected: Expected phantom values dict (optional)

    Returns:
        Dictionary of all multi-modal features
    """
    from .ebm_features import extract_all_features as extract_mosaiq_features

    # 1. Extract Mosaiq features
    mosaiq_features = extract_mosaiq_features(cursor, ttx_id)

    # 2. Extract TRF features (if available)
    if trf_path and trf_path.exists():
        # Get Mosaiq delivery data for comparison
        mosaiq_delivery = {
            'meterset': mosaiq_features.get('mu_planned', 0.0),
            'mlc_a': None,  # Would extract from Mosaiq
            'gantry_angles': None,  # Would extract from Mosaiq
            'duration_seconds': mosaiq_features.get('duration_seconds', 0.0),
        }
        trf_features = extract_trf_features(trf_path, mosaiq_delivery)
    else:
        trf_features = {f'trf_{k}': 0.0 for k in [
            'file_readable', 'data_complete', 'mu_delivered', 'mu_mosaiq_diff',
            'mu_mosaiq_percent_diff', 'mu_tolerance_exceeded', 'mlc_max_deviation',
            'mlc_mean_deviation', 'mlc_rms_deviation', 'mlc_tolerance_exceeded',
            'gantry_max_deviation', 'gantry_mean_deviation', 'gantry_tolerance_exceeded',
            'beam_hold_count', 'beam_hold_detected', 'duration_seconds',
            'duration_mosaiq_diff', 'duration_tolerance_exceeded',
        ]}

    # 3. Extract portal dosimetry features (if available)
    if portal_image_path and portal_image_path.exists():
        portal_features = extract_portal_dosimetry_features(
            portal_image_path, portal_reference_path
        )
    else:
        portal_features = _portal_features_unavailable()

    # 4. Extract phantom dosimetry features (if available)
    if phantom_measurements:
        phantom_features = extract_phantom_dosimetry_features(
            phantom_measurements, phantom_expected or {}
        )
    else:
        phantom_features = {f'phantom_{k}': 0.0 for k in [
            'dose_measured', 'dose_expected', 'dose_diff_cGy', 'dose_diff_percent',
            'dose_tolerance_exceeded', 'gamma_pass_rate', 'gamma_tolerance_exceeded',
            'gamma_mean', 'output_factor_measured', 'output_factor_expected',
            'output_factor_diff_percent', 'output_tolerance_exceeded',
            'measurements_available', 'measurement_count',
        ]}

    # 5. Extract cross-validation features
    cv_features = extract_cross_validation_features(
        mosaiq_features, trf_features, portal_features, phantom_features
    )

    # Combine all features
    all_features = {}
    all_features.update(mosaiq_features)
    all_features.update(trf_features)
    all_features.update(portal_features)
    all_features.update(phantom_features)
    all_features.update(cv_features)

    return all_features


# Standard feature names for multi-modal EBM
MULTIMODAL_FEATURE_NAMES = [
    # ... (would include all Mosaiq features from ebm_features.STANDARD_FEATURE_NAMES)
    # Plus TRF features (18)
    "trf_file_readable",
    "trf_data_complete",
    "trf_mu_delivered",
    "trf_mu_mosaiq_diff",
    "trf_mu_mosaiq_percent_diff",
    "trf_mu_tolerance_exceeded",
    "trf_mlc_max_deviation",
    "trf_mlc_mean_deviation",
    "trf_mlc_rms_deviation",
    "trf_mlc_tolerance_exceeded",
    "trf_gantry_max_deviation",
    "trf_gantry_mean_deviation",
    "trf_gantry_tolerance_exceeded",
    "trf_beam_hold_count",
    "trf_beam_hold_detected",
    "trf_duration_seconds",
    "trf_duration_mosaiq_diff",
    "trf_duration_tolerance_exceeded",
    # Portal features (11)
    "portal_file_readable",
    "portal_mean_dose",
    "portal_max_dose",
    "portal_dose_range",
    "portal_central_dose",
    "portal_uniformity_std",
    "portal_uniformity_cov",
    "portal_gamma_pass_rate",
    "portal_gamma_mean",
    "portal_gamma_max",
    "portal_gamma_tolerance_exceeded",
    # Phantom features (13)
    "phantom_measurements_available",
    "phantom_measurement_count",
    "phantom_dose_measured",
    "phantom_dose_expected",
    "phantom_dose_diff_cGy",
    "phantom_dose_diff_percent",
    "phantom_dose_tolerance_exceeded",
    "phantom_gamma_pass_rate",
    "phantom_gamma_tolerance_exceeded",
    "phantom_gamma_mean",
    "phantom_output_factor_measured",
    "phantom_output_factor_expected",
    "phantom_output_factor_diff_percent",
    "phantom_output_tolerance_exceeded",
    # Cross-validation features (9)
    "cv_mosaiq_available",
    "cv_trf_available",
    "cv_portal_available",
    "cv_phantom_available",
    "cv_data_sources_count",
    "cv_mu_consistent",
    "cv_gamma_consistent",
    "cv_multi_modal_failure",
    "cv_confidence_score",
]

# Total: 72 (Mosaiq) + 18 (TRF) + 11 (Portal) + 14 (Phantom) + 9 (CV) = 124 features
