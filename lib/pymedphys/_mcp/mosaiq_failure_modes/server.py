"""MCP Server for Mosaiq Database Failure Mode Testing.

This server provides tools to simulate database corruption and third-party write failures
for defensive testing purposes.
"""

import logging
import struct
from datetime import datetime, timedelta
from typing import Any, Sequence

import pymssql
from mcp.server import Server
from mcp.types import Resource, TextContent, Tool

from pymedphys._mosaiq import connect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("mosaiq-failure-modes")


# =============================================================================
# FAILURE MODE DEFINITIONS
# =============================================================================

# Severity scale for EBM training:
# 0.0 = Normal data (low energy)
# 0.5-0.8 = Low severity (data quality issues, non-critical)
# 1.0-1.5 = Medium severity (data integrity, workflow issues)
# 1.8-2.3 = High severity (dose delivery errors, treatment integrity)
# 2.5-3.0 = Critical severity (patient safety risk, wrong patient/position/dose)

FAILURE_MODES = {
    "corrupt_mlc_data": {
        "description": "Corrupt MLC binary data (A_Leaf_Set/B_Leaf_Set) in TxFieldPoint",
        "severity": {
            "base": 1.5,  # High - affects dose delivery
            "variants": {
                "odd_bytes": 0.6,  # Low - parsing issue, may not affect treatment
                "out_of_range": 2.0,  # High - dose calculation errors
                "negative_gap": 2.2,  # High - physical impossibility, delivery fails
                "random_bytes": 1.5,  # Medium-High - unpredictable effects
            },
        },
        "qa_checks": [
            "Verify MLC byte array length is even (2 bytes per leaf)",
            "Verify MLC positions are within physical limits (-20cm to +20cm typical)",
            "Verify leaf pairs don't create negative gap (A_Leaf > B_Leaf)",
            "Verify all control points have same number of leaves",
        ],
        "additional_data_needed": [
            "MLC model specification (number of leaves, physical limits per leaf)",
            "Expected leaf count for treatment machine",
            "Historical MLC position ranges for normal operation",
        ],
    },
    "invalid_angles": {
        "description": "Set gantry/collimator angles outside valid range (0-360°)",
        "severity": {
            "base": 0.7,  # Low - likely data entry error, delivery has safety constraints
            "variants": {
                "gantry": 0.7,
                "collimator": 0.6,
                "both": 0.8,
            },
        },
        "qa_checks": [
            "Verify Gantry_Ang is within 0-360° range",
            "Verify Coll_Ang is within 0-360° range",
            "Verify angle changes between control points are physically achievable",
            "Check for discontinuous angle jumps (e.g., 359° to 1°)",
        ],
        "additional_data_needed": [
            "Machine maximum rotation speed (degrees/second)",
            "Control point timing information for speed validation",
            "Machine mechanical limits (some machines may have restricted ranges)",
        ],
    },
    "duplicate_treatments": {
        "description": "Create duplicate TrackTreatment entries with conflicting data",
        "severity": {
            "base": 1.2,  # Medium - billing/record issues, treatment integrity unclear
            "variants": {
                "same_field": 1.0,  # Medium-Low - likely duplicate recording
                "different_field": 1.5,  # Medium-High - conflicting treatment records
            },
        },
        "qa_checks": [
            "Check for multiple TrackTreatment entries within time buffer",
            "Verify no conflicting FLD_ID for same patient/machine/time",
            "Detect overlapping Create_DtTm to Edit_DtTm intervals",
            "Validate uniqueness constraints on treatment delivery",
        ],
        "additional_data_needed": [
            "Expected treatment duration for given field type",
            "Machine interlock logs (to confirm actual single delivery)",
            "ARIA/R&V system records for cross-validation",
        ],
    },
    "missing_control_points": {
        "description": "Delete control points from TxFieldPoint to create gaps",
        "severity": {
            "base": 2.0,  # High - incomplete dose delivery data
            "variants": {
                "single_gap": 1.8,  # High - one missing segment
                "multiple_gaps": 2.2,  # High - severe data loss
                "all_deleted": 2.5,  # Critical - complete data loss
            },
        },
        "qa_checks": [
            "Verify control point indices are sequential (0, 1, 2, ...)",
            "Verify total control points match expected for field type",
            "Check that Point values are consecutive",
            "Validate at least 2 control points exist (start/end)",
        ],
        "additional_data_needed": [
            "Treatment plan file (DICOM RT Plan) for expected control point count",
            "Field delivery type (static vs IMRT vs VMAT) to determine expected CPs",
            "MU weighting to validate cumulative dose across control points",
        ],
    },
    "null_required_fields": {
        "description": "Set critical fields to NULL (Pat_ID1, FLD_ID, timestamps)",
        "severity": {
            "base": 1.8,  # High - data corruption, queries fail
            "variants": {
                "pat_id": 2.7,  # Critical - patient identification lost
                "fld_id": 2.3,  # High-Critical - treatment plan reference lost
                "timestamp": 1.2,  # Medium - temporal reference lost
                "other": 1.0,  # Medium-Low - data quality issue
            },
        },
        "qa_checks": [
            "Verify all foreign keys are non-NULL",
            "Verify all timestamp fields are non-NULL",
            "Check for NULL in indexed fields used for joins",
            "Validate data type constraints",
        ],
        "additional_data_needed": [
            "Database schema constraints (NOT NULL, FOREIGN KEY definitions)",
            "Index definitions to identify critical fields",
            "Application-level field requirements beyond DB schema",
        ],
    },
    "timestamp_inconsistencies": {
        "description": "Create invalid timestamp relationships (Edit_DtTm < Create_DtTm)",
        "severity": {
            "base": 1.0,  # Medium - record keeping issue
            "variants": {
                "edit_before_create": 1.2,  # Medium - logical impossibility
                "future_timestamp": 1.0,  # Medium - clock sync issue
                "negative_duration": 1.3,  # Medium - data integrity issue
            },
        },
        "qa_checks": [
            "Verify Edit_DtTm >= Create_DtTm for all TrackTreatment records",
            "Check for future timestamps (beyond current datetime)",
            "Validate treatment duration is within acceptable range",
            "Cross-check with dose history (Dose_Hst.Tx_DtTm) for consistency",
        ],
        "additional_data_needed": [
            "Expected treatment duration ranges by field type",
            "System clock synchronization logs (NTP status)",
            "Treatment timeline from other systems (machine logs, ARIA)",
        ],
    },
    "orphaned_records": {
        "description": "Create records with invalid foreign key references",
        "severity": {
            "base": 2.6,  # Critical - patient safety risk (wrong patient association)
            "variants": {
                "patient_id": 2.8,  # Critical - treatment may be for wrong patient
                "field_id": 2.3,  # High-Critical - treatment plan reference invalid
                "site_id": 2.5,  # Critical - treatment site reference invalid
            },
        },
        "qa_checks": [
            "Verify all Pat_ID1 references exist in Patient/Ident tables",
            "Verify all FLD_ID references exist in TxField table",
            "Verify all SIT_ID references exist in Site table",
            "Check referential integrity across all joins",
        ],
        "additional_data_needed": [
            "Foreign key constraint definitions",
            "Cascade delete rules",
            "Application-level relationship requirements",
        ],
    },
    "invalid_offset_data": {
        "description": "Corrupt patient positioning offset values (third-party writes)",
        "severity": {
            "base": 2.4,  # High-Critical - wrong treatment position
            "variants": {
                "extreme_values": 2.7,  # Critical - patient safety (geometric miss)
                "invalid_type": 1.0,  # Medium - data quality issue
                "invalid_state": 0.9,  # Medium-Low - workflow tracking issue
                "future_study_time": 1.1,  # Medium - temporal tracking issue
            },
        },
        "qa_checks": [
            "Verify offset magnitudes are within acceptable range (±5cm typical)",
            "Check Offset_Type is valid enumeration (2=Localization, 3=Portal, 4=ThirdParty)",
            "Verify Offset_State is valid (1=Active, 2=Complete)",
            "Validate Study_DtTm is within reasonable time window of treatment",
        ],
        "additional_data_needed": [
            "Acceptable offset magnitude limits by treatment site",
            "Expected offset workflow timing (how long before treatment)",
            "Third-party system integration logs (CBCT, portal imaging)",
            "Action limits and tolerance tables from clinical protocols",
        ],
    },
    "meterset_inconsistency": {
        "description": "Mismatch between planned meterset and delivered MU",
        "severity": {
            "base": 2.3,  # High-Critical - dose delivery error
            "variants": {
                "negative_meterset": 2.8,  # Critical - physically impossible, safety interlock
                "extreme_value": 2.6,  # Critical - wrong dose delivered
                "mismatch_with_cp": 1.5,  # Medium-High - documentation inconsistency
            },
        },
        "qa_checks": [
            "Verify TxField.Meterset matches sum of control point MU",
            "Check for negative meterset values",
            "Validate meterset is within acceptable range for field type",
            "Compare planned vs delivered MU within tolerance",
        ],
        "additional_data_needed": [
            "MU tolerance limits from clinical protocols",
            "Control point MU weighting from treatment plan",
            "Machine output factors for dose verification",
            "Historical MU ranges by treatment technique",
        ],
    },
    "mlc_leaf_count_mismatch": {
        "description": "Inconsistent number of MLC leaves across control points",
        "severity": {
            "base": 1.9,  # High - dose calculation errors
            "variants": {
                "intra_field": 2.1,  # High - inconsistent within same field
                "vs_machine": 2.0,  # High - doesn't match machine config
                "vs_plan": 1.8,  # High - doesn't match treatment plan
            },
        },
        "qa_checks": [
            "Verify all control points have same leaf count",
            "Verify leaf count matches machine specification",
            "Check byte array length consistency (length = leaves * 2)",
            "Validate against treatment plan MLC configuration",
        ],
        "additional_data_needed": [
            "Machine MLC configuration (80-leaf, 120-leaf, etc.)",
            "Treatment plan DICOM RT Plan BeamLimitingDeviceSequence",
            "Machine manufacturer MLC specifications",
        ],
    },
    # =============================================================================
    # MALICIOUS ACTOR FAILURE MODES
    # =============================================================================
    # These represent intentional sabotage attempts designed to cause patient harm
    # while avoiding detection. All have critical severity (2.5-3.0).
    "subtle_dose_escalation": {
        "description": "Incrementally increase MU across multiple fractions to deliver excess dose while staying within daily tolerances",
        "severity": {
            "base": 2.9,  # Critical - deliberate patient harm
            "variants": {
                "linear_ramp": 2.8,  # Critical - gradual increase over treatment course
                "random_walk": 2.9,  # Critical - random ±1-2% changes that accumulate
                "fraction_selective": 3.0,  # Critical - only certain fractions modified
                "field_distributed": 2.7,  # Critical - spread across multiple fields
            },
        },
        "qa_checks": [
            "Track cumulative MU across all fractions, compare to prescription",
            "Statistical analysis of MU variance (detect systematic bias vs random)",
            "Field-by-field longitudinal tracking (detect gradual drift)",
            "Cross-reference with TRF files for delivered vs planned MU",
            "Calculate total dose from delivered MU, verify against prescription",
            "Detect outliers in MU per fraction time series",
            "Compare current fraction MU to initial baseline (first 3 fractions)",
        ],
        "additional_data_needed": [
            "Complete treatment history (all fractions) for statistical analysis",
            "Prescription dose and fractionation schedule",
            "Expected MU per fraction baseline (from plan)",
            "TRF files for independent verification of delivered dose",
            "Statistical models of normal MU variance per technique",
            "Longitudinal trending data from previous patients with same protocol",
        ],
    },
    "targeted_patient_selection": {
        "description": "Selectively corrupt data for specific patients to avoid statistical detection across population",
        "severity": {
            "base": 3.0,  # Critical - targeted harm, difficult to detect
            "variants": {
                "demographic_targeting": 3.0,  # Critical - based on age, diagnosis, etc.
                "random_sampling": 2.9,  # Critical - small % of patients to avoid pattern
                "temporal_spacing": 2.9,  # Critical - long intervals between attacks
                "site_specific": 2.8,  # Critical - only certain treatment sites
            },
        },
        "qa_checks": [
            "Per-patient anomaly detection (not just population statistics)",
            "Longitudinal patient tracking from simulation through completion",
            "Demographic analysis of anomalies (detect biased targeting)",
            "Temporal clustering analysis (detect suspicious gaps or patterns)",
            "Cross-patient comparison within same protocol/site",
            "Audit all data modifications with user attribution",
            "Flag patients with multiple independent anomalies (low probability)",
        ],
        "additional_data_needed": [
            "Patient demographics and diagnosis codes for bias detection",
            "Complete audit trail of database modifications with timestamps and users",
            "Historical baseline of anomaly rates by demographic group",
            "Treatment protocol and site information for stratified analysis",
            "Multi-modal QA results (TRF, portal, phantom) for independent verification",
            "Social network analysis of treating staff (detect insider threats)",
        ],
    },
    "coordinated_multifield_attack": {
        "description": "Distribute dose errors across multiple fields such that each field appears normal but cumulative dose is incorrect",
        "severity": {
            "base": 2.9,  # Critical - sophisticated, evades field-level QA
            "variants": {
                "compensatory_errors": 3.0,  # Critical - systematic opposite biases that sum
                "geometric_shift": 2.9,  # Critical - shift dose away from target
                "aperture_erosion": 2.8,  # Critical - subtle MLC changes across all fields
                "hotspot_creation": 2.9,  # Critical - create overlap regions with excess dose
            },
        },
        "qa_checks": [
            "3D dose reconstruction from all fields (not individual field QA)",
            "Gamma analysis of composite plan vs delivered (portal dosimetry)",
            "DVH metrics for targets and OARs from delivered dose",
            "Geometric analysis of field aperture overlaps and gaps",
            "MLC aperture evolution tracking across fractions",
            "Cumulative fluence maps (detect systematic shifts)",
            "Independent dose calculation from delivered parameters",
        ],
        "additional_data_needed": [
            "CT dataset for dose reconstruction",
            "Structure set (targets, OARs) for DVH analysis",
            "Treatment plan for baseline comparison",
            "Portal dosimetry for all fields (independent measurement)",
            "TRF files with MLC positions for all fractions",
            "Gamma analysis tolerance levels (e.g., 3%/3mm)",
            "3D dose calculation engine (e.g., Monte Carlo)",
        ],
    },
    "time_delayed_corruption": {
        "description": "Modify future fractions in database so errors manifest days/weeks after modification (audit trail obfuscation)",
        "severity": {
            "base": 2.8,  # Critical - difficult attribution, delayed detection
            "variants": {
                "scheduled_activation": 2.9,  # Critical - triggers on specific date/fraction
                "conditional_execution": 2.8,  # Critical - only if certain conditions met
                "database_trigger": 2.7,  # Critical - uses DB triggers for delayed execution
                "gradual_onset": 2.8,  # Critical - slow manifestation over multiple fractions
            },
        },
        "qa_checks": [
            "Pre-treatment data integrity check immediately before each fraction",
            "Version control and checksumming of treatment parameters",
            "Detect unexpected data modifications (compare to previous fraction)",
            "Audit trail temporal analysis (flag modifications of future fractions)",
            "Database trigger and stored procedure audit",
            "Diff analysis: planned vs current database state before delivery",
            "Lock treatment parameters after approval (detect lock bypass)",
        ],
        "additional_data_needed": [
            "Database audit logs with write operation timestamps",
            "Baseline checksums/hashes of approved treatment parameters",
            "Database trigger and stored procedure definitions",
            "Treatment parameter version history (track all modifications)",
            "User access logs (who modified what and when)",
            "Pre-treatment verification workflow timestamps",
            "Lock status and approval chain for treatment parameters",
        ],
    },
    "statistical_camouflage": {
        "description": "Introduce systematic bias that stays within normal statistical ranges but accumulates to harmful levels",
        "severity": {
            "base": 2.9,  # Critical - sophisticated, designed to evade statistical QA
            "variants": {
                "within_tolerance_bias": 2.9,  # Critical - always <2% but always positive
                "noise_injection": 2.7,  # Critical - adds noise to hide systematic signal
                "correlation_masking": 2.8,  # Critical - correlates errors with normal variance
                "outlier_avoidance": 2.9,  # Critical - carefully avoids statistical outliers
            },
        },
        "qa_checks": [
            "Statistical process control (SPC) charts with CUSUM analysis",
            "Detect systematic bias (t-test against zero mean)",
            "Correlation analysis between errors and legitimate variance sources",
            "Non-parametric trend tests (Mann-Kendall, Sen's slope)",
            "Benford's law analysis (detect artificial data)",
            "Run tests for randomness (detect non-random patterns)",
            "Bayesian anomaly detection (update beliefs over time)",
        ],
        "additional_data_needed": [
            "Historical baseline data (>100 patients) for statistical models",
            "Known sources of legitimate variance (machine, technique, patient)",
            "Statistical tolerance limits from clinical protocols",
            "Time series data for trend analysis",
            "Control group data (unmodified patients) for comparison",
            "Multi-modal verification (TRF, portal, phantom) to break correlations",
            "Machine QA data (detect correlated equipment issues)",
        ],
    },
    "audit_trail_manipulation": {
        "description": "Modify database audit tables, timestamps, or user attribution to hide malicious changes",
        "severity": {
            "base": 2.7,  # Critical - covers tracks, enables other attacks
            "variants": {
                "timestamp_forgery": 2.8,  # Critical - backdates malicious modifications
                "user_impersonation": 2.9,  # Critical - attributes changes to legitimate users
                "log_deletion": 3.0,  # Critical - removes evidence entirely
                "audit_disable": 2.7,  # Critical - disables logging temporarily
            },
        },
        "qa_checks": [
            "External audit logging (write-once, append-only log off-database)",
            "Database binary log analysis (transaction log forensics)",
            "Timestamp consistency checks (cross-reference with network time)",
            "User session analysis (detect impossible user activity patterns)",
            "File system timestamps on database files (detect out-of-band modifications)",
            "Audit table integrity checks (checksums, digital signatures)",
            "Compare database state to external backups (detect retroactive changes)",
        ],
        "additional_data_needed": [
            "Write-once external audit log (WORM storage, blockchain)",
            "Database transaction logs (binary logs, redo logs)",
            "Network time protocol (NTP) logs for timestamp validation",
            "User authentication and session logs",
            "Database file system metadata and backups",
            "Cryptographic audit trail (digital signatures on critical records)",
            "Immutable backup schedule (detect restoration attacks)",
        ],
    },
    "field_aperture_manipulation": {
        "description": "Subtly modify MLC apertures to shift dose away from target toward critical structures",
        "severity": {
            "base": 2.9,  # Critical - deliberate geometric miss
            "variants": {
                "systematic_shift": 3.0,  # Critical - all leaves shifted same direction
                "margin_erosion": 2.9,  # Critical - reduce PTV margins
                "oar_expansion": 2.8,  # Critical - move dose toward critical structures
                "asymmetric_blur": 2.7,  # Critical - non-uniform aperture degradation
            },
        },
        "qa_checks": [
            "MLC aperture centroid tracking (detect systematic shifts)",
            "Field size metrics (area, perimeter, equivalent square)",
            "PTV margin analysis (calculate effective margin from aperture)",
            "OAR proximity analysis (distance from aperture to critical structures)",
            "Aperture shape analysis (detect asymmetric changes)",
            "Leaf end position histograms (detect systematic bias)",
            "Portal imaging analysis (compare to planned aperture)",
        ],
        "additional_data_needed": [
            "Treatment plan DICOM RT Plan (baseline apertures)",
            "Structure set with PTV and OAR contours",
            "Portal images for geometric verification",
            "MLC calibration and positional accuracy data",
            "Expected PTV margins by treatment site",
            "TRF files with delivered MLC positions",
            "Beam's eye view (BEV) geometry from plan",
        ],
    },
    "gradual_parameter_drift": {
        "description": "Slowly modify treatment parameters over multiple fractions to create cumulative errors that evade daily QA",
        "severity": {
            "base": 2.8,  # Critical - long-term harm, difficult detection
            "variants": {
                "gantry_angle_drift": 2.7,  # Critical - gradual rotation errors
                "isocenter_shift": 2.9,  # Critical - slow positional drift
                "mu_creep": 2.8,  # Critical - gradual dose escalation
                "mlc_drift": 2.7,  # Critical - slow aperture changes
            },
        },
        "qa_checks": [
            "Longitudinal parameter tracking (plot all parameters vs fraction)",
            "Regression analysis to detect trends (linear, exponential fits)",
            "Change point detection algorithms (identify when drift started)",
            "Cumulative deviation analysis (sum of all deviations from baseline)",
            "Velocity analysis (rate of change between fractions)",
            "Kalman filtering for state estimation (predict expected values)",
            "Compare to initial baseline (first 3 fractions after approval)",
        ],
        "additional_data_needed": [
            "Complete treatment history (all fractions) for trend analysis",
            "Initial approved baseline parameters",
            "Expected parameter stability tolerances",
            "TRF files for independent verification of all fractions",
            "Statistical models of normal inter-fraction variance",
            "Machine QA trends (separate equipment drift from malicious drift)",
            "Environmental data (temperature, humidity) to account for physical drift",
        ],
    },
    "selective_fraction_sabotage": {
        "description": "Modify only specific fractions (e.g., every 5th fraction) to avoid detection while accumulating dose errors",
        "severity": {
            "base": 2.8,  # Critical - sparse attacks evade detection
            "variants": {
                "periodic_attacks": 2.7,  # Critical - regular intervals (every Nth)
                "random_selection": 2.9,  # Critical - unpredictable fraction choice
                "calendar_based": 2.8,  # Critical - specific days/times (weekend, night shift)
                "staff_correlated": 2.8,  # Critical - only when certain staff on duty
            },
        },
        "qa_checks": [
            "Per-fraction anomaly scoring (flag individual fractions)",
            "Frequency domain analysis (detect periodic patterns)",
            "Temporal autocorrelation (detect non-random fraction selection)",
            "Staff correlation analysis (detect insider threat patterns)",
            "Calendar analysis (detect temporal biases)",
            "Poisson process modeling (detect non-random event timing)",
            "Cumulative dose tracking (verify total dose regardless of fraction pattern)",
        ],
        "additional_data_needed": [
            "Per-fraction QA data (TRF, portal dosimetry) for all fractions",
            "Staff schedules and shift assignments",
            "Calendar data (weekday, time of day, holidays)",
            "Facility access logs (detect unauthorized access patterns)",
            "Complete audit trail of database modifications",
            "Multi-modal verification for every fraction (not just periodic QA)",
            "Statistical baseline of normal fraction-to-fraction variance",
        ],
    },
    "collimator_jaw_manipulation": {
        "description": "Modify collimator jaw positions to create field size errors or dose gradient shifts",
        "severity": {
            "base": 2.6,  # Critical - affects entire field dose distribution
            "variants": {
                "jaw_asymmetry": 2.7,  # Critical - unequal jaw positions
                "field_size_reduction": 2.8,  # Critical - reduce field to underdose target
                "field_size_expansion": 2.5,  # Critical - overdose surrounding tissue
                "gradient_shift": 2.6,  # Critical - change dose fall-off characteristics
            },
        },
        "qa_checks": [
            "Verify collimator X1, X2, Y1, Y2 jaw positions match plan",
            "Field size calculation from jaw positions (verify against prescription)",
            "Jaw symmetry analysis (detect asymmetric modifications)",
            "Comparison with MLC aperture (jaws should exceed MLC aperture)",
            "Portal imaging field size verification",
            "Dose profile analysis (detect gradient changes)",
            "TRF file verification of delivered jaw positions",
        ],
        "additional_data_needed": [
            "Treatment plan jaw positions (DICOM RT Plan)",
            "Expected field sizes by beam",
            "Portal images for geometric verification",
            "TRF files with delivered jaw positions",
            "Jaw position tolerances from clinical protocol",
            "Beam profile measurements (for gradient verification)",
            "Collimator calibration data (jaw position accuracy)",
        ],
    },
}


# =============================================================================
# MCP RESOURCES
# =============================================================================


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available failure mode documentation resources."""
    return [
        Resource(
            uri=f"failure-mode://mosaiq/{mode_id}",
            name=f"Failure Mode: {mode_id}",
            mimeType="text/plain",
            description=mode_info["description"],
        )
        for mode_id, mode_info in FAILURE_MODES.items()
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read detailed information about a failure mode including QA checks."""
    if not uri.startswith("failure-mode://mosaiq/"):
        raise ValueError(f"Unknown resource URI: {uri}")

    mode_id = uri.replace("failure-mode://mosaiq/", "")
    if mode_id not in FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {mode_id}")

    mode_info = FAILURE_MODES[mode_id]

    doc = f"""# Failure Mode: {mode_id}

## Description
{mode_info['description']}

## QA Checks to Detect This Failure Mode

"""
    for i, check in enumerate(mode_info["qa_checks"], 1):
        doc += f"{i}. {check}\n"

    doc += "\n## Additional Data Required for QA\n\n"
    for i, data in enumerate(mode_info["additional_data_needed"], 1):
        doc += f"{i}. {data}\n"

    return doc


# =============================================================================
# MCP TOOLS - Database Modification Functions
# =============================================================================


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available failure mode simulation tools."""
    return [
        Tool(
            name="corrupt_mlc_data",
            description=FAILURE_MODES["corrupt_mlc_data"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string", "description": "Mosaiq SQL Server hostname"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "fld_id": {"type": "integer", "description": "Field ID to corrupt"},
                    "corruption_type": {
                        "type": "string",
                        "enum": ["odd_bytes", "out_of_range", "negative_gap", "random_bytes"],
                        "description": "Type of MLC corruption to apply",
                    },
                },
                "required": ["hostname", "fld_id", "corruption_type"],
            },
        ),
        Tool(
            name="create_invalid_angles",
            description=FAILURE_MODES["invalid_angles"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "fld_id": {"type": "integer"},
                    "angle_type": {
                        "type": "string",
                        "enum": ["gantry", "collimator", "both"],
                    },
                    "invalid_value": {
                        "type": "number",
                        "description": "Invalid angle value (e.g., 400, -50)",
                    },
                },
                "required": ["hostname", "fld_id", "angle_type", "invalid_value"],
            },
        ),
        Tool(
            name="create_duplicate_treatment",
            description=FAILURE_MODES["duplicate_treatments"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "ttx_id": {
                        "type": "integer",
                        "description": "Original TrackTreatment ID to duplicate",
                    },
                    "time_offset_seconds": {
                        "type": "integer",
                        "description": "Time offset for duplicate (seconds)",
                        "default": 5,
                    },
                    "change_field": {
                        "type": "boolean",
                        "description": "Whether to assign different FLD_ID",
                        "default": False,
                    },
                },
                "required": ["hostname", "ttx_id"],
            },
        ),
        Tool(
            name="delete_control_points",
            description=FAILURE_MODES["missing_control_points"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "fld_id": {"type": "integer"},
                    "points_to_delete": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of point indices to delete",
                    },
                },
                "required": ["hostname", "fld_id", "points_to_delete"],
            },
        ),
        Tool(
            name="nullify_required_fields",
            description=FAILURE_MODES["null_required_fields"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "table": {
                        "type": "string",
                        "enum": ["TrackTreatment", "TxField", "TxFieldPoint", "Offset"],
                    },
                    "record_id": {"type": "integer", "description": "Primary key of record"},
                    "field_to_nullify": {
                        "type": "string",
                        "description": "Field name to set to NULL",
                    },
                },
                "required": ["hostname", "table", "record_id", "field_to_nullify"],
            },
        ),
        Tool(
            name="create_timestamp_inconsistency",
            description=FAILURE_MODES["timestamp_inconsistencies"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "ttx_id": {"type": "integer"},
                    "inconsistency_type": {
                        "type": "string",
                        "enum": ["edit_before_create", "future_timestamp", "negative_duration"],
                    },
                },
                "required": ["hostname", "ttx_id", "inconsistency_type"],
            },
        ),
        Tool(
            name="create_orphaned_record",
            description=FAILURE_MODES["orphaned_records"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "record_type": {
                        "type": "string",
                        "enum": ["TrackTreatment", "TxFieldPoint", "Offset"],
                    },
                    "invalid_fk_field": {
                        "type": "string",
                        "description": "Foreign key field to corrupt (e.g., Pat_ID1, FLD_ID)",
                    },
                    "invalid_value": {
                        "type": "integer",
                        "description": "Non-existent ID value",
                    },
                },
                "required": ["hostname", "record_type", "invalid_fk_field", "invalid_value"],
            },
        ),
        Tool(
            name="corrupt_offset_data",
            description=FAILURE_MODES["invalid_offset_data"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "off_id": {"type": "integer"},
                    "corruption_type": {
                        "type": "string",
                        "enum": [
                            "extreme_values",
                            "invalid_type",
                            "invalid_state",
                            "future_study_time",
                        ],
                    },
                    "value": {
                        "type": "number",
                        "description": "Value to set (for extreme_values or invalid enums)",
                    },
                },
                "required": ["hostname", "off_id", "corruption_type"],
            },
        ),
        Tool(
            name="create_meterset_inconsistency",
            description=FAILURE_MODES["meterset_inconsistency"]["description"],
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "database": {"type": "string", "default": "MOSAIQ"},
                    "fld_id": {"type": "integer"},
                    "inconsistency_type": {
                        "type": "string",
                        "enum": ["negative_meterset", "extreme_value", "mismatch_with_cp"],
                    },
                    "value": {"type": "number", "description": "Meterset value to set"},
                },
                "required": ["hostname", "fld_id", "inconsistency_type"],
            },
        ),
        Tool(
            name="list_failure_modes",
            description="List all available failure modes with descriptions and QA requirements",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


def execute_update(
    hostname: str, database: str, query: str, params: dict[str, Any] | None = None
) -> str:
    """Execute an UPDATE query on Mosaiq database."""
    try:
        with connect(hostname=hostname, database=database, read_only=False) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return f"Successfully executed: {cursor.rowcount} rows affected"
    except Exception as e:
        logger.error(f"Database error: {e}")
        return f"Error: {str(e)}"


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Handle tool calls for failure mode simulation."""
    if name == "list_failure_modes":
        result = "# Available Failure Modes\n\n"
        for mode_id, mode_info in FAILURE_MODES.items():
            result += f"## {mode_id}\n"
            result += f"{mode_info['description']}\n\n"
            result += "**QA Checks:**\n"
            for check in mode_info["qa_checks"]:
                result += f"- {check}\n"
            result += "\n**Additional Data Needed:**\n"
            for data in mode_info["additional_data_needed"]:
                result += f"- {data}\n"
            result += "\n---\n\n"
        return [TextContent(type="text", text=result)]

    elif name == "corrupt_mlc_data":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        fld_id = arguments["fld_id"]
        corruption_type = arguments["corruption_type"]

        # Different corruption strategies
        if corruption_type == "odd_bytes":
            # Corrupt A_Leaf_Set to have odd number of bytes
            query = """
            UPDATE TxFieldPoint
            SET A_Leaf_Set = A_Leaf_Set + CAST(0x00 AS VARBINARY(1))
            WHERE FLD_ID = %(fld_id)s AND Point = 0
            """
        elif corruption_type == "out_of_range":
            # Set extreme MLC values (e.g., 10000 = 100cm when packed as short)
            bad_mlc = struct.pack("<h", 10000)  # Way out of range
            query = """
            UPDATE TxFieldPoint
            SET A_Leaf_Set = %(bad_mlc)s
            WHERE FLD_ID = %(fld_id)s AND Point = 0
            """
            params = {"fld_id": fld_id, "bad_mlc": bad_mlc}
        elif corruption_type == "random_bytes":
            # Completely random binary data
            import random

            bad_mlc = bytes([random.randint(0, 255) for _ in range(20)])
            query = """
            UPDATE TxFieldPoint
            SET A_Leaf_Set = %(bad_mlc)s
            WHERE FLD_ID = %(fld_id)s
            """
            params = {"fld_id": fld_id, "bad_mlc": bad_mlc}
        else:
            return [TextContent(type="text", text=f"Unknown corruption type: {corruption_type}")]

        result = execute_update(hostname, database, query, params if "params" in locals() else {"fld_id": fld_id})
        return [TextContent(type="text", text=result)]

    elif name == "create_invalid_angles":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        fld_id = arguments["fld_id"]
        angle_type = arguments["angle_type"]
        invalid_value = arguments["invalid_value"]

        if angle_type == "gantry":
            query = "UPDATE TxFieldPoint SET Gantry_Ang = %(value)s WHERE FLD_ID = %(fld_id)s"
        elif angle_type == "collimator":
            query = "UPDATE TxFieldPoint SET Coll_Ang = %(value)s WHERE FLD_ID = %(fld_id)s"
        else:  # both
            query = """
            UPDATE TxFieldPoint
            SET Gantry_Ang = %(value)s, Coll_Ang = %(value)s
            WHERE FLD_ID = %(fld_id)s
            """

        result = execute_update(hostname, database, query, {"fld_id": fld_id, "value": invalid_value})
        return [TextContent(type="text", text=result)]

    elif name == "create_duplicate_treatment":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        ttx_id = arguments["ttx_id"]
        time_offset = arguments.get("time_offset_seconds", 5)
        change_field = arguments.get("change_field", False)

        # This requires INSERT which is more complex - provide SQL template
        result = f"""
To create duplicate treatment, execute:

INSERT INTO TrackTreatment (Pat_ID1, FLD_ID, SIT_ID, Create_DtTm, Edit_DtTm, Machine_ID_Staff_ID, WasQAMode, WasBeamComplete)
SELECT Pat_ID1,
       {'FLD_ID + 1' if change_field else 'FLD_ID'},
       SIT_ID,
       DATEADD(second, {time_offset}, Create_DtTm),
       DATEADD(second, {time_offset}, Edit_DtTm),
       Machine_ID_Staff_ID,
       WasQAMode,
       WasBeamComplete
FROM TrackTreatment
WHERE TTX_ID = {ttx_id}

Note: This requires INSERT permissions. Tool provides template for manual execution.
"""
        return [TextContent(type="text", text=result)]

    elif name == "delete_control_points":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        fld_id = arguments["fld_id"]
        points = arguments["points_to_delete"]

        query = f"""
        DELETE FROM TxFieldPoint
        WHERE FLD_ID = %(fld_id)s AND Point IN ({','.join(map(str, points))})
        """

        result = execute_update(hostname, database, query, {"fld_id": fld_id})
        return [TextContent(type="text", text=result)]

    elif name == "nullify_required_fields":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        table = arguments["table"]
        record_id = arguments["record_id"]
        field = arguments["field_to_nullify"]

        # Determine primary key column based on table
        pk_map = {
            "TrackTreatment": "TTX_ID",
            "TxField": "FLD_ID",
            "TxFieldPoint": "TFP_ID",
            "Offset": "OFF_ID",
        }
        pk_col = pk_map.get(table)
        if not pk_col:
            return [TextContent(type="text", text=f"Unknown table: {table}")]

        query = f"UPDATE {table} SET {field} = NULL WHERE {pk_col} = %(record_id)s"
        result = execute_update(hostname, database, query, {"record_id": record_id})
        return [TextContent(type="text", text=result)]

    elif name == "create_timestamp_inconsistency":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        ttx_id = arguments["ttx_id"]
        inconsistency_type = arguments["inconsistency_type"]

        if inconsistency_type == "edit_before_create":
            query = """
            UPDATE TrackTreatment
            SET Edit_DtTm = DATEADD(hour, -2, Create_DtTm)
            WHERE TTX_ID = %(ttx_id)s
            """
        elif inconsistency_type == "future_timestamp":
            query = """
            UPDATE TrackTreatment
            SET Create_DtTm = DATEADD(year, 1, GETDATE()),
                Edit_DtTm = DATEADD(year, 1, DATEADD(hour, 1, GETDATE()))
            WHERE TTX_ID = %(ttx_id)s
            """
        else:  # negative_duration
            query = """
            UPDATE TrackTreatment
            SET Edit_DtTm = Create_DtTm
            WHERE TTX_ID = %(ttx_id)s
            """

        result = execute_update(hostname, database, query, {"ttx_id": ttx_id})
        return [TextContent(type="text", text=result)]

    elif name == "create_orphaned_record":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        record_type = arguments["record_type"]
        fk_field = arguments["invalid_fk_field"]
        invalid_value = arguments["invalid_value"]

        # Find a record to corrupt
        result = f"""
To create orphaned record, execute:

UPDATE {record_type}
SET {fk_field} = {invalid_value}
WHERE <select appropriate record with WHERE clause>

Note: Requires specifying which record to corrupt. Use with caution.
"""
        return [TextContent(type="text", text=result)]

    elif name == "corrupt_offset_data":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        off_id = arguments["off_id"]
        corruption_type = arguments["corruption_type"]
        value = arguments.get("value")

        if corruption_type == "extreme_values":
            query = """
            UPDATE Offset
            SET Superior_Offset = %(value)s,
                Anterior_Offset = %(value)s,
                Lateral_Offset = %(value)s
            WHERE OFF_ID = %(off_id)s
            """
            params = {"off_id": off_id, "value": value or 999.9}
        elif corruption_type == "invalid_type":
            query = "UPDATE Offset SET Offset_Type = %(value)s WHERE OFF_ID = %(off_id)s"
            params = {"off_id": off_id, "value": int(value or 99)}
        elif corruption_type == "invalid_state":
            query = "UPDATE Offset SET Offset_State = %(value)s WHERE OFF_ID = %(off_id)s"
            params = {"off_id": off_id, "value": int(value or 99)}
        else:  # future_study_time
            query = """
            UPDATE Offset
            SET Study_DtTm = DATEADD(year, 1, GETDATE())
            WHERE OFF_ID = %(off_id)s
            """
            params = {"off_id": off_id}

        result = execute_update(hostname, database, query, params)
        return [TextContent(type="text", text=result)]

    elif name == "create_meterset_inconsistency":
        hostname = arguments["hostname"]
        database = arguments.get("database", "MOSAIQ")
        fld_id = arguments["fld_id"]
        inconsistency_type = arguments["inconsistency_type"]
        value = arguments.get("value")

        if inconsistency_type == "negative_meterset":
            query = "UPDATE TxField SET Meterset = -100 WHERE FLD_ID = %(fld_id)s"
        elif inconsistency_type == "extreme_value":
            query = "UPDATE TxField SET Meterset = %(value)s WHERE FLD_ID = %(fld_id)s"
            params = {"fld_id": fld_id, "value": value or 999999}
        else:  # mismatch_with_cp
            query = "UPDATE TxField SET Meterset = 1.0 WHERE FLD_ID = %(fld_id)s"

        result = execute_update(
            hostname, database, query, {"fld_id": fld_id} if "params" not in locals() else params
        )
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Run the MCP server."""
    import asyncio

    from mcp.server.stdio import stdio_server

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
