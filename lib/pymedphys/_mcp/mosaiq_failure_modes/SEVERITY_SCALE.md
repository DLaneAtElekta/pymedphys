# Mosaiq Failure Mode Severity Scale

## Overview

The severity scale assigns continuous scores (0.0-3.0) to failure modes based on their clinical impact and patient safety risk. This enables the Energy-Based Model (EBM) to:

1. **Detect anomalies** - Any energy > 0.4 indicates potential issues
2. **Assess severity** - Energy value indicates clinical risk level
3. **Prioritize alerts** - Critical failures trigger immediate alerts, low severity issues may only log
4. **Reduce alert fatigue** - Physics team sees risk-stratified notifications

## Severity Categories

### 0.0 - Normal Data
**Energy Target**: 0.1-0.3

Clean data with no detected issues. This is the target energy for all normal treatment records.

**Examples**:
- All QA checks pass
- No data anomalies detected
- Standard treatment delivery

---

### 0.5-0.8 - Low Severity
**Impact**: Minimal clinical risk, primarily data quality concerns

**Characteristics**:
- Data parsing or formatting issues
- Non-critical data quality problems
- Issues that don't affect treatment delivery
- Safety systems would prevent actual clinical impact

**Failure Modes**:

| Failure Mode | Variant | Severity | Rationale |
|--------------|---------|----------|-----------|
| `invalid_angles` | `collimator` | 0.6 | Likely data entry error; linac has safety interlocks |
| `invalid_angles` | `gantry` | 0.7 | Same as above, gantry has mechanical limits |
| `corrupt_mlc_data` | `odd_bytes` | 0.6 | Parsing issue, may not affect actual treatment |
| `invalid_offset_data` | `invalid_state` | 0.9 | Workflow tracking issue, not patient safety |

**Recommended Action**:
- Log for trending analysis
- Fix during routine data quality review
- No immediate clinical intervention required

---

### 1.0-1.5 - Medium Severity
**Impact**: Moderate administrative/workflow impact, low clinical risk

**Characteristics**:
- Data integrity issues
- Workflow tracking problems
- Billing or record-keeping errors
- May cause confusion but low patient safety risk

**Failure Modes**:

| Failure Mode | Variant | Severity | Rationale |
|--------------|---------|----------|-----------|
| `duplicate_treatments` | `same_field` | 1.0 | Likely duplicate recording, billing issue |
| `timestamp_inconsistencies` | `future_timestamp` | 1.0 | Clock sync issue, temporal reference problem |
| `invalid_offset_data` | `invalid_type` | 1.0 | Data quality, not affecting positioning |
| `timestamp_inconsistencies` | `edit_before_create` | 1.2 | Logical impossibility, data integrity |
| `null_required_fields` | `timestamp` | 1.2 | Temporal reference lost |
| `timestamp_inconsistencies` | `negative_duration` | 1.3 | Data integrity issue |
| `duplicate_treatments` | `different_field` | 1.5 | Conflicting records, unclear which correct |
| `meterset_inconsistency` | `mismatch_with_cp` | 1.5 | Documentation inconsistency |
| `corrupt_mlc_data` | `random_bytes` | 1.5 | Unpredictable effects, moderate concern |

**Recommended Action**:
- Review within 24 hours
- Investigate root cause
- Implement corrective action to prevent recurrence
- May require data correction

---

### 1.8-2.3 - High Severity
**Impact**: High clinical risk, potential for incorrect dose delivery

**Characteristics**:
- Dose delivery errors
- Treatment integrity compromised
- MLC or control point corruption affecting dose
- Data losses that impact dose calculation

**Failure Modes**:

| Failure Mode | Variant | Severity | Rationale |
|--------------|---------|----------|-----------|
| `null_required_fields` | `other` | 1.0 | General data corruption, queries may fail |
| `mlc_leaf_count_mismatch` | `vs_plan` | 1.8 | Doesn't match treatment plan |
| `missing_control_points` | `single_gap` | 1.8 | One missing segment |
| `mlc_leaf_count_mismatch` | `vs_machine` | 2.0 | Doesn't match machine config |
| `corrupt_mlc_data` | `out_of_range` | 2.0 | Dose calculation errors likely |
| `mlc_leaf_count_mismatch` | `intra_field` | 2.1 | Inconsistent within field |
| `missing_control_points` | `multiple_gaps` | 2.2 | Severe data loss |
| `corrupt_mlc_data` | `negative_gap` | 2.2 | Physical impossibility, delivery would fail |
| `orphaned_records` | `field_id` | 2.3 | Treatment plan reference invalid |
| `null_required_fields` | `fld_id` | 2.3 | Treatment plan reference lost |

**Recommended Action**:
- **Immediate review required** (within 1-4 hours)
- Hold patient treatment if not yet delivered
- Verify treatment against independent records (DICOM plan, machine logs)
- Clinical physicist review mandatory
- Document investigation and resolution

---

### 2.5-3.0 - Critical Severity
**Impact**: **Critical patient safety risk, immediate intervention required**

**Characteristics**:
- Wrong patient association
- Wrong treatment position (geometric miss)
- Wrong dose (under/over by significant margin)
- Data corruption that could lead to treating wrong patient

**Failure Modes**:

| Failure Mode | Variant | Severity | Rationale |
|--------------|---------|----------|-----------|
| `missing_control_points` | `all_deleted` | 2.5 | Complete data loss, no treatment record |
| `orphaned_records` | `site_id` | 2.5 | Treatment site reference invalid |
| `meterset_inconsistency` | `extreme_value` | 2.6 | Wrong dose delivered, patient harm possible |
| `orphaned_records` | `patient_id` | 2.8 | **CRITICAL**: Treatment may be for wrong patient |
| `null_required_fields` | `pat_id` | 2.7 | **CRITICAL**: Patient identification lost |
| `invalid_offset_data` | `extreme_values` | 2.7 | **CRITICAL**: Wrong treatment position, geometric miss |
| `meterset_inconsistency` | `negative_meterset` | 2.8 | **CRITICAL**: Physically impossible, safety interlock failure |

**Recommended Action**:
- **IMMEDIATE action required** (within 15-60 minutes)
- **STOP**: Halt any pending treatments for this patient
- Page on-call clinical physicist immediately
- Verify patient identity through independent means
- Cross-check with:
  - Original DICOM RT Plan
  - Treatment management system (ARIA/Mosaiq)
  - Machine interlock logs
  - Portal images / CBCT
- Document incident per institutional policy
- May require:
  - Treatment re-planning
  - Additional imaging
  - Physician notification
  - Incident reporting to safety committee

---

## Severity Assignment Rationale

### Patient Safety First
Severities are assigned with patient safety as the primary consideration:

**Critical (2.5-3.0)**: Direct patient harm possible
- Wrong patient → radiation to wrong person
- Wrong position → miss target, hit critical structures
- Wrong dose → under-treatment (tumor) or over-treatment (toxicity)

**High (1.8-2.3)**: Dose delivery integrity compromised
- MLC errors → incorrect dose distribution
- Missing control points → incomplete treatment record
- May affect tumor control or normal tissue sparing

**Medium (1.0-1.5)**: Workflow and documentation
- Duplicate records → billing issues, confusion
- Timestamp errors → audit trail compromised
- Low probability of patient harm

**Low (0.5-0.8)**: Data quality only
- Parsing errors → may auto-correct or be caught by validation
- Invalid angles → safety interlocks prevent delivery
- No realistic patient harm scenario

### Variant-Specific Severity

Many failure modes have different severities based on the specific corruption:

**Example: `corrupt_mlc_data`**
- `odd_bytes` (0.6): Parsing issue, may decode correctly or fail safe
- `random_bytes` (1.5): Unpredictable, could decode to anything
- `out_of_range` (2.0): Dose calculation errors, wrong DVH
- `negative_gap` (2.2): Physical impossibility, delivery fails or delivers wrong shape

**Example: `invalid_offset_data`**
- `invalid_type` (1.0): Data quality, doesn't affect actual shift
- `invalid_state` (0.9): Workflow tracking, not safety
- `future_study_time` (1.1): Temporal tracking issue
- `extreme_values` (2.7): **CRITICAL** - 10cm shift = geometric miss

---

## EBM Training with Severity

### Loss Function

The EBM is trained with Mean Squared Error (MSE) loss between predicted energy and target severity:

```
L = MSE(E(x), severity)

where:
  E(x) = predicted energy for input features x
  severity = 0.0 for normal, 0.5-3.0 for anomalies
```

This encourages the model to:
1. Output low energy (~0.1-0.3) for normal data
2. Output energy matching severity for anomalies
3. Distinguish between severity levels (not just binary)

### Benefits

**1. Risk Stratification**
```python
if energy > 2.5:
    alert_critical()  # Page physicist
elif energy > 1.8:
    alert_high()  # Review within hours
elif energy > 1.0:
    log_medium()  # Review daily
elif energy > 0.4:
    log_low()  # Trend analysis
```

**2. Reduced Alert Fatigue**
- Only critical/high severity issues trigger immediate alerts
- Medium/low severity logged for batch review
- Physics team sees what matters most

**3. Continuous Assessment**
- Not just "anomaly yes/no"
- Provides quantitative risk score
- Can track severity trends over time

**4. Prioritized Workflow**
- Triage anomalies by severity
- Address critical issues first
- Schedule medium/low severity for routine review

---

## Validation and Calibration

### Expected Energy Distributions

After training, the EBM should produce:

| Category | Expected Energy | Std Dev | Example Count (N=1000) |
|----------|----------------|---------|------------------------|
| Normal | 0.1-0.3 | ±0.1 | 500 |
| Low | 0.5-0.8 | ±0.15 | 150 |
| Medium | 1.0-1.5 | ±0.2 | 150 |
| High | 1.8-2.3 | ±0.25 | 130 |
| Critical | 2.5-3.0 | ±0.3 | 70 |

### Calibration Metrics

**Mean Absolute Error (MAE)**: Target < 0.3
- Average absolute difference between predicted energy and true severity
- Lower is better

**Category Accuracy**: Target > 90%
- Percentage correctly classified into severity category
- Binary classification (normal vs anomaly): Target > 95%

**Energy Separation**:
```
mean(critical_energy) - mean(high_energy) > 0.5
mean(high_energy) - mean(medium_energy) > 0.5
mean(medium_energy) - mean(low_energy) > 0.3
mean(low_energy) - mean(normal_energy) > 0.2
```

### Clinical Validation

Before deployment:
1. **Retrospective Analysis**: Apply to historical data, verify physics review matches severity
2. **Prospective Pilot**: Run in shadow mode, compare alerts to actual incidents
3. **Threshold Tuning**: Adjust based on local false positive tolerance
4. **Continuous Monitoring**: Track performance metrics over time

---

## Implementation Examples

### Basic Detection

```python
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import AdversarialTrainer
from pymedphys._mcp.mosaiq_failure_modes.severity import categorize_severity

trainer = AdversarialTrainer(model_path="mosaiq_ebm.pt")
trainer.load_checkpoint()

# Predict on new treatment
result = trainer.predict(features)
energy = result['energies'][0]

category = categorize_severity(energy)
print(f"Energy: {energy:.2f} ({category})")
```

### Risk-Stratified Alerting

```python
def process_treatment_alert(ttx_id: int, energy: float):
    """Handle treatment anomaly based on severity."""
    if energy >= 2.5:
        # Critical: Immediate action
        page_physicist(ttx_id, energy, urgency="STAT")
        halt_pending_treatments(patient_id)
        create_incident_report(ttx_id, severity="CRITICAL")

    elif energy >= 1.8:
        # High: Urgent review
        email_physics_team(ttx_id, energy, urgency="HIGH")
        flag_for_review(ttx_id, deadline="4_hours")

    elif energy >= 1.0:
        # Medium: Next day review
        add_to_qa_queue(ttx_id, priority="MEDIUM")
        log_anomaly(ttx_id, energy)

    elif energy >= 0.4:
        # Low: Trending only
        log_anomaly(ttx_id, energy)
        update_trending_dashboard(ttx_id)
```

### Batch Analysis

```python
def daily_qa_report():
    """Generate daily QA report with severity breakdown."""
    # Get all treatments from last 24 hours
    treatments = query_recent_treatments(hours=24)

    severity_counts = {"normal": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}

    for treatment in treatments:
        features = extract_all_features(cursor, treatment.ttx_id)
        result = trainer.predict(features.reshape(1, -1))
        category = categorize_severity(result['energies'][0])
        severity_counts[category] += 1

    # Generate report
    print(f"Daily QA Report ({date.today()})")
    print(f"  Normal: {severity_counts['normal']}")
    print(f"  Low: {severity_counts['low']}")
    print(f"  Medium: {severity_counts['medium']}")
    print(f"  High: {severity_counts['high']}")
    print(f"  Critical: {severity_counts['critical']}")

    if severity_counts['critical'] > 0:
        print("\n**CRITICAL ISSUES DETECTED - IMMEDIATE REVIEW REQUIRED**")
```

---

## Severity Scale Summary Table

| Category | Range | Count (example) | Response Time | Action |
|----------|-------|----------------|---------------|--------|
| Normal | 0.0-0.4 | 500 | None | Standard workflow |
| Low | 0.5-0.8 | 150 | Days | Trending, routine QA |
| Medium | 1.0-1.5 | 150 | 24 hours | Investigate, document |
| High | 1.8-2.3 | 130 | 1-4 hours | Clinical physicist review |
| Critical | 2.5-3.0 | 70 | 15-60 min | IMMEDIATE intervention |

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Author**: Claude Code (with clinical input from domain experts)
**Approved By**: [Pending physics team review]
