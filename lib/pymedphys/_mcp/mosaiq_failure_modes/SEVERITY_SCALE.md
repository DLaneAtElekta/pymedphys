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

## Malicious Intent Severity Factors

### Overview

**All malicious actor failure modes have severity ≥ 2.5** (critical range) regardless of technical characteristics. The severity scale for malicious failures considers multiple factors beyond clinical impact:

### Severity Factors for Malicious Failures

| Factor | Weight | Rationale |
|--------|--------|-----------|
| **Clinical Impact** | 30% | Actual patient harm potential |
| **Intent to Harm** | 30% | Deliberate malice vs accidental error |
| **Evasion Sophistication** | 20% | How difficult to detect |
| **Persistence** | 10% | Can attack continue over time |
| **Attribution Difficulty** | 10% | How hard to identify perpetrator |

### Why Malicious = Critical Severity

1. **Deliberate Intent**: Unlike accidental errors, malicious attacks are **designed to cause harm**
2. **Evasion Built-In**: Attacks are crafted to **avoid detection**, making them more dangerous
3. **Systematic Nature**: Malicious actors can **repeat attacks** on multiple patients
4. **Trust Violation**: Indicates **compromised security** requiring incident response
5. **Legal/Regulatory**: Potential for **criminal prosecution** and **regulatory sanctions**

### Malicious Failure Mode Severity Breakdown

#### Severity 2.5-2.8 (Critical - Lower Range)

**Characteristics**: High clinical risk with some detectability

| Failure Mode | Base Severity | Variants | Rationale |
|--------------|--------------|----------|-----------|
| `collimator_jaw_manipulation` | 2.6 | jaw_asymmetry (2.7), field_size_reduction (2.8) | Affects field geometry, but portal imaging can detect |
| `audit_trail_manipulation` | 2.7 | audit_disable (2.7), timestamp_forgery (2.8) | Enables other attacks, but external logs can catch |
| `gradual_parameter_drift` | 2.8 | gantry_drift (2.7), mu_creep (2.8), isocenter_shift (2.9) | Longitudinal analysis can detect trends |
| `selective_fraction_sabotage` | 2.8 | periodic (2.7), random (2.9), calendar (2.8) | Per-fraction QA can identify anomalies |
| `time_delayed_corruption` | 2.8 | db_trigger (2.7), scheduled (2.9), conditional (2.8) | Pre-treatment verification can catch |

**Response**: Immediate physicist review, enhanced monitoring, forensic investigation

---

#### Severity 2.9-3.0 (Critical - Upper Range)

**Characteristics**: Maximum clinical risk and/or extreme evasion sophistication

| Failure Mode | Base Severity | Variants | Rationale |
|--------------|--------------|----------|-----------|
| `subtle_dose_escalation` | 2.9 | random_walk (2.9), within_tolerance (2.9) | Statistical camouflage, requires CUSUM detection |
| `coordinated_multifield_attack` | 2.9 | compensatory_errors (3.0), geometric_shift (2.9) | Evades field-level QA, needs 3D reconstruction |
| `statistical_camouflage` | 2.9 | within_tolerance_bias (2.9), outlier_avoidance (2.9) | Designed to evade statistical detection |
| `field_aperture_manipulation` | 2.9 | systematic_shift (3.0), margin_erosion (2.9) | Direct geometric miss, high patient harm |
| `targeted_patient_selection` | 3.0 | demographic_targeting (3.0), random_sampling (2.9) | **Maximum severity**: Vulnerable populations, difficult population-level detection |

**Special Cases (Severity 3.0)**:
- **Log Deletion** (3.0): Removes all evidence
- **Systematic MLC Shift** (3.0): Guaranteed geometric miss
- **Compensatory Errors** (3.0): Sophisticated multi-field coordination
- **Fraction-Selective Overdose** (3.0): Deliberate targeting specific fractions
- **Demographic Targeting** (3.0): Ethical violation + detection evasion

**Response**: IMMEDIATE treatment halt, incident command activation, law enforcement notification, regulatory reporting

---

### Detection Difficulty vs Clinical Impact Matrix

```
                Low Detection Difficulty              High Detection Difficulty
                (Standard QA catches it)               (Needs advanced analytics)
High Clinical ┌─────────────────────────┬─────────────────────────────────┐
Impact        │ Severity: 2.5-2.7       │ Severity: 2.9-3.0               │
              │                         │                                 │
              │ Examples:               │ Examples:                       │
              │ - Jaw manipulation      │ - Statistical camouflage        │
              │ - Parameter drift       │ - Multi-field coordinated       │
              │ - Audit disable         │ - Targeted patient selection    │
              │                         │ - Subtle dose escalation        │
              │ Response: Hours         │ Response: Minutes               │
              └─────────────────────────┴─────────────────────────────────┘

Low Clinical  ┌─────────────────────────┬─────────────────────────────────┐
Impact        │ Severity: 0.5-1.5       │ Severity: 1.8-2.3               │
              │ (Accidental errors)     │ (Sophisticated accidents)       │
              │                         │                                 │
              │ Not applicable for      │ Not applicable for              │
              │ malicious actors        │ malicious actors                │
              │ (intent → ↑ severity)   │ (intent → ↑ severity)           │
              └─────────────────────────┴─────────────────────────────────┘
```

**Key Insight**: The combination of **high clinical impact** + **high evasion sophistication** results in **maximum severity (3.0)**, requiring the most urgent response.

---

### Severity Escalation Due to Malicious Intent

**Example**: Consider MLC aperture shift

| Scenario | Severity | Rationale |
|----------|----------|-----------|
| **Accidental**: Planning software bug causes 1mm systematic shift | 1.8 | High severity, but detectable via portal imaging |
| **Malicious**: Attacker deliberately shifts aperture 3mm to avoid target | 3.0 | **Critical**: Deliberate harm + designed to appear normal |

**Difference**: Same technical failure, but malicious intent:
- ✓ Systematic across all fields (coordinated)
- ✓ Carefully chosen magnitude (evades tolerance checks)
- ✓ Targets vulnerable patients (selective)
- ✓ May manipulate audit trail (cover tracks)

**Result**: Severity escalated from 1.8 → 3.0 due to malicious factors

---

### Response Time Requirements for Malicious Failures

| Severity Range | Response Time | Actions Required |
|----------------|--------------|------------------|
| **2.5-2.6** | 30-60 min | • Physicist review<br>• Enhanced QA next fraction<br>• Forensic investigation initiated |
| **2.7-2.8** | 15-30 min | • Same as above<br>• Incident commander assigned<br>• Security team notified |
| **2.9-3.0** | **IMMEDIATE** (< 15 min) | • **Treatment halt for all affected patients**<br>• **Emergency physicist review**<br>• **Incident command activated**<br>• **Law enforcement notified**<br>• **Regulatory reporting (within hours)**<br>• **Forensic data preservation** |

---

### Legal and Regulatory Considerations

#### Criminal Liability

Malicious modification of radiotherapy data may constitute:
- **Assault with intent to injure** (if patient harmed)
- **Attempted murder** (if intent to kill)
- **Computer fraud and abuse** (unauthorized database access)
- **HIPAA violations** (if patient data accessed)

**Severity 3.0 failures should trigger law enforcement notification.**

#### Regulatory Reporting

In the United States:
- **FDA**: Medical device adverse event reporting (MedWatch)
- **State radiation control program**: Immediate notification
- **Joint Commission**: Sentinel event reporting
- **NRC**: If radioactive materials involved

**Timeline**: Severity 3.0 failures require regulatory notification within **24 hours**.

---

### Training the EBM with Malicious Intent

#### Challenge: Malicious Failures are Rare

- **Accidental errors**: Common, plentiful training data
- **Malicious attacks**: Rare (hopefully zero in real data)

**Solution**: Use MCP server to generate synthetic malicious examples

#### Severity-Balanced Training

Ensure EBM learns the full severity spectrum:

```python
# Training data distribution
severity_distribution = {
    "normal (0.0-0.4)": 1000,      # 50%
    "low (0.5-0.8)": 300,          # 15%
    "medium (1.0-1.5)": 300,       # 15%
    "high (1.8-2.3)": 200,         # 10%
    "critical_accidental (2.5)": 100,  # 5%
    "critical_malicious (2.7-3.0)": 100  # 5% - synthetic via MCP
}
```

**Key**: Even though real malicious attacks are rare, EBM must learn to recognize them → Use MCP server to generate realistic malicious examples

---

### Confidence Scoring for Malicious Detection

Not all high-energy detections are malicious. Use confidence scoring:

```python
def assess_malicious_probability(features, energy, multi_modal_data):
    """Estimate probability that high energy is malicious vs accidental."""

    confidence_factors = {}

    # 1. Multiple independent anomalies (suspicious)
    if count_anomaly_types(features) >= 3:
        confidence_factors['multiple_anomalies'] = 0.8

    # 2. Statistical signature (designed evasion)
    if detect_outlier_avoidance(features):
        confidence_factors['statistical_evasion'] = 0.7

    # 3. Temporal patterns (selective attacks)
    if detect_temporal_clustering(features):
        confidence_factors['temporal_pattern'] = 0.6

    # 4. Audit trail anomalies
    if detect_audit_manipulation(features):
        confidence_factors['audit_issues'] = 0.9

    # 5. Cross-patient patterns
    if detect_demographic_bias(features):
        confidence_factors['targeting'] = 0.9

    # Combine factors (Bayesian update)
    malicious_probability = bayesian_combine(confidence_factors)

    return {
        'energy': energy,
        'malicious_probability': malicious_probability,
        'confidence_factors': confidence_factors,
        'recommendation': get_recommendation(energy, malicious_probability)
    }
```

**Output Example**:
```
Energy: 2.8 (Critical)
Malicious Probability: 0.85 (High confidence)
Factors:
  - Multiple anomalies detected (MLC + MU + position)
  - Statistical evasion signature present
  - Audit trail timestamp inconsistencies

RECOMMENDATION: IMMEDIATE INCIDENT RESPONSE
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
