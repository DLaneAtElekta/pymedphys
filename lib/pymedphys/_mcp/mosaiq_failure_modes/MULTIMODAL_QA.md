# Multi-Modal QA Framework for Anomaly Detection

## Overview

**Key Principle**: Independent QA strands provide stronger validation than any single data source.

This framework integrates **four QA modalities** to create a robust anomaly detection system:

1. **Mosaiq Database** - Treatment planning and recording system
2. **TRF Files** - Machine log files from Elekta linacs (ground truth delivery)
3. **Portal Dosimetry** - EPID-based dose measurements
4. **Phantom Dosimetry** - Physical QA measurements

### Why Multi-Modal?

**Single-source QA limitations**:
- Mosaiq only: Database corruption may not reflect actual delivery
- TRF only: Machine logs don't capture dose accuracy
- Portal only: EPID measurements have uncertainties
- Phantom only: Periodic measurements miss daily variations

**Multi-modal advantages**:
- **Independent verification**: Cross-validate across sources
- **Failure mode localization**: Identify where problem occurred
- **Higher confidence**: Agreement across modalities increases certainty
- **Comprehensive coverage**: Detect issues invisible to single modality

---

## QA Strand Descriptions

### 1. Mosaiq Database Analysis

**What it provides**:
- Treatment plan parameters (MLC, angles, MU)
- Patient demographics and treatment site
- Workflow and timestamp data
- Foreign key relationships

**Strengths**:
- Comprehensive treatment record
- Links to clinical context
- Temporal tracking

**Weaknesses**:
- Database corruption possible
- May not reflect actual delivery
- Third-party writes can corrupt data

**Features extracted**: 72 features
- MLC positions, angles, control points
- Timestamps, foreign keys, meterset
- See `ebm_features.py`

---

### 2. TRF (Treatment Record File) Analysis

**What it provides**:
- **Ground truth** machine log data
- Actual MLC positions delivered
- Actual gantry/collimator angles
- Actual MU delivered
- Beam hold events and interlocks

**Strengths**:
- Independent of Mosaiq
- Recorded by linac control system
- Cannot be retroactively modified
- High temporal resolution

**Weaknesses**:
- Doesn't measure dose directly
- Machine-specific format
- May not be available for older treatments

**Features extracted**: 18 features

| Feature Category | Count | Examples |
|------------------|-------|----------|
| MU comparison | 4 | `trf_mu_delivered`, `trf_mu_mosaiq_diff` |
| MLC comparison | 4 | `trf_mlc_max_deviation`, `trf_mlc_rms_deviation` |
| Angle comparison | 3 | `trf_gantry_max_deviation` |
| Beam interruptions | 2 | `trf_beam_hold_count`, `trf_beam_hold_detected` |
| Duration | 3 | `trf_duration_seconds`, `trf_duration_mosaiq_diff` |
| Integrity | 2 | `trf_file_readable`, `trf_data_complete` |

**Key Cross-Validation**:
```python
# If Mosaiq says 200 MU but TRF shows 180 MU delivered
if abs(trf_mu - mosaiq_mu) / mosaiq_mu > 0.01:  # >1% difference
    flag_as_critical()  # Mosaiq-TRF disagreement
```

---

### 3. Portal Dosimetry (EPID-based)

**What it provides**:
- **Measured dose** at EPID detector
- 2D dose distribution
- Gamma analysis vs predicted
- In vivo dosimetry capability

**Strengths**:
- Direct dose measurement
- Every field can be measured
- Detects delivery errors (MLC, output)
- Independent of Mosaiq/TRF

**Weaknesses**:
- EPID response differs from tissue
- Requires calibration
- May not detect all failure modes
- Image quality varies

**Features extracted**: 11 features

| Feature Category | Count | Examples |
|------------------|-------|----------|
| Dose metrics | 3 | `portal_mean_dose`, `portal_max_dose` |
| Uniformity | 3 | `portal_central_dose`, `portal_uniformity_cov` |
| Gamma analysis | 4 | `portal_gamma_pass_rate`, `portal_gamma_mean` |
| Integrity | 1 | `portal_file_readable` |

**Gamma Criteria**: Typically 3%/3mm (global) or 2%/2mm (local)

**Key Cross-Validation**:
```python
# Portal gamma fails but Mosaiq/TRF agree
if portal_gamma_pass_rate < 95 and trf_mu_diff < 1.0:
    flag_as_delivery_error()  # Problem is in delivery, not planning
```

---

### 4. Phantom Dosimetry

**What it provides**:
- End-to-end system verification
- Absolute dose accuracy
- IMRT/VMAT QA (ArcCHECK, Delta4, etc.)
- Output factors and dose linearity

**Strengths**:
- **Gold standard** for dose verification
- Tests entire delivery chain
- Well-established tolerances
- Regulatory requirement (TG-142, TG-218)

**Weaknesses**:
- Periodic (not every patient)
- Doesn't catch patient-specific errors
- Time and resource intensive
- May not represent all delivery conditions

**Features extracted**: 14 features

| Feature Category | Count | Examples |
|------------------|-------|----------|
| Absolute dose | 5 | `phantom_dose_measured`, `phantom_dose_diff_percent` |
| Gamma analysis | 3 | `phantom_gamma_pass_rate`, `phantom_gamma_mean` |
| Output factors | 4 | `phantom_output_factor_measured`, `phantom_output_factor_diff_percent` |
| Availability | 2 | `phantom_measurements_available`, `phantom_measurement_count` |

**Typical Frequencies**:
- Daily: Output constancy (ion chamber)
- Monthly: IMRT/VMAT QA (composite plan)
- Annual: Full commissioning tests

**Key Cross-Validation**:
```python
# Phantom QA fails for machine
if phantom_gamma_pass_rate < 90:
    # Check if recent patient TRF/portal also failing
    if patient_portal_gamma < 95 or patient_trf_mu_diff > 2.0:
        flag_as_systemic_issue()  # Machine-wide problem
```

---

## Cross-Validation Strategy

### Feature Extraction

**9 cross-validation features** assess agreement across modalities:

```python
cv_features = {
    # Data availability
    'cv_mosaiq_available': 1.0,
    'cv_trf_available': 1.0,
    'cv_portal_available': 1.0,
    'cv_phantom_available': 0.0,  # Not available for every patient
    'cv_data_sources_count': 3.0,

    # Consistency flags
    'cv_mu_consistent': 1.0,      # Mosaiq-TRF MU agree
    'cv_gamma_consistent': 1.0,   # Portal-phantom gamma agree

    # Multi-modal failure detection
    'cv_multi_modal_failure': 0.0,  # ≥2 modalities failing

    # Confidence score
    'cv_confidence_score': 1.0,   # High when ≥3 modalities available and agree
}
```

### Decision Matrix

| Mosaiq | TRF | Portal | Phantom | Interpretation | Severity | Action |
|--------|-----|--------|---------|----------------|----------|--------|
| ✓ | ✓ | ✓ | ✓ | All agree - normal | 0.1 | None |
| ✓ | ✓ | ✓ | ✗ | Patient delivery OK | 0.2 | Check phantom setup |
| ✓ | ✗ | - | - | Mosaiq-TRF mismatch | 2.5 | **CRITICAL**: Data corruption |
| ✓ | ✓ | ✗ | - | Delivery issue | 1.8 | Review portal image |
| ✗ | - | - | - | Mosaiq corruption | 2.7 | **CRITICAL**: Database issue |
| ✓ | ✓ | ✗ | ✗ | Systemic delivery | 2.3 | Machine calibration |
| ✗ | ✗ | ✗ | ✗ | Complete failure | 3.0 | **CRITICAL**: All systems |

**Key**: ✓ = passes, ✗ = fails, - = not available

### Confidence Scoring

**High Confidence (1.0)**:
- ≥3 modalities available
- All agree within tolerance
- Action: Trust the data

**Medium Confidence (0.7)**:
- 2 modalities available
- Both agree
- Action: Proceed but log for trending

**Low Confidence (0.4)**:
- Only 1 modality available
- No independent verification
- Action: Manual review recommended

**No Confidence (0.0)**:
- No data available or major inconsistencies
- Action: **Stop**, investigate immediately

---

## Failure Mode Localization

Multi-modal QA helps identify **where** the failure occurred:

### Database Corruption (Mosaiq)

**Signature**:
- Mosaiq features show anomalies
- TRF shows normal delivery
- Portal dosimetry passes

**Example**:
```
Mosaiq: 500 MU, negative MLC gap
TRF: 200 MU delivered, normal MLC
Portal: Gamma 98% pass rate
→ Diagnosis: Mosaiq database corruption, actual delivery was correct
→ Severity: High (2.3) - data integrity compromised
```

### Delivery Error (Linac)

**Signature**:
- Mosaiq and TRF agree on plan
- Portal dosimetry fails
- Phantom QA (recent) shows issues

**Example**:
```
Mosaiq: 200 MU, MLC positions correct
TRF: 200 MU delivered, MLC matches Mosaiq
Portal: Gamma 85% pass rate
Phantom: Last monthly QA gamma 88%
→ Diagnosis: Linac delivery problem (MLC calibration drift)
→ Severity: High (2.0) - dose delivery accuracy compromised
```

### Planning Error (TPS/Mosaiq Interface)

**Signature**:
- Mosaiq shows incorrect parameters
- TRF matches Mosaiq (delivered as planned)
- Portal fails because delivered ≠ intended

**Example**:
```
Mosaiq: 150 MU (should be 200 MU)
TRF: 150 MU delivered (matches Mosaiq)
Portal: Gamma 75% vs intended plan
→ Diagnosis: Transfer error from TPS to Mosaiq
→ Severity: Critical (2.8) - wrong dose delivered
```

### Workflow Error (Wrong Field Delivered)

**Signature**:
- Mosaiq shows Field A delivered
- TRF shows Field B parameters
- Portal shows Field B pattern

**Example**:
```
Mosaiq: Field "AP" (0° gantry, specific MLC)
TRF: Gantry 180°, different MLC pattern
Portal: Dose pattern matches PA field
→ Diagnosis: Wrong field delivered (human error)
→ Severity: Critical (3.0) - wrong treatment delivered
```

---

## EBM Architecture for Multi-Modal Inputs

### Input Layer Expansion

**Total Features**: 72 (Mosaiq) + 18 (TRF) + 11 (Portal) + 14 (Phantom) + 9 (CV) = **124 features**

**Network Architecture**:
```
Input (124 features)
    ↓
Dense(256) + LayerNorm + ReLU + Dropout(0.2)  [Larger to handle more features]
    ↓
Dense(128) + LayerNorm + ReLU + Dropout(0.2)
    ↓
Dense(64) + LayerNorm + ReLU + Dropout(0.2)
    ↓
Dense(1) → Energy output
```

### Attention Mechanism (Advanced)

For interpretability, add attention to weight modality importance:

```python
class MultiModalEBM(nn.Module):
    def __init__(self, feature_counts: dict):
        """
        feature_counts = {
            'mosaiq': 72,
            'trf': 18,
            'portal': 11,
            'phantom': 14,
            'cv': 9
        }
        """
        self.modality_encoders = nn.ModuleDict({
            modality: nn.Linear(count, 32)
            for modality, count in feature_counts.items()
        })

        self.attention = nn.MultiheadAttention(32, num_heads=4)
        self.output = nn.Linear(32 * 5, 1)  # 5 modalities

    def forward(self, x_dict):
        # Encode each modality
        encoded = [
            encoder(x_dict[modality])
            for modality, encoder in self.modality_encoders.items()
        ]

        # Apply attention (learn which modalities are most informative)
        attended, weights = self.attention(encoded, encoded, encoded)

        # Combine and predict
        combined = torch.cat(attended, dim=-1)
        energy = self.output(combined)
        return energy, weights  # Return attention weights for interpretability
```

This allows the model to learn which modalities are most important for each failure type.

---

## Implementation Examples

### Basic Multi-Modal Feature Extraction

```python
from pymedphys._mcp.mosaiq_failure_modes.multimodal_features import (
    extract_all_multimodal_features,
    MULTIMODAL_FEATURE_NAMES
)

# Extract features for a treatment
features = extract_all_multimodal_features(
    ttx_id=12345,
    cursor=mosaiq_cursor,
    trf_path=Path("/data/trf/12345.trf"),
    portal_image_path=Path("/data/portal/12345.dcm"),
    portal_reference_path=Path("/data/portal/12345_ref.dcm"),
    phantom_measurements={
        'gamma_pass_rate': 97.5,
        'center_dose': 100.3,
        'output_factor': 1.001,
    },
    phantom_expected={
        'center_dose': 100.0,
        'output_factor': 1.000,
    }
)

# Check cross-validation
if features['cv_multi_modal_failure'] > 0:
    print("WARNING: Multiple QA modalities failing!")
    print(f"Confidence: {features['cv_confidence_score']:.2f}")
```

### Risk-Stratified Alerting with Multi-Modal Context

```python
def evaluate_treatment_with_context(ttx_id, energy, features):
    """Evaluate treatment with multi-modal context."""

    # Base severity from energy
    base_severity = categorize_severity(energy)

    # Escalate if multi-modal failure
    if features['cv_multi_modal_failure'] > 0:
        severity = "critical"  # Override to critical
        message = "Multiple independent QA modalities failing"

    # High confidence in anomaly detection
    elif features['cv_confidence_score'] > 0.8 and energy > 2.5:
        severity = "critical"
        message = "High-confidence critical anomaly"

    # Low confidence - may be false alarm
    elif features['cv_confidence_score'] < 0.5 and energy > 1.8:
        severity = "medium"  # Downgrade due to low confidence
        message = "Potential anomaly, but low confidence (limited data)"

    else:
        severity = base_severity
        message = f"Standard {severity} severity"

    # Localize failure
    if features['cv_mosaiq_available'] and not features['cv_trf_available']:
        location = "Cannot verify with TRF - Mosaiq only"
    elif features['trf_mu_tolerance_exceeded'] and not features['portal_gamma_tolerance_exceeded']:
        location = "Mosaiq-TRF mismatch, but portal OK → Database corruption"
    elif features['portal_gamma_tolerance_exceeded'] and not features['trf_mu_tolerance_exceeded']:
        location = "Portal gamma failure, but TRF OK → Delivery issue"
    else:
        location = "Multiple modalities affected → Systemic issue"

    return {
        'severity': severity,
        'message': message,
        'location': location,
        'confidence': features['cv_confidence_score']
    }
```

### Daily QA Report with Multi-Modal Summary

```python
def generate_daily_multimodal_qa_report():
    """Generate daily QA report showing multi-modal coverage."""

    treatments = query_recent_treatments(hours=24)

    report = {
        'total': len(treatments),
        'mosaiq_only': 0,
        'mosaiq_trf': 0,
        'mosaiq_trf_portal': 0,
        'full_coverage': 0,  # All 4 modalities
        'anomalies_detected': 0,
        'multi_modal_failures': 0,
    }

    for treatment in treatments:
        features = extract_all_multimodal_features(...)

        # Track data coverage
        sources = features['cv_data_sources_count']
        if sources == 1:
            report['mosaiq_only'] += 1
        elif sources == 2:
            report['mosaiq_trf'] += 1
        elif sources == 3:
            report['mosaiq_trf_portal'] += 1
        elif sources == 4:
            report['full_coverage'] += 1

        # Detect anomalies
        energy = predict_energy(features)
        if energy > 0.4:
            report['anomalies_detected'] += 1

        if features['cv_multi_modal_failure']:
            report['multi_modal_failures'] += 1

    # Print report
    print(f"Daily QA Report - {date.today()}")
    print(f"Total treatments: {report['total']}")
    print(f"\nData Coverage:")
    print(f"  Mosaiq only: {report['mosaiq_only']} ({report['mosaiq_only']/report['total']*100:.1f}%)")
    print(f"  Mosaiq + TRF: {report['mosaiq_trf']} ({report['mosaiq_trf']/report['total']*100:.1f}%)")
    print(f"  Mosaiq + TRF + Portal: {report['mosaiq_trf_portal']} ({report['mosaiq_trf_portal']/report['total']*100:.1f}%)")
    print(f"  Full coverage (all 4): {report['full_coverage']} ({report['full_coverage']/report['total']*100:.1f}%)")

    print(f"\nAnomalies:")
    print(f"  Total detected: {report['anomalies_detected']}")
    print(f"  Multi-modal failures: {report['multi_modal_failures']}")

    if report['multi_modal_failures'] > 0:
        print("\n**CRITICAL: Multi-modal failures detected - immediate review required**")
```

---

## Benefits Summary

### 1. **Higher Detection Accuracy**
- Independent verification reduces false positives
- Cross-validation increases true positive rate
- Confidence scoring guides response

### 2. **Failure Localization**
- Identify whether issue is in:
  - Database (Mosaiq corruption)
  - Delivery (linac problem)
  - Planning (TPS-Mosaiq transfer)
  - Workflow (wrong field/patient)

### 3. **Reduced Alert Fatigue**
- High-confidence alerts prioritized
- Low-confidence anomalies flagged for review, not paging
- Multi-modal agreement gives strong signal

### 4. **Comprehensive Coverage**
- Mosaiq: Planning intent and workflow
- TRF: Actual delivery parameters
- Portal: Dose accuracy (2D)
- Phantom: System integrity (3D)

### 5. **Regulatory Compliance**
- TG-142: Portal dosimetry and phantom QA
- TG-218: Treatment log file analysis
- Multi-modal approach exceeds minimum requirements

---

## Recommended Implementation Roadmap

### Phase 1: Mosaiq + TRF (Months 1-2)
- Implement TRF feature extraction
- Train EBM with Mosaiq + TRF features (90 total)
- Focus on MU and MLC cross-validation
- **Expected improvement**: 15-20% better anomaly detection

### Phase 2: Add Portal Dosimetry (Months 3-4)
- Integrate portal DICOM images
- Implement gamma analysis features
- Train with 101 features (Mosaiq + TRF + Portal)
- **Expected improvement**: 25-30% better detection

### Phase 3: Add Phantom QA (Months 5-6)
- Link phantom measurements to treatment periods
- Implement cross-validation with patient treatments
- Full 124-feature model
- **Expected improvement**: 35-40% better detection

### Phase 4: Attention Mechanism (Months 7-8)
- Implement multi-head attention for modality weighting
- Interpretability: which modalities contributed to each detection
- Fine-tune on clinical cases

---

## Clinical Validation

Before deployment, validate with retrospective cases:

1. **Known Incidents** (N=20-50):
   - Pull incident reports from last 2 years
   - Extract multi-modal features
   - Verify EBM detects with high energy and correct localization

2. **Clean Cases** (N=500-1000):
   - Random sample of normal treatments
   - Verify low energy and high confidence
   - Establish false positive rate

3. **Borderline Cases** (N=50-100):
   - Cases that required physics review but were OK
   - Check if model appropriately assigns medium severity
   - Tune thresholds based on clinical tolerance

**Success Criteria**:
- Sensitivity ≥ 95% for critical failures (severity ≥ 2.5)
- Specificity ≥ 98% for normal cases (avoid alert fatigue)
- Failure localization accuracy ≥ 85%

---

**Document Version**: 1.0
**Author**: Claude Code with clinical QA domain guidance
**Next Review**: After Phase 1 implementation
