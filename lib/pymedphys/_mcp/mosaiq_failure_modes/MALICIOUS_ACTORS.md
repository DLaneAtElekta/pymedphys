# Malicious Actor Failure Modes: Detection and Mitigation

## Overview

This document describes intentional sabotage scenarios where malicious actors with database access attempt to cause patient harm while evading detection. These failure modes represent the most sophisticated threats to radiotherapy data integrity and require advanced detection strategies.

**Critical Context**: All malicious failure modes have severity scores of 2.5-3.0 (critical) because they represent **deliberate intent to harm**. The severity reflects not just clinical impact, but the sophistication required to evade detection.

## Threat Model

### Attacker Capabilities

Malicious actors are assumed to have:

1. **Database Access**: Write permissions to Mosaiq database tables (TrackTreatment, TxFieldPoint, etc.)
2. **Domain Knowledge**: Understanding of radiotherapy workflows, tolerances, and QA procedures
3. **Technical Sophistication**: Ability to craft subtle modifications that evade statistical detection
4. **Persistence**: Multiple opportunities to execute attacks over time
5. **Operational Security**: Awareness of audit trails and desire to avoid attribution

### Attack Objectives

- **Patient Harm**: Deliver incorrect dose, target wrong tissue, create geometric misses
- **Evasion**: Avoid triggering standard QA checks and statistical outlier detection
- **Attribution Avoidance**: Obscure who made changes and when
- **Persistence**: Maintain ability to attack multiple patients over time

### Attack Surface

The primary attack vectors are:

- Direct SQL writes to Mosaiq database
- Compromised third-party integration systems (CBCT, portal imaging, planning systems)
- Insider threats (authorized users with malicious intent)
- Supply chain attacks (compromised software updates)

## Malicious Failure Mode Taxonomy

### Category 1: Temporal Evasion

**Goal**: Avoid detection by spreading attacks across time or targeting specific time windows.

#### 1.1 Subtle Dose Escalation (Severity: 2.9)

**Description**: Gradually increase delivered dose across multiple fractions, staying within per-fraction tolerances while accumulating to harmful cumulative dose.

**Attack Variants**:
- **Linear Ramp** (2.8): Increase MU by 1-2% each fraction (e.g., fraction 1 = 200 MU, fraction 20 = 240 MU)
- **Random Walk** (2.9): Add random ±1-2% noise but with positive bias (mean = +1.5%)
- **Fraction Selective** (3.0): Modify only certain fractions (e.g., every 5th) to avoid longitudinal trends
- **Field Distributed** (2.7): Spread excess dose across multiple fields to avoid per-field detection

**Example Attack**:
```sql
-- Increase MU by 2% for fraction 15 of a 30-fraction treatment
-- Daily QA checks 5% tolerance, so 2% appears normal
UPDATE TxField
SET Meterset = Meterset * 1.02
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TxField
    WHERE Pat_ID1 = '12345' AND Type_Enum = 2  -- Treatment fields only
)
-- Result: 30 fractions × 2% = 60% cumulative overdose
```

**Clinical Impact**:
- 20-60% cumulative dose escalation
- Increased acute toxicity (skin, mucositis)
- Increased late effects (fibrosis, secondary cancers)
- Potential for severe organ damage if OAR tolerance exceeded

**Detection Strategies**:

1. **Cumulative Dose Tracking**:
   ```sql
   -- Track cumulative MU vs prescription
   SELECT
       Pat_ID1,
       SUM(Meterset) AS cumulative_MU,
       MAX(expected_total_MU) AS prescribed_MU,
       (SUM(Meterset) - MAX(expected_total_MU)) / MAX(expected_total_MU) * 100 AS percent_deviation
   FROM TxField tf
   JOIN TrackTreatment tt ON tf.FLD_ID = tt.FLD_ID
   GROUP BY Pat_ID1
   HAVING ABS((SUM(Meterset) - MAX(expected_total_MU)) / MAX(expected_total_MU)) > 0.05
   ```

2. **Statistical Process Control (CUSUM)**:
   - Track cumulative sum of deviations from expected MU
   - Alert when CUSUM exceeds control limit (typically ±5σ)
   - Detects systematic bias even if individual fractions are within tolerance

3. **Longitudinal Trend Analysis**:
   - Fit linear regression to MU vs fraction number
   - Alert if slope significantly different from zero (p < 0.01)
   - Use Mann-Kendall test for non-parametric trend detection

4. **Multi-Modal Verification**:
   - Cross-reference with TRF files (independent machine log)
   - Portal dosimetry for cumulative dose measurement
   - Phantom dosimetry for periodic absolute dose verification

**Required Data**:
- Complete treatment history (all fractions)
- Prescription dose and fractionation
- TRF files for all fractions
- Statistical baseline of normal MU variance

---

#### 1.2 Time-Delayed Corruption (Severity: 2.8)

**Description**: Modify treatment parameters for future fractions, creating temporal separation between attack and manifestation to obscure audit trail.

**Attack Variants**:
- **Scheduled Activation** (2.9): Changes trigger on specific date (e.g., fraction 20)
- **Conditional Execution** (2.8): Modify only if certain conditions met (e.g., weekend treatment)
- **Database Trigger** (2.7): Use SQL triggers to execute modifications at future time
- **Gradual Onset** (2.8): Slow manifestation over 5-10 fractions after modification

**Example Attack**:
```sql
-- Modify fractions 15-20 on Monday for treatments occurring Friday-Wednesday
-- 4-day gap obscures attribution
UPDATE TxFieldPoint
SET Gantry_Ang = Gantry_Ang + 5.0  -- 5° systematic error
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TrackTreatment
    WHERE Create_DtTm BETWEEN '2024-02-16' AND '2024-02-21'  -- Future dates
)
-- Modification made Monday, errors manifest Friday onward
```

**Clinical Impact**:
- Delayed detection means multiple fractions delivered incorrectly
- Difficult attribution due to temporal gap
- May affect multiple patients if timed to treatment schedule

**Detection Strategies**:

1. **Pre-Treatment Integrity Checks**:
   ```python
   def verify_treatment_integrity(pat_id, fld_id, fraction_number):
       """Check data immediately before delivery."""
       current_checksum = calculate_checksum(fld_id)
       baseline_checksum = get_approved_baseline(fld_id)

       if current_checksum != baseline_checksum:
           alert(f"Data modification detected for {pat_id} field {fld_id}")
           return False
       return True
   ```

2. **Version Control and Checksumming**:
   - Store cryptographic hash of approved treatment parameters
   - Verify hash before each fraction delivery
   - Alert on any mismatch (indicates modification)

3. **Audit Trail Temporal Analysis**:
   ```sql
   -- Flag modifications to future fractions
   SELECT
       table_name,
       record_id,
       modified_timestamp,
       user_id,
       fraction_delivery_date
   FROM audit_log
   WHERE modified_timestamp < (fraction_delivery_date - INTERVAL '1 day')
     AND table_name IN ('TxField', 'TxFieldPoint')
   ORDER BY modified_timestamp DESC
   ```

4. **Parameter Locking**:
   - Lock treatment parameters after clinical approval
   - Require multi-person authentication to unlock
   - Log all lock/unlock events with justification

**Required Data**:
- Database audit logs with write timestamps
- Baseline checksums/hashes of approved parameters
- Treatment schedule (planned delivery dates)
- User access and authentication logs

---

#### 1.3 Gradual Parameter Drift (Severity: 2.8)

**Description**: Slowly modify treatment parameters (gantry angle, isocenter, MLC positions) over multiple fractions to evade daily QA while accumulating clinically significant errors.

**Attack Variants**:
- **Gantry Angle Drift** (2.7): ±0.5° per fraction → 10° total shift
- **Isocenter Shift** (2.9): 0.5mm per fraction → 10mm geometric miss
- **MU Creep** (2.8): +1% per fraction → 30% cumulative dose error
- **MLC Drift** (2.7): 0.2mm per fraction → 4mm margin erosion

**Example Attack**:
```sql
-- Gradually shift isocenter 0.5mm anterior each fraction
-- After 20 fractions: 10mm shift (significant geometric miss)
DECLARE @fraction INT = 0
WHILE @fraction < 20
BEGIN
    UPDATE Offset
    SET Anterior_Offset = Anterior_Offset + 0.5  -- mm
    WHERE Pat_ID1 = '12345'
      AND Offset_Type = 2  -- Localization offset
      AND fraction_number = @fraction

    SET @fraction = @fraction + 1
END
```

**Clinical Impact**:
- Cumulative geometric errors (10-20mm)
- Underdosing of target volume
- Overdosing of organs at risk
- May not be detected until end-of-treatment imaging

**Detection Strategies**:

1. **Longitudinal Parameter Tracking**:
   ```python
   def detect_parameter_drift(pat_id, parameter_name):
       """Detect systematic trends in treatment parameters."""
       data = get_parameter_history(pat_id, parameter_name)

       # Fit linear regression
       slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
           data['fraction_number'], data[parameter_name]
       )

       # Alert if significant trend detected
       if p_value < 0.01 and abs(slope) > tolerance:
           alert(f"Parameter drift detected: {parameter_name}")
           return True
       return False
   ```

2. **Change Point Detection**:
   - Use Bayesian change point detection (e.g., PELT algorithm)
   - Identify when systematic drift started
   - Correlate with audit trail to identify responsible user

3. **Cumulative Deviation Analysis**:
   ```python
   def cumulative_deviation(parameter_values, baseline):
       """Track cumulative sum of deviations from baseline."""
       deviations = [v - baseline for v in parameter_values]
       cumsum = np.cumsum(deviations)

       # Alert if cumulative deviation exceeds tolerance
       if abs(cumsum[-1]) > cumulative_tolerance:
           return True
       return False
   ```

4. **Kalman Filtering**:
   - Use state space models to predict expected parameter values
   - Flag observations that deviate from predicted state
   - Accounts for legitimate variance while detecting systematic bias

**Required Data**:
- Complete fraction-by-fraction parameter history
- Initial approved baseline values
- Statistical models of normal inter-fraction variance
- TRF files for independent verification

---

#### 1.4 Selective Fraction Sabotage (Severity: 2.8)

**Description**: Modify only specific fractions (e.g., every 5th, random selection, weekends only) to reduce attack frequency and evade detection.

**Attack Variants**:
- **Periodic Attacks** (2.7): Every Nth fraction (e.g., every 5th)
- **Random Selection** (2.9): Unpredictable fraction choice (e.g., 20% of fractions)
- **Calendar-Based** (2.8): Specific days/times (weekends, night shifts)
- **Staff-Correlated** (2.8): Only when certain staff on duty (insider threat)

**Example Attack**:
```sql
-- Modify every 5th fraction only (6 out of 30 fractions)
-- Random daily QA unlikely to catch periodic pattern
UPDATE TxField
SET Meterset = Meterset * 1.15  -- 15% overdose
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TrackTreatment
    WHERE Pat_ID1 = '12345'
      AND fraction_number % 5 = 0  -- Fractions 5, 10, 15, 20, 25, 30
)
-- Cumulative: (6 × 15% + 24 × 0%) / 30 = 3% average overdose
-- But 6 fractions receive 15% overdose
```

**Clinical Impact**:
- Partial course errors difficult to detect
- Cumulative dose still affected but less obvious
- Creates "hot fractions" with locally high dose

**Detection Strategies**:

1. **Per-Fraction Anomaly Scoring**:
   ```python
   def score_fraction_anomaly(fraction_data, historical_baseline):
       """Assign anomaly score to each fraction."""
       z_score = (fraction_data - historical_baseline['mean']) / historical_baseline['std']

       # Multi-feature anomaly score
       anomaly_score = np.abs(z_score).max()

       if anomaly_score > 3.0:  # 3 sigma threshold
           return True, anomaly_score
       return False, anomaly_score
   ```

2. **Frequency Domain Analysis**:
   - FFT analysis to detect periodic patterns
   - Autocorrelation function to detect temporal structure
   - Alert if significant periodicity detected (p < 0.05)

3. **Temporal Autocorrelation**:
   ```python
   def detect_temporal_autocorrelation(parameter_series):
       """Detect non-random temporal patterns."""
       acf = statsmodels.tsa.stattools.acf(parameter_series, nlags=10)

       # Test for significant autocorrelation
       ljung_box = statsmodels.stats.diagnostic.acorr_ljungbox(parameter_series)

       if ljung_box['lb_pvalue'].min() < 0.05:
           alert("Non-random temporal pattern detected")
           return True
       return False
   ```

4. **Staff Correlation Analysis**:
   ```sql
   -- Detect correlation between anomalies and staff assignments
   SELECT
       staff_id,
       COUNT(*) AS anomaly_count,
       BINOM_TEST(COUNT(*), total_shifts, expected_rate) AS p_value
   FROM anomalies a
   JOIN staff_schedule s ON a.treatment_date = s.shift_date
   GROUP BY staff_id
   HAVING p_value < 0.01
   ```

**Required Data**:
- Per-fraction QA data (all fractions, not periodic sampling)
- Staff schedules and shift assignments
- Calendar metadata (day of week, time of day)
- Facility access logs
- Multi-modal verification (TRF, portal) for every fraction

---

### Category 2: Spatial Evasion

**Goal**: Distribute errors across multiple spatial dimensions (fields, control points, patients) to evade single-point detection.

#### 2.1 Coordinated Multi-Field Attack (Severity: 2.9)

**Description**: Distribute dose errors across multiple treatment fields such that each field individually appears normal, but cumulative 3D dose is incorrect.

**Attack Variants**:
- **Compensatory Errors** (3.0): Systematic opposite biases that sum (field 1: +10%, field 2: -8%, net: +2%)
- **Geometric Shift** (2.9): All fields shifted same direction → dose displaced from target
- **Aperture Erosion** (2.8): Small MLC changes in all fields → margin reduction
- **Hotspot Creation** (2.9): Increase overlap regions → localized overdose

**Example Attack**:
```sql
-- Shift all AP/PA field apertures 3mm posterior
-- Individually within tolerance, but cumulatively shifts dose 3mm
UPDATE TxFieldPoint
SET A_Leaf_Set = shift_mlc_positions(A_Leaf_Set, direction='posterior', amount_mm=3),
    B_Leaf_Set = shift_mlc_positions(B_Leaf_Set, direction='posterior', amount_mm=3)
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TxField
    WHERE Pat_ID1 = '12345'
      AND beam_orientation IN ('AP', 'PA')  -- Only anterior/posterior beams
)
-- Result: 3mm systematic posterior shift in composite dose
```

**Clinical Impact**:
- Systematic geometric miss (underdose target, overdose OAR)
- May pass individual field QA but fail composite plan verification
- Difficult to detect without 3D dose reconstruction

**Detection Strategies**:

1. **3D Dose Reconstruction**:
   ```python
   def reconstruct_delivered_dose(trf_files, ct_dataset, structure_set):
       """Calculate 3D dose from delivered parameters."""
       dose_grid = initialize_dose_grid(ct_dataset)

       for trf in trf_files:
           # Extract delivered MLC, MU, angles from TRF
           delivered_params = parse_trf(trf)

           # Calculate dose contribution
           field_dose = dose_calculation_engine(
               ct_dataset,
               delivered_params,
               algorithm='monte_carlo'
           )

           dose_grid += field_dose

       return dose_grid
   ```

2. **Gamma Analysis (Composite)**:
   - Compare delivered 3D dose to planned dose
   - Use 3%/3mm or 2%/2mm criteria
   - Alert if gamma pass rate < 95%

3. **DVH Metrics**:
   ```python
   def verify_dvh_metrics(delivered_dose, structure_set, constraints):
       """Check dose-volume metrics vs clinical constraints."""
       violations = []

       for structure, constraint in constraints.items():
           dvh = calculate_dvh(delivered_dose, structure_set[structure])

           # Check constraint (e.g., "V20Gy < 30%" for lung)
           if not evaluate_constraint(dvh, constraint):
               violations.append((structure, constraint, dvh))

       return violations
   ```

4. **Aperture Centroid Tracking**:
   - Calculate geometric centroid of each field aperture
   - Detect systematic shifts in same direction across all fields
   - Flag if mean shift > 2mm (p < 0.01)

**Required Data**:
- CT dataset for dose reconstruction
- Structure set (targets and OARs)
- Treatment plan (DICOM RT Plan)
- Portal dosimetry for independent verification
- TRF files for all fractions
- 3D dose calculation engine

---

#### 2.2 Field Aperture Manipulation (Severity: 2.9)

**Description**: Subtly modify MLC apertures to shift dose distribution away from target toward critical structures.

**Attack Variants**:
- **Systematic Shift** (3.0): All leaves shifted same direction (e.g., 3mm superior)
- **Margin Erosion** (2.9): Reduce PTV margins from 5mm to 2mm
- **OAR Expansion** (2.8): Move aperture edge closer to critical structures
- **Asymmetric Blur** (2.7): Preferentially expand aperture in one direction

**Example Attack**:
```sql
-- Erode PTV margin from 5mm to 2mm (3mm reduction)
-- Visually subtle but geometrically significant
UPDATE TxFieldPoint
SET A_Leaf_Set = erode_mlc_aperture(A_Leaf_Set, amount_mm=3),
    B_Leaf_Set = erode_mlc_aperture(B_Leaf_Set, amount_mm=3)
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TxField WHERE Pat_ID1 = '12345'
)
-- Result: 3mm margin erosion → increased risk of geographic miss
```

**Clinical Impact**:
- Underdosing of target periphery
- Increased local recurrence risk
- Violation of treatment planning margins

**Detection Strategies**:

1. **MLC Aperture Centroid Tracking**:
   ```python
   def calculate_aperture_centroid(mlc_positions):
       """Calculate geometric centroid of MLC aperture."""
       aperture_points = []
       for leaf_pair in mlc_positions:
           if leaf_pair['A'] < leaf_pair['B']:  # Leaf pair open
               center = (leaf_pair['A'] + leaf_pair['B']) / 2
               aperture_points.append(center)

       centroid = np.mean(aperture_points, axis=0)
       return centroid
   ```

2. **Field Size Metrics**:
   - Calculate aperture area, perimeter, equivalent square
   - Compare to baseline (treatment plan)
   - Alert if area reduced by >5%

3. **PTV Margin Analysis**:
   ```python
   def calculate_effective_margin(delivered_aperture, planned_ptv, beam_geometry):
       """Calculate effective PTV margin from aperture."""
       # Project aperture onto patient anatomy (BEV → patient coordinates)
       aperture_patient_coords = beams_eye_view_to_patient(
           delivered_aperture, beam_geometry
       )

       # Calculate minimum distance from aperture edge to PTV surface
       margin = calculate_min_distance(aperture_patient_coords, planned_ptv)

       return margin
   ```

4. **Portal Imaging Verification**:
   - Compare EPID image to planned aperture
   - Detect systematic shifts or margin changes
   - Use auto-contouring to extract aperture from portal image

**Required Data**:
- Treatment plan (baseline apertures)
- Structure set (PTV contours)
- Portal images for geometric verification
- TRF files (delivered MLC positions)
- Beam geometry (gantry angle, SAD)

---

#### 2.3 Collimator Jaw Manipulation (Severity: 2.6)

**Description**: Modify collimator jaw positions (X1, X2, Y1, Y2) to create field size errors or unintended dose spillage.

**Attack Variants**:
- **Jaw Asymmetry** (2.7): Unequal jaw positions (e.g., X1 = -5cm, X2 = +7cm)
- **Field Size Reduction** (2.8): Close jaws to underdose target
- **Field Size Expansion** (2.5): Open jaws to overdose surrounding tissue
- **Gradient Shift** (2.6): Alter penumbra characteristics

**Example Attack**:
```sql
-- Reduce field size by 1cm in each dimension
-- 2cm total reduction may underdose target periphery
UPDATE TxFieldPoint
SET X1_Jaw = X1_Jaw + 0.5,  -- Close by 5mm
    X2_Jaw = X2_Jaw - 0.5,  -- Close by 5mm
    Y1_Jaw = Y1_Jaw + 0.5,
    Y2_Jaw = Y2_Jaw - 0.5
WHERE FLD_ID IN (
    SELECT FLD_ID FROM TxField WHERE Pat_ID1 = '12345'
)
-- Result: 1cm reduction in field size
```

**Clinical Impact**:
- Geometric miss if field size reduced
- Increased normal tissue dose if expanded
- Altered dose gradients (penumbra)

**Detection Strategies**:

1. **Jaw Position Verification**:
   ```sql
   SELECT
       tf.FLD_ID,
       tf.Pat_ID1,
       tfp.X1_Jaw, tfp.X2_Jaw, tfp.Y1_Jaw, tfp.Y2_Jaw,
       plan.X1_Jaw_Plan, plan.X2_Jaw_Plan, plan.Y1_Jaw_Plan, plan.Y2_Jaw_Plan,
       ABS(tfp.X1_Jaw - plan.X1_Jaw_Plan) AS x1_deviation
   FROM TxFieldPoint tfp
   JOIN TxField tf ON tfp.FLD_ID = tf.FLD_ID
   JOIN Plan_Jaws plan ON tf.FLD_ID = plan.FLD_ID
   WHERE ABS(tfp.X1_Jaw - plan.X1_Jaw_Plan) > 0.5  -- >5mm deviation
   ```

2. **Field Size Calculation**:
   ```python
   def verify_field_size(jaw_positions, planned_field_size):
       """Verify field size from jaw positions."""
       delivered_x = abs(jaw_positions['X2'] - jaw_positions['X1'])
       delivered_y = abs(jaw_positions['Y2'] - jaw_positions['Y1'])

       planned_x = planned_field_size['X']
       planned_y = planned_field_size['Y']

       if abs(delivered_x - planned_x) > 0.5 or abs(delivered_y - planned_y) > 0.5:
           alert("Field size deviation detected")
           return False
       return True
   ```

3. **Jaw Symmetry Analysis**:
   ```python
   def check_jaw_symmetry(jaw_positions):
       """Detect asymmetric jaw modifications."""
       x_center = (jaw_positions['X1'] + jaw_positions['X2']) / 2
       y_center = (jaw_positions['Y1'] + jaw_positions['Y2']) / 2

       if abs(x_center) > 0.3 or abs(y_center) > 0.3:  # >3mm asymmetry
           alert("Jaw asymmetry detected")
           return False
       return True
   ```

4. **Portal Imaging**:
   - Extract field edges from EPID images
   - Compare to planned field size
   - Detect systematic deviations

**Required Data**:
- Treatment plan jaw positions
- Expected field sizes by beam
- Portal images
- TRF files (delivered jaw positions)
- Jaw position tolerances (typically ±5mm)

---

### Category 3: Statistical Evasion

**Goal**: Design attacks that evade statistical detection by staying within normal ranges, avoiding outliers, or mimicking legitimate variance.

#### 3.1 Statistical Camouflage (Severity: 2.9)

**Description**: Introduce systematic bias that stays within normal statistical ranges but accumulates to clinically harmful levels.

**Attack Variants**:
- **Within-Tolerance Bias** (2.9): Always <2% error but always positive (mean = +1.5%)
- **Noise Injection** (2.7): Add random noise to mask systematic signal
- **Correlation Masking** (2.8): Correlate errors with legitimate variance sources
- **Outlier Avoidance** (2.9): Carefully stay below 3σ threshold

**Example Attack**:
```python
# Add systematic +1.5% bias with random noise to mask signal
import numpy as np

def camouflaged_dose_escalation(baseline_mu, num_fractions):
    """Add subtle systematic bias masked by noise."""
    systematic_bias = 1.015  # +1.5% per fraction
    noise_std = 0.01  # ±1% random noise

    delivered_mu = []
    for fraction in range(num_fractions):
        # Systematic bias + random noise
        mu = baseline_mu * systematic_bias * (1 + np.random.normal(0, noise_std))

        # Ensure stays within tolerance (clip to ±3%)
        mu = np.clip(mu, baseline_mu * 0.97, baseline_mu * 1.03)
        delivered_mu.append(mu)

    # Result: mean overdose +1.5% but appears as random variance
    return delivered_mu
```

**Clinical Impact**:
- Cumulative dose errors (10-30%)
- Difficult to detect with standard statistical tests
- Appears as normal variance to outlier detection

**Detection Strategies**:

1. **CUSUM (Cumulative Sum) Analysis**:
   ```python
   def cusum_analysis(parameter_series, target_value, sensitivity=0.5, threshold=5):
       """Detect small systematic shifts using CUSUM."""
       cusum_pos = 0
       cusum_neg = 0
       alerts = []

       for i, value in enumerate(parameter_series):
           deviation = value - target_value

           # Update CUSUM
           cusum_pos = max(0, cusum_pos + deviation - sensitivity)
           cusum_neg = min(0, cusum_neg + deviation + sensitivity)

           # Check thresholds
           if cusum_pos > threshold or cusum_neg < -threshold:
               alerts.append(i)

       return alerts
   ```

2. **One-Sample t-Test** (systematic bias detection):
   ```python
   def detect_systematic_bias(parameter_series, expected_mean):
       """Test if mean significantly different from expected."""
       t_statistic, p_value = scipy.stats.ttest_1samp(
           parameter_series,
           expected_mean
       )

       if p_value < 0.01:
           alert(f"Systematic bias detected (p={p_value:.4f})")
           return True
       return False
   ```

3. **Non-Parametric Trend Tests**:
   ```python
   def mann_kendall_trend_test(parameter_series):
       """Detect monotonic trends (robust to outliers)."""
       import pymannkendall as mk

       result = mk.original_test(parameter_series)

       if result.p < 0.01 and result.trend != 'no trend':
           alert(f"Trend detected: {result.trend} (p={result.p:.4f})")
           return True
       return False
   ```

4. **Benford's Law Analysis**:
   ```python
   def benford_law_test(data):
       """Detect artificial data using first-digit distribution."""
       first_digits = [int(str(abs(x))[0]) for x in data if x != 0]

       # Expected Benford distribution
       benford = [np.log10(1 + 1/d) for d in range(1, 10)]

       # Chi-square test
       observed = np.histogram(first_digits, bins=range(1, 11))[0]
       expected = len(first_digits) * np.array(benford)

       chi2, p_value = scipy.stats.chisquare(observed, expected)

       if p_value < 0.01:
           alert("Benford's law violation (possible artificial data)")
           return True
       return False
   ```

**Required Data**:
- Large historical baseline (>100 patients)
- Known sources of legitimate variance
- Statistical tolerance limits
- Time series data for trend analysis
- Multi-modal verification to break correlations

---

### Category 4: Attribution Evasion

**Goal**: Avoid detection by obscuring who made changes, when they were made, or whether they were legitimate.

#### 4.1 Audit Trail Manipulation (Severity: 2.7)

**Description**: Modify database audit tables, timestamps, or user attribution to hide evidence of malicious changes.

**Attack Variants**:
- **Timestamp Forgery** (2.8): Backdate modifications to appear legitimate
- **User Impersonation** (2.9): Attribute changes to authorized users
- **Log Deletion** (3.0): Remove evidence entirely from audit tables
- **Audit Disable** (2.7): Temporarily disable logging during attack

**Example Attack**:
```sql
-- Step 1: Disable audit trigger
ALTER TABLE TrackTreatment DISABLE TRIGGER audit_trigger;

-- Step 2: Make malicious modification
UPDATE TxField SET Meterset = Meterset * 1.2 WHERE FLD_ID = 12345;

-- Step 3: Re-enable audit trigger
ALTER TABLE TrackTreatment ENABLE TRIGGER audit_trigger;

-- Step 4: Forge audit entry with backdated timestamp and false user
INSERT INTO audit_log (table_name, record_id, modified_timestamp, user_id, operation)
VALUES ('TxField', 12345, '2024-01-15 08:30:00', 'legitimate_user', 'UPDATE');

-- Result: Malicious change appears as legitimate modification by authorized user
```

**Clinical Impact**:
- Enables all other attacks by hiding evidence
- Difficult forensic investigation
- Undermines trust in audit system

**Detection Strategies**:

1. **External Write-Once Audit Log**:
   ```python
   class ImmutableAuditLog:
       """Write-once audit log on separate system."""

       def __init__(self, blockchain_client):
           self.blockchain = blockchain_client

       def log_database_write(self, table, record_id, user, timestamp, operation):
           """Log to blockchain (immutable)."""
           entry = {
               'table': table,
               'record_id': record_id,
               'user': user,
               'timestamp': timestamp,
               'operation': operation,
               'hash': self.calculate_hash(table, record_id, timestamp)
           }

           # Write to blockchain (cannot be modified retroactively)
           self.blockchain.append_block(entry)
   ```

2. **Database Transaction Log Forensics**:
   ```sql
   -- Analyze SQL Server transaction log for undocumented changes
   SELECT
       [Transaction ID],
       [Begin Time],
       [Operation],
       [Context],
       [Transaction Name]
   FROM fn_dblog(NULL, NULL)
   WHERE [Operation] = 'LOP_MODIFY_ROW'
     AND [Context] NOT IN (SELECT context_id FROM audit_log)
   ORDER BY [Begin Time] DESC
   ```

3. **Timestamp Consistency Checks**:
   ```python
   def verify_timestamp_consistency(database_timestamp, ntp_timestamp, filesystem_timestamp):
       """Cross-reference multiple time sources."""
       max_skew = 2.0  # seconds

       if abs(database_timestamp - ntp_timestamp) > max_skew:
           alert("Database timestamp inconsistent with NTP")
           return False

       if abs(filesystem_timestamp - ntp_timestamp) > max_skew:
           alert("Filesystem timestamp inconsistent with NTP")
           return False

       return True
   ```

4. **User Session Analysis**:
   ```python
   def detect_impossible_user_activity(user_id, audit_entries):
       """Detect physically impossible user activity patterns."""
       for i in range(len(audit_entries) - 1):
           current = audit_entries[i]
           next_entry = audit_entries[i + 1]

           time_diff = next_entry['timestamp'] - current['timestamp']
           location_diff = geodesic_distance(current['location'], next_entry['location'])

           # Check if user could physically travel between locations
           max_speed = 120  # km/h (driving)
           required_speed = location_diff / (time_diff / 3600)  # km/h

           if required_speed > max_speed:
               alert(f"Impossible user activity for {user_id}")
               return False

       return True
   ```

**Required Data**:
- External immutable audit log (blockchain, WORM storage)
- Database transaction logs (binary logs)
- NTP logs for timestamp validation
- User authentication and session logs
- Geolocation data (IP addresses, physical access)
- File system metadata on database files

---

#### 4.2 Targeted Patient Selection (Severity: 3.0)

**Description**: Selectively attack specific patients to avoid statistical detection across patient population.

**Attack Variants**:
- **Demographic Targeting** (3.0): Based on age, diagnosis, prognosis
- **Random Sampling** (2.9): Small percentage of patients (e.g., 5%)
- **Temporal Spacing** (2.9): Long intervals between attacks
- **Site-Specific** (2.8): Only certain treatment sites (e.g., head & neck)

**Example Attack**:
```python
# Target only elderly patients with poor prognosis
# Reasoning: Less likely to be followed long-term, deaths attributed to disease

def select_vulnerable_patients(database_connection):
    """Select patients least likely to trigger suspicion."""
    query = """
    SELECT Pat_ID1
    FROM Patient p
    JOIN Diagnosis d ON p.Pat_ID1 = d.Pat_ID1
    WHERE p.Age > 75  -- Elderly
      AND d.Prognosis = 'Poor'  -- Poor prognosis
      AND d.Site_Code IN ('Lung', 'Pancreas')  -- Aggressive cancers
      AND RAND() < 0.05  -- 5% sampling to avoid pattern
    ORDER BY NEWID()
    LIMIT 10
    """

    return execute_query(database_connection, query)
```

**Clinical Impact**:
- Difficult to detect via population statistics (small sample)
- Vulnerable populations targeted
- May appear as normal variation in outcomes

**Detection Strategies**:

1. **Per-Patient Anomaly Detection**:
   ```python
   def patient_level_anomaly_detection(pat_id):
       """Detect anomalies at individual patient level."""
       features = extract_all_multimodal_features(pat_id)

       # Train patient-specific model on first 3 fractions (baseline)
       baseline_features = features[:3]

       # Detect anomalies in subsequent fractions
       for fraction_features in features[3:]:
           anomaly_score = isolation_forest.score_samples([fraction_features])

           if anomaly_score < threshold:
               alert(f"Anomaly detected for patient {pat_id}")
   ```

2. **Demographic Bias Analysis**:
   ```python
   def detect_demographic_bias(anomaly_database):
       """Test if anomalies disproportionately affect certain demographics."""
       for demographic_group in ['age_bins', 'diagnosis', 'treatment_site']:
           # Chi-square test for independence
           contingency_table = create_contingency_table(
               anomaly_database,
               demographic_group
           )

           chi2, p_value = scipy.stats.chi2_contingency(contingency_table)

           if p_value < 0.01:
               alert(f"Demographic bias detected in {demographic_group}")
   ```

3. **Temporal Clustering Analysis**:
   ```python
   def detect_temporal_clustering(anomaly_timestamps):
       """Detect non-random temporal distribution."""
       # Test if inter-arrival times follow exponential distribution (random)
       inter_arrival_times = np.diff(sorted(anomaly_timestamps))

       ks_statistic, p_value = scipy.stats.kstest(
           inter_arrival_times,
           'expon',
           args=(0, np.mean(inter_arrival_times))
       )

       if p_value < 0.01:
           alert("Non-random temporal pattern detected")
   ```

4. **Multi-Anomaly Flagging**:
   ```python
   def flag_multiple_independent_anomalies(pat_id):
       """Flag patients with multiple independent anomalies (low probability)."""
       anomaly_types = []

       # Check different anomaly types
       if detect_mlc_anomaly(pat_id):
           anomaly_types.append('mlc')
       if detect_mu_anomaly(pat_id):
           anomaly_types.append('mu')
       if detect_position_anomaly(pat_id):
           anomaly_types.append('position')

       # Probability of multiple independent anomalies is very low
       if len(anomaly_types) >= 2:
           probability = (0.05 ** len(anomaly_types))  # Assuming 5% base rate
           alert(f"Multiple anomalies for {pat_id} (p={probability:.6f})")
   ```

**Required Data**:
- Patient demographics and diagnosis
- Complete audit trail with user attribution
- Historical anomaly rates by demographic
- Multi-modal QA results for independent verification
- Staff access patterns and schedules

---

## Detection Framework Summary

### Multi-Layer Defense Strategy

Effective detection of malicious actors requires **defense in depth** with multiple independent verification layers:

| Layer | Detection Method | Evasion Difficulty | False Positive Rate | Implementation Cost |
|-------|------------------|-------------------|---------------------|-------------------|
| **Layer 1: Real-Time Field QA** | Portal dosimetry, jaw position checks | Low | Low | Medium |
| **Layer 2: Statistical Process Control** | CUSUM, trend analysis, t-tests | Medium | Medium | Low |
| **Layer 3: Multi-Modal Verification** | TRF + Portal + Phantom cross-validation | High | Low | High |
| **Layer 4: 3D Dose Reconstruction** | Independent dose calculation from delivered params | High | Low | High |
| **Layer 5: Audit Trail Forensics** | External immutable logs, transaction analysis | Very High | Very Low | Very High |

### Recommended Implementation Phasing

**Phase 1: Foundation** (Weeks 1-4)
- Implement external immutable audit logging
- Deploy per-fraction TRF file archival
- Establish baseline statistical models

**Phase 2: Statistical Detection** (Weeks 5-8)
- CUSUM charts for MU, angles, positions
- Longitudinal trend analysis
- Demographic bias detection

**Phase 3: Multi-Modal QA** (Weeks 9-16)
- Automated portal dosimetry analysis
- TRF-to-Mosaiq cross-validation
- Periodic phantom dosimetry

**Phase 4: Advanced Analytics** (Weeks 17-24)
- 3D dose reconstruction engine
- Patient-level anomaly detection
- Machine learning models for stealth pattern detection

**Phase 5: Forensics** (Weeks 25-30)
- Transaction log analysis tools
- User behavioral analytics
- Incident response workflows

---

## Clinical Response Protocols

### Alert Triage Workflow

```
Anomaly Detected
    ↓
[High Confidence] → IMMEDIATE ACTION
    • Halt treatment
    • Verify with independent measurement
    • Escalate to physicist + radiation oncologist

[Medium Confidence] → ENHANCED REVIEW
    • Continue treatment with enhanced monitoring
    • Schedule independent verification (portal/phantom)
    • Daily physicist review

[Low Confidence] → WATCHLIST
    • Flag patient for increased monitoring
    • Review at next chart check
    • Trend analysis over multiple fractions
```

### Severity-Based Response Times

| Severity | Response Time | Actions |
|----------|--------------|---------|
| **Critical (3.0)** | 15-30 min | Immediate treatment halt, emergency physicist review, independent verification |
| **High (2.5-2.9)** | 1-2 hours | Same-day physicist review, enhanced QA, verify next fraction |
| **Medium (1.8-2.4)** | 4-24 hours | Scheduled physicist review, trending analysis |
| **Low (0.5-1.7)** | 1-7 days | Routine QA review, documentation |

### Forensic Investigation

When malicious activity is suspected:

1. **Preserve Evidence**
   - Freeze database backups
   - Archive transaction logs
   - Collect all TRF files and portal images
   - Document audit trail

2. **Isolate Scope**
   - Identify all affected patients
   - Determine time window of attack
   - Correlate with staff schedules

3. **Independent Verification**
   - Recalculate dose from first principles
   - Phantom measurements for affected beams
   - Clinical assessment of patient impact

4. **Attribution Analysis**
   - Transaction log forensics
   - User session analysis
   - Geolocation and access logs

5. **Clinical Remediation**
   - Assess patient dose discrepancies
   - Adaptive re-planning if needed
   - Enhanced follow-up imaging

6. **System Hardening**
   - Patch vulnerabilities
   - Enhance access controls
   - Update detection algorithms

---

## Conclusion

Malicious actor scenarios represent the most sophisticated threats to radiotherapy data integrity. Detection requires:

- **Multiple Independent Verification Layers**: TRF + Portal + Phantom + 3D Dose
- **Advanced Statistical Methods**: CUSUM, trend analysis, Bayesian detection
- **Immutable Audit Trails**: Blockchain or WORM storage for forensics
- **Per-Patient Anomaly Detection**: Not just population statistics
- **Rapid Clinical Response**: Severity-based protocols with <30 min for critical alerts

The EBM training framework with malicious failure modes enables machine learning models to learn sophisticated attack patterns and develop robust detection capabilities that complement traditional QA methods.

**Key Principle**: Assume adversarial actors have domain knowledge and database access. Design detection systems that are robust to evasion attempts through multi-modal, multi-layer verification with cryptographic audit trails.
