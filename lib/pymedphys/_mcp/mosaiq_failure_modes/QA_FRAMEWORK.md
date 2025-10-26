# Mosaiq Database QA Framework

This document outlines the Quality Assurance framework for detecting data corruption and third-party write failures in Mosaiq databases. It is organized by failure mode with corresponding detection strategies.

## Table of Contents

1. [MLC Data Corruption](#1-mlc-data-corruption)
2. [Invalid Angle Values](#2-invalid-angle-values)
3. [Duplicate Treatment Entries](#3-duplicate-treatment-entries)
4. [Missing Control Points](#4-missing-control-points)
5. [NULL Required Fields](#5-null-required-fields)
6. [Timestamp Inconsistencies](#6-timestamp-inconsistencies)
7. [Orphaned Records](#7-orphaned-records)
8. [Invalid Offset Data](#8-invalid-offset-data)
9. [Meterset Inconsistency](#9-meterset-inconsistency)
10. [MLC Leaf Count Mismatch](#10-mlc-leaf-count-mismatch)
11. [Malicious Actor Detection](#11-malicious-actor-detection)

---

## 1. MLC Data Corruption

### Failure Mode
Corrupted binary MLC data in `TxFieldPoint.A_Leaf_Set` and `TxFieldPoint.B_Leaf_Set` fields.

### QA Checks

#### 1.1 Byte Array Length Validation
```python
def validate_mlc_byte_length(leaf_set_binary: bytes) -> bool:
    """Verify MLC byte array has even length (2 bytes per leaf)."""
    return len(leaf_set_binary) % 2 == 0
```

**SQL Query:**
```sql
SELECT FLD_ID, Point, DATALENGTH(A_Leaf_Set) AS A_Bytes, DATALENGTH(B_Leaf_Set) AS B_Bytes
FROM TxFieldPoint
WHERE DATALENGTH(A_Leaf_Set) % 2 != 0 OR DATALENGTH(B_Leaf_Set) % 2 != 0
```

#### 1.2 Physical Limit Validation
```python
def validate_mlc_positions(leaf_positions: list[float], mlc_config: dict) -> bool:
    """Check all leaf positions are within physical machine limits."""
    min_limit = mlc_config.get('min_position', -20.0)  # cm
    max_limit = mlc_config.get('max_position', 20.0)   # cm
    return all(min_limit <= pos <= max_limit for pos in leaf_positions)
```

#### 1.3 Leaf Pair Gap Validation
```python
def validate_leaf_gaps(a_leaves: list[float], b_leaves: list[float]) -> bool:
    """Ensure A bank leaves don't exceed B bank (no negative gaps)."""
    return all(a <= b for a, b in zip(a_leaves, b_leaves))
```

#### 1.4 Leaf Count Consistency
```sql
-- All control points should have same number of leaves
SELECT FLD_ID, Point, DATALENGTH(A_Leaf_Set)/2 AS LeafCount
FROM TxFieldPoint
WHERE FLD_ID = @FieldID
GROUP BY FLD_ID, Point
HAVING COUNT(DISTINCT DATALENGTH(A_Leaf_Set)) > 1
```

### Additional Data Requirements

1. **MLC Configuration File**: JSON/database defining:
   - MLC model (e.g., "Agility", "Millennium 120")
   - Number of leaf pairs (40, 60, 80, etc.)
   - Physical travel limits per leaf pair
   - Maximum leaf speed

2. **Machine-Specific Limits**: Table mapping `Staff.Last_Name` to MLC specifications

3. **Historical Position Statistics**:
   - 95th percentile MLC positions for each machine
   - Typical range for different treatment techniques (IMRT, VMAT, 3D)

**Example MLC Config:**
```json
{
  "LinacA": {
    "mlc_model": "Agility",
    "leaf_pairs": 80,
    "leaf_width": [0.5, 0.5],  // cm, by section
    "max_position": 20.0,      // cm
    "min_position": -20.0,
    "max_speed": 2.5           // cm/s
  }
}
```

---

## 2. Invalid Angle Values

### Failure Mode
Gantry angle (`Gantry_Ang`) or collimator angle (`Coll_Ang`) outside valid 0-360° range.

### QA Checks

#### 2.1 Range Validation
```sql
SELECT FLD_ID, Point, Gantry_Ang, Coll_Ang
FROM TxFieldPoint
WHERE Gantry_Ang < 0 OR Gantry_Ang > 360
   OR Coll_Ang < 0 OR Coll_Ang > 360
```

#### 2.2 Angle Transition Speed
```python
def validate_angle_speed(
    angles: list[float],
    times: list[float],
    max_speed: float
) -> bool:
    """Check angle changes are physically achievable."""
    for i in range(1, len(angles)):
        delta_angle = min(
            abs(angles[i] - angles[i-1]),
            360 - abs(angles[i] - angles[i-1])  # Handle wraparound
        )
        delta_time = times[i] - times[i-1]
        if delta_time > 0 and (delta_angle / delta_time) > max_speed:
            return False
    return True
```

#### 2.3 Discontinuity Detection
```sql
-- Find large angle jumps between consecutive control points
SELECT t1.FLD_ID, t1.Point, t1.Gantry_Ang, t2.Gantry_Ang AS Next_Gantry
FROM TxFieldPoint t1
JOIN TxFieldPoint t2 ON t1.FLD_ID = t2.FLD_ID AND t1.Point + 1 = t2.Point
WHERE ABS(t1.Gantry_Ang - t2.Gantry_Ang) > 180  -- Suspicious jump
  AND ABS(t1.Gantry_Ang - t2.Gantry_Ang) < 180  -- But not wraparound
```

### Additional Data Requirements

1. **Machine Rotation Speeds**:
   - Gantry: typically 6°/s for therapy machines
   - Collimator: typically 12°/s
   - Couch rotation: typically 3°/s

2. **Control Point Timing**:
   - Currently not stored in Mosaiq (major gap!)
   - Could be derived from:
     - Total beam-on time / number of control points
     - MU weighting per control point
     - Machine log files

3. **Mechanical Constraints**:
   - Some machines have restricted gantry ranges (e.g., 180° ± 60°)
   - Collision avoidance angles for specific couch positions

**Recommended Enhancement:**
```sql
-- Add timing information to TxFieldPoint (requires schema change)
ALTER TABLE TxFieldPoint ADD CP_Duration_Ms INT;
ALTER TABLE TxFieldPoint ADD Cumulative_Time_Ms INT;
```

---

## 3. Duplicate Treatment Entries

### Failure Mode
Multiple `TrackTreatment` entries for same delivery with conflicting data.

### QA Checks

#### 3.1 Time-Based Duplicate Detection
```sql
SELECT t1.TTX_ID, t2.TTX_ID, t1.Pat_ID1, t1.FLD_ID, t1.Create_DtTm
FROM TrackTreatment t1
JOIN TrackTreatment t2
  ON t1.Pat_ID1 = t2.Pat_ID1
  AND t1.Machine_ID_Staff_ID = t2.Machine_ID_Staff_ID
  AND t1.TTX_ID < t2.TTX_ID
  AND ABS(DATEDIFF(second, t1.Create_DtTm, t2.Create_DtTm)) <= 10
```

#### 3.2 Overlapping Delivery Windows
```python
def detect_overlapping_deliveries(treatments: list[dict]) -> list[tuple]:
    """Find treatments with overlapping time windows."""
    overlaps = []
    for i, t1 in enumerate(treatments):
        for t2 in treatments[i+1:]:
            if (t1['create_time'] <= t2['edit_time'] and
                t1['edit_time'] >= t2['create_time']):
                overlaps.append((t1, t2))
    return overlaps
```

#### 3.3 Conflicting Field Assignment
```sql
-- Same patient/time but different field IDs
SELECT t1.TTX_ID, t1.FLD_ID, t2.TTX_ID, t2.FLD_ID, t1.Create_DtTm
FROM TrackTreatment t1
JOIN TrackTreatment t2
  ON t1.Pat_ID1 = t2.Pat_ID1
  AND t1.TTX_ID < t2.TTX_ID
  AND t1.FLD_ID != t2.FLD_ID
  AND ABS(DATEDIFF(second, t1.Create_DtTm, t2.Create_DtTm)) <= 5
```

### Additional Data Requirements

1. **Machine Interlock Logs**:
   - Beam-on timestamps from linac control system
   - Should show only ONE delivery per time window
   - Provides ground truth for duplicate detection

2. **ARIA/Varian R&V System Records**:
   - Cross-reference with Record & Verify system
   - Captures intended delivery vs. actual
   - Provides independent verification

3. **Treatment Duration Statistics**:
   - Expected duration by field type:
     - Static field: 1-3 minutes
     - IMRT: 3-10 minutes
     - VMAT: 2-4 minutes
   - Used to set reasonable time buffer

4. **Workflow Timing Rules**:
   ```json
   {
     "min_time_between_fields": 30,     // seconds
     "max_beam_on_time": 600,           // seconds
     "duplicate_detection_window": 10   // seconds
   }
   ```

**Recommended Table:**
```sql
CREATE TABLE MachineInterlockLog (
    Log_ID INT PRIMARY KEY,
    Machine_ID_Staff_ID INT,
    Beam_On_DtTm DATETIME,
    Beam_Off_DtTm DATETIME,
    Total_MU DECIMAL(9,2),
    FLD_ID INT FOREIGN KEY REFERENCES TxField(FLD_ID)
);
```

---

## 4. Missing Control Points

### Failure Mode
Deleted or missing control points in `TxFieldPoint` creating gaps in sequence.

### QA Checks

#### 4.1 Sequential Index Validation
```sql
-- Check for gaps in Point sequence
WITH ControlPointSequence AS (
    SELECT FLD_ID, Point,
           ROW_NUMBER() OVER (PARTITION BY FLD_ID ORDER BY Point) - 1 AS Expected_Point
    FROM TxFieldPoint
)
SELECT FLD_ID, Point, Expected_Point
FROM ControlPointSequence
WHERE Point != Expected_Point
```

#### 4.2 Minimum Control Point Count
```sql
SELECT FLD_ID, COUNT(*) AS CP_Count
FROM TxFieldPoint
GROUP BY FLD_ID
HAVING COUNT(*) < 2  -- At least start and end
```

#### 4.3 Control Point Count vs Plan
```python
def validate_cp_count(mosaiq_cps: int, dicom_plan_cps: int, field_type: str) -> bool:
    """Compare actual vs expected control point count."""
    if field_type in ['STATIC', '3D']:
        return mosaiq_cps >= 2  # Start and end minimum
    elif field_type == 'IMRT':
        return mosaiq_cps == dicom_plan_cps  # Exact match expected
    elif field_type == 'VMAT':
        return mosaiq_cps >= 10  # Typical minimum for arc
    return False
```

### Additional Data Requirements

1. **DICOM RT Plan**:
   - `ControlPointSequence` from BeamSequence
   - Expected number of control points per beam
   - MU weighting per control point

2. **Field Type Classification**:
   ```sql
   ALTER TABLE TxField ADD Technique VARCHAR(20);
   -- Values: 'STATIC', 'IMRT', 'VMAT', '3DCRT'
   ```

3. **Treatment Plan File Archive**:
   - Store DICOM RT Plan files linked to FLD_ID
   - Enables verification of delivered vs planned

4. **Control Point Metadata**:
   ```sql
   CREATE TABLE ControlPointExpectation (
       FLD_ID INT,
       Expected_CP_Count INT,
       Source VARCHAR(50),  -- 'DICOM_Plan', 'Monaco_Export', etc.
       Import_DtTm DATETIME
   );
   ```

---

## 5. NULL Required Fields

### Failure Mode
Critical fields set to NULL breaking foreign key relationships or queries.

### QA Checks

#### 5.1 Foreign Key Integrity
```sql
-- Check for NULL in foreign key fields
SELECT 'TrackTreatment' AS TableName, COUNT(*) AS NullCount
FROM TrackTreatment
WHERE Pat_ID1 IS NULL OR FLD_ID IS NULL OR SIT_ID IS NULL
UNION ALL
SELECT 'TxFieldPoint', COUNT(*)
FROM TxFieldPoint
WHERE FLD_ID IS NULL
UNION ALL
SELECT 'Offset', COUNT(*)
FROM Offset
WHERE SIT_SET_ID IS NULL
```

#### 5.2 Timestamp NULL Check
```sql
SELECT TableName, FieldName, NullCount
FROM (
    SELECT 'TrackTreatment' AS TableName, 'Create_DtTm' AS FieldName, COUNT(*) AS NullCount
    FROM TrackTreatment WHERE Create_DtTm IS NULL
    UNION ALL
    SELECT 'TrackTreatment', 'Edit_DtTm', COUNT(*)
    FROM TrackTreatment WHERE Edit_DtTm IS NULL
    UNION ALL
    SELECT 'Offset', 'Study_DtTm', COUNT(*)
    FROM Offset WHERE Study_DtTm IS NULL
) AS NullChecks
WHERE NullCount > 0
```

#### 5.3 Index Field Validation
```python
def validate_indexed_fields(cursor) -> list[str]:
    """Check all indexed fields for NULL values."""
    # Get all indexed columns
    cursor.execute("""
        SELECT t.name AS TableName, c.name AS ColumnName
        FROM sys.indexes i
        JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        JOIN sys.tables t ON i.object_id = t.object_id
        WHERE t.name IN ('TrackTreatment', 'TxField', 'TxFieldPoint', 'Offset')
    """)

    issues = []
    for table, column in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            issues.append(f"{table}.{column}: {null_count} NULL values")
    return issues
```

### Additional Data Requirements

1. **Database Schema Constraints**:
   - Export of all NOT NULL constraints
   - Foreign key definitions
   - Default value specifications

   ```sql
   SELECT
       t.name AS TableName,
       c.name AS ColumnName,
       c.is_nullable,
       dc.definition AS DefaultValue,
       fk.name AS ForeignKeyName
   FROM sys.columns c
   JOIN sys.tables t ON c.object_id = t.object_id
   LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id
   LEFT JOIN sys.foreign_key_columns fkc ON c.object_id = fkc.parent_object_id
                                          AND c.column_id = fkc.parent_column_id
   LEFT JOIN sys.foreign_keys fk ON fkc.constraint_object_id = fk.object_id
   WHERE t.name IN ('TrackTreatment', 'TxField', 'TxFieldPoint', 'Offset')
   ```

2. **Application-Level Requirements**:
   - Document which fields PyMedPhys assumes non-NULL
   - Fields that should be NOT NULL but aren't in schema

3. **Validation Rules Table**:
   ```sql
   CREATE TABLE FieldValidationRules (
       TableName VARCHAR(50),
       ColumnName VARCHAR(50),
       Rule VARCHAR(20),  -- 'NOT_NULL', 'FOREIGN_KEY', 'UNIQUE'
       ErrorSeverity VARCHAR(10),  -- 'CRITICAL', 'WARNING'
       Description TEXT
   );
   ```

---

## 6. Timestamp Inconsistencies

### Failure Mode
Invalid timestamp relationships: `Edit_DtTm < Create_DtTm`, future dates, etc.

### QA Checks

#### 6.1 Edit Before Create Detection
```sql
SELECT TTX_ID, Pat_ID1, Create_DtTm, Edit_DtTm,
       DATEDIFF(second, Edit_DtTm, Create_DtTm) AS Inconsistency_Seconds
FROM TrackTreatment
WHERE Edit_DtTm < Create_DtTm
```

#### 6.2 Future Timestamp Detection
```sql
SELECT TTX_ID, Create_DtTm, Edit_DtTm
FROM TrackTreatment
WHERE Create_DtTm > GETDATE() OR Edit_DtTm > GETDATE()
```

#### 6.3 Treatment Duration Validation
```python
def validate_treatment_duration(
    create_time: datetime,
    edit_time: datetime,
    field_type: str,
    expected_durations: dict
) -> bool:
    """Check treatment duration is within expected range."""
    duration = (edit_time - create_time).total_seconds()
    min_duration, max_duration = expected_durations[field_type]
    return min_duration <= duration <= max_duration
```

```sql
-- Find suspiciously short or long treatments
SELECT TTX_ID, FLD_ID, Create_DtTm, Edit_DtTm,
       DATEDIFF(second, Create_DtTm, Edit_DtTm) AS Duration_Seconds
FROM TrackTreatment
WHERE DATEDIFF(second, Create_DtTm, Edit_DtTm) < 10  -- Too short
   OR DATEDIFF(second, Create_DtTm, Edit_DtTm) > 1800  -- Too long (>30min)
```

#### 6.4 Cross-System Time Validation
```python
def validate_against_dose_history(track_treatment: dict, dose_history: dict) -> bool:
    """Compare TrackTreatment times with Dose_Hst."""
    time_diff = abs((track_treatment['create_time'] - dose_history['tx_time']).total_seconds())
    return time_diff <= 300  # Within 5 minutes
```

### Additional Data Requirements

1. **Expected Duration Ranges**:
   ```json
   {
     "field_durations": {
       "STATIC": {"min": 30, "max": 300},
       "IMRT": {"min": 180, "max": 900},
       "VMAT": {"min": 120, "max": 600},
       "SETUP": {"min": 5, "max": 60}
     }
   }
   ```

2. **System Clock Synchronization**:
   - NTP server logs for Mosaiq database server
   - Time drift monitoring
   - Timezone configuration documentation

3. **External System Timestamps**:
   - **Linac Log Files**: Actual beam-on/beam-off times
   - **ARIA Treatment Records**: Independent timestamp source
   - **Portal Imaging System**: Image acquisition times

4. **Workflow Checkpoints**:
   ```sql
   CREATE TABLE TreatmentCheckpoints (
       Checkpoint_ID INT PRIMARY KEY,
       TTX_ID INT,
       Checkpoint_Type VARCHAR(50),  -- 'PATIENT_IN', 'SETUP_COMPLETE', 'BEAM_ON', 'BEAM_OFF'
       Checkpoint_DtTm DATETIME,
       Source_System VARCHAR(50)
   );
   ```

---

## 7. Orphaned Records

### Failure Mode
Records with foreign key references to non-existent parent records.

### QA Checks

#### 7.1 Patient Reference Validation
```sql
-- TrackTreatment with non-existent patient
SELECT tt.TTX_ID, tt.Pat_ID1
FROM TrackTreatment tt
LEFT JOIN Patient p ON tt.Pat_ID1 = p.Pat_ID1
WHERE p.Pat_ID1 IS NULL

UNION ALL

-- TrackTreatment with non-existent Ident
SELECT tt.TTX_ID, tt.Pat_ID1
FROM TrackTreatment tt
LEFT JOIN Ident i ON tt.Pat_ID1 = i.Pat_ID1
WHERE i.Pat_ID1 IS NULL
```

#### 7.2 Field Reference Validation
```sql
-- TrackTreatment with non-existent field
SELECT tt.TTX_ID, tt.FLD_ID
FROM TrackTreatment tt
LEFT JOIN TxField f ON tt.FLD_ID = f.FLD_ID
WHERE f.FLD_ID IS NULL

UNION ALL

-- TxFieldPoint with non-existent field
SELECT tfp.TFP_ID, tfp.FLD_ID
FROM TxFieldPoint tfp
LEFT JOIN TxField f ON tfp.FLD_ID = f.FLD_ID
WHERE f.FLD_ID IS NULL
```

#### 7.3 Site Reference Validation
```sql
SELECT tt.TTX_ID, tt.SIT_ID
FROM TrackTreatment tt
LEFT JOIN Site s ON tt.SIT_ID = s.SIT_ID
WHERE s.SIT_ID IS NULL
```

#### 7.4 Comprehensive FK Check
```python
def validate_all_foreign_keys(cursor) -> dict:
    """Check all foreign key relationships."""
    issues = {}

    fk_checks = [
        ('TrackTreatment', 'Pat_ID1', 'Patient', 'Pat_ID1'),
        ('TrackTreatment', 'FLD_ID', 'TxField', 'FLD_ID'),
        ('TrackTreatment', 'SIT_ID', 'Site', 'SIT_ID'),
        ('TxFieldPoint', 'FLD_ID', 'TxField', 'FLD_ID'),
        ('TxField', 'Pat_ID1', 'Patient', 'Pat_ID1'),
        ('Offset', 'SIT_SET_ID', 'Site', 'SIT_SET_ID'),
    ]

    for child_table, child_col, parent_table, parent_col in fk_checks:
        query = f"""
        SELECT COUNT(*) FROM {child_table} c
        LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
        WHERE p.{parent_col} IS NULL
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        if count > 0:
            issues[f"{child_table}.{child_col}"] = count

    return issues
```

### Additional Data Requirements

1. **Foreign Key Constraint Definitions**:
   ```sql
   SELECT
       fk.name AS FK_Name,
       tp.name AS Child_Table,
       cp.name AS Child_Column,
       tr.name AS Parent_Table,
       cr.name AS Parent_Column,
       fk.delete_referential_action_desc AS On_Delete_Action
   FROM sys.foreign_keys fk
   JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
   JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
   JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id
                       AND fkc.parent_column_id = cp.column_id
   JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
   JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id
                       AND fkc.referenced_column_id = cr.column_id
   ```

2. **Cascade Rules**:
   - Document what should happen on parent delete
   - ON DELETE CASCADE vs ON DELETE SET NULL vs RESTRICT

3. **Data Lineage**:
   - Track data import sources
   - Identify third-party write operations
   - Audit log of record creation/deletion

4. **Referential Integrity Monitoring**:
   ```sql
   CREATE TABLE ReferentialIntegrityLog (
       Log_ID INT PRIMARY KEY IDENTITY,
       Check_DtTm DATETIME DEFAULT GETDATE(),
       Child_Table VARCHAR(50),
       Child_Column VARCHAR(50),
       Orphaned_Count INT,
       Sample_Orphaned_IDs VARCHAR(500)  -- Comma-separated list
   );
   ```

---

## 8. Invalid Offset Data

### Failure Mode
Corrupted patient positioning offset values from third-party systems (CBCT, portal imaging).

### QA Checks

#### 8.1 Magnitude Validation
```sql
-- Offsets beyond reasonable limits (±10cm warning, ±20cm error)
SELECT OFF_ID, Superior_Offset, Anterior_Offset, Lateral_Offset,
       SQRT(POWER(Superior_Offset, 2) + POWER(Anterior_Offset, 2) + POWER(Lateral_Offset, 2)) AS Vector_Magnitude
FROM Offset
WHERE ABS(Superior_Offset) > 10
   OR ABS(Anterior_Offset) > 10
   OR ABS(Lateral_Offset) > 10
```

#### 8.2 Enumeration Validation
```sql
-- Invalid Offset_Type (valid: 2=Localization, 3=Portal, 4=ThirdParty)
SELECT OFF_ID, Offset_Type
FROM Offset
WHERE Offset_Type NOT IN (2, 3, 4)

UNION ALL

-- Invalid Offset_State (valid: 1=Active, 2=Complete)
SELECT OFF_ID, Offset_State
FROM Offset
WHERE Offset_State NOT IN (1, 2)
```

#### 8.3 Temporal Validation
```sql
-- Offsets with unreasonable timing relative to treatment
SELECT o.OFF_ID, o.Study_DtTm, dh.Tx_DtTm,
       DATEDIFF(minute, o.Study_DtTm, dh.Tx_DtTm) AS Minutes_Before_Treatment
FROM Offset o
JOIN Dose_Hst dh ON o.SIT_SET_ID = dh.SIT_SET_ID
WHERE DATEDIFF(minute, o.Study_DtTm, dh.Tx_DtTm) > 120  -- More than 2 hours before
   OR DATEDIFF(minute, o.Study_DtTm, dh.Tx_DtTm) < -30  -- More than 30 min after
```

#### 8.4 Site-Specific Limits
```python
def validate_offset_by_site(offset: dict, site_info: dict) -> bool:
    """Check offset is within site-specific action limits."""
    action_limits = {
        'BRAIN': {'vector_max': 3.0},      # mm precision required
        'PROSTATE': {'vector_max': 10.0},  # More tolerance
        'LUNG': {'vector_max': 20.0},      # Respiratory motion
    }

    site_type = site_info.get('site_name', 'PROSTATE')
    limits = action_limits.get(site_type, {'vector_max': 10.0})

    vector_mag = (offset['superior']**2 + offset['anterior']**2 + offset['lateral']**2)**0.5
    return vector_mag <= limits['vector_max']
```

### Additional Data Requirements

1. **Clinical Action Limits by Site**:
   ```sql
   CREATE TABLE SiteOffsetLimits (
       Site_Type VARCHAR(50),
       Direction VARCHAR(20),  -- 'SUPERIOR', 'ANTERIOR', 'LATERAL', 'VECTOR'
       Warning_Limit_cm DECIMAL(4,1),
       Action_Limit_cm DECIMAL(4,1),
       Protocol_Reference VARCHAR(100)
   );

   INSERT INTO SiteOffsetLimits VALUES
   ('BRAIN', 'VECTOR', 0.2, 0.3, 'SRS_Protocol_v2.1'),
   ('PROSTATE', 'VECTOR', 0.5, 1.0, 'IGRT_Prostate_v3.0'),
   ('LUNG_SBRT', 'VECTOR', 0.5, 1.0, 'SBRT_Lung_v1.5');
   ```

2. **Third-Party System Integration Logs**:
   - **CBCT System**: Image acquisition metadata
     - Image quality scores
     - Registration algorithm used
     - User who approved shift
   - **Portal Imaging**:
     - kV/MV image pairs
     - Auto-match confidence scores
   - **Surface Guidance** (AlignRT, etc.):
     - Real-time position tracking
     - Threshold alerts

3. **Offset Workflow Metadata**:
   ```sql
   ALTER TABLE Offset ADD Image_Quality_Score DECIMAL(3,2);
   ALTER TABLE Offset ADD Registration_Method VARCHAR(50);
   ALTER TABLE Offset ADD Approved_By_Staff_ID INT;
   ALTER TABLE Offset ADD Approval_DtTm DATETIME;
   ALTER TABLE Offset ADD Override_Reason TEXT;
   ```

4. **Historical Offset Statistics**:
   ```python
   # Calculate baseline offset distributions per patient
   def get_offset_baseline(patient_id: int, site_id: int) -> dict:
       """Get historical mean and std dev of offsets for this patient."""
       query = """
       SELECT AVG(Superior_Offset), STDEV(Superior_Offset),
              AVG(Anterior_Offset), STDEV(Anterior_Offset),
              AVG(Lateral_Offset), STDEV(Lateral_Offset)
       FROM Offset o
       JOIN Site s ON o.SIT_SET_ID = s.SIT_SET_ID
       WHERE s.Pat_ID1 = ? AND s.SIT_ID = ?
       """
       # Use for outlier detection: new offset > mean + 3*stdev
   ```

---

## 9. Meterset Inconsistency

### Failure Mode
Mismatch between `TxField.Meterset` (planned MU) and delivered/control point MU.

### QA Checks

#### 9.1 Negative Meterset Detection
```sql
SELECT FLD_ID, Field_Label, Meterset
FROM TxField
WHERE Meterset < 0
```

#### 9.2 Extreme Value Detection
```sql
-- Suspiciously high meterset (typical max ~500 MU for single field)
SELECT FLD_ID, Field_Label, Meterset, Type_Enum
FROM TxField
WHERE Meterset > 1000
```

#### 9.3 Control Point MU Sum Validation
```python
def validate_cp_meterset(
    field_meterset: float,
    cp_mu_weights: list[float],
    tolerance_pct: float = 2.0
) -> bool:
    """Verify control point MU sum matches field meterset."""
    cp_total = sum(cp_mu_weights)
    diff_pct = abs(cp_total - field_meterset) / field_meterset * 100
    return diff_pct <= tolerance_pct
```

**Note**: Mosaiq's `TxFieldPoint` table doesn't store MU per control point directly. This must be derived from:
- DICOM RT Plan's `CumulativeMetersetWeight`
- Machine log files
- Treatment plan export

#### 9.4 Historical Range Validation
```sql
-- Compare to historical meterset for similar field types
WITH FieldStats AS (
    SELECT Type_Enum, AVG(Meterset) AS Avg_MU, STDEV(Meterset) AS Std_MU
    FROM TxField
    GROUP BY Type_Enum
)
SELECT f.FLD_ID, f.Meterset, fs.Avg_MU, fs.Std_MU,
       (f.Meterset - fs.Avg_MU) / fs.Std_MU AS Z_Score
FROM TxField f
JOIN FieldStats fs ON f.Type_Enum = fs.Type_Enum
WHERE ABS((f.Meterset - fs.Avg_MU) / fs.Std_MU) > 3  -- More than 3 std deviations
```

### Additional Data Requirements

1. **MU Tolerance Limits**:
   ```json
   {
     "mu_tolerance": {
       "photon_static": {"warning": 2.0, "action": 5.0},
       "photon_imrt": {"warning": 3.0, "action": 5.0},
       "electron": {"warning": 2.0, "action": 3.0}
     }
   }
   ```

2. **Control Point MU Weighting** (currently missing from Mosaiq):
   - **Source**: DICOM RT Plan
     - `BeamSequence` → `ControlPointSequence` → `CumulativeMetersetWeight`
   - **Recommended Addition**:
     ```sql
     ALTER TABLE TxFieldPoint ADD Cumulative_MU DECIMAL(9,2);
     ALTER TABLE TxFieldPoint ADD Fractional_MU DECIMAL(9,2);
     ```

3. **Machine Output Factors**:
   ```sql
   CREATE TABLE MachineOutputFactors (
       Machine_ID_Staff_ID INT,
       Energy_MV DECIMAL(4,1),
       Field_Size_cm DECIMAL(5,2),
       Output_Factor DECIMAL(6,4),
       Measurement_DtTm DATETIME,
       QA_Physicist_Staff_ID INT
   );
   ```

4. **Planned vs Delivered MU**:
   ```sql
   CREATE TABLE DeliveredMU (
       TTX_ID INT,
       FLD_ID INT,
       Planned_MU DECIMAL(9,2),
       Delivered_MU DECIMAL(9,2),
       Difference_MU DECIMAL(9,2),
       Difference_Pct DECIMAL(5,2),
       Source VARCHAR(50)  -- 'MACHINE_LOG', 'MOSAIQ_TRACK', 'ARIA_RV'
   );
   ```

5. **Machine Log File Integration**:
   - Varian DynaLog files (`.dlg`)
   - Elekta iCom streams
   - Capture actual delivered MU per segment/control point

---

## 10. MLC Leaf Count Mismatch

### Failure Mode
Inconsistent number of MLC leaves across control points or mismatch with machine configuration.

### QA Checks

#### 10.1 Intra-Field Leaf Count Consistency
```sql
-- Check all control points in a field have same leaf count
SELECT FLD_ID, COUNT(DISTINCT DATALENGTH(A_Leaf_Set)) AS Unique_A_Lengths,
       COUNT(DISTINCT DATALENGTH(B_Leaf_Set)) AS Unique_B_Lengths
FROM TxFieldPoint
GROUP BY FLD_ID
HAVING COUNT(DISTINCT DATALENGTH(A_Leaf_Set)) > 1
    OR COUNT(DISTINCT DATALENGTH(B_Leaf_Set)) > 1
```

#### 10.2 Machine Configuration Validation
```python
def validate_leaf_count_vs_machine(
    leaf_count: int,
    machine_config: dict
) -> bool:
    """Verify leaf count matches machine specification."""
    expected_leaf_count = machine_config['mlc_leaf_pairs']
    return leaf_count == expected_leaf_count
```

```sql
-- Cross-reference with machine MLC config
SELECT f.FLD_ID, s.Last_Name AS Machine,
       DATALENGTH(tfp.A_Leaf_Set)/2 AS Actual_Leaves,
       mc.Expected_Leaf_Pairs
FROM TxField f
JOIN TxFieldPoint tfp ON f.FLD_ID = tfp.FLD_ID AND tfp.Point = 0
JOIN TrackTreatment tt ON f.FLD_ID = tt.FLD_ID
JOIN Staff s ON tt.Machine_ID_Staff_ID = s.Staff_ID
JOIN MachineMLCConfig mc ON REPLACE(s.Last_Name, ' ', '') = mc.Machine_Name
WHERE DATALENGTH(tfp.A_Leaf_Set)/2 != mc.Expected_Leaf_Pairs
```

#### 10.3 DICOM Plan Verification
```python
def compare_with_dicom_plan(
    mosaiq_leaf_count: int,
    dicom_rt_plan_path: str,
    beam_number: int
) -> bool:
    """Compare Mosaiq leaf count with DICOM RT Plan."""
    import pydicom
    ds = pydicom.dcmread(dicom_rt_plan_path)
    beam = ds.BeamSequence[beam_number]

    for device in beam.BeamLimitingDeviceSequence:
        if device.RTBeamLimitingDeviceType == 'MLCX':
            expected_leaves = device.NumberOfLeafJawPairs
            return mosaiq_leaf_count == expected_leaves
    return False
```

### Additional Data Requirements

1. **Machine MLC Configuration Database**:
   ```sql
   CREATE TABLE MachineMLCConfig (
       Machine_ID INT PRIMARY KEY,
       Machine_Name VARCHAR(50),
       MLC_Model VARCHAR(50),
       Expected_Leaf_Pairs INT,
       Leaf_Width_cm VARCHAR(100),  -- JSON array: [0.5, 0.5, 1.0, ...]
       Commission_DtTm DATETIME,
       Last_QA_DtTm DATETIME
   );

   INSERT INTO MachineMLCConfig VALUES
   (1, 'LinacA', 'Agility', 80, '[0.5,0.5]', '2020-01-15', '2024-01-10'),
   (2, 'LinacB', 'Millennium120', 60, '[1.0,0.5,0.5,1.0]', '2018-06-20', '2023-12-15');
   ```

2. **DICOM RT Plan Archive**:
   ```sql
   CREATE TABLE DicomRTPlanArchive (
       Plan_Archive_ID INT PRIMARY KEY,
       FLD_ID INT,
       Dicom_File_Path VARCHAR(500),
       SOP_Instance_UID VARCHAR(100),
       Import_DtTm DATETIME,
       File_Hash VARCHAR(64)  -- SHA256 for integrity
   );
   ```

3. **MLC Model Specifications** (manufacturer data):
   ```json
   {
     "mlc_models": {
       "Varian_Millennium120": {
         "total_leaves": 120,
         "leaf_pairs": 60,
         "leaf_widths": [1.0, 0.5, 0.5, 1.0],
         "max_field_size": [40.0, 40.0]
       },
       "Elekta_Agility": {
         "total_leaves": 160,
         "leaf_pairs": 80,
         "leaf_widths": [0.5, 0.5],
         "max_field_size": [40.0, 40.0]
       }
     }
   }
   ```

4. **Treatment Plan MLC Configuration**:
   - From DICOM RT Plan:
     - `BeamLimitingDeviceSequence`
     - `NumberOfLeafJawPairs`
     - `LeafPositionBoundaries`

   **Recommended Storage**:
   ```sql
   ALTER TABLE TxField ADD Expected_Leaf_Count INT;
   ALTER TABLE TxField ADD MLC_Model VARCHAR(50);
   ALTER TABLE TxField ADD Plan_SOP_Instance_UID VARCHAR(100);
   ```

---

## Summary: Critical Data Gaps

Based on this analysis, the following data is **critically missing** from Mosaiq for comprehensive QA:

### High Priority
1. **Control Point Timing**: Duration/cumulative time per control point
2. **Control Point MU**: Fractional and cumulative MU per control point
3. **Machine Log Integration**: Link to Dynalog/iCom files
4. **DICOM Plan Reference**: SOP Instance UID linking TxField to RT Plan
5. **Third-Party Integration Metadata**: CBCT/portal imaging approval workflow

### Medium Priority
6. **Machine Configuration Database**: MLC specs, output factors
7. **Expected Value Ranges**: Treatment duration, offset limits by site
8. **Validation Rules Table**: Centralized business logic for QA
9. **Audit Logging**: Track who modified what and when

### Recommended Schema Enhancements

```sql
-- Add to TxFieldPoint
ALTER TABLE TxFieldPoint ADD CP_Duration_Ms INT;
ALTER TABLE TxFieldPoint ADD Cumulative_MU DECIMAL(9,2);

-- Add to TxField
ALTER TABLE TxField ADD Plan_SOP_Instance_UID VARCHAR(100);
ALTER TABLE TxField ADD Expected_CP_Count INT;
ALTER TABLE TxField ADD Technique VARCHAR(20);

-- Add to Offset
ALTER TABLE Offset ADD Image_Quality_Score DECIMAL(3,2);
ALTER TABLE Offset ADD Approved_By_Staff_ID INT;

-- New tables
CREATE TABLE MachineMLCConfig (...);
CREATE TABLE SiteOffsetLimits (...);
CREATE TABLE TreatmentCheckpoints (...);
CREATE TABLE DeliveredMU (...);
```

---

## 11. Malicious Actor Detection

### Overview

Malicious actor failure modes represent intentional sabotage attempts by adversaries with database access. Unlike accidental corruption, these attacks are designed to evade detection through:

- **Statistical camouflage**: Staying within normal variance ranges
- **Temporal evasion**: Spreading attacks across time or targeting specific windows
- **Spatial distribution**: Coordinating errors across fields/patients
- **Audit trail manipulation**: Hiding evidence of modifications

**All malicious failure modes have severity 2.5-3.0** (critical) due to deliberate intent to harm.

**See [MALICIOUS_ACTORS.md](MALICIOUS_ACTORS.md) for complete documentation.**

---

### 11.1 Subtle Dose Escalation Detection

**Threat**: Gradual MU increases across fractions that stay within daily tolerance but accumulate to harmful cumulative dose.

#### QA Checks

##### 11.1.1 Cumulative Dose Tracking

```sql
-- Track cumulative delivered MU vs prescription
WITH CumulativeMU AS (
    SELECT
        tt.Pat_ID1,
        tf.FLD_ID,
        tf.Field_Name,
        SUM(tt.Delivered_MU) AS cumulative_mu,
        tf.Meterset * COUNT(*) AS expected_cumulative_mu,
        COUNT(*) AS fraction_count
    FROM TrackTreatment tt
    JOIN TxField tf ON tt.FLD_ID = tf.FLD_ID
    GROUP BY tt.Pat_ID1, tf.FLD_ID, tf.Field_Name, tf.Meterset
)
SELECT
    Pat_ID1,
    FLD_ID,
    Field_Name,
    cumulative_mu,
    expected_cumulative_mu,
    (cumulative_mu - expected_cumulative_mu) / expected_cumulative_mu * 100 AS percent_deviation,
    fraction_count
FROM CumulativeMU
WHERE ABS((cumulative_mu - expected_cumulative_mu) / expected_cumulative_mu) > 0.05  -- >5% deviation
ORDER BY ABS(percent_deviation) DESC;
```

##### 11.1.2 CUSUM (Cumulative Sum) Analysis

```python
def cusum_mu_analysis(pat_id: str, fld_id: int, target_mu: float, threshold: float = 5.0):
    """Detect systematic MU bias using CUSUM control chart.

    Args:
        pat_id: Patient ID
        fld_id: Field ID
        target_mu: Expected MU per fraction
        threshold: CUSUM threshold for alert (default 5.0)

    Returns:
        List of fraction numbers where CUSUM exceeds threshold
    """
    import numpy as np

    # Fetch MU history
    mu_history = get_mu_history(pat_id, fld_id)  # [fraction_1_mu, fraction_2_mu, ...]

    # CUSUM parameters
    sensitivity = 0.5  # Detect shifts of 0.5% or more
    cusum_pos = 0
    cusum_neg = 0
    alerts = []

    for i, delivered_mu in enumerate(mu_history):
        deviation = (delivered_mu - target_mu) / target_mu * 100  # Percent deviation

        # Update CUSUM
        cusum_pos = max(0, cusum_pos + deviation - sensitivity)
        cusum_neg = min(0, cusum_neg + deviation + sensitivity)

        # Check thresholds
        if cusum_pos > threshold:
            alerts.append({
                'fraction': i + 1,
                'type': 'positive_drift',
                'cusum': cusum_pos,
                'delivered_mu': delivered_mu
            })

        if cusum_neg < -threshold:
            alerts.append({
                'fraction': i + 1,
                'type': 'negative_drift',
                'cusum': cusum_neg,
                'delivered_mu': delivered_mu
            })

    return alerts
```

##### 11.1.3 Longitudinal Trend Detection

```python
def detect_mu_trend(pat_id: str, fld_id: int):
    """Detect systematic trends in MU using Mann-Kendall test."""
    import scipy.stats
    import pymannkendall as mk

    mu_history = get_mu_history(pat_id, fld_id)
    fraction_numbers = list(range(1, len(mu_history) + 1))

    # Mann-Kendall trend test (non-parametric, robust to outliers)
    result = mk.original_test(mu_history)

    if result.p < 0.01 and result.trend in ['increasing', 'decreasing']:
        # Calculate Sen's slope (magnitude of trend)
        slope, intercept = scipy.stats.theilslopes(mu_history, fraction_numbers)[:2]

        return {
            'alert': True,
            'trend': result.trend,
            'p_value': result.p,
            'slope_per_fraction': slope,
            'cumulative_change': slope * len(mu_history)
        }

    return {'alert': False}
```

---

### 11.2 Time-Delayed Corruption Detection

**Threat**: Modifications made to future fractions, creating temporal gap between attack and manifestation.

#### QA Checks

##### 11.2.1 Pre-Treatment Integrity Verification

```python
import hashlib
import json

def verify_treatment_integrity(pat_id: str, fld_id: int, fraction_number: int) -> bool:
    """Verify treatment parameters haven't been modified since approval.

    Returns:
        True if integrity verified, False if modification detected
    """
    # Fetch current treatment parameters
    current_params = get_treatment_parameters(fld_id)

    # Calculate checksum
    param_json = json.dumps(current_params, sort_keys=True)
    current_checksum = hashlib.sha256(param_json.encode()).hexdigest()

    # Retrieve baseline checksum from approval
    baseline_checksum = get_approved_baseline_checksum(fld_id)

    if current_checksum != baseline_checksum:
        alert_modification_detected(pat_id, fld_id, fraction_number)
        return False

    return True
```

##### 11.2.2 Audit Trail Temporal Analysis

```sql
-- Flag modifications to future fractions (suspicious pattern)
WITH FutureFractionMods AS (
    SELECT
        al.table_name,
        al.record_id,
        al.modified_timestamp,
        al.user_id,
        tt.Create_DtTm AS treatment_datetime,
        DATEDIFF(hour, al.modified_timestamp, tt.Create_DtTm) AS hours_before_treatment
    FROM audit_log al
    JOIN TxFieldPoint tfp ON al.record_id = tfp.TFP_ID AND al.table_name = 'TxFieldPoint'
    JOIN TxField tf ON tfp.FLD_ID = tf.FLD_ID
    JOIN TrackTreatment tt ON tf.FLD_ID = tt.FLD_ID
    WHERE al.modified_timestamp < (tt.Create_DtTm - INTERVAL '24 hours')  -- Modified >24h before delivery
)
SELECT
    user_id,
    COUNT(*) AS future_modification_count,
    AVG(hours_before_treatment) AS avg_hours_before,
    MIN(modified_timestamp) AS first_occurrence,
    MAX(modified_timestamp) AS last_occurrence
FROM FutureFractionMods
GROUP BY user_id
HAVING COUNT(*) > 5  -- More than 5 occurrences is suspicious
ORDER BY future_modification_count DESC;
```

##### 11.2.3 Parameter Version Control

```python
def track_parameter_versions(fld_id: int):
    """Maintain version history of treatment parameters."""
    import datetime

    # Snapshot current parameters
    current_params = get_treatment_parameters(fld_id)

    # Store in version history table
    version_entry = {
        'fld_id': fld_id,
        'timestamp': datetime.datetime.now(),
        'parameters': current_params,
        'checksum': calculate_checksum(current_params),
        'user_id': get_current_user()
    }

    store_version_history(version_entry)

    # Compare to previous version
    previous_version = get_latest_version(fld_id, before=version_entry['timestamp'])

    if previous_version:
        diff = compute_parameter_diff(previous_version['parameters'], current_params)
        if diff:
            log_parameter_change(fld_id, diff, version_entry['timestamp'])
```

---

### 11.3 Coordinated Multi-Field Attack Detection

**Threat**: Distribute errors across fields such that individual fields appear normal but composite dose is incorrect.

#### QA Checks

##### 11.3.1 3D Dose Reconstruction

```python
def reconstruct_delivered_dose_3d(pat_id: str, trf_files: list, ct_dataset, structure_set):
    """Calculate 3D dose distribution from delivered parameters.

    Args:
        pat_id: Patient ID
        trf_files: List of TRF file paths (machine logs)
        ct_dataset: CT image dataset
        structure_set: DICOM RT Structure Set

    Returns:
        3D dose grid
    """
    from pymedphys import Delivery

    dose_grid = initialize_dose_grid(ct_dataset)

    for trf_file in trf_files:
        # Parse TRF file to get delivered parameters
        delivery = Delivery.from_trf(trf_file)

        # Calculate dose contribution using Monte Carlo
        field_dose = calculate_dose_monte_carlo(
            ct_dataset=ct_dataset,
            delivery=delivery,
            grid=dose_grid.shape
        )

        dose_grid += field_dose

    return dose_grid
```

##### 11.3.2 Composite Gamma Analysis

```python
def composite_gamma_analysis(planned_dose, delivered_dose, dose_threshold=3, distance_mm=3):
    """Compare delivered composite dose to plan using gamma analysis.

    Args:
        planned_dose: 3D dose array from treatment plan
        delivered_dose: 3D dose array from reconstruction
        dose_threshold: Dose difference threshold (%)
        distance_mm: Distance to agreement (mm)

    Returns:
        Gamma pass rate and failure regions
    """
    from pymedphys import gamma

    gamma_result = gamma(
        reference_dose=planned_dose,
        evaluation_dose=delivered_dose,
        dose_percent_threshold=dose_threshold,
        distance_mm_threshold=distance_mm,
        lower_percent_dose_cutoff=10  # Ignore low dose regions
    )

    pass_rate = (gamma_result <= 1.0).sum() / gamma_result.size * 100

    if pass_rate < 95:  # Alert if <95% pass rate
        failure_regions = identify_failure_regions(gamma_result)
        return {
            'alert': True,
            'pass_rate': pass_rate,
            'failure_regions': failure_regions
        }

    return {'alert': False, 'pass_rate': pass_rate}
```

##### 11.3.3 Aperture Centroid Tracking

```python
def detect_systematic_aperture_shift(pat_id: str, fld_ids: list):
    """Detect coordinated aperture shifts across multiple fields.

    Args:
        pat_id: Patient ID
        fld_ids: List of field IDs for the patient

    Returns:
        Alert if systematic shift detected across all fields
    """
    import numpy as np

    centroids = []
    for fld_id in fld_ids:
        mlc_data = get_mlc_positions(fld_id)
        centroid = calculate_aperture_centroid(mlc_data)
        centroids.append(centroid)

    # Check if all centroids shifted in same direction
    centroid_shifts = np.array(centroids) - np.array(get_planned_centroids(fld_ids))
    mean_shift = np.mean(centroid_shifts, axis=0)
    shift_consistency = np.linalg.norm(np.std(centroid_shifts, axis=0))

    # Alert if mean shift > 2mm and highly consistent (low variance)
    if np.linalg.norm(mean_shift) > 2.0 and shift_consistency < 1.0:
        return {
            'alert': True,
            'mean_shift_mm': mean_shift,
            'shift_direction': mean_shift / np.linalg.norm(mean_shift),
            'consistency': shift_consistency
        }

    return {'alert': False}
```

---

### 11.4 Statistical Camouflage Detection

**Threat**: Systematic bias hidden within normal variance ranges through careful statistical manipulation.

#### QA Checks

##### 11.4.1 One-Sample t-Test for Systematic Bias

```python
def detect_systematic_bias(parameter_series: list, expected_mean: float):
    """Test if parameter mean significantly differs from expected value.

    Args:
        parameter_series: List of parameter values across fractions
        expected_mean: Expected mean value (from treatment plan)

    Returns:
        Alert if significant bias detected (p < 0.01)
    """
    import scipy.stats
    import numpy as np

    # One-sample t-test
    t_statistic, p_value = scipy.stats.ttest_1samp(parameter_series, expected_mean)

    actual_mean = np.mean(parameter_series)
    bias = actual_mean - expected_mean

    if p_value < 0.01:  # Statistically significant bias
        return {
            'alert': True,
            'p_value': p_value,
            't_statistic': t_statistic,
            'expected_mean': expected_mean,
            'actual_mean': actual_mean,
            'bias': bias,
            'bias_percent': bias / expected_mean * 100
        }

    return {'alert': False}
```

##### 11.4.2 Benford's Law Analysis

```python
def benford_law_test(data: list):
    """Detect artificial data using first-digit distribution analysis.

    Natural data follows Benford's Law: P(d) = log10(1 + 1/d)
    Artificial data often violates this distribution.

    Args:
        data: List of numerical values

    Returns:
        Alert if significant deviation from Benford's Law detected
    """
    import numpy as np
    import scipy.stats

    # Extract first digits
    first_digits = [int(str(abs(x)).replace('.', '')[0]) for x in data if x != 0]

    # Expected Benford distribution
    benford_expected = [np.log10(1 + 1/d) for d in range(1, 10)]

    # Observed distribution
    observed_counts = np.histogram(first_digits, bins=range(1, 11))[0]
    observed_freq = observed_counts / len(first_digits)

    # Expected counts
    expected_counts = len(first_digits) * np.array(benford_expected)

    # Chi-square test
    chi2_statistic, p_value = scipy.stats.chisquare(observed_counts, expected_counts)

    if p_value < 0.01:  # Significant deviation from Benford's Law
        return {
            'alert': True,
            'p_value': p_value,
            'chi2_statistic': chi2_statistic,
            'interpretation': 'Data may be artificially generated or manipulated'
        }

    return {'alert': False}
```

##### 11.4.3 Run Test for Randomness

```python
def runs_test_randomness(parameter_series: list, median: float = None):
    """Test if sequence is random or has systematic patterns.

    A 'run' is a sequence of consecutive values above/below median.
    Non-random data has too few or too many runs.

    Args:
        parameter_series: List of parameter values
        median: Expected median (if None, use sample median)

    Returns:
        Alert if non-random pattern detected
    """
    import numpy as np
    from statsmodels.sandbox.stats.runs import runstest_1samp

    if median is None:
        median = np.median(parameter_series)

    # Convert to binary sequence (above/below median)
    binary_series = [1 if x > median else 0 for x in parameter_series]

    # Runs test
    z_statistic, p_value = runstest_1samp(binary_series)

    if p_value < 0.01:  # Non-random pattern
        return {
            'alert': True,
            'p_value': p_value,
            'z_statistic': z_statistic,
            'interpretation': 'Too few runs' if z_statistic < 0 else 'Too many runs'
        }

    return {'alert': False}
```

---

### 11.5 Audit Trail Manipulation Detection

**Threat**: Modification of audit logs to hide malicious database changes.

#### QA Checks

##### 11.5.1 External Immutable Audit Log

```python
class BlockchainAuditLog:
    """Write-once audit log using blockchain for immutability."""

    def __init__(self):
        self.chain = []

    def log_database_modification(self, table: str, record_id: int, user: str,
                                  timestamp, operation: str, old_value, new_value):
        """Log database write to immutable blockchain.

        This log cannot be retroactively modified without detection.
        """
        import hashlib

        # Previous block hash (chain integrity)
        previous_hash = self.chain[-1]['hash'] if self.chain else '0' * 64

        # Create entry
        entry = {
            'table': table,
            'record_id': record_id,
            'user': user,
            'timestamp': timestamp.isoformat(),
            'operation': operation,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'previous_hash': previous_hash
        }

        # Calculate hash (includes previous hash for chain integrity)
        entry_json = json.dumps(entry, sort_keys=True)
        entry['hash'] = hashlib.sha256(entry_json.encode()).hexdigest()

        # Append to chain
        self.chain.append(entry)

        # Write to WORM storage (Write Once Read Many)
        self.persist_to_worm_storage(entry)

    def verify_chain_integrity(self):
        """Verify no blocks have been tampered with."""
        for i in range(1, len(self.chain)):
            if self.chain[i]['previous_hash'] != self.chain[i-1]['hash']:
                raise IntegrityError(f"Chain broken at block {i}")
```

##### 11.5.2 Transaction Log Forensics

```sql
-- Detect undocumented database modifications using SQL Server transaction log
-- Compare transaction log to audit table
WITH TransactionLogEntries AS (
    SELECT
        [Transaction ID] AS txn_id,
        [Begin Time] AS txn_time,
        [Operation] AS operation,
        [AllocUnitName] AS table_name,
        [Page ID] AS page_id,
        [Slot ID] AS slot_id
    FROM sys.fn_dblog(NULL, NULL)
    WHERE [Operation] IN ('LOP_MODIFY_ROW', 'LOP_INSERT_ROWS', 'LOP_DELETE_ROWS')
      AND [AllocUnitName] LIKE '%TxField%' OR [AllocUnitName] LIKE '%TrackTreatment%'
),
AuditedTransactions AS (
    SELECT
        transaction_id,
        timestamp
    FROM audit_log
)
SELECT
    tl.txn_id,
    tl.txn_time,
    tl.operation,
    tl.table_name,
    'NOT IN AUDIT LOG' AS audit_status
FROM TransactionLogEntries tl
LEFT JOIN AuditedTransactions at ON tl.txn_id = at.transaction_id
WHERE at.transaction_id IS NULL  -- Transactions not in audit log (suspicious)
ORDER BY tl.txn_time DESC;
```

##### 11.5.3 User Session Impossibility Detection

```python
def detect_impossible_user_activity(user_id: str, audit_entries: list):
    """Detect physically impossible user activity patterns.

    Examples:
    - User logged in from two locations simultaneously
    - User traveled impossibly fast between locations
    - User active during documented vacation/sick leave

    Args:
        user_id: User ID to analyze
        audit_entries: List of audit log entries with timestamp and location

    Returns:
        List of impossible activity alerts
    """
    from geopy.distance import geodesic
    from datetime import timedelta

    alerts = []

    for i in range(len(audit_entries) - 1):
        current = audit_entries[i]
        next_entry = audit_entries[i + 1]

        time_diff = next_entry['timestamp'] - current['timestamp']
        location_diff_km = geodesic(
            current['location_coords'],
            next_entry['location_coords']
        ).kilometers

        # Maximum credible speed (driving + some buffer)
        max_speed_kmh = 120
        required_speed = location_diff_km / (time_diff.total_seconds() / 3600)

        if required_speed > max_speed_kmh:
            alerts.append({
                'user_id': user_id,
                'entry_1': current,
                'entry_2': next_entry,
                'time_diff_minutes': time_diff.total_seconds() / 60,
                'distance_km': location_diff_km,
                'required_speed_kmh': required_speed,
                'interpretation': f'User would need to travel {required_speed:.1f} km/h'
            })

    return alerts
```

---

### 11.6 Targeted Patient Selection Detection

**Threat**: Selective attacks on specific patients to avoid population-level statistical detection.

#### QA Checks

##### 11.6.1 Per-Patient Anomaly Detection

```python
def patient_level_anomaly_detection(pat_id: str):
    """Detect anomalies at individual patient level (not just population).

    Uses patient's own baseline (first 3 fractions) rather than
    population statistics.

    Args:
        pat_id: Patient ID

    Returns:
        Anomaly alerts for specific fractions
    """
    from sklearn.ensemble import IsolationForest

    # Extract features for all fractions
    all_features = extract_multimodal_features(pat_id)

    # Baseline: first 3 fractions (after setup, before potential attack)
    baseline_features = all_features[:3]

    # Train isolation forest on baseline
    detector = IsolationForest(contamination=0.1, random_state=42)
    detector.fit(baseline_features)

    # Score all fractions
    anomaly_scores = detector.score_samples(all_features)

    # Alert on fractions with low scores
    alerts = []
    for i, score in enumerate(anomaly_scores):
        if score < -0.5:  # Anomaly threshold
            alerts.append({
                'fraction': i + 1,
                'anomaly_score': score,
                'features': all_features[i]
            })

    return alerts
```

##### 11.6.2 Demographic Bias Analysis

```python
def detect_demographic_bias(anomaly_database):
    """Test if anomalies disproportionately affect certain demographics.

    Insider threats may target vulnerable populations (elderly, poor prognosis).

    Args:
        anomaly_database: DataFrame with anomaly flags and patient demographics

    Returns:
        Alert if significant demographic bias detected
    """
    import pandas as pd
    import scipy.stats

    # Test each demographic variable
    demographic_vars = ['age_bin', 'diagnosis', 'treatment_site', 'prognosis']

    alerts = []
    for var in demographic_vars:
        # Create contingency table: anomaly (yes/no) × demographic group
        contingency = pd.crosstab(
            anomaly_database['has_anomaly'],
            anomaly_database[var]
        )

        # Chi-square test for independence
        chi2, p_value, dof, expected = scipy.stats.chi2_contingency(contingency)

        if p_value < 0.01:  # Significant association
            # Calculate which groups are over/under-represented
            observed = contingency.loc[True]  # Anomaly counts
            expected_anomalies = expected[1]
            bias_ratio = observed / expected_anomalies

            alerts.append({
                'demographic_variable': var,
                'p_value': p_value,
                'chi2_statistic': chi2,
                'bias_ratios': bias_ratio.to_dict(),
                'interpretation': f'Anomalies not uniformly distributed across {var}'
            })

    return alerts
```

##### 11.6.3 Multi-Anomaly Flagging

```python
def flag_multiple_independent_anomalies(pat_id: str, base_rate: float = 0.05):
    """Flag patients with multiple independent anomalies.

    The probability of multiple independent anomalies occurring by chance
    is very low (product of individual probabilities).

    Args:
        pat_id: Patient ID
        base_rate: Expected rate of individual anomalies (default 5%)

    Returns:
        Alert if multiple anomalies detected (low probability)
    """
    anomaly_types = {}

    # Check different independent anomaly types
    if detect_mlc_anomaly(pat_id):
        anomaly_types['mlc'] = True

    if detect_mu_anomaly(pat_id):
        anomaly_types['mu'] = True

    if detect_position_anomaly(pat_id):
        anomaly_types['position'] = True

    if detect_angle_anomaly(pat_id):
        anomaly_types['angle'] = True

    # Calculate probability of this many anomalies occurring by chance
    num_anomalies = sum(anomaly_types.values())

    if num_anomalies >= 2:
        # Probability = base_rate^num_anomalies
        probability = base_rate ** num_anomalies

        return {
            'alert': True,
            'pat_id': pat_id,
            'num_anomalies': num_anomalies,
            'anomaly_types': [k for k, v in anomaly_types.items() if v],
            'probability_by_chance': probability,
            'interpretation': f'{num_anomalies} independent anomalies is highly suspicious (p={probability:.6f})'
        }

    return {'alert': False}
```

---

### Additional Data Requirements for Malicious Detection

1. **External Immutable Audit Log**: Blockchain or WORM storage off-database
2. **Complete TRF Files**: Machine logs for all fractions (not just periodic)
3. **Portal Dosimetry Images**: EPID images for every fraction
4. **Network Time Protocol Logs**: For timestamp validation
5. **User Geolocation Data**: IP addresses, badge swipes, physical access logs
6. **Staff Schedules**: Shift assignments for correlation analysis
7. **Patient Demographics**: Age, diagnosis, prognosis for bias detection
8. **3D Dose Calculation Engine**: Monte Carlo or collapse cone for reconstruction
9. **CT Datasets**: For delivered dose reconstruction
10. **Structure Sets**: DICOM RT Struct for DVH analysis

---

### Malicious Detection Implementation Priority

**Critical (Implement Immediately)**:
1. External immutable audit log (blockchain/WORM)
2. Pre-treatment integrity verification (checksums)
3. CUSUM charts for cumulative dose tracking
4. Per-patient anomaly detection

**High Priority (Implement Within 3 Months)**:
5. 3D dose reconstruction and gamma analysis
6. Transaction log forensics
7. Demographic bias detection
8. Multi-modal cross-validation (TRF + portal + phantom)

**Medium Priority (Implement Within 6 Months)**:
9. User session impossibility detection
10. Benford's Law and randomness tests
11. Aperture centroid tracking
12. Temporal clustering analysis

---

## Implementation Roadmap

### Phase 1: Immediate (No Schema Changes)
- Implement queries for existing field validation
- Set up scheduled SQL Agent jobs for daily QA checks
- Create dashboard for QA metrics

### Phase 2: External Data Integration (2-3 months)
- Parse and import machine log files
- Link DICOM RT Plans to TxField records
- Integrate CBCT/portal imaging metadata

### Phase 3: Schema Enhancements (3-6 months)
- Add control point timing and MU fields
- Create machine configuration tables
- Implement audit logging

### Phase 4: Real-Time Monitoring (6-12 months)
- Database triggers for immediate validation
- Integration with clinical workflow (block treatment on critical errors)
- Automated reporting to physics team
