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
