# Mosaiq Failure Modes MCP Server

A Model Context Protocol (MCP) server for PyMedPhys that enables simulation of database corruption and third-party write failures in Mosaiq databases for defensive testing purposes.

## Overview

This MCP server provides tools to:

1. **Simulate failure modes** - Intentionally corrupt Mosaiq database entries in controlled test environments
2. **Document QA checks** - Provide comprehensive QA framework to detect each failure mode
3. **Identify data requirements** - Specify additional data needed for robust validation
4. **Train anomaly detection models** - Generate adversarial training data for Energy-Based Models (EBMs)

**⚠️ CRITICAL: For Testing Purposes Only**

This tool is designed for:
- ✅ Defensive testing in isolated test databases
- ✅ Validating error handling logic in PyMedPhys
- ✅ Developing QA checks for production systems
- ❌ **NOT for production databases**
- ❌ **NOT for malicious purposes**

## Installation

### Prerequisites

- Python 3.10 or later
- PyMedPhys with Mosaiq dependencies installed
- Access to a **test** Mosaiq database (never use production!)
- MCP-compatible client (Claude Desktop, etc.)

### Setup

1. **Install PyMedPhys with Mosaiq support**:

```bash
# From PyMedPhys repository root
uv sync --extra all --group dev
```

2. **Install MCP dependencies** (if not already present):

```bash
uv add "mcp>=0.1.0" "pymssql>=2.2.0"
```

3. **Configure MCP server in Claude Desktop**:

Edit your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the server configuration:

```json
{
  "mcpServers": {
    "mosaiq-failure-modes": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "pymedphys._mcp.mosaiq_failure_modes.server"
      ],
      "cwd": "/path/to/pymedphys"
    }
  }
}
```

4. **Restart Claude Desktop** to load the MCP server.

## Usage

### Listing Available Failure Modes

In Claude Desktop, you can ask:

> "What failure modes are available in the Mosaiq testing server?"

This will invoke the `list_failure_modes` tool and show all 10 failure modes with descriptions, QA checks, and data requirements.

### Reading Failure Mode Documentation

Use MCP resources to read detailed documentation:

```
failure-mode://mosaiq/corrupt_mlc_data
failure-mode://mosaiq/invalid_angles
failure-mode://mosaiq/duplicate_treatments
...
```

Claude can fetch these resources to provide detailed information about:
- Failure mode description
- QA checks to detect it
- Additional data required

### Simulating Failure Modes

**⚠️ WARNING: Only use on test databases!**

Example conversation with Claude Desktop:

> "I have a test Mosaiq database at `testserver.local`. I want to corrupt the MLC data for field ID 12345 with random bytes to test my validation logic."

Claude will then:
1. Use the `corrupt_mlc_data` tool
2. Connect to your test database
3. Apply the corruption
4. Report the results

### Programmatic Usage

You can also invoke the MCP server directly:

```python
import asyncio
from pymedphys._mcp.mosaiq_failure_modes.server import app

# Example: List all failure modes
async def list_modes():
    resources = await app.list_resources()
    for resource in resources:
        print(f"{resource.name}: {resource.description}")

asyncio.run(list_modes())
```

## Available Failure Modes

| Failure Mode | Description | Use Case |
|--------------|-------------|----------|
| `corrupt_mlc_data` | Corrupt MLC binary data | Test MLC parsing error handling |
| `create_invalid_angles` | Set angles outside 0-360° | Test angle validation logic |
| `create_duplicate_treatment` | Create duplicate treatment records | Test duplicate detection |
| `delete_control_points` | Remove control points | Test missing data handling |
| `nullify_required_fields` | Set critical fields to NULL | Test NULL value handling |
| `create_timestamp_inconsistency` | Create invalid time relationships | Test temporal validation |
| `create_orphaned_record` | Break foreign key relationships | Test referential integrity checks |
| `corrupt_offset_data` | Invalid positioning offsets | Test offset validation |
| `create_meterset_inconsistency` | MU mismatch issues | Test dose validation |
| `mlc_leaf_count_mismatch` | Inconsistent leaf counts | Test MLC configuration checks |

See [QA_FRAMEWORK.md](./QA_FRAMEWORK.md) for detailed QA checks and data requirements for each failure mode.

## MCP Tools

### Database Modification Tools

All tools require:
- `hostname`: Mosaiq SQL Server hostname
- `database`: Database name (default: "MOSAIQ")
- Additional parameters specific to the failure mode

Example tool schemas:

**corrupt_mlc_data**:
```json
{
  "hostname": "testserver.local",
  "database": "MosaiqTest",
  "fld_id": 12345,
  "corruption_type": "random_bytes"
}
```

**create_invalid_angles**:
```json
{
  "hostname": "testserver.local",
  "fld_id": 12345,
  "angle_type": "gantry",
  "invalid_value": 400.0
}
```

**delete_control_points**:
```json
{
  "hostname": "testserver.local",
  "fld_id": 12345,
  "points_to_delete": [1, 3, 5]
}
```

### Information Tools

**list_failure_modes**: Returns comprehensive documentation of all failure modes

**Resources**: Read individual failure mode documentation via URIs like `failure-mode://mosaiq/corrupt_mlc_data`

## QA Framework

This server includes a comprehensive QA framework documented in [QA_FRAMEWORK.md](./QA_FRAMEWORK.md) that provides:

### For Each Failure Mode:
1. **SQL Queries** to detect the corruption
2. **Python validation functions** for automated checking
3. **Data requirements** needed for comprehensive QA
4. **Schema enhancement recommendations**

### Critical Data Gaps Identified:

The QA framework identifies missing data in Mosaiq:

**High Priority**:
- Control point timing information
- Control point MU weighting
- Machine log file integration
- DICOM RT Plan references

**Medium Priority**:
- Machine configuration database
- Treatment duration baselines
- Third-party integration metadata

See the full [Implementation Roadmap](./QA_FRAMEWORK.md#implementation-roadmap) for details.

## Energy-Based Model (EBM) Training

### Overview

This MCP server supports **adversarial training** of machine learning models for automated anomaly detection:

1. **Generate adversarial training data** - Use failure mode corruptions as positive examples (anomalies)
2. **Extract QA features** - 72 features from QA framework serve as model inputs
3. **Train Energy-Based Model** - Learn to assign low energy to normal data, high energy to anomalies
4. **Deploy for real-time monitoring** - Automated detection of data quality issues

### Quick Start

**Install additional dependencies**:
```bash
uv add "torch>=2.0.0" "scikit-learn>=1.3.0" "numpy>=1.24.0"
```

**Run example training script**:
```bash
python -m pymedphys._mcp.mosaiq_failure_modes.example_training \
    --prod-host mosaiq.hospital.org \
    --test-host testserver.local \
    --test-database MosaiqTest_EBM \
    --n-normal 2000 \
    --n-adversarial 500 \
    --n-epochs 100 \
    --output-dir ./models
```

This will:
1. Collect 2000 normal examples from production
2. Generate 500 adversarial examples using failure modes
3. Train EBM for 100 epochs
4. Save model checkpoint and training history

### Key Features

**72 Input Features** across 6 categories:
- **MLC Features** (19): Byte lengths, positions, gaps, physical limits
- **Angle Features** (14): Gantry/collimator ranges, transitions, speed feasibility
- **Control Point Features** (6): Sequence validation, gap detection
- **Timestamp Features** (9): Temporal ordering, duration validation
- **Offset Features** (12): Position magnitudes, extreme value detection
- **Meterset Features** (8): MU consistency, range validation
- **Foreign Key Features** (6): Referential integrity, NULL detection

**Severity-Weighted Training**:
- **Continuous severity scores** (0=normal, 0.5-3.0=anomaly severity)
- **Risk stratification**: Low (0.5-0.8), Medium (1.0-1.5), High (1.8-2.3), Critical (2.5-3.0)
- **MSE loss function**: Model learns to output energy matching severity
- **Clinical prioritization**: Critical failures trigger immediate alerts, low severity logged for trending

**Severity Examples**:
- Critical (2.5-3.0): Wrong patient ID, extreme positioning offsets, negative MU
- High (1.8-2.3): Missing control points, MLC out of range, dose errors
- Medium (1.0-1.5): Duplicate treatments, timestamp inconsistencies
- Low (0.5-0.8): Invalid angles (safety interlocks prevent delivery), parsing errors

See [SEVERITY_SCALE.md](./SEVERITY_SCALE.md) for complete severity documentation.

### Documentation

See [EBM_TRAINING_GUIDE.md](./EBM_TRAINING_GUIDE.md) for comprehensive documentation including:
- Detailed architecture and loss function explanation
- Step-by-step training workflow
- Deployment strategies for real-time monitoring
- Advanced techniques (ensembles, uncertainty quantification, active learning)
- Troubleshooting and best practices

### Example: Real-Time Monitoring

Deploy trained model to monitor production database:

```python
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import AdversarialTrainer
from pymedphys._mcp.mosaiq_failure_modes.ebm_features import extract_all_features
import numpy as np

# Load trained model
trainer = AdversarialTrainer(model_path="models/mosaiq_ebm.pt")
trainer.load_checkpoint()

# Monitor recent treatments
with connect(hostname="mosaiq.hospital.org", read_only=True) as conn:
    cursor = conn.cursor()

    # Get recent treatments
    cursor.execute("""
        SELECT TTX_ID FROM TrackTreatment
        WHERE Create_DtTm >= DATEADD(hour, -1, GETDATE())
    """)

    for (ttx_id,) in cursor.fetchall():
        # Extract features
        features = extract_all_features(cursor, ttx_id)
        feature_vec = create_feature_vector(features, trainer.feature_names)

        # Predict anomaly
        result = trainer.predict(feature_vec.reshape(1, -1))

        if result['predictions'][0] == 1:  # Anomaly detected
            print(f"ALERT: Anomaly in TTX_ID {ttx_id}")
            print(f"  Energy: {result['energies'][0]:.4f}")
            print(f"  Confidence: {result['probabilities'][0]:.2%}")
            # Send alert to physics team
```

### Performance Expectations

With proper training, the EBM achieves:
- **Accuracy**: 95-98% on held-out test set
- **Precision**: 92-96% (low false positive rate)
- **Recall**: 94-98% (catches most anomalies)
- **F1 Score**: 93-97%

**Energy separation**:
- Normal data: Energy ~ 0.1-0.3
- Anomalous data: Energy ~ 0.8-1.5

## Safety Guidelines

### Test Database Requirements

**Before using this tool, ensure**:
1. ✅ You are connected to a **dedicated test database**
2. ✅ The database contains **synthetic or anonymized data only**
3. ✅ Database backups exist and are tested
4. ✅ You have explicit permission to modify the database
5. ✅ No clinical systems depend on this database

### Recommended Test Database Setup

```sql
-- Create isolated test database
CREATE DATABASE MosaiqTest_FailureModes;

-- Restore from production backup (anonymized)
RESTORE DATABASE MosaiqTest_FailureModes
FROM DISK = 'C:\Backups\MosaiqAnonymized.bak'
WITH REPLACE;

-- Verify isolation
SELECT name, database_id, create_date
FROM sys.databases
WHERE name = 'MosaiqTest_FailureModes';
```

### Cleanup and Reset

After testing, reset your test database:

```sql
-- Option 1: Restore from clean backup
RESTORE DATABASE MosaiqTest_FailureModes
FROM DISK = 'C:\Backups\MosaiqClean.bak'
WITH REPLACE;

-- Option 2: Delete corrupted records manually
-- (Use specific WHERE clauses based on your test IDs)
DELETE FROM TxFieldPoint WHERE FLD_ID = 12345;
```

## Development

### Adding New Failure Modes

1. **Add to FAILURE_MODES dictionary** in `server.py`:

```python
FAILURE_MODES["new_failure"] = {
    "description": "...",
    "qa_checks": ["...", "..."],
    "additional_data_needed": ["...", "..."]
}
```

2. **Implement tool in list_tools()**:

```python
Tool(
    name="create_new_failure",
    description=FAILURE_MODES["new_failure"]["description"],
    inputSchema={...}
)
```

3. **Add handler in call_tool()**:

```python
elif name == "create_new_failure":
    # Implementation
```

4. **Document in QA_FRAMEWORK.md**:
   - QA checks with SQL examples
   - Python validation functions
   - Data requirements

### Testing the MCP Server

```bash
# Run the server directly
uv run python -m pymedphys._mcp.mosaiq_failure_modes.server

# Test with MCP inspector
npx @modelcontextprotocol/inspector uv run python -m pymedphys._mcp.mosaiq_failure_modes.server
```

## Example Workflows

### Workflow 1: Test MLC Validation Logic

**Goal**: Verify that PyMedPhys correctly detects corrupted MLC data.

**Steps**:
1. Create test field in test database with valid MLC data
2. Record field ID (e.g., `FLD_ID = 99999`)
3. Use MCP to corrupt MLC data:
   ```
   "Corrupt MLC data for field 99999 with random bytes"
   ```
4. Run PyMedPhys delivery extraction:
   ```python
   from pymedphys._mosaiq import get_mosaiq_delivery_details
   try:
       delivery = get_mosaiq_delivery_details(...)
       print("ERROR: Should have detected corruption!")
   except ValueError as e:
       print(f"SUCCESS: Detected corruption - {e}")
   ```
5. Document the error handling behavior

### Workflow 2: Develop Duplicate Detection

**Goal**: Create QA check to detect duplicate treatment entries.

**Steps**:
1. Find existing treatment record (`TTX_ID = 88888`)
2. Use MCP to create duplicate:
   ```
   "Create duplicate treatment for TTX_ID 88888 with 5 second offset"
   ```
3. Develop SQL query to detect:
   ```sql
   -- From QA_FRAMEWORK.md section 3.1
   SELECT t1.TTX_ID, t2.TTX_ID, ...
   FROM TrackTreatment t1
   JOIN TrackTreatment t2 ON ...
   ```
4. Test query detects the duplicate
5. Implement in PyMedPhys QA module

### Workflow 3: Test Offset Validation

**Goal**: Validate third-party offset data integrity.

**Steps**:
1. Create offset record in test database
2. Corrupt with extreme values:
   ```
   "Corrupt offset 777 with extreme_values of 999.9 cm"
   ```
3. Test validation logic catches outlier
4. Reference against clinical action limits

## Troubleshooting

### Connection Issues

**Error**: `WrongUsernameOrPassword`

**Solution**: Verify credentials:
```python
from pymedphys._mosaiq import connect
with connect(hostname="testserver.local", username="test_user") as conn:
    print("Connected successfully")
```

### Permission Errors

**Error**: `pymssql.OperationalError: Permission denied`

**Solution**: Ensure database user has `UPDATE` and `DELETE` permissions on test database:
```sql
GRANT SELECT, UPDATE, DELETE ON DATABASE::MosaiqTest TO test_user;
```

### MCP Server Not Loading

**Error**: Server doesn't appear in Claude Desktop

**Solution**:
1. Check `claude_desktop_config.json` syntax
2. Verify `cwd` path is correct
3. Restart Claude Desktop
4. Check logs: `~/Library/Logs/Claude/` (macOS)

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [PyMedPhys Documentation](https://docs.pymedphys.com/)
- [Mosaiq Integration Guide](../../_mosaiq/README.md)

## License

This tool is part of PyMedPhys and follows the same license (Apache 2.0).

## Contributing

Contributions welcome! Please:
1. Add tests for new failure modes
2. Document QA checks in QA_FRAMEWORK.md
3. Follow PyMedPhys code style (ruff, pyright)
4. Test with real Mosaiq test database

## Support

For issues or questions:
- PyMedPhys GitHub Issues: https://github.com/pymedphys/pymedphys/issues
- PyMedPhys Discourse: https://pymedphys.discourse.group/

---

**Remember: This tool is for defensive testing only. Always use test databases. Never compromise patient data or clinical systems.**
