# PyMedPhys MCP Server

This module provides a Model Context Protocol (MCP) interface for PyMedPhys,
enabling AI assistants (like Claude) to interact with medical physics data and tools.

## Installation

```bash
pip install pymedphys[mcp]
```

## Quick Start

### Starting the Server

```bash
# Basic server start
pymedphys mcp serve

# With DICOM and TRF directories
pymedphys mcp serve --dicom-dir /path/to/dicom --trf-dir /path/to/trf

# With Mosaiq connection
pymedphys mcp serve --mosaiq-server sql-server.hospital.org
```

### Claude Desktop Configuration

Generate configuration for Claude Desktop:

```bash
pymedphys mcp config --format claude-desktop
```

This outputs JSON to add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pymedphys": {
      "command": "python",
      "args": ["-m", "pymedphys", "mcp", "serve"]
    }
  }
}
```

## Available Resources

### Mosaiq Resources (when connected)

- `mosaiq://patients` - List of patients in Mosaiq
- `mosaiq://machines` - List of treatment machines
- `mosaiq://patient/{id}` - Specific patient data
- `mosaiq://field/{id}` - Treatment field data

### DICOM Resources

- `dicom://{directory}/{path}` - DICOM files in configured directories

### TRF Resources

- `trf://{directory}/{path}` - TRF log files in configured directories

## Available Tools

### Analysis Tools

#### `gamma_analysis`
Perform gamma analysis comparing two dose distributions.

```json
{
  "name": "gamma_analysis",
  "arguments": {
    "reference_dose_path": "/path/to/reference.dcm",
    "evaluation_dose_path": "/path/to/evaluation.dcm",
    "dose_threshold_percent": 3,
    "distance_threshold_mm": 3
  }
}
```

#### `calculate_metersetmap`
Calculate fluence maps from delivery data.

```json
{
  "name": "calculate_metersetmap",
  "arguments": {
    "source": "trf",
    "source_path": "/path/to/logfile.trf",
    "grid_resolution": 1.0,
    "output_format": "json"
  }
}
```

### DICOM Tools

#### `read_dicom`
Read and parse DICOM files.

```json
{
  "name": "read_dicom",
  "arguments": {
    "file_path": "/path/to/rtplan.dcm",
    "include_pixel_data": false
  }
}
```

#### `anonymize_dicom`
Anonymize DICOM files.

```json
{
  "name": "anonymize_dicom",
  "arguments": {
    "input_path": "/path/to/original.dcm",
    "output_path": "/path/to/anonymized.dcm",
    "delete_private_tags": true
  }
}
```

#### `create_rt_plan`
Create RT Plan DICOM from delivery data.

```json
{
  "name": "create_rt_plan",
  "arguments": {
    "source_type": "trf",
    "source_data": {"file_path": "/path/to/logfile.trf"},
    "template_path": "/path/to/template.dcm",
    "output_path": "/path/to/new_plan.dcm"
  }
}
```

#### `create_rtpconnect`
Create prescription files for Mosaiq import.

```json
{
  "name": "create_rtpconnect",
  "arguments": {
    "prescription": {
      "site_name": "Brain Met",
      "fractions": 5,
      "dose_per_fraction": 600,
      "technique": "SRS"
    },
    "patient_info": {
      "patient_id": "12345",
      "patient_name": "DOE^JOHN"
    },
    "output_path": "/path/to/prescription.dcm",
    "instructions": true
  }
}
```

### TRF Tools

#### `read_trf`
Parse Elekta TRF log files.

```json
{
  "name": "read_trf",
  "arguments": {
    "file_path": "/path/to/logfile.trf",
    "output_format": "summary"
  }
}
```

#### `compare_trf_to_plan`
Compare delivery logs to planned parameters.

```json
{
  "name": "compare_trf_to_plan",
  "arguments": {
    "trf_path": "/path/to/logfile.trf",
    "plan_source": "dicom",
    "plan_path_or_id": "/path/to/rtplan.dcm"
  }
}
```

### Mosaiq Tools (when connected)

#### `query_mosaiq`
Query the Mosaiq database.

```json
{
  "name": "query_mosaiq",
  "arguments": {
    "query_type": "patient",
    "parameters": {
      "patient_id": "12345"
    }
  }
}
```

#### `get_patient_chart`
Retrieve comprehensive patient data.

```json
{
  "name": "get_patient_chart",
  "arguments": {
    "patient_id": "12345",
    "include_sections": ["demographics", "prescriptions", "treatments"]
  }
}
```

## Prompt Templates

### `patient_qa_review`
Review patient QA data and provide analysis summary.

### `treatment_comparison`
Compare planned vs delivered treatment parameters.

### `create_prescription`
Guide for creating a prescription RTPCONNECT file for Mosaiq import.

## Example Conversations

### Creating a Prescription

**User:** Create a prescription for patient John Doe (ID: 12345) for brain metastasis
treatment - 30 Gy in 5 fractions using SRS technique.

**Claude:** I'll create an RTPCONNECT prescription file for this patient.
*Uses create_rtpconnect tool*

The prescription file has been created at `/output/prescription_12345.dcm`.

To import into Mosaiq:
1. Open Mosaiq and navigate to patient John Doe's chart
2. Go to File > Import > DICOM RT Plan
3. Select the file: prescription_12345.dcm
4. Review and verify:
   - Site Name: Brain Met
   - Fractions: 5
   - Dose per Fraction: 600 cGy
   - Total Dose: 3000 cGy
5. Have the physician approve the prescription

### Analyzing a Treatment Log

**User:** Compare the TRF log file from today's treatment to the planned values.

**Claude:** I'll analyze the TRF file and compare it to the plan.
*Uses read_trf and compare_trf_to_plan tools*

Analysis Results:
- MU Agreement: 99.8% (planned: 245.3, delivered: 245.8)
- Gantry Range: Matches plan (178° - 182°)
- MLC Deviation: Max 0.3mm, Mean 0.1mm
- Overall Status: PASS

All delivery parameters are within tolerance.

## Programmatic Usage

```python
import asyncio
from pymedphys._mcp import create_server, run_server

# Create server with custom configuration
server = create_server(
    dicom_directories=["/data/dicom"],
    trf_directories=["/data/trf"],
)

# Or run directly
asyncio.run(run_server(
    dicom_directories=["/data/dicom"],
    trf_directories=["/data/trf"],
))
```

## Security Considerations

- Mosaiq queries use parameterized SQL to prevent injection
- Custom SQL queries block dangerous operations (DROP, DELETE, etc.)
- Patient data should be handled according to institutional policies
- Consider network isolation when exposing MCP server

## Troubleshooting

### Server won't start
- Ensure `mcp` package is installed: `pip install mcp>=1.0.0`
- Check Python version (requires 3.10+)

### Mosaiq connection fails
- Verify SQL Server hostname and port
- Check credentials in keyring
- Ensure pymssql is installed

### DICOM tools fail
- Verify pydicom is installed
- Check file paths are accessible
- Ensure DICOM files are valid
