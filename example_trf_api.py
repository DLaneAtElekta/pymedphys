#!/usr/bin/env python
"""
Example demonstrating the PyMedPhys TRF reading API.

This shows how to read Elekta linac TRF (treatment log) files.
"""

import pymedphys

print("="*70)
print("PyMedPhys TRF File Reading API")
print("="*70)

print("""
TRF files are binary log files from Elekta linear accelerators that record
treatment delivery information including:
  - MLC leaf positions
  - Jaw positions
  - Gantry angles
  - Collimator angles
  - Monitor units delivered
  - Control point information

""")

print("METHOD 1: Read TRF into pandas DataFrames")
print("-" * 70)
print("""
header_df, table_df = pymedphys.trf.read('/path/to/file.trf')

# header_df contains:
#   - Patient ID
#   - Plan information
#   - Machine details
#   - Treatment date/time
#
# table_df contains control point data with columns like:
#   - Step Dose/Actual Value (Mu)
#   - Gantry Rtn/Actual Angle (deg)
#   - Collimator Rtn/Actual Angle (deg)
#   - MLC Leaf positions (A1-A80, B1-B80 for Agility)
#   - Jaw positions (X1, X2, Y1, Y2)
""")

print("\nMETHOD 2: Create a Delivery object")
print("-" * 70)
print("""
delivery = pymedphys.Delivery.from_trf('/path/to/file.trf')

# Access delivery data:
monitor_units = delivery.monitor_units  # numpy array of cumulative MU
gantry = delivery.gantry                # gantry angles (degrees)
collimator = delivery.collimator        # collimator angles (degrees)
mlc = delivery.mlc                      # MLC positions (shape: [cps, leaves, 2])
jaw = delivery.jaw                      # jaw positions (shape: [cps, 4])

# Use for further analysis:
metersetmap = delivery.metersetmap()    # Create a meterset map
""")

print("\nMETHOD 3: Using the CLI")
print("-" * 70)
print("""
# Convert TRF to CSV:
$ pymedphys trf to-csv input.trf output.csv

# This creates a CSV file with all the control point data
""")

print("\nEXAMPLE USE CASE:")
print("-" * 70)
print("""
# Compare planned vs delivered treatment
from pymedphys import Delivery

# Load planned treatment from DICOM RT Plan
planned = Delivery.from_dicom('rtplan.dcm', fraction_group=1)

# Load delivered treatment from TRF log file
delivered = Delivery.from_trf('treatment.trf')

# Calculate meterset maps
planned_map = planned.metersetmap()
delivered_map = delivered.metersetmap()

# Compare using gamma analysis
import pymedphys
gamma = pymedphys.gamma(
    planned_map, delivered_map,
    dose_percent_threshold=3,
    distance_mm_threshold=2
)
""")

print("\n" + "="*70)
print("To try with real data, you need a .trf file from an Elekta linac.")
print("Test data is available but requires network access to download.")
print("="*70)
