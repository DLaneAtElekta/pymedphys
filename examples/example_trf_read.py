#!/usr/bin/env python
"""Example script demonstrating how to read an Elekta TRF file."""

import pymedphys

# Get path to example TRF file from test data
print("Downloading test data (if needed)...")
all_paths = pymedphys.zip_data_paths("delivery_test_data.zip")

# Find the TRF file
trf_paths = [path for path in all_paths if path.suffix == ".trf"]
print(f"\nFound {len(trf_paths)} TRF files in test data:")
for path in trf_paths:
    print(f"  - {path.parent.name}/{path.name}")

# Read the first TRF file
trf_file = str(trf_paths[0])
print(f"\nReading TRF file: {trf_file}")

# Method 1: Using pymedphys.trf.read
header_df, table_df = pymedphys.trf.read(trf_file)

print("\n" + "=" * 60)
print("TRF HEADER INFORMATION")
print("=" * 60)
print(header_df.T)

print("\n" + "=" * 60)
print("TRF TABLE DATA (first 10 rows)")
print("=" * 60)
print(table_df.head(10))

print(f"\nTotal number of control points: {len(table_df)}")
print(f"Number of columns: {len(table_df.columns)}")

# Method 2: Create a Delivery object
print("\n" + "=" * 60)
print("CREATING DELIVERY OBJECT")
print("=" * 60)
delivery = pymedphys.Delivery.from_trf(trf_file)

print(
    f"Monitor units range: {delivery.monitor_units.min():.2f} - {delivery.monitor_units.max():.2f} MU"
)
print(f"Gantry angles: {delivery.gantry.min():.1f}° - {delivery.gantry.max():.1f}°")
print(
    f"Collimator angles: {delivery.collimator.min():.1f}° - {delivery.collimator.max():.1f}°"
)
print(f"MLC shape: {delivery.mlc.shape} (control points, leaves, banks)")
print(f"Jaw shape: {delivery.jaw.shape} (control points, jaws)")

print("\n✓ TRF file successfully read!")
