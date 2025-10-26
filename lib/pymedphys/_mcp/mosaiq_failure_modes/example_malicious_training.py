"""Example: Adversarial EBM Training with Malicious Failure Modes

This script demonstrates how to train an Energy-Based Model to detect both
accidental errors AND sophisticated malicious attacks on radiotherapy databases.

Key Features:
- Balanced dataset: 50% normal, 45% accidental, 5% malicious
- Severity-weighted training (MSE loss)
- Multi-modal feature extraction (Mosaiq + TRF + Portal + Phantom)
- Confidence scoring (malicious vs accidental classification)
- Comprehensive evaluation metrics

Usage:
    python example_malicious_training.py --hostname test-mosaiq --num-samples 2000
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pymedphys._mosaiq import connect
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import EnergyBasedModel, train_epoch
from pymedphys._mcp.mosaiq_failure_modes.multimodal_features import extract_all_multimodal_features
from pymedphys._mcp.mosaiq_failure_modes.malicious_features import (
    assess_malicious_probability,
    cusum_analysis,
    detect_systematic_bias,
    mann_kendall_trend_test,
)
from pymedphys._mcp.mosaiq_failure_modes.severity import get_severity

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# ADVERSARIAL DATASET GENERATION
# =============================================================================


def generate_balanced_dataset(cursor, num_samples: int = 2000) -> dict:
    """Generate balanced dataset: 50% normal, 45% accidental, 5% malicious.

    Args:
        cursor: Database cursor (TEST DATABASE ONLY)
        num_samples: Total number of samples to generate

    Returns:
        Dictionary with features, labels, and metadata
    """
    logger.info(f"Generating {num_samples} samples...")

    # Distribution
    num_normal = int(num_samples * 0.50)  # 1000
    num_accidental = int(num_samples * 0.45)  # 900
    num_malicious = int(num_samples * 0.05)  # 100

    logger.info(f"  Normal: {num_normal} (50%)")
    logger.info(f"  Accidental: {num_accidental} (45%)")
    logger.info(f"  Malicious: {num_malicious} (5%)")

    dataset = {"features": [], "severities": [], "metadata": []}

    # -------------------------------------------------------------------------
    # 1. Normal Data (Severity 0.0-0.4)
    # -------------------------------------------------------------------------
    logger.info("Extracting normal treatment records...")
    cursor.execute(
        """
        SELECT TOP (?) TTX_ID
        FROM TrackTreatment
        WHERE WasBeamComplete = 1
          AND Edit_DtTm >= Create_DtTm  -- Basic sanity check
        ORDER BY NEWID()
    """,
        (num_normal,),
    )

    for row in cursor.fetchall():
        ttx_id = row[0]
        features = extract_all_multimodal_features(ttx_id, cursor)

        dataset["features"].append(features)
        dataset["severities"].append(0.2)  # Normal energy target
        dataset["metadata"].append({"ttx_id": ttx_id, "type": "normal", "failure_mode": None, "variant": None})

    logger.info(f"  Extracted {len(dataset['features'])} normal records")

    # -------------------------------------------------------------------------
    # 2. Accidental Errors (Severity 0.5-2.5)
    # -------------------------------------------------------------------------
    logger.info("Generating accidental failure mode examples...")

    accidental_modes = [
        # Low severity (15% of accidental = 135 samples)
        ("corrupt_mlc_data", "odd_bytes", 0.6, 45),
        ("invalid_angles", "gantry", 0.7, 45),
        ("invalid_angles", "collimator", 0.6, 45),
        # Medium severity (30% of accidental = 270 samples)
        ("duplicate_treatments", "same_field", 1.0, 70),
        ("timestamp_inconsistencies", "future_timestamp", 1.0, 70),
        ("timestamp_inconsistencies", "edit_before_create", 1.2, 70),
        ("corrupt_mlc_data", "random_bytes", 1.5, 60),
        # High severity (40% of accidental = 360 samples)
        ("corrupt_mlc_data", "out_of_range", 2.0, 90),
        ("missing_control_points", "single_gap", 1.8, 90),
        ("corrupt_mlc_data", "negative_gap", 2.2, 90),
        ("mlc_leaf_count_mismatch", "intra_field", 2.1, 90),
        # Critical accidental (15% of accidental = 135 samples)
        ("missing_control_points", "all_deleted", 2.5, 45),
        ("null_required_fields", "fld_id", 2.3, 45),
        ("orphaned_records", "patient_id", 2.8, 45),
    ]

    for failure_mode, variant, severity, count in accidental_modes:
        logger.info(f"  Generating {count} × {failure_mode}:{variant} (severity {severity})")

        # Apply failure mode to random records
        # (In practice, use MCP server tools or direct SQL corruption)
        for _ in range(count):
            # Placeholder: In real implementation, corrupt database and extract features
            # For now, use normal features with label
            features = extract_sample_features(cursor)  # Simplified

            dataset["features"].append(features)
            dataset["severities"].append(severity)
            dataset["metadata"].append({"ttx_id": None, "type": "accidental", "failure_mode": failure_mode, "variant": variant})

    logger.info(f"  Generated {num_accidental} accidental failure examples")

    # -------------------------------------------------------------------------
    # 3. Malicious Attacks (Severity 2.5-3.0)
    # -------------------------------------------------------------------------
    logger.info("Generating malicious failure mode examples...")

    malicious_modes = [
        # Critical malicious (100 samples distributed across sophistication levels)
        ("subtle_dose_escalation", "random_walk", 2.9, 15),
        ("subtle_dose_escalation", "linear_ramp", 2.8, 10),
        ("coordinated_multifield_attack", "geometric_shift", 2.9, 12),
        ("coordinated_multifield_attack", "compensatory_errors", 3.0, 8),
        ("field_aperture_manipulation", "systematic_shift", 3.0, 10),
        ("field_aperture_manipulation", "margin_erosion", 2.9, 10),
        ("statistical_camouflage", "within_tolerance_bias", 2.9, 12),
        ("statistical_camouflage", "outlier_avoidance", 2.9, 8),
        ("gradual_parameter_drift", "isocenter_shift", 2.9, 8),
        ("time_delayed_corruption", "scheduled_activation", 2.9, 7),
    ]

    for failure_mode, variant, severity, count in malicious_modes:
        logger.info(f"  Generating {count} × {failure_mode}:{variant} (severity {severity})")

        for _ in range(count):
            # Use MCP server tools to generate malicious examples
            # For demonstration, we'll simulate the features
            features = generate_malicious_features(cursor, failure_mode, variant)

            dataset["features"].append(features)
            dataset["severities"].append(severity)
            dataset["metadata"].append({"ttx_id": None, "type": "malicious", "failure_mode": failure_mode, "variant": variant})

    logger.info(f"  Generated {num_malicious} malicious failure examples")

    # -------------------------------------------------------------------------
    # Convert to arrays
    # -------------------------------------------------------------------------
    dataset["features"] = np.array(dataset["features"], dtype=np.float32)
    dataset["severities"] = np.array(dataset["severities"], dtype=np.float32)

    logger.info(f"Dataset generation complete: {len(dataset['features'])} total samples")
    logger.info(f"  Feature dimensions: {dataset['features'].shape}")

    return dataset


def extract_sample_features(cursor) -> np.ndarray:
    """Extract features from a random normal treatment (simplified)."""
    cursor.execute(
        """
        SELECT TOP 1 TTX_ID
        FROM TrackTreatment
        WHERE WasBeamComplete = 1
        ORDER BY NEWID()
    """
    )
    row = cursor.fetchone()
    if row:
        ttx_id = row[0]
        features = extract_all_multimodal_features(ttx_id, cursor)
        return features
    else:
        # Fallback: return zero features
        return np.zeros(124, dtype=np.float32)


def generate_malicious_features(cursor, failure_mode: str, variant: str) -> np.ndarray:
    """Generate features for malicious failure mode (synthetic).

    In real implementation, use MCP server tools to corrupt database and extract.
    For demonstration, we simulate characteristic feature patterns.
    """
    # Start with normal features
    features = extract_sample_features(cursor)

    # Add malicious signatures based on failure mode
    if failure_mode == "subtle_dose_escalation":
        # Gradual MU increase → systematic bias in MU features
        # Features 64-71 are meterset features
        features[64:72] *= 1.3  # 30% cumulative overdose signature

    elif failure_mode == "statistical_camouflage":
        # Small systematic bias + noise
        # Add +1.5% bias to MU with ±1% noise
        features[64:72] *= 1.015 * (1 + np.random.normal(0, 0.01, 8))

    elif failure_mode == "field_aperture_manipulation":
        # MLC centroid shift
        # Features 0-18 are MLC features
        features[0:19] += 3.0  # 3mm systematic shift

    elif failure_mode == "gradual_parameter_drift":
        # Isocenter drift
        # Features 48-59 are offset features
        features[48:60] += 10.0  # 10mm cumulative shift

    elif failure_mode == "coordinated_multifield_attack":
        # Distributed errors across fields
        # Multiple feature categories affected
        features[0:19] *= 1.05  # MLC changes
        features[64:72] *= 1.02  # Small MU changes
        # Net effect: composite dose error

    return features


# =============================================================================
# MODEL TRAINING
# =============================================================================


def train_adversarial_ebm(dataset: dict, num_epochs: int = 100, batch_size: int = 64) -> EnergyBasedModel:
    """Train EBM with severity-weighted loss.

    Args:
        dataset: Dictionary with features, severities, metadata
        num_epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained EBM model
    """
    logger.info("Initializing EBM model...")

    input_dim = dataset["features"].shape[1]  # 124 features
    model = EnergyBasedModel(input_dim=input_dim, hidden_dim=256, num_layers=4)

    # Convert to tensors
    features_tensor = torch.FloatTensor(dataset["features"])
    severities_tensor = torch.FloatTensor(dataset["severities"])

    # Create dataset and dataloader
    train_dataset = TensorDataset(features_tensor, severities_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Training EBM for {num_epochs} epochs...")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Hidden dim: 256")
    logger.info(f"  Batch size: {batch_size}")

    # Training loop
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, use_severity=True)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

    logger.info("Training complete!")
    return model


# =============================================================================
# EVALUATION AND DETECTION
# =============================================================================


def evaluate_detection(model: EnergyBasedModel, dataset: dict) -> dict:
    """Evaluate detection performance with malicious probability assessment.

    Args:
        model: Trained EBM
        dataset: Test dataset

    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating detection performance...")

    features_tensor = torch.FloatTensor(dataset["features"])
    energies = model.predict(features_tensor)

    # Classification by severity thresholds
    results = {"normal": [], "low": [], "medium": [], "high": [], "critical_accidental": [], "critical_malicious": []}

    for i, (energy, metadata) in enumerate(zip(energies, dataset["metadata"])):
        # Categorize by energy
        if energy < 0.4:
            category = "normal"
        elif energy < 0.8:
            category = "low"
        elif energy < 1.5:
            category = "medium"
        elif energy < 2.3:
            category = "high"
        elif metadata["type"] == "malicious":
            category = "critical_malicious"
        else:
            category = "critical_accidental"

        results[category].append({"energy": energy, "metadata": metadata})

    # Calculate metrics
    metrics = {}
    for category, samples in results.items():
        metrics[category] = {
            "count": len(samples),
            "mean_energy": np.mean([s["energy"] for s in samples]) if samples else 0,
            "std_energy": np.std([s["energy"] for s in samples]) if samples else 0,
        }

    # Malicious vs Accidental Classification
    logger.info("Assessing malicious probability for high-energy detections...")

    malicious_classifications = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}

    for i, (energy, metadata) in enumerate(zip(energies, dataset["metadata"])):
        if energy >= 2.5:  # Critical severity
            # Assess malicious probability
            features_dict = {
                "multiple_anomalies": False,  # Would check in real implementation
                "statistical_evasion": "camouflage" in metadata.get("failure_mode", ""),
                "audit_inconsistency": "audit" in metadata.get("failure_mode", ""),
                "temporal_clustering": "delayed" in metadata.get("failure_mode", ""),
                "demographic_bias": False,
            }

            assessment = assess_malicious_probability(features_dict, energy)

            # Ground truth
            is_actually_malicious = metadata["type"] == "malicious"
            is_classified_malicious = assessment["malicious_probability"] > 0.5

            if is_actually_malicious and is_classified_malicious:
                malicious_classifications["true_positive"] += 1
            elif is_actually_malicious and not is_classified_malicious:
                malicious_classifications["false_negative"] += 1
            elif not is_actually_malicious and is_classified_malicious:
                malicious_classifications["false_positive"] += 1
            else:
                malicious_classifications["true_negative"] += 1

    # Calculate precision, recall, F1
    tp = malicious_classifications["true_positive"]
    fp = malicious_classifications["false_positive"]
    fn = malicious_classifications["false_negative"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["malicious_classification"] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": malicious_classifications["true_negative"],
    }

    logger.info("Evaluation complete!")
    return metrics


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train adversarial EBM with malicious failure modes")
    parser.add_argument("--hostname", required=True, help="Mosaiq SQL Server hostname (TEST DATABASE ONLY)")
    parser.add_argument("--database", default="MOSAIQ_TEST", help="Database name (must be test database)")
    parser.add_argument("--num-samples", type=int, default=2000, help="Total number of samples to generate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--output-dir", type=Path, default=Path("./training_results"), help="Output directory")

    args = parser.parse_args()

    # Safety check: Verify test database
    if "test" not in args.database.lower() and "dev" not in args.database.lower():
        logger.error("ERROR: Database name must contain 'test' or 'dev'")
        logger.error("NEVER train on production databases!")
        return

    logger.info("=" * 80)
    logger.info("Adversarial EBM Training with Malicious Failure Modes")
    logger.info("=" * 80)
    logger.info(f"Hostname: {args.hostname}")
    logger.info(f"Database: {args.database}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info("")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    logger.info("Connecting to database...")
    with connect(hostname=args.hostname, database=args.database) as conn:
        cursor = conn.cursor()

        # Generate dataset
        dataset = generate_balanced_dataset(cursor, num_samples=args.num_samples)

        # Save dataset metadata
        metadata_file = args.output_dir / f"dataset_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(
                {
                    "num_samples": len(dataset["features"]),
                    "feature_dim": dataset["features"].shape[1],
                    "severity_distribution": {
                        "mean": float(np.mean(dataset["severities"])),
                        "std": float(np.std(dataset["severities"])),
                        "min": float(np.min(dataset["severities"])),
                        "max": float(np.max(dataset["severities"])),
                    },
                    "type_distribution": {
                        "normal": sum(1 for m in dataset["metadata"] if m["type"] == "normal"),
                        "accidental": sum(1 for m in dataset["metadata"] if m["type"] == "accidental"),
                        "malicious": sum(1 for m in dataset["metadata"] if m["type"] == "malicious"),
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Saved dataset metadata to {metadata_file}")

    # Train model
    model = train_adversarial_ebm(dataset, num_epochs=args.num_epochs)

    # Save model
    model_file = args.output_dir / f"ebm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model.save(str(model_file))
    logger.info(f"Saved model to {model_file}")

    # Evaluate
    metrics = evaluate_detection(model, dataset)

    # Print metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 80)

    for category, stats in metrics.items():
        if category != "malicious_classification":
            logger.info(f"\n{category.upper()}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Mean Energy: {stats['mean_energy']:.3f}")
            logger.info(f"  Std Energy: {stats['std_energy']:.3f}")

    logger.info("\nMALICIOUS vs ACCIDENTAL CLASSIFICATION:")
    mc = metrics["malicious_classification"]
    logger.info(f"  Precision: {mc['precision']:.3f}")
    logger.info(f"  Recall: {mc['recall']:.3f}")
    logger.info(f"  F1 Score: {mc['f1_score']:.3f}")
    logger.info(f"  True Positives: {mc['true_positive']}")
    logger.info(f"  False Positives: {mc['false_positive']}")
    logger.info(f"  False Negatives: {mc['false_negative']}")

    # Save metrics
    metrics_file = args.output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nSaved metrics to {metrics_file}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
