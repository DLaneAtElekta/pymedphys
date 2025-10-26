"""Example end-to-end training script for Mosaiq anomaly detection EBM.

This script demonstrates the complete workflow:
1. Collect normal examples from production database
2. Generate adversarial examples using failure modes
3. Extract features
4. Train EBM
5. Evaluate performance
6. Save model for deployment

Usage:
    python example_training.py --prod-host mosaiq.hospital.org \
                              --test-host testserver.local \
                              --output-dir ./models
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from pymedphys._mcp.mosaiq_failure_modes.ebm_features import (
    STANDARD_FEATURE_NAMES,
    create_feature_vector,
    extract_all_features,
)
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import (
    AdversarialTrainer,
    create_balanced_dataset,
)
from pymedphys._mosaiq import connect

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def collect_normal_data(
    hostname: str,
    database: str,
    n_samples: int,
    recent_days: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect and extract features from normal treatment records.

    Args:
        hostname: Production Mosaiq hostname
        database: Database name
        n_samples: Number of samples to collect
        recent_days: Only use recent records
        feature_names: List of feature names to extract

    Returns:
        Tuple of (features, labels)
    """
    logger.info(f"Connecting to production database: {hostname}/{database}")

    with connect(hostname=hostname, database=database, read_only=True) as conn:
        cursor = conn.cursor()

        # Query recent normal treatments
        logger.info(f"Querying {n_samples} recent treatments from last {recent_days} days")
        cursor.execute(
            f"""
            SELECT TOP {n_samples} TTX_ID
            FROM TrackTreatment
            WHERE Create_DtTm >= DATEADD(day, -{recent_days}, GETDATE())
              AND WasBeamComplete = 1
              AND WasQAMode = 0
            ORDER BY NEWID()
            """
        )

        ttx_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(ttx_ids)} treatment records")

        # Extract features
        features_list = []
        for i, ttx_id in enumerate(ttx_ids):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(ttx_ids)}...")

            try:
                features = extract_all_features(cursor, ttx_id)
                feature_vec = create_feature_vector(features, feature_names)
                features_list.append(feature_vec)
            except Exception as e:
                logger.warning(f"Failed to extract features for TTX_ID {ttx_id}: {e}")

        features = np.array(features_list)
        labels = np.zeros(len(features))

        logger.info(f"Successfully extracted {len(features)} normal examples")
        return features, labels


def generate_adversarial_data(
    hostname: str,
    database: str,
    n_samples: int,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate adversarial examples using failure mode corruptions.

    Args:
        hostname: Test database hostname
        database: Test database name
        n_samples: Number of adversarial examples to generate
        feature_names: List of feature names to extract

    Returns:
        Tuple of (features, labels, corrupted_ttx_ids)
    """
    logger.info(f"Connecting to test database: {hostname}/{database}")

    with connect(hostname=hostname, database=database, read_only=False) as conn:
        cursor = conn.cursor()

        # Get candidate records to corrupt
        logger.info(f"Selecting {n_samples} candidate records for corruption")
        cursor.execute(
            f"""
            SELECT TOP {n_samples} tt.TTX_ID, tt.FLD_ID, o.OFF_ID
            FROM TrackTreatment tt
            LEFT JOIN Site s ON tt.SIT_ID = s.SIT_ID
            LEFT JOIN Offset o ON s.SIT_SET_ID = o.SIT_SET_ID
            WHERE tt.Create_DtTm >= DATEADD(day, -60, GETDATE())
            ORDER BY NEWID()
            """
        )

        candidates = cursor.fetchall()
        logger.info(f"Found {len(candidates)} candidates")

        # Define failure modes to apply (distribute evenly)
        failure_modes = [
            # MLC corruptions
            ("mlc_random", lambda ttx, fld, off: corrupt_mlc_random(cursor, fld)),
            ("mlc_out_of_range", lambda ttx, fld, off: corrupt_mlc_out_of_range(cursor, fld)),
            # Angle corruptions
            ("angle_invalid", lambda ttx, fld, off: corrupt_angles(cursor, fld, 400.0)),
            # Control point corruptions
            ("cp_gaps", lambda ttx, fld, off: delete_control_points(cursor, fld, [1, 3])),
            # Timestamp corruptions
            (
                "timestamp_invalid",
                lambda ttx, fld, off: corrupt_timestamp(cursor, ttx, "edit_before_create"),
            ),
            # Offset corruptions
            (
                "offset_extreme",
                lambda ttx, fld, off: corrupt_offset(cursor, off, 999.9)
                if off
                else None,
            ),
            # Meterset corruptions
            ("meterset_negative", lambda ttx, fld, off: corrupt_meterset(cursor, fld, -100.0)),
            # FK corruptions
            ("orphaned", lambda ttx, fld, off: corrupt_foreign_key(cursor, ttx, 999999)),
        ]

        corrupted_ttx_ids = []
        records_per_mode = len(candidates) // len(failure_modes)

        for i, (mode_name, corruption_func) in enumerate(failure_modes):
            start_idx = i * records_per_mode
            end_idx = (
                start_idx + records_per_mode
                if i < len(failure_modes) - 1
                else len(candidates)
            )

            logger.info(
                f"Applying {mode_name} to records {start_idx}-{end_idx} "
                f"({end_idx - start_idx} records)"
            )

            for ttx_id, fld_id, off_id in candidates[start_idx:end_idx]:
                try:
                    corruption_func(ttx_id, fld_id, off_id)
                    corrupted_ttx_ids.append(ttx_id)
                except Exception as e:
                    logger.warning(f"Failed to corrupt TTX_ID {ttx_id} with {mode_name}: {e}")

        logger.info(f"Successfully corrupted {len(corrupted_ttx_ids)} records")

        # Extract features from corrupted records
        logger.info("Extracting features from corrupted records")
        features_list = []
        for i, ttx_id in enumerate(corrupted_ttx_ids):
            if i % 50 == 0:
                logger.info(f"Processing {i}/{len(corrupted_ttx_ids)}...")

            try:
                features = extract_all_features(cursor, ttx_id)
                feature_vec = create_feature_vector(features, feature_names)
                features_list.append(feature_vec)
            except Exception as e:
                logger.warning(f"Failed to extract features for corrupted TTX_ID {ttx_id}: {e}")

        features = np.array(features_list)
        labels = np.ones(len(features))

        logger.info(f"Successfully extracted {len(features)} adversarial examples")
        return features, labels, corrupted_ttx_ids


# Corruption helper functions
def corrupt_mlc_random(cursor, fld_id):
    """Corrupt MLC with random bytes."""
    random_bytes = bytes(np.random.randint(0, 256, 20, dtype=np.uint8))
    cursor.execute(
        "UPDATE TxFieldPoint SET A_Leaf_Set = %s WHERE FLD_ID = %s AND Point = 0",
        (random_bytes, fld_id),
    )
    cursor.connection.commit()


def corrupt_mlc_out_of_range(cursor, fld_id):
    """Corrupt MLC with out-of-range values."""
    import struct

    bad_mlc = struct.pack("<h", 10000)  # 100cm - way out of range
    cursor.execute(
        "UPDATE TxFieldPoint SET A_Leaf_Set = %s WHERE FLD_ID = %s AND Point = 0",
        (bad_mlc, fld_id),
    )
    cursor.connection.commit()


def corrupt_angles(cursor, fld_id, invalid_angle):
    """Set invalid gantry angle."""
    cursor.execute(
        "UPDATE TxFieldPoint SET Gantry_Ang = %s WHERE FLD_ID = %s",
        (invalid_angle, fld_id),
    )
    cursor.connection.commit()


def delete_control_points(cursor, fld_id, points_to_delete):
    """Delete specific control points."""
    cursor.execute(
        f"DELETE FROM TxFieldPoint WHERE FLD_ID = %s AND Point IN ({','.join(map(str, points_to_delete))})",
        (fld_id,),
    )
    cursor.connection.commit()


def corrupt_timestamp(cursor, ttx_id, corruption_type):
    """Corrupt timestamp relationship."""
    if corruption_type == "edit_before_create":
        cursor.execute(
            "UPDATE TrackTreatment SET Edit_DtTm = DATEADD(hour, -2, Create_DtTm) WHERE TTX_ID = %s",
            (ttx_id,),
        )
    cursor.connection.commit()


def corrupt_offset(cursor, off_id, extreme_value):
    """Set extreme offset values."""
    if off_id:
        cursor.execute(
            """
            UPDATE Offset
            SET Superior_Offset = %s, Anterior_Offset = %s, Lateral_Offset = %s
            WHERE OFF_ID = %s
            """,
            (extreme_value, extreme_value, extreme_value, off_id),
        )
        cursor.connection.commit()


def corrupt_meterset(cursor, fld_id, negative_value):
    """Set negative meterset."""
    cursor.execute("UPDATE TxField SET Meterset = %s WHERE FLD_ID = %s", (negative_value, fld_id))
    cursor.connection.commit()


def corrupt_foreign_key(cursor, ttx_id, invalid_id):
    """Set invalid foreign key."""
    cursor.execute("UPDATE TrackTreatment SET Pat_ID1 = %s WHERE TTX_ID = %s", (invalid_id, ttx_id))
    cursor.connection.commit()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train Mosaiq anomaly detection EBM")

    parser.add_argument("--prod-host", required=True, help="Production Mosaiq hostname")
    parser.add_argument("--prod-database", default="MOSAIQ", help="Production database name")
    parser.add_argument("--test-host", required=True, help="Test database hostname")
    parser.add_argument("--test-database", required=True, help="Test database name")

    parser.add_argument("--n-normal", type=int, default=2000, help="Number of normal examples")
    parser.add_argument(
        "--n-adversarial", type=int, default=500, help="Number of adversarial examples"
    )
    parser.add_argument("--recent-days", type=int, default=60, help="Use data from recent N days")

    parser.add_argument("--n-epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0, help="Contrastive loss margin")

    parser.add_argument("--output-dir", type=Path, default="./models", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "mosaiq_ebm.pt"

    logger.info("=" * 80)
    logger.info("Mosaiq Anomaly Detection EBM Training")
    logger.info("=" * 80)

    # Step 1: Collect normal examples
    logger.info("\n[1/6] Collecting normal examples from production database")
    normal_features, normal_labels = collect_normal_data(
        args.prod_host, args.prod_database, args.n_normal, args.recent_days, STANDARD_FEATURE_NAMES
    )

    # Step 2: Generate adversarial examples
    logger.info("\n[2/6] Generating adversarial examples in test database")
    adv_features, adv_labels, corrupted_ids = generate_adversarial_data(
        args.test_host, args.test_database, args.n_adversarial, STANDARD_FEATURE_NAMES
    )

    # Save corrupted IDs for reference
    corrupted_ids_file = args.output_dir / "corrupted_ttx_ids.txt"
    with open(corrupted_ids_file, "w") as f:
        for ttx_id in corrupted_ids:
            f.write(f"{ttx_id}\n")
    logger.info(f"Saved corrupted TTX_IDs to {corrupted_ids_file}")

    # Step 3: Create balanced dataset
    logger.info("\n[3/6] Creating balanced dataset")
    all_features, all_labels = create_balanced_dataset(
        normal_features, normal_labels, adv_features, adv_labels, balance_ratio=1.0
    )

    # Step 4: Split train/test
    logger.info("\n[4/6] Splitting train/test sets")
    train_features, test_features, train_labels, test_labels = train_test_split(
        all_features, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    logger.info(f"Training set: {len(train_features)} examples")
    logger.info(f"  Normal: {(train_labels == 0).sum()}")
    logger.info(f"  Anomaly: {(train_labels == 1).sum()}")
    logger.info(f"Test set: {len(test_features)} examples")
    logger.info(f"  Normal: {(test_labels == 0).sum()}")
    logger.info(f"  Anomaly: {(test_labels == 1).sum()}")

    # Step 5: Train EBM
    logger.info("\n[5/6] Training EBM")
    trainer = AdversarialTrainer(
        feature_names=STANDARD_FEATURE_NAMES, model_path=model_path, device=args.device
    )

    history = trainer.train(
        train_features,
        train_labels,
        test_features,
        test_labels,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        margin=args.margin,
    )

    # Step 6: Final evaluation
    logger.info("\n[6/6] Final evaluation")
    checkpoint = trainer.load_checkpoint()

    logger.info("\nBest Model Performance:")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    for metric, value in checkpoint["metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")

    # Save training history
    import json

    history_file = args.output_dir / "training_history.json"
    with open(history_file, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {
            "train_loss": [float(x) for x in history["train_loss"]],
            "test_metrics": [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in m.items()}
                for m in history["test_metrics"]
            ],
        }
        json.dump(serializable_history, f, indent=2)

    logger.info(f"\nSaved training history to {history_file}")
    logger.info(f"Saved model checkpoint to {model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
