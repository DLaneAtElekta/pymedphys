# Energy-Based Model (EBM) Training for Mosaiq Anomaly Detection

This guide explains how to use the Mosaiq Failure Modes MCP server to train an adversarial Energy-Based Model for automated anomaly detection.

## Overview

### What is an Energy-Based Model?

An Energy-Based Model (EBM) learns to assign a scalar "energy" value to each input:
- **Low energy** = data is normal/valid
- **High energy** = data is anomalous/corrupted

Unlike classifiers that output probabilities, EBMs learn to shape an energy landscape where valid data occupies low-energy valleys and anomalies sit on high-energy peaks.

### Adversarial Training Approach

Our training strategy uses the MCP failure mode tools to create a robust anomaly detector:

1. **Collect normal examples** from production Mosaiq database
2. **Generate adversarial examples** using MCP failure mode corruption tools
3. **Extract QA features** from both normal and corrupted records
4. **Train EBM** to distinguish low-energy (normal) from high-energy (anomaly)
5. **Deploy detector** to monitor production database in real-time

## Architecture

### Feature Engineering

The QA checks documented in [QA_FRAMEWORK.md](./QA_FRAMEWORK.md) become input features for the EBM:

**72 total features across 6 categories:**

| Category | Features | Examples |
|----------|----------|----------|
| **MLC** (19) | Byte lengths, positions, gaps, ranges | `a_mean_pos`, `negative_gap_count`, `byte_length_even` |
| **Angles** (14) | Gantry/collimator ranges, transitions | `gantry_out_of_range`, `gantry_max_delta`, `coll_suspicious_jumps` |
| **Control Points** (6) | Count, sequence, gaps | `cp_sequential`, `cp_gap_count`, `cp_minimum_met` |
| **Timestamps** (9) | Ordering, duration, consistency | `edit_before_create`, `duration_negative`, `dose_time_mismatch` |
| **Offsets** (12) | Magnitudes, extremes, validity | `offset_vector_magnitude`, `vector_extreme`, `type_valid` |
| **Meterset** (8) | MU values, consistency | `mu_negative`, `mu_cp_mismatch`, `mu_extreme_high` |
| **Foreign Keys** (6) | Existence, NULLs | `patient_exists`, `fld_id_null`, `site_exists` |

### Network Architecture

```
Input (72 features)
    ↓
Dense(128) + LayerNorm + ReLU + Dropout(0.2)
    ↓
Dense(64) + LayerNorm + ReLU + Dropout(0.2)
    ↓
Dense(32) + LayerNorm + ReLU + Dropout(0.2)
    ↓
Dense(1) → Energy output
```

### Loss Function

**Contrastive Energy Loss:**

```
L = (1 - y) * E(x) + y * max(0, margin - E(x))

where:
  y = 0 for normal examples
  y = 1 for adversarial examples
  E(x) = energy output by network
  margin = separation threshold (default: 1.0)
```

**Intuition:**
- For normal data (y=0): minimize energy E(x)
- For anomalies (y=1): maximize energy, but only up to `margin`
- Creates separation between normal and anomalous data

## Training Workflow

### Step 1: Prepare Test Database

Create an isolated test database with anonymized production data:

```sql
-- Create test database
CREATE DATABASE MosaiqTest_EBM;

-- Restore from anonymized backup
RESTORE DATABASE MosaiqTest_EBM
FROM DISK = 'C:\Backups\MosaiqAnonymized.bak';

-- Create backup of clean state
BACKUP DATABASE MosaiqTest_EBM
TO DISK = 'C:\Backups\MosaiqTest_Clean.bak';
```

### Step 2: Collect Normal Examples

Extract features from recent, validated treatment records:

```python
import pymssql
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import AdversarialTrainer
from pymedphys._mosaiq import connect

# Initialize trainer
trainer = AdversarialTrainer(
    model_path="models/mosaiq_ebm_v1.pt",
    device="cuda"  # or "cpu"
)

# Connect to production DB (read-only!)
with connect(hostname="mosaiq.hospital.org", database="MOSAIQ", read_only=True) as conn:
    cursor = conn.cursor()

    # Collect 5000 normal examples from last 60 days
    normal_features, normal_labels = trainer.collect_normal_examples(
        cursor,
        n_samples=5000,
        recent_days=60
    )

print(f"Collected {len(normal_features)} normal examples")
```

### Step 3: Generate Adversarial Examples

Use the MCP server to systematically corrupt test database records:

```python
from pymedphys._mcp.mosaiq_failure_modes.server import call_tool

# Connect to TEST database (writable)
with connect(hostname="testserver.local", database="MosaiqTest_EBM", read_only=False) as conn:
    cursor = conn.cursor()

    # Get candidate records to corrupt
    cursor.execute("""
        SELECT TOP 500 TTX_ID, FLD_ID
        FROM TrackTreatment
        WHERE Create_DtTm >= DATEADD(day, -60, GETDATE())
        ORDER BY NEWID()
    """)

    candidates = cursor.fetchall()

    # Track corrupted IDs
    corrupted_ttx_ids = []

    # Apply each failure mode to different subsets
    failure_modes = [
        ("corrupt_mlc_data", {"corruption_type": "random_bytes"}),
        ("create_invalid_angles", {"angle_type": "gantry", "invalid_value": 400.0}),
        ("delete_control_points", {"points_to_delete": [1, 3]}),
        ("create_timestamp_inconsistency", {"inconsistency_type": "edit_before_create"}),
        ("corrupt_offset_data", {"corruption_type": "extreme_values", "value": 999.9}),
        ("create_meterset_inconsistency", {"inconsistency_type": "negative_meterset"}),
        # ... more failure modes
    ]

    records_per_mode = len(candidates) // len(failure_modes)

    for i, (mode_name, params) in enumerate(failure_modes):
        start_idx = i * records_per_mode
        end_idx = start_idx + records_per_mode

        for ttx_id, fld_id in candidates[start_idx:end_idx]:
            # Apply corruption
            params_copy = params.copy()
            params_copy["hostname"] = "testserver.local"
            params_copy["database"] = "MosaiqTest_EBM"

            if mode_name in ["corrupt_mlc_data", "create_invalid_angles", "delete_control_points"]:
                params_copy["fld_id"] = fld_id
            elif mode_name in ["create_timestamp_inconsistency"]:
                params_copy["ttx_id"] = ttx_id
            # ... handle other failure modes

            try:
                # Call MCP tool to corrupt
                result = await call_tool(mode_name, params_copy)
                corrupted_ttx_ids.append(ttx_id)
                print(f"Corrupted TTX_ID {ttx_id} with {mode_name}")
            except Exception as e:
                print(f"Failed to corrupt {ttx_id}: {e}")

    print(f"Generated {len(corrupted_ttx_ids)} adversarial examples")
```

### Step 4: Extract Adversarial Features

Extract features from the corrupted records:

```python
# Collect adversarial examples
adv_features, adv_labels = trainer.collect_adversarial_examples(
    cursor,
    failure_mode_ttx_ids=corrupted_ttx_ids
)

print(f"Extracted features from {len(adv_features)} adversarial examples")
```

### Step 5: Create Training/Test Split

```python
from sklearn.model_selection import train_test_split
from pymedphys._mcp.mosaiq_failure_modes.ebm_training import create_balanced_dataset

# Combine and balance
all_features, all_labels = create_balanced_dataset(
    normal_features, normal_labels,
    adv_features, adv_labels,
    balance_ratio=1.0  # Equal normal and adversarial
)

# Split 80/20
train_features, test_features, train_labels, test_labels = train_test_split(
    all_features, all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)

print(f"Training set: {len(train_features)} examples")
print(f"Test set: {len(test_features)} examples")
```

### Step 6: Train the EBM

```python
# Train
history = trainer.train(
    train_features, train_labels,
    test_features, test_labels,
    n_epochs=100,
    batch_size=32,
    margin=1.0
)

# Plot training curve
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
f1_scores = [m['f1'] for m in history['test_metrics']]
plt.plot(f1_scores)
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Test F1 Score')

plt.tight_layout()
plt.savefig('training_history.png')
```

### Step 7: Evaluate Performance

```python
# Load best checkpoint
checkpoint = trainer.load_checkpoint()

print(f"Best model from epoch {checkpoint['epoch']}")
print(f"Metrics: {checkpoint['metrics']}")

# Detailed evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
predictions = trainer.predict(test_features)

print("\nClassification Report:")
print(classification_report(
    test_labels,
    predictions['predictions'],
    target_names=['Normal', 'Anomaly']
))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, predictions['predictions'])
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# Analyze energy distributions
normal_energies = predictions['energies'][test_labels == 0]
anomaly_energies = predictions['energies'][test_labels == 1]

plt.figure(figsize=(10, 6))
plt.hist(normal_energies, bins=50, alpha=0.5, label='Normal', density=True)
plt.hist(anomaly_energies, bins=50, alpha=0.5, label='Anomaly', density=True)
plt.axvline(predictions['threshold'], color='r', linestyle='--', label='Threshold')
plt.xlabel('Energy')
plt.ylabel('Density')
plt.legend()
plt.title('Energy Distribution by Class')
plt.savefig('energy_distribution.png')
```

## Deployment

### Real-Time Monitoring

Deploy the trained EBM to monitor production database:

```python
import time
from datetime import datetime, timedelta

def monitor_recent_treatments(
    cursor,
    trainer: AdversarialTrainer,
    lookback_minutes: int = 60,
    check_interval: int = 300  # 5 minutes
):
    """Monitor recent treatments for anomalies."""

    while True:
        # Query recent treatments
        cursor.execute(f"""
            SELECT TTX_ID
            FROM TrackTreatment
            WHERE Create_DtTm >= DATEADD(minute, -{lookback_minutes}, GETDATE())
              AND WasBeamComplete = 1
        """)

        recent_ttx_ids = [row[0] for row in cursor.fetchall()]

        # Extract features
        features_list = []
        for ttx_id in recent_ttx_ids:
            try:
                features = extract_all_features(cursor, ttx_id)
                feature_vec = create_feature_vector(features, trainer.feature_names)
                features_list.append((ttx_id, feature_vec))
            except Exception as e:
                print(f"Error extracting features for {ttx_id}: {e}")

        if features_list:
            ttx_ids, features = zip(*features_list)
            features = np.array(features)

            # Predict
            results = trainer.predict(features)

            # Flag anomalies
            for ttx_id, energy, pred, prob in zip(
                ttx_ids,
                results['energies'],
                results['predictions'],
                results['probabilities']
            ):
                if pred == 1:  # Anomaly detected
                    print(f"ALERT: Anomaly detected in TTX_ID {ttx_id}")
                    print(f"  Energy: {energy:.4f}")
                    print(f"  Probability: {prob:.4f}")

                    # Send alert to physics team
                    send_alert(ttx_id, energy, prob)

        # Wait before next check
        time.sleep(check_interval)

# Run monitoring
with connect(hostname="mosaiq.hospital.org", database="MOSAIQ", read_only=True) as conn:
    cursor = conn.cursor()
    monitor_recent_treatments(cursor, trainer)
```

### Feature Importance Analysis

Identify which QA features are most important for detection:

```python
import torch

def compute_feature_importance(
    trainer: AdversarialTrainer,
    test_features: np.ndarray,
    n_samples: int = 100
) -> dict[str, float]:
    """Compute feature importance using gradient-based saliency."""

    trainer.model.eval()

    # Random sample
    indices = np.random.choice(len(test_features), n_samples, replace=False)
    sample_features = torch.FloatTensor(test_features[indices]).to(trainer.device)
    sample_features.requires_grad = True

    # Forward pass
    energies = trainer.model(sample_features)
    energy_sum = energies.sum()

    # Backward pass
    energy_sum.backward()

    # Gradient magnitudes = feature importance
    gradients = sample_features.grad.abs().mean(dim=0).cpu().numpy()

    # Create importance dictionary
    importance = {
        name: float(grad)
        for name, grad in zip(trainer.feature_names, gradients)
    }

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return importance

# Compute and display
importance = compute_feature_importance(trainer, test_features)

print("\nTop 20 Most Important Features:")
for i, (feature, score) in enumerate(list(importance.items())[:20], 1):
    print(f"{i:2d}. {feature:30s} {score:.6f}")
```

## Continual Learning

As new failure modes are discovered, retrain the EBM:

```python
def continual_training(
    trainer: AdversarialTrainer,
    new_normal_features: np.ndarray,
    new_adversarial_features: np.ndarray,
    n_epochs: int = 20
):
    """Fine-tune existing model with new examples."""

    # Load existing checkpoint
    trainer.load_checkpoint()

    # Create new dataset
    new_features, new_labels = create_balanced_dataset(
        new_normal_features, np.zeros(len(new_normal_features)),
        new_adversarial_features, np.ones(len(new_adversarial_features))
    )

    # Split
    train_features, test_features, train_labels, test_labels = train_test_split(
        new_features, new_labels,
        test_size=0.2,
        stratify=new_labels
    )

    # Fine-tune with lower learning rate
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=1e-5,  # Lower learning rate for fine-tuning
        weight_decay=1e-5
    )

    # Train
    history = trainer.train(
        train_features, train_labels,
        test_features, test_labels,
        n_epochs=n_epochs,
        batch_size=32
    )

    return history
```

## Advanced Techniques

### 1. Ensemble Models

Train multiple EBMs with different architectures or random seeds:

```python
class EBMEnsemble:
    """Ensemble of EBM models for robust detection."""

    def __init__(self, n_models: int = 5):
        self.models = [
            AdversarialTrainer(
                model_path=f"models/ebm_{i}.pt"
            )
            for i in range(n_models)
        ]

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Predict using ensemble voting."""

        all_predictions = []
        all_energies = []

        for model in self.models:
            result = model.predict(features)
            all_predictions.append(result['predictions'])
            all_energies.append(result['energies'])

        # Majority vote
        ensemble_predictions = (
            np.mean(all_predictions, axis=0) > 0.5
        ).astype(int)

        # Average energy
        ensemble_energies = np.mean(all_energies, axis=0)

        return {
            'predictions': ensemble_predictions,
            'energies': ensemble_energies,
            'individual_predictions': all_predictions,
            'agreement': np.std(all_predictions, axis=0)  # Uncertainty measure
        }
```

### 2. Uncertainty Quantification

Use Monte Carlo Dropout for uncertainty estimation:

```python
def predict_with_uncertainty(
    trainer: AdversarialTrainer,
    features: np.ndarray,
    n_samples: int = 50
) -> dict[str, np.ndarray]:
    """Predict with uncertainty using MC Dropout."""

    trainer.model.train()  # Keep dropout active

    all_energies = []

    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(trainer.device)

        for _ in range(n_samples):
            energy = trainer.model(features_tensor).squeeze().cpu().numpy()
            all_energies.append(energy)

    all_energies = np.array(all_energies)

    return {
        'energy_mean': all_energies.mean(axis=0),
        'energy_std': all_energies.std(axis=0),
        'energy_quantiles': np.percentile(all_energies, [5, 50, 95], axis=0)
    }
```

### 3. Active Learning

Identify uncertain examples for manual review:

```python
def active_learning_selection(
    trainer: AdversarialTrainer,
    unlabeled_features: np.ndarray,
    n_select: int = 100
) -> np.ndarray:
    """Select most uncertain examples for labeling."""

    uncertainty_results = predict_with_uncertainty(
        trainer,
        unlabeled_features,
        n_samples=50
    )

    # Select examples with highest uncertainty (energy std)
    uncertainty_scores = uncertainty_results['energy_std']
    top_indices = np.argsort(uncertainty_scores)[-n_select:]

    return top_indices
```

## Best Practices

### Data Collection

1. **Temporal Coverage**: Collect normal examples from different time periods (weekdays, weekends, different shifts)
2. **Machine Coverage**: Include data from all treatment machines
3. **Treatment Diversity**: Sample across different treatment sites, techniques (IMRT, VMAT, 3D)
4. **Anonymization**: Always use anonymized data for model development

### Training

1. **Balance Classes**: Equal normal and adversarial examples
2. **Cross-Validation**: Use k-fold CV for robust performance estimation
3. **Regularization**: Use dropout and weight decay to prevent overfitting
4. **Early Stopping**: Monitor validation F1 score and stop when plateauing

### Deployment

1. **Threshold Calibration**: Set threshold based on acceptable false positive rate
2. **Alert Fatigue**: Don't alert on marginal cases - require high confidence
3. **Human in Loop**: Always have physics review before acting on alerts
4. **Monitoring**: Track model performance over time, retrain when degraded

### Validation

1. **Prospective Testing**: Test on future data not seen during training
2. **Clinical Validation**: Have physics review detected anomalies
3. **False Positive Analysis**: Understand what causes false alarms
4. **Ablation Studies**: Test importance of different failure modes

## Troubleshooting

### Poor Performance

**Issue**: Low F1 score on test set

**Solutions**:
- Collect more normal examples (5000+ recommended)
- Generate more diverse failure modes
- Check feature normalization
- Increase model capacity (larger hidden layers)
- Reduce learning rate
- Train for more epochs

### High False Positive Rate

**Issue**: Too many false alarms on normal data

**Solutions**:
- Increase energy threshold
- Collect more representative normal examples
- Check for data drift between training and production
- Use uncertainty quantification
- Ensemble models for robustness

### Missing Anomalies

**Issue**: Known corruptions not detected

**Solutions**:
- Ensure failure mode is represented in training set
- Check if relevant QA features are extracted
- Lower energy threshold
- Add failure mode-specific features
- Retrain with new adversarial examples

## References

- [EBM Tutorial](https://arxiv.org/abs/1903.08689) - "Implicit Generation and Generalization in Energy-Based Models"
- [Contrastive Divergence](http://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf) - Original CD algorithm by Hinton
- [Adversarial Training](https://arxiv.org/abs/1412.6572) - "Explaining and Harnessing Adversarial Examples"

---

**Next Steps**: See [README.md](./README.md) for MCP server usage and [QA_FRAMEWORK.md](./QA_FRAMEWORK.md) for complete list of failure modes and QA checks.
