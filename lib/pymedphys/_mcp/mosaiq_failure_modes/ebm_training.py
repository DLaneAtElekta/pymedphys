"""Energy-Based Model (EBM) for Mosaiq anomaly detection.

This module implements adversarial training of an EBM using:
1. Normal Mosaiq records as negative examples (low energy)
2. Failure mode corruptions as positive examples (high energy)
3. QA features as input dimensions

The EBM learns to assign low energy to valid data and high energy to anomalies.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pymssql

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. EBM training will not work.")

from .ebm_features import STANDARD_FEATURE_NAMES, create_feature_vector, extract_all_features


class MosaiqEBM(nn.Module):
    """Energy-Based Model for Mosaiq data quality assessment.

    The network outputs a scalar energy value for input features.
    Lower energy = more likely to be valid data
    Higher energy = more likely to be anomalous data
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = None):
        """Initialize EBM architecture.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions. Default: [128, 64, 32]
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        # Output layer: single energy value
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy for input features.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Energy values [batch_size, 1]
        """
        return self.network(x)


class MosaiqDataset(Dataset):
    """PyTorch dataset for Mosaiq training data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
    ):
        """Initialize dataset.

        Args:
            features: Feature matrix [n_samples, n_features]
            labels: Binary labels (0=normal, 1=anomaly)
            feature_names: List of feature names for documentation
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class AdversarialTrainer:
    """Adversarial training framework for Mosaiq EBM.

    This trainer:
    1. Collects normal examples from production Mosaiq DB
    2. Generates adversarial examples using failure mode MCP tools
    3. Trains EBM to distinguish normal from anomalous data
    4. Supports continual learning as new failure modes are discovered
    """

    def __init__(
        self,
        feature_names: list[str] = None,
        model_path: Path | None = None,
        device: str = "cpu",
    ):
        """Initialize trainer.

        Args:
            feature_names: List of feature names to use
            model_path: Path to save/load model checkpoints
            device: PyTorch device ('cpu' or 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for EBM training")

        self.feature_names = feature_names or STANDARD_FEATURE_NAMES
        self.input_dim = len(self.feature_names)
        self.model_path = model_path or Path("mosaiq_ebm_checkpoint.pt")
        self.device = torch.device(device)

        # Initialize model
        self.model = MosaiqEBM(input_dim=self.input_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def collect_normal_examples(
        self,
        cursor: pymssql.Cursor,
        n_samples: int = 1000,
        recent_days: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect normal treatment records from production database.

        Args:
            cursor: Database cursor (read-only connection)
            n_samples: Number of samples to collect
            recent_days: Only use recent records (for temporal relevance)

        Returns:
            Tuple of (features, labels) where labels are all 0 (normal)
        """
        # Query recent normal treatments (no known issues)
        cursor.execute(
            f"""
            SELECT TOP {n_samples} TTX_ID
            FROM TrackTreatment
            WHERE Create_DtTm >= DATEADD(day, -{recent_days}, GETDATE())
              AND WasBeamComplete = 1
              AND WasQAMode = 0
            ORDER BY NEWID()  -- Random sampling
            """
        )

        ttx_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Collected {len(ttx_ids)} normal treatment records")

        # Extract features for each
        features_list = []
        for ttx_id in ttx_ids:
            try:
                features = extract_all_features(cursor, ttx_id)
                feature_vec = create_feature_vector(features, self.feature_names)
                features_list.append(feature_vec)
            except Exception as e:
                logger.warning(f"Failed to extract features for TTX_ID {ttx_id}: {e}")

        features = np.array(features_list)
        labels = np.zeros(len(features))  # All normal

        return features, labels

    def collect_adversarial_examples(
        self,
        cursor: pymssql.Cursor,
        failure_mode_records: list[tuple[int, str, str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect adversarial examples from failure mode corruptions.

        Args:
            cursor: Database cursor
            failure_mode_records: List of tuples (ttx_id, failure_mode, variant, severity)

        Returns:
            Tuple of (features, severity_scores)
            - features: Feature matrix
            - severity_scores: Severity for each example (0=normal, 0.5-3.0=anomaly)
        """
        features_list = []
        severity_list = []

        for ttx_id, failure_mode, variant, severity in failure_mode_records:
            try:
                features = extract_all_features(cursor, ttx_id)
                feature_vec = create_feature_vector(features, self.feature_names)
                features_list.append(feature_vec)
                severity_list.append(severity)
            except Exception as e:
                logger.warning(f"Failed to extract features for corrupted TTX_ID {ttx_id}: {e}")

        features = np.array(features_list)
        severities = np.array(severity_list)

        logger.info(f"Collected {len(features)} adversarial examples")
        logger.info(f"  Severity range: {severities.min():.2f} - {severities.max():.2f}")
        logger.info(f"  Mean severity: {severities.mean():.2f}")
        return features, severities

    def train_epoch(
        self,
        train_loader: DataLoader,
        margin: float = 1.0,
        use_severity: bool = True,
    ) -> float:
        """Train for one epoch using severity-weighted loss.

        Loss function (severity-weighted):
        - For normal examples (severity=0): E(x) should be low (~0.1-0.3)
        - For anomalies: E(x) should match severity score
          - Low severity (0.5-0.8): Minor data quality issues
          - Medium severity (1.0-1.5): Data integrity issues
          - High severity (1.8-2.3): Dose delivery errors
          - Critical severity (2.5-3.0): Patient safety risks

        Loss: L = MSE(E(x), target_energy) for severity-weighted
              L = (1-y)*E(x) + y*max(0, margin-E(x)) for binary (backward compat)

        Args:
            train_loader: DataLoader with training data
            margin: Margin for binary contrastive loss (unused if use_severity=True)
            use_severity: Use severity scores as target energy (recommended)

        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: compute energy
            energy = self.model(features).squeeze()

            if use_severity:
                # Severity-weighted loss: MSE between predicted and target energy
                # labels contains severity scores (0=normal, 0.5-3.0=anomaly severity)
                loss = torch.nn.functional.mse_loss(energy, labels)
            else:
                # Binary contrastive loss (backward compatibility)
                # labels are binary (0=normal, 1=anomaly)
                normal_loss = (1 - labels) * energy
                anomaly_loss = labels * torch.clamp(margin - energy, min=0)
                loss = (normal_loss + anomaly_loss).mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float | None = None,
        use_severity: bool = True,
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: DataLoader with test data
            threshold: Energy threshold for binary classification (auto-compute if None)
            use_severity: If True, compute severity-based metrics; if False, binary metrics

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_energies = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                energy = self.model(features).squeeze()

                all_energies.extend(energy.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_energies = np.array(all_energies)
        all_labels = np.array(all_labels)

        metrics = {}

        if use_severity:
            # Severity-based metrics (continuous)
            # Mean Absolute Error between predicted energy and target severity
            mae = np.abs(all_energies - all_labels).mean()
            mse = np.mean((all_energies - all_labels) ** 2)
            rmse = np.sqrt(mse)

            # Separate by severity category
            normal_mask = all_labels < 0.4
            low_mask = (all_labels >= 0.4) & (all_labels < 1.0)
            medium_mask = (all_labels >= 1.0) & (all_labels < 1.8)
            high_mask = (all_labels >= 1.8) & (all_labels < 2.5)
            critical_mask = all_labels >= 2.5

            metrics.update({
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "normal_energy_mean": float(all_energies[normal_mask].mean()) if normal_mask.any() else 0.0,
                "low_severity_mean": float(all_energies[low_mask].mean()) if low_mask.any() else 0.0,
                "medium_severity_mean": float(all_energies[medium_mask].mean()) if medium_mask.any() else 0.0,
                "high_severity_mean": float(all_energies[high_mask].mean()) if high_mask.any() else 0.0,
                "critical_severity_mean": float(all_energies[critical_mask].mean()) if critical_mask.any() else 0.0,
            })

            # Binary classification metrics using threshold
            if threshold is None:
                # Set threshold between normal and lowest anomaly
                normal_max = all_energies[normal_mask].max() if normal_mask.any() else 0.3
                anomaly_min = all_energies[~normal_mask].min() if (~normal_mask).any() else 0.5
                threshold = (normal_max + anomaly_min) / 2

        else:
            # Binary metrics (backward compatibility)
            if threshold is None:
                normal_mean = all_energies[all_labels == 0].mean()
                anomaly_mean = all_energies[all_labels == 1].mean()
                threshold = (normal_mean + anomaly_mean) / 2

        # Binary classification metrics (useful for both modes)
        binary_labels = (all_labels > 0.4).astype(int)  # Threshold to convert severity to binary
        predictions = (all_energies > threshold).astype(int)
        accuracy = (predictions == binary_labels).mean()

        # True/False positives/negatives
        tp = ((predictions == 1) & (binary_labels == 1)).sum()
        fp = ((predictions == 1) & (binary_labels == 0)).sum()
        tn = ((predictions == 0) & (binary_labels == 0)).sum()
        fn = ((predictions == 0) & (binary_labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.update({
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
        })

        return metrics

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        margin: float = 1.0,
        use_severity: bool = True,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_features: Training features
            train_labels: Training severity scores (0=normal, 0.5-3.0=anomaly severity)
            test_features: Test features
            test_labels: Test severity scores
            n_epochs: Number of training epochs
            batch_size: Batch size
            margin: Contrastive loss margin (unused if use_severity=True)
            use_severity: Use severity-weighted loss (recommended)

        Returns:
            Dictionary with training history and final metrics
        """
        # Create datasets
        train_dataset = MosaiqDataset(train_features, train_labels, self.feature_names)
        test_dataset = MosaiqDataset(test_features, test_labels, self.feature_names)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Training loop
        history = {"train_loss": [], "test_metrics": []}

        best_f1 = 0.0

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, margin=margin, use_severity=use_severity)
            history["train_loss"].append(train_loss)

            # Evaluate
            test_metrics = self.evaluate(test_loader, use_severity=use_severity)
            history["test_metrics"].append(test_metrics)

            # Log progress
            if epoch % 10 == 0:
                if use_severity:
                    logger.info(
                        f"Epoch {epoch}/{n_epochs} - "
                        f"Loss: {train_loss:.4f}, "
                        f"MAE: {test_metrics['mae']:.4f}, "
                        f"F1: {test_metrics['f1']:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}/{n_epochs} - "
                        f"Loss: {train_loss:.4f}, "
                        f"F1: {test_metrics['f1']:.4f}, "
                        f"Acc: {test_metrics['accuracy']:.4f}"
                    )

            # Save best model (use MAE for severity, F1 for binary)
            metric_key = "mae" if use_severity else "f1"
            metric_value = test_metrics[metric_key]

            # For MAE, lower is better; for F1, higher is better
            if use_severity:
                is_better = (best_f1 == 0.0) or (metric_value < best_f1)
            else:
                is_better = metric_value > best_f1

            if is_better:
                best_f1 = metric_value
                self.save_checkpoint(epoch, test_metrics)

        return history

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "feature_names": self.feature_names,
        }

        torch.save(checkpoint, self.model_path)
        logger.info(f"Saved checkpoint to {self.model_path}")

    def load_checkpoint(self) -> dict[str, Any]:
        """Load model checkpoint.

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"with F1={checkpoint['metrics']['f1']:.4f}"
        )

        return checkpoint

    def predict(self, features: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
        """Predict anomaly scores for new data.

        Args:
            features: Feature matrix [n_samples, n_features]
            threshold: Energy threshold (use saved threshold if None)

        Returns:
            Dictionary with energies, predictions, and probabilities
        """
        self.model.eval()

        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            energies = self.model(features_tensor).squeeze().cpu().numpy()

        if threshold is None:
            # Load threshold from checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            threshold = checkpoint["metrics"]["threshold"]

        predictions = (energies > threshold).astype(int)

        # Convert energies to pseudo-probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-energies))

        return {
            "energies": energies,
            "predictions": predictions,
            "probabilities": probabilities,
            "threshold": threshold,
        }


def create_balanced_dataset(
    normal_features: np.ndarray,
    normal_labels: np.ndarray,
    adversarial_features: np.ndarray,
    adversarial_labels: np.ndarray,
    balance_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced dataset from normal and adversarial examples.

    Args:
        normal_features: Normal example features
        normal_labels: Normal example labels (all 0)
        adversarial_features: Adversarial example features
        adversarial_labels: Adversarial example labels (all 1)
        balance_ratio: Ratio of anomalies to normal (default 1.0 = equal)

    Returns:
        Tuple of (features, labels) with balanced classes
    """
    n_normal = len(normal_features)
    n_adversarial = len(adversarial_features)

    # Determine target counts
    target_adversarial = int(n_normal * balance_ratio)

    # Resample adversarial examples if needed
    if n_adversarial > target_adversarial:
        # Downsample
        indices = np.random.choice(n_adversarial, target_adversarial, replace=False)
        adversarial_features = adversarial_features[indices]
        adversarial_labels = adversarial_labels[indices]
    elif n_adversarial < target_adversarial:
        # Upsample with replacement
        indices = np.random.choice(n_adversarial, target_adversarial, replace=True)
        adversarial_features = adversarial_features[indices]
        adversarial_labels = adversarial_labels[indices]

    # Combine
    features = np.vstack([normal_features, adversarial_features])
    labels = np.concatenate([normal_labels, adversarial_labels])

    # Shuffle
    shuffle_idx = np.random.permutation(len(features))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    logger.info(
        f"Created balanced dataset: {len(normal_features)} normal, "
        f"{len(adversarial_features)} adversarial"
    )

    return features, labels
