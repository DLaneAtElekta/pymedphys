# Copyright (C) 2026 PyMedPhys Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A NumPy site trainer that fits a mean vector, for tests and teaching.

The model is the smallest thing with a closed form answer, which is the point:
the federated mean over non-IID sites must equal the pooled mean exactly, so a
failure is unambiguously the plumbing rather than the optimiser.

It also carries a site-local parameter that is deliberately excluded from
``shared_keys``, which is the FedBN pattern in miniature.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pymedphys._imports import numpy as np

from .aperture import Aperture
from .protocol import EvalResult, FitResult, SiteManifest


class MeanVectorTrainer:
    """Fit the mean of a site's local data, sharing only that mean.

    Parameters
    ----------
    data
        Local data, shaped ``(num_examples, num_features)``. Never leaves.
    manifest
        The site's declaration of its representation.
    aperture
        The only egress. ``fit`` and ``evaluate`` return what it returns.
    leak_debug_volume
        For teaching and tests: attach a data-grid shaped array to the
        payload, as a careless debugging change would. The aperture is
        expected to reject it.
    leak_metric_key
        For teaching and tests: attach a metric under a key that is not on the
        whitelist. The aperture is expected to reject it.
    """

    def __init__(
        self,
        data: "np.ndarray",
        manifest: SiteManifest,
        aperture: Aperture,
        leak_debug_volume: bool = False,
        leak_metric_key: str | None = None,
    ):
        self._data = np.asarray(data, dtype=float)

        if self._data.ndim != 2:
            raise ValueError(
                f"`data` must be shaped (num_examples, num_features), got "
                f"{self._data.shape}."
            )

        self._manifest = manifest
        self._aperture = aperture
        self._leak_debug_volume = leak_debug_volume
        self._leak_metric_key = leak_metric_key

        self._mean = np.zeros(self._data.shape[1], dtype=float)

        # Site-local, never shared. Stands in for a normalisation statistic
        # that would be meaningless averaged across scanners.
        self._local_scale = np.std(self._data, axis=0)

    def manifest(self) -> SiteManifest:
        return self._manifest

    def shared_keys(self) -> list[str]:
        return ["mean"]

    def get_weights(self) -> list["np.ndarray"]:
        return [self._mean.copy()]

    def set_weights(self, weights: Sequence["np.ndarray"]) -> None:
        (mean,) = weights
        mean = np.asarray(mean, dtype=float)

        if mean.shape != self._mean.shape:
            raise ValueError(
                f"Expected a mean shaped {self._mean.shape}, got {mean.shape}."
            )

        self._mean = mean.copy()

    def fit(self, config: Mapping[str, Any]) -> FitResult:
        learning_rate = float(config.get("learning_rate", 1.0))
        steps = int(config.get("local_steps", 1))

        local_mean = np.mean(self._data, axis=0)

        for _ in range(steps):
            self._mean = self._mean - learning_rate * (self._mean - local_mean)

        weights: list["np.ndarray"] = [self._mean.copy()]

        if self._leak_debug_volume:
            weights.append(np.zeros(self._manifest.grid_shape, dtype=float))

        metrics: dict[str, Any] = {"train_loss": self._loss()}

        if self._leak_metric_key is not None:
            metrics[self._leak_metric_key] = 1.0

        return self._aperture.emit(
            weights=weights,
            num_examples=self._data.shape[0],
            metrics=metrics,
            round_number=int(config.get("round", 0)),
        )

    def evaluate(self, config: Mapping[str, Any]) -> EvalResult:
        return self._aperture.emit_evaluation(
            loss=self._loss(),
            num_examples=self._data.shape[0],
            metrics={},
            round_number=int(config.get("round", 0)),
        )

    @property
    def local_scale(self) -> "np.ndarray":
        """The site-local parameter, exposed so tests can show it stays local."""

        return self._local_scale.copy()

    def _loss(self) -> float:
        return float(np.mean((self._data - self._mean) ** 2))
