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

"""The framework agnostic contract a clinic implements in order to take part
in a federation.

Nothing in this module imports a deep learning framework or a federated
learning framework. It is NumPy in and NumPy out, so that a site trainer can
be exercised end to end in a unit test with no server running.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pymedphys._imports import numpy as np

_SHA256_LENGTH = 64


class ManifestError(ValueError):
    """Raised when a :class:`SiteManifest` is internally inconsistent."""


@dataclasses.dataclass(frozen=True)
class SiteManifest:
    """A site's declaration of the representation it will train on.

    Every field other than ``site_id`` and ``notes`` contributes to
    :attr:`compatibility_key`. Two sites whose keys differ are not training on
    the same thing, whatever their loss curves suggest.

    Parameters
    ----------
    site_id
        Free text label for the clinic. Deliberately excluded from the
        compatibility key -- sites are meant to differ here.
    grid_shape
        Shape of the canonical voxel grid, for example ``(128, 128, 128)``.
    voxel_spacing_mm
        Spacing of that grid in millimetres, one entry per axis of
        ``grid_shape``.
    structure_keys
        The canonical (TG-263) structure names the site will present, sorted
        and without duplicates.
    structure_vocabulary_sha256
        SHA-256 of the *shared* canonical vocabulary the site maps its local
        structure names onto. See :mod:`pymedphys._dicom.structure.tg263`.
    alias_table_sha256
        SHA-256 of the site's own, local alias table. Recorded for provenance
        and audit; deliberately *not* part of the compatibility key, because
        every site's alias table legitimately differs.
    notes
        Free text. Not hashed.
    """

    site_id: str
    grid_shape: tuple[int, ...]
    voxel_spacing_mm: tuple[float, ...]
    structure_keys: tuple[str, ...]
    structure_vocabulary_sha256: str
    alias_table_sha256: str = ""
    notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "grid_shape", tuple(int(x) for x in self.grid_shape))
        object.__setattr__(
            self, "voxel_spacing_mm", tuple(float(x) for x in self.voxel_spacing_mm)
        )
        object.__setattr__(
            self, "structure_keys", tuple(str(x) for x in self.structure_keys)
        )

        if not self.site_id:
            raise ManifestError("A site manifest requires a non-empty `site_id`.")

        if not self.grid_shape:
            raise ManifestError("`grid_shape` must have at least one axis.")

        if any(axis <= 0 for axis in self.grid_shape):
            raise ManifestError(
                f"`grid_shape` must be positive, got {self.grid_shape}."
            )

        if len(self.voxel_spacing_mm) != len(self.grid_shape):
            raise ManifestError(
                "`voxel_spacing_mm` must have one entry per axis of `grid_shape`, "
                f"got {self.voxel_spacing_mm} for {self.grid_shape}."
            )

        if any(spacing <= 0 for spacing in self.voxel_spacing_mm):
            raise ManifestError(
                f"`voxel_spacing_mm` must be positive, got {self.voxel_spacing_mm}."
            )

        if len(set(self.structure_keys)) != len(self.structure_keys):
            raise ManifestError(
                f"`structure_keys` contains duplicates: {self.structure_keys}."
            )

        if list(self.structure_keys) != sorted(self.structure_keys):
            raise ManifestError(
                "`structure_keys` must be sorted so that the compatibility key is "
                f"insensitive to declaration order, got {self.structure_keys}."
            )

        _validate_sha256(
            "structure_vocabulary_sha256", self.structure_vocabulary_sha256
        )

        if self.alias_table_sha256:
            _validate_sha256("alias_table_sha256", self.alias_table_sha256)

    @property
    def compatibility_fields(self) -> dict[str, Any]:
        """The subset of the manifest that every site must agree on."""

        return {
            "grid_shape": list(self.grid_shape),
            "voxel_spacing_mm": list(self.voxel_spacing_mm),
            "structure_keys": list(self.structure_keys),
            "structure_vocabulary_sha256": self.structure_vocabulary_sha256,
        }

    @property
    def compatibility_key(self) -> str:
        """A single hash standing in for "we mean the same thing by a batch"."""

        canonical = json.dumps(
            self.compatibility_fields, sort_keys=True, separators=(",", ":")
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """A JSON friendly representation, including the derived key."""

        record = dataclasses.asdict(self)
        record["grid_shape"] = list(self.grid_shape)
        record["voxel_spacing_mm"] = list(self.voxel_spacing_mm)
        record["structure_keys"] = list(self.structure_keys)
        record["compatibility_key"] = self.compatibility_key

        return record


def _validate_sha256(name: str, value: str):
    if len(value) != _SHA256_LENGTH:
        raise ManifestError(
            f"`{name}` must be a 64 character SHA-256 hex digest, got {value!r}."
        )

    try:
        int(value, 16)
    except ValueError as error:
        raise ManifestError(
            f"`{name}` must be a SHA-256 hex digest, got {value!r}."
        ) from error


@dataclasses.dataclass(frozen=True)
class FitResult:
    """What a site returns from a round of local training.

    This is the object that crosses the clinic boundary, so it is deliberately
    small: an array per shared parameter, a cohort size for weighting, and
    whitelisted scalar metrics.
    """

    weights: tuple["np.ndarray", ...]
    num_examples: int
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """What a site returns from a round of local evaluation."""

    loss: float
    num_examples: int
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "loss", float(self.loss))
        object.__setattr__(self, "metrics", dict(self.metrics))


@runtime_checkable
class SiteTrainer(Protocol):
    """The six methods a clinic implements.

    Implementations own their data loading, their model, and their optimiser.
    They do not own the network, and they do not decide what is allowed to
    leave -- ``fit`` and ``evaluate`` are expected to return whatever their
    :class:`~pymedphys._federated.aperture.Aperture` returns, so that
    bypassing the aperture is a visible code change rather than an omission.
    """

    def manifest(self) -> SiteManifest:
        """Declare the representation this site trains on."""

    def shared_keys(self) -> Sequence[str]:
        """Name the parameters that cross the boundary, in weight order.

        Excluding a parameter here is how FedBN (keep normalisation
        statistics local), partial federation (share a decoder, keep an
        encoder private), and staged unfreezing are expressed.
        """

    def get_weights(self) -> Sequence["np.ndarray"]:
        """Return the shared parameters, ordered to match ``shared_keys``."""

    def set_weights(self, weights: Sequence["np.ndarray"]) -> None:
        """Adopt aggregated shared parameters, leaving local ones untouched."""

    def fit(self, config: Mapping[str, Any]) -> FitResult:
        """Train locally for a round and emit through the aperture."""

    def evaluate(self, config: Mapping[str, Any]) -> EvalResult:
        """Evaluate locally for a round and emit through the aperture."""
