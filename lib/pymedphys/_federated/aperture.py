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

"""The aperture: the single, auditable egress point of a clinic.

A ``DataLoader`` exists to widen a pipe. An aperture exists to narrow one. It
is the precise, enforced, logged specification of which bytes are permitted to
leave a hospital, and it is intended to be reviewable on its own -- a
physicist or an ethics committee should be able to approve one policy object
and one audit log without reading a training loop.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import numbers
import pathlib
from typing import Any, Callable, Iterable, Mapping, Sequence

from pymedphys._imports import numpy as np

from .protocol import EvalResult, FitResult


class ApertureViolation(Exception):
    """Raised when a payload is not permitted to leave the clinic.

    This is deliberately not a warning. A payload that violates policy is
    dropped, recorded, and raised on -- there is no partial emission.
    """


@dataclasses.dataclass(frozen=True)
class AperturePolicy:
    """What a site permits to leave, per round.

    Parameters
    ----------
    max_bytes_per_round
        Hard cap on the total size of a single emission, counting array
        payloads and the encoded metrics.
    forbidden_shapes
        Shapes that must never leave. In practice this is the data grid: any
        array shaped like a patient is not a gradient. A payload array is
        rejected when its shape equals a forbidden shape, or when its trailing
        axes do (which catches a batch of volumes).
    allowed_metric_keys
        The only metric names that may travel. ``patient_mrn`` is not a
        metric.
    min_examples
        Minimum cohort size behind any emitted statistic. A mean over ``n=1``
        is a data leak wearing a moustache.
    max_arrays
        Optional cap on the number of arrays in a payload. Catches the case
        where a debugging list grows a member per patient.
    allow_non_finite
        Whether NaN or infinite values may leave. Off by default: a NaN
        payload poisons an aggregate silently, and finiteness is close to free
        to check.
    """

    max_bytes_per_round: int
    forbidden_shapes: tuple[tuple[int, ...], ...] = ()
    allowed_metric_keys: frozenset[str] = frozenset()
    min_examples: int = 1
    max_arrays: int | None = None
    allow_non_finite: bool = False

    def __post_init__(self):
        object.__setattr__(
            self,
            "forbidden_shapes",
            tuple(
                tuple(int(axis) for axis in shape) for shape in self.forbidden_shapes
            ),
        )
        object.__setattr__(
            self, "allowed_metric_keys", frozenset(self.allowed_metric_keys)
        )

        if self.max_bytes_per_round <= 0:
            raise ValueError("`max_bytes_per_round` must be positive.")

        if self.min_examples < 1:
            raise ValueError("`min_examples` must be at least 1.")

        if self.max_arrays is not None and self.max_arrays < 1:
            raise ValueError("`max_arrays` must be at least 1 when set.")

    def to_dict(self) -> dict[str, Any]:
        """A JSON friendly representation, for the audit log's header."""

        return {
            "max_bytes_per_round": self.max_bytes_per_round,
            "forbidden_shapes": [list(shape) for shape in self.forbidden_shapes],
            "allowed_metric_keys": sorted(self.allowed_metric_keys),
            "min_examples": self.min_examples,
            "max_arrays": self.max_arrays,
            "allow_non_finite": self.allow_non_finite,
        }


def policy_from_manifest(
    manifest,
    max_bytes_per_round: int,
    allowed_metric_keys: Iterable[str],
    min_examples: int = 10,
    max_arrays: int | None = None,
) -> AperturePolicy:
    """Build a policy whose forbidden shape is the site's own data grid.

    The grid shape is the one shape a site knows is dangerous without having
    to think about it, and it is already declared in the manifest, so deriving
    it here removes a chance to get the two out of step.
    """

    grid = tuple(manifest.grid_shape)
    structures = len(manifest.structure_keys)

    forbidden = [grid, (1,) + grid]
    if structures:
        forbidden.append((structures,) + grid)

    return AperturePolicy(
        max_bytes_per_round=max_bytes_per_round,
        forbidden_shapes=tuple(dict.fromkeys(forbidden)),
        allowed_metric_keys=frozenset(allowed_metric_keys),
        min_examples=min_examples,
        max_arrays=max_arrays,
    )


class Aperture:
    """Enforce an :class:`AperturePolicy` and record what passed through it.

    Parameters
    ----------
    policy
        The rules to enforce.
    site_id
        Which clinic this aperture belongs to. Written to every audit record.
    audit_log_path
        Where to append audit records, one JSON object per line. ``None``
        keeps the log in memory only, which is appropriate for tests and not
        for anything else.
    clock
        Callable returning the current time. Injectable so that tests are
        deterministic.
    """

    def __init__(
        self,
        policy: AperturePolicy,
        site_id: str,
        audit_log_path: str | pathlib.Path | None = None,
        clock: Callable[[], datetime.datetime] | None = None,
    ):
        self._policy = policy
        self._site_id = site_id
        self._audit_log_path = (
            pathlib.Path(audit_log_path) if audit_log_path is not None else None
        )
        self._clock = clock if clock is not None else _utc_now
        self._records: list[dict[str, Any]] = []

        if self._audit_log_path is not None:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def policy(self) -> AperturePolicy:
        return self._policy

    @property
    def site_id(self) -> str:
        return self._site_id

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """The audit records written by this aperture, in order."""

        return tuple(self._records)

    def emit(
        self,
        weights: Sequence["np.ndarray"],
        num_examples: int,
        metrics: Mapping[str, Any] | None = None,
        round_number: int = 0,
    ) -> FitResult:
        """Check and record a training payload, or raise.

        The return value is the only thing a trainer's ``fit`` should hand
        back, so that an emission which skipped the aperture is a visible code
        change rather than an omission.
        """

        arrays = _as_arrays(weights)
        metrics = dict(metrics or {})

        self._check(
            arrays=arrays,
            num_examples=num_examples,
            metrics=metrics,
            round_number=round_number,
            kind="fit",
        )

        return FitResult(
            weights=arrays, num_examples=num_examples, metrics=_as_floats(metrics)
        )

    def emit_evaluation(
        self,
        loss: float,
        num_examples: int,
        metrics: Mapping[str, Any] | None = None,
        round_number: int = 0,
    ) -> EvalResult:
        """Check and record an evaluation payload, or raise.

        Evaluation runs through the same aperture as training, because a
        per-site loss computed over a handful of patients is a statistic about
        those patients.
        """

        metrics = dict(metrics or {})

        self._check(
            arrays=(),
            num_examples=num_examples,
            metrics={**metrics, "loss": loss},
            round_number=round_number,
            kind="evaluate",
            implicit_metric_keys=("loss",),
        )

        return EvalResult(
            loss=float(loss), num_examples=num_examples, metrics=_as_floats(metrics)
        )

    def _check(
        self,
        arrays: tuple["np.ndarray", ...],
        num_examples: int,
        metrics: Mapping[str, Any],
        round_number: int,
        kind: str,
        implicit_metric_keys: Sequence[str] = (),
    ):
        try:
            self._raise_on_violation(
                arrays=arrays,
                num_examples=num_examples,
                metrics=metrics,
                implicit_metric_keys=implicit_metric_keys,
            )
        except ApertureViolation as violation:
            self._record(
                arrays=arrays,
                num_examples=num_examples,
                metrics=metrics,
                round_number=round_number,
                kind=kind,
                status="rejected",
                reason=str(violation),
            )
            raise

        self._record(
            arrays=arrays,
            num_examples=num_examples,
            metrics=metrics,
            round_number=round_number,
            kind=kind,
            status="emitted",
            reason=None,
        )

    def _raise_on_violation(
        self,
        arrays: tuple["np.ndarray", ...],
        num_examples: int,
        metrics: Mapping[str, Any],
        implicit_metric_keys: Sequence[str],
    ):
        policy = self._policy

        if int(num_examples) < policy.min_examples:
            raise ApertureViolation(
                f"Payload summarises {num_examples} example(s); policy requires at "
                f"least {policy.min_examples}."
            )

        if policy.max_arrays is not None and len(arrays) > policy.max_arrays:
            raise ApertureViolation(
                f"Payload holds {len(arrays)} arrays; policy permits at most "
                f"{policy.max_arrays}."
            )

        for index, array in enumerate(arrays):
            for forbidden in policy.forbidden_shapes:
                if _shape_matches(array.shape, forbidden):
                    raise ApertureViolation(
                        f"Array {index} has shape {tuple(array.shape)}, which matches "
                        f"the forbidden shape {forbidden}. Voxel shaped arrays do not "
                        "leave the clinic."
                    )

            if not policy.allow_non_finite and not np.all(np.isfinite(array)):
                raise ApertureViolation(
                    f"Array {index} holds non-finite values, which the policy forbids."
                )

        allowed = set(policy.allowed_metric_keys) | set(implicit_metric_keys)
        disallowed = sorted(set(metrics) - allowed)
        if disallowed:
            raise ApertureViolation(
                f"Metric key(s) {disallowed} are not on the whitelist "
                f"{sorted(allowed)}."
            )

        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ApertureViolation(
                    f"Metric {key!r} is {type(value).__name__}; only real scalars "
                    "may leave."
                )

            if not policy.allow_non_finite and not np.isfinite(float(value)):
                raise ApertureViolation(
                    f"Metric {key!r} is not finite, which the policy forbids."
                )

        byte_count = _byte_count(arrays, metrics)
        if byte_count > policy.max_bytes_per_round:
            raise ApertureViolation(
                f"Payload is {byte_count} bytes; policy caps a round at "
                f"{policy.max_bytes_per_round} bytes."
            )

    def _record(
        self,
        arrays: tuple["np.ndarray", ...],
        num_examples: int,
        metrics: Mapping[str, Any],
        round_number: int,
        kind: str,
        status: str,
        reason: str | None,
    ):
        record = {
            "timestamp": self._clock().isoformat(),
            "site_id": self._site_id,
            "round": int(round_number),
            "kind": kind,
            "status": status,
            "array_count": len(arrays),
            "byte_count": _byte_count(arrays, metrics),
            "example_count": int(num_examples),
            "metric_keys": sorted(metrics),
            "payload_sha256": payload_digest(arrays, metrics, num_examples),
        }

        if reason is not None:
            record["reason"] = reason

        self._records.append(record)

        if self._audit_log_path is not None:
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            with open(self._audit_log_path, "a", encoding="utf-8") as audit_log:
                audit_log.write(line + "\n")


def read_audit_log(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read an append-only audit log back into a list of records."""

    records = []
    with open(path, encoding="utf-8") as audit_log:
        for line in audit_log:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def payload_digest(
    arrays: Sequence["np.ndarray"], metrics: Mapping[str, Any], num_examples: int
) -> str:
    """A stable SHA-256 over exactly what would leave the clinic.

    The digest covers dtype, shape and bytes of each array as well as the
    metrics and cohort size, so that an aggregator's copy of a payload can be
    checked against the site's own audit record.
    """

    digest = hashlib.sha256()

    for array in arrays:
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())

    digest.update(
        json.dumps(
            {key: _jsonable(value) for key, value in metrics.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(str(int(num_examples)).encode("utf-8"))

    return digest.hexdigest()


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _as_arrays(weights: Sequence["np.ndarray"]) -> tuple["np.ndarray", ...]:
    return tuple(np.asarray(weight) for weight in weights)


def _as_floats(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)

    return repr(value)


def _byte_count(arrays: Sequence["np.ndarray"], metrics: Mapping[str, Any]) -> int:
    array_bytes = sum(int(array.nbytes) for array in arrays)
    metric_bytes = len(
        json.dumps(
            {key: _jsonable(value) for key, value in metrics.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    return array_bytes + metric_bytes


def _shape_matches(shape: tuple[int, ...], forbidden: tuple[int, ...]) -> bool:
    if not forbidden:
        return False

    if tuple(shape) == forbidden:
        return True

    return len(shape) > len(forbidden) and tuple(shape[-len(forbidden) :]) == forbidden
