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

"""Canonicalisation of structure names onto a shared TG-263 vocabulary.

The literature frames non-IID federated learning as a statistical problem. In
radiotherapy it is overwhelmingly a naming and geometry problem::

    Parotid_L   vs   L Parotid   vs   PAROTID LT   vs   Lt_Parotid_gland

If sites silently disagree, training does not crash. It produces a worse model
and a plausible loss curve, which is far more expensive than a crash.

A mapping file therefore has two parts, and they are hashed separately:

``vocabulary``
    The shared target list. Every participating site must hold the same one,
    so its hash is part of a site's compatibility key.
``aliases``
    The site's own local names for those targets. Every site's alias table
    legitimately differs, so its hash is recorded for provenance and audit but
    is *not* compared between sites.

This split is a deliberate departure from hashing the mapping file as a whole:
a whole-file hash could never match across two clinics, since differing local
names are the entire reason the file exists.

Automatic normalisation is deliberately modest. It resolves case, punctuation,
whitespace and laterality wording, and nothing else. Anything beyond that is
clinical knowledge, belongs in a hand-maintained, version-controlled alias
table, and should be reviewed by a human at each clinic once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
from typing import Any, Iterable, Mapping

import tomlkit

_LATERALITY = {
    "L": "L",
    "LT": "L",
    "LEFT": "L",
    "R": "R",
    "RT": "R",
    "RIGHT": "R",
}

_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")


class UnmappedStructure(KeyError):
    """Raised when a local structure name has no canonical target."""


class MappingError(ValueError):
    """Raised when a mapping file is inconsistent."""


def normalise(name: str) -> str:
    """Reduce a structure name to a case, punctuation and laterality normal form.

    Laterality is recognised in either the leading or trailing token and moved
    to a trailing ``_L`` or ``_R``.

    Examples
    --------
    >>> from pymedphys._dicom.structure.tg263 import normalise
    >>> normalise("Parotid_L")
    'PAROTID_L'
    >>> normalise("L Parotid")
    'PAROTID_L'
    >>> normalise("PAROTID LT")
    'PAROTID_L'
    >>> normalise("Brainstem")
    'BRAINSTEM'
    """

    tokens = [token.upper() for token in _TOKEN_SPLIT.split(name) if token]

    if not tokens:
        raise MappingError(f"Structure name {name!r} holds no alphanumeric characters.")

    laterality = None

    if len(tokens) > 1 and tokens[-1] in _LATERALITY:
        laterality = _LATERALITY[tokens.pop()]
    elif len(tokens) > 1 and tokens[0] in _LATERALITY:
        laterality = _LATERALITY[tokens.pop(0)]

    stem = "".join(tokens)

    if laterality is None:
        return stem

    return f"{stem}_{laterality}"


@dataclasses.dataclass(frozen=True)
class StructureVocabulary:
    """The shared list of canonical structure names, and its hash.

    Parameters
    ----------
    version
        Version label for the vocabulary, so that a hash mismatch can be
        explained without diffing two files.
    structures
        The canonical names, sorted and unique.
    """

    version: str
    structures: tuple[str, ...]

    def __post_init__(self):
        structures = tuple(str(name) for name in self.structures)

        if not structures:
            raise MappingError("A vocabulary must name at least one structure.")

        if len(set(structures)) != len(structures):
            raise MappingError(f"Vocabulary holds duplicate names: {structures}.")

        normalised = [normalise(name) for name in structures]
        if len(set(normalised)) != len(normalised):
            raise MappingError(
                "Vocabulary holds names that normalise to the same key: "
                f"{sorted(structures)}."
            )

        object.__setattr__(self, "structures", tuple(sorted(structures)))

    @property
    def sha256(self) -> str:
        """SHA-256 over the vocabulary, insensitive to file formatting."""

        canonical = json.dumps(
            {"version": self.version, "structures": list(self.structures)},
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class StructureMapping:
    """A shared vocabulary plus one site's local alias table."""

    vocabulary: StructureVocabulary
    aliases: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    source_path: str = ""

    def __post_init__(self):
        aliases = {
            str(canonical): tuple(str(alias) for alias in alias_list)
            for canonical, alias_list in dict(self.aliases).items()
        }

        unknown = sorted(set(aliases) - set(self.vocabulary.structures))
        if unknown:
            raise MappingError(
                f"Alias table targets {unknown}, which are not in the vocabulary "
                f"{list(self.vocabulary.structures)}."
            )

        lookup: dict[str, str] = {}
        for canonical in self.vocabulary.structures:
            lookup[normalise(canonical)] = canonical

        for canonical, alias_list in aliases.items():
            for alias in alias_list:
                key = normalise(alias)
                existing = lookup.get(key)
                if existing is not None and existing != canonical:
                    raise MappingError(
                        f"Alias {alias!r} maps to both {existing!r} and {canonical!r}."
                    )
                lookup[key] = canonical

        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "_lookup", lookup)

    @property
    def vocabulary_sha256(self) -> str:
        """The hash every participating site must agree on."""

        return self.vocabulary.sha256

    @property
    def alias_table_sha256(self) -> str:
        """The hash of this site's local aliases, for provenance only."""

        canonical = json.dumps(
            {key: sorted(value) for key, value in sorted(self.aliases.items())},
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def structure_keys(self) -> tuple[str, ...]:
        """The canonical names, sorted, ready for a :class:`SiteManifest`."""

        return self.vocabulary.structures

    def canonicalise(self, name: str) -> str:
        """Map a local structure name onto its canonical name.

        Raises
        ------
        UnmappedStructure
            If the name is neither a canonical name nor a declared alias. This
            is a hard error on purpose: silently dropping a structure that a
            clinician contoured is exactly the failure this module exists to
            prevent.
        """

        try:
            return self._lookup[normalise(name)]  # type: ignore[attr-defined]
        except KeyError as error:
            raise UnmappedStructure(
                f"{name!r} is not in the vocabulary and has no alias. Add it to the "
                f"alias table at {self.source_path or '<in memory>'}, or exclude it "
                "from the cohort."
            ) from error

    def canonicalise_all(self, names: Iterable[str]) -> dict[str, str]:
        """Canonicalise many names, reporting every failure at once."""

        mapped: dict[str, str] = {}
        unmapped: list[str] = []

        for name in names:
            try:
                mapped[name] = self.canonicalise(name)
            except UnmappedStructure:
                unmapped.append(name)

        if unmapped:
            raise UnmappedStructure(
                f"{len(unmapped)} structure name(s) have no canonical target: "
                f"{sorted(unmapped)}."
            )

        return mapped

    def manifest_fields(self) -> dict[str, Any]:
        """The fields a :class:`~pymedphys._federated.protocol.SiteManifest` needs."""

        return {
            "structure_keys": self.structure_keys,
            "structure_vocabulary_sha256": self.vocabulary_sha256,
            "alias_table_sha256": self.alias_table_sha256,
        }

    @classmethod
    def from_dict(
        cls, contents: Mapping[str, Any], source_path: str = ""
    ) -> "StructureMapping":
        """Build a mapping from an already parsed document."""

        try:
            vocabulary_section = contents["vocabulary"]
        except KeyError as error:
            raise MappingError(
                "A mapping file needs a [vocabulary] table holding `version` and "
                "`structures`."
            ) from error

        try:
            vocabulary = StructureVocabulary(
                version=str(vocabulary_section["version"]),
                structures=tuple(vocabulary_section["structures"]),
            )
        except KeyError as error:
            raise MappingError(f"[vocabulary] is missing the key {error}.") from error

        aliases = contents.get("aliases", {})

        return cls(
            vocabulary=vocabulary,
            aliases={
                str(canonical): tuple(alias_list)
                for canonical, alias_list in dict(aliases).items()
            },
            source_path=source_path,
        )

    @classmethod
    def from_toml_file(cls, path: str | pathlib.Path) -> "StructureMapping":
        """Load a mapping from a TOML file.

        The expected shape is::

            [vocabulary]
            version = "2026-01"
            structures = ["Brainstem", "Parotid_L", "Parotid_R"]

            [aliases]
            Parotid_L = ["Lt_Parotid_gland", "L PAROTID"]
        """

        path = pathlib.Path(path)
        contents = tomlkit.parse(path.read_text(encoding="utf-8"))

        return cls.from_dict(contents, source_path=str(path))
