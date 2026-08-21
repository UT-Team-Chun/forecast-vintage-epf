"""Lightweight on-disk dataset registry.

We avoid bringing in DVC for a solo-author research project. Instead, every
external dataset we fetch records an entry into ``data/registry.yaml``
capturing:

- ``name``         : a stable identifier (e.g. ``jepx_spot_2024``)
- ``relative_path``: location under ``EpfConfig.raw_data_dir``
- ``sha256``       : full SHA-256 of the file
- ``source_url``   : the URL we fetched from
- ``fetched_at``   : ISO-8601 timestamp with timezone
- ``bytes``        : file size in bytes

This makes "which exact bytes was this experiment trained on?" a one-line
lookup, without the operational cost of DVC.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

REGISTRY_FILENAME = "registry.yaml"


class DatasetEntry(BaseModel):
    """One row of the dataset registry."""

    model_config = ConfigDict(frozen=True)

    name: str
    relative_path: Path
    sha256: str = Field(min_length=64, max_length=64)
    source_url: str
    fetched_at: datetime
    size_bytes: int = Field(ge=0)


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetRegistry:
    """Append-friendly YAML-backed registry of downloaded datasets."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._entries: dict[str, DatasetEntry] = {}
        if registry_path.exists():
            self._load()

    def _load(self) -> None:
        raw = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        datasets = raw.get("datasets", {}) or {}
        for name, payload in datasets.items():
            payload_with_name = {"name": name, **payload}
            payload_with_name["relative_path"] = Path(payload_with_name["relative_path"])
            self._entries[name] = DatasetEntry(**payload_with_name)

    def find(self, name: str) -> DatasetEntry | None:
        return self._entries.get(name)

    def upsert(self, entry: DatasetEntry) -> None:
        self._entries[entry.name] = entry
        self._write()

    def all_entries(self) -> Iterable[DatasetEntry]:
        return tuple(self._entries.values())

    def _write(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "datasets": {
                name: {
                    "relative_path": str(entry.relative_path),
                    "sha256": entry.sha256,
                    "source_url": entry.source_url,
                    "fetched_at": entry.fetched_at.isoformat(),
                    "size_bytes": entry.size_bytes,
                }
                for name, entry in sorted(self._entries.items())
            },
        }
        self.registry_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, indent=2),
            encoding="utf-8",
        )


def make_entry(
    *,
    name: str,
    raw_data_dir: Path,
    file_path: Path,
    source_url: str,
) -> DatasetEntry:
    """Build a ``DatasetEntry`` from an already-downloaded file."""
    relative = file_path.relative_to(raw_data_dir)
    return DatasetEntry(
        name=name,
        relative_path=relative,
        sha256=hash_file(file_path),
        source_url=source_url,
        fetched_at=datetime.now(tz=UTC),
        size_bytes=file_path.stat().st_size,
    )
