from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import hashlib
import json


MANIFEST_FILE = "manifest.json"


@dataclass
class Manifest:
    atlasvault_version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    files: list[dict[str, Any]] = field(default_factory=list)
    pipeline: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_or_create(cls, library_path: Path) -> "Manifest":
        path = library_path / MANIFEST_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, library_path: Path) -> None:
        self.updated_at = datetime.now(UTC).isoformat()
        (library_path / MANIFEST_FILE).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def file_record(path: Path, relative_path: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



