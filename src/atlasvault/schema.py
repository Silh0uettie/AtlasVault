from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceChunk:
    id: str
    text: str
    score: float | None = None
    source_path: str | None = None
    file_name: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path | None:
        return Path(self.source_path) if self.source_path else None


@dataclass
class AtlasAnswer:
    text: str
    sources: list[SourceChunk] = field(default_factory=list)
    raw_response: Any = None



