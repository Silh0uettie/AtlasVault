from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tomllib


CONFIG_FILE = "atlasvault.toml"


@dataclass
class LibrarySettings:
    name: str = "default"
    type: str = "general"
    preset: str = "custom"
    portable: bool = True


@dataclass
class InputSettings:
    mode: str = "case"
    input_dir: str | None = None
    archive_sources: bool = True
    duplicate_check: bool = True


@dataclass
class PathSettings:
    sources_dir: str = "sources"
    parsed_dir: str = "parsed"
    index_dir: str = "index"
    metadata_dir: str = "metadata"
    eval_dir: str = "eval"


@dataclass
class ParserSettings:
    provider: str = "llamaindex"
    output_format: str = "markdown"


@dataclass
class ChunkingSettings:
    strategy: str = "sentence"
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class EmbeddingSettings:
    provider: str = "llamaindex_default"
    model: str = "default"
    api_key_env: str | None = None
    base_url: str | None = None
    batch_size: int = 32


@dataclass
class VectorStoreSettings:
    provider: str = "llamaindex_simple"
    collection: str = "default"


@dataclass
class RetrievalSettings:
    mode: str = "fast"
    top_k: int = 10
    chunks_per_document: int = 3
    max_context_chunks: int = 40
    max_context_chars: int = 120000
    max_document_chars: int = 8000


@dataclass
class LlmSettings:
    provider: str = "llamaindex_default"
    model: str = "default"
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = None
    context_window: int | None = None
    request_timeout: float = 360.0


@dataclass
class AtlasVaultConfig:
    library: LibrarySettings = field(default_factory=LibrarySettings)
    input: InputSettings = field(default_factory=InputSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    parser: ParserSettings = field(default_factory=ParserSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)

    @classmethod
    def default(cls, name: str, library_type: str = "general") -> "AtlasVaultConfig":
        config = cls()
        config.library.name = name
        config.library.type = library_type
        config.vector_store.collection = name
        return config

    def apply_preset(self, preset: str) -> None:
        if preset == "custom":
            self.library.preset = "custom"
            return
        if self.library.type == "general":
            self.library.type = preset
        if preset == "notes":
            self.parser.provider = "llamaindex"
            self.parser.output_format = "markdown"
            self.chunking.strategy = "sentence"
            self.retrieval.mode = "fast"
        elif preset == "manuals":
            self.parser.provider = "docling"
            self.parser.output_format = "markdown"
            self.chunking.strategy = "sentence"
            self.retrieval.mode = "fast"
        elif preset == "papers":
            self.parser.provider = "docling"
            self.parser.output_format = "json"
            self.chunking.strategy = "docling"
            self.retrieval.mode = "comprehensive"
        else:
            raise ValueError(f"Unsupported preset: {preset}")
        self.library.preset = preset

    @classmethod
    def load(cls, library_path: Path) -> "AtlasVaultConfig":
        path = library_path / CONFIG_FILE
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AtlasVaultConfig":
        config = cls()
        for key, value in raw.items():
            if hasattr(config, key) and isinstance(value, dict):
                section = getattr(config, key)
                for section_key, section_value in value.items():
                    if hasattr(section, section_key):
                        setattr(section, section_key, section_value)
        return config

    def save(self, library_path: Path) -> None:
        self.validate()
        library_path.mkdir(parents=True, exist_ok=True)
        (library_path / CONFIG_FILE).write_text(self.to_toml(), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        lines: list[str] = []
        for section, values in self.to_dict().items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                if value is None:
                    continue
                lines.append(f"{key} = {format_toml_value(value)}")
            lines.append("")
        return "\n".join(lines)

    def validate(self) -> None:
        choices = {
            "library.preset": (self.library.preset, {"custom", "notes", "manuals", "papers"}),
            "input.mode": (self.input.mode, {"case", "sync"}),
            "parser.provider": (self.parser.provider, {"llamaindex", "docling"}),
            "parser.output_format": (self.parser.output_format, {"markdown", "json"}),
            "chunking.strategy": (self.chunking.strategy, {"sentence", "docling"}),
            "embedding.provider": (
                self.embedding.provider,
                {"llamaindex_default", "default", "", "openai", "huggingface", "ollama"},
            ),
            "vector_store.provider": (self.vector_store.provider, {"llamaindex_simple", "chroma"}),
            "retrieval.mode": (self.retrieval.mode, {"fast", "comprehensive", "per_document"}),
            "llm.provider": (self.llm.provider, {"llamaindex_default", "default", "", "openai", "ollama"}),
        }
        for label, (value, allowed) in choices.items():
            if value not in allowed:
                allowed_text = ", ".join(sorted(str(item) for item in allowed if item != ""))
                raise ValueError(f"Unsupported {label}: {value}. Choices: {allowed_text}")

        if self.parser.provider == "llamaindex" and self.parser.output_format != "markdown":
            raise ValueError("parser.output_format=json requires parser.provider=docling.")
        if self.chunking.strategy == "docling":
            if self.parser.provider != "docling" or self.parser.output_format != "json":
                raise ValueError(
                    "chunking.strategy=docling requires parser.provider=docling "
                    "and parser.output_format=json."
                )
        if self.chunking.chunk_size <= 0:
            raise ValueError("chunking.chunk_size must be greater than 0.")
        if self.chunking.chunk_overlap < 0:
            raise ValueError("chunking.chunk_overlap must be 0 or greater.")
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            raise ValueError("chunking.chunk_overlap must be smaller than chunking.chunk_size.")
        if self.embedding.batch_size <= 0:
            raise ValueError("embedding.batch_size must be greater than 0.")
        if self.retrieval.top_k <= 0:
            raise ValueError("retrieval.top_k must be greater than 0.")
        if self.retrieval.chunks_per_document <= 0:
            raise ValueError("retrieval.chunks_per_document must be greater than 0.")
        if self.retrieval.max_context_chunks <= 0:
            raise ValueError("retrieval.max_context_chunks must be greater than 0.")
        if self.retrieval.max_context_chars <= 0:
            raise ValueError("retrieval.max_context_chars must be greater than 0.")
        if self.retrieval.max_document_chars <= 0:
            raise ValueError("retrieval.max_document_chars must be greater than 0.")


def format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


