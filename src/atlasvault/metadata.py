from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import sqlite3


DB_FILE = "atlasvault.db"


class MetadataStore:
    def __init__(self, library_path: Path, metadata_dir: str = "metadata") -> None:
        self.library_path = library_path
        self.path = library_path / metadata_dir / DB_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    file_type TEXT,
                    sha256 TEXT,
                    size INTEGER,
                    modified_at TEXT,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    authors_json TEXT,
                    year INTEGER,
                    doi TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    token_count INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path);

                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    parser TEXT,
                    embedding_provider TEXT,
                    embedding_model TEXT,
                    vector_store TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def upsert_document(self, record: dict[str, Any]) -> int:
        with self._connect() as conn:
            return self._upsert_document(conn, record)

    def replace_documents_and_chunks(
        self,
        records: Iterable[dict[str, Any]],
        nodes: Iterable[object],
    ) -> None:
        grouped = self._group_nodes_by_source(nodes)
        with self._connect() as conn:
            for record in records:
                self._upsert_document(conn, record)
            for source_path, source_nodes in grouped.items():
                document_id = self._document_id_for_source(conn, source_path)
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self._insert_chunks(conn, document_id, source_nodes)

    def _upsert_document(self, conn: sqlite3.Connection, record: dict[str, Any]) -> int:
        source_path = str(record["path"])
        file_name = Path(source_path).name
        metadata = dict(record.get("metadata") or {})
        authors = metadata.get("authors", [])
        conn.execute(
            """
            INSERT INTO documents (
                source_path, file_name, file_type, sha256, size, modified_at,
                title, authors_json, year, doi, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_name=excluded.file_name,
                file_type=excluded.file_type,
                sha256=excluded.sha256,
                size=excluded.size,
                modified_at=excluded.modified_at,
                title=excluded.title,
                authors_json=excluded.authors_json,
                year=excluded.year,
                doi=excluded.doi,
                metadata_json=excluded.metadata_json
            """,
            (
                source_path,
                file_name,
                metadata.get("file_type"),
                record.get("sha256"),
                record.get("size"),
                record.get("modified_at"),
                metadata.get("title"),
                json.dumps(authors if isinstance(authors, list) else []),
                metadata.get("year"),
                metadata.get("doi"),
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT id FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        return int(row["id"])

    def replace_chunks_for_nodes(self, nodes: Iterable[object]) -> None:
        grouped = self._group_nodes_by_source(nodes)
        with self._connect() as conn:
            for source_path, source_nodes in grouped.items():
                document_id = self._document_id_for_source(conn, source_path)
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self._insert_chunks(conn, document_id, source_nodes)

    def _group_nodes_by_source(self, nodes: Iterable[object]) -> dict[str, list[object]]:
        grouped: dict[str, list[object]] = {}
        for node in nodes:
            metadata = dict(getattr(node, "metadata", {}) or {})
            source_path = metadata.get("source_path") or metadata.get("file_path") or "unknown"
            grouped.setdefault(str(source_path), []).append(node)
        return grouped

    def _document_id_for_source(self, conn: sqlite3.Connection, source_path: str) -> int:
        row = conn.execute(
            "SELECT id FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if row is None:
            return self._upsert_document(conn, {"path": source_path, "metadata": {}})
        return int(row["id"])

    def _insert_chunks(
        self,
        conn: sqlite3.Connection,
        document_id: int,
        source_nodes: Iterable[object],
    ) -> None:
        for index, node in enumerate(source_nodes):
            metadata = dict(getattr(node, "metadata", {}) or {})
            chunk_index = int(metadata.get("chunk_index", index))
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            page = parse_optional_int(metadata.get("page_label") or metadata.get("page"))
            conn.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    id, document_id, chunk_index, text, page, section,
                    token_count, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(getattr(node, "node_id", metadata.get("id", ""))),
                    document_id,
                    chunk_index,
                    text,
                    page,
                    metadata.get("section"),
                    metadata.get("token_count"),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_path AS path, sha256, size, modified_at
                FROM documents
                ORDER BY source_path
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunk_records(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    chunks.id,
                    chunks.text,
                    chunks.metadata_json,
                    documents.source_path
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                ORDER BY documents.source_path, chunks.chunk_index
                """
            ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.setdefault("source_path", row["source_path"])
            records.append({"id": row["id"], "text": row["text"], "metadata": metadata})
        return records

    def remove_document(self, source_path: str) -> list[dict[str, Any]]:
        matches = self.find_documents(source_path)
        with self._connect() as conn:
            for record in matches:
                conn.execute("DELETE FROM documents WHERE source_path = ?", (record["path"],))
        return matches

    def find_documents(self, source_path: str) -> list[dict[str, Any]]:
        target = normalize_source_key(source_path)
        return [
            record
            for record in self.list_documents()
            if source_key_matches(record["path"], target)
        ]

    def clear_chunks(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")

    def reset(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM ingestion_runs")


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_source_key(source_path: str | Path) -> str:
    return str(source_path).replace("\\", "/").strip()


def source_key_matches(candidate: str, target: str) -> bool:
    normalized = normalize_source_key(candidate)
    return normalized == target or normalized.endswith(f"/{target}") or Path(normalized).name == target



