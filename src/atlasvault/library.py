from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import os
import shutil
import sqlite3

from atlasvault.config import CONFIG_FILE, AtlasVaultConfig
from atlasvault.manifest import Manifest, file_record
from atlasvault.metadata import MetadataStore, normalize_source_key
from atlasvault.schema import AtlasAnswer, SourceChunk


class AtlasVaultError(RuntimeError):
    """Base exception for AtlasVault errors."""


class MissingDependencyError(AtlasVaultError):
    """Raised when an optional RAG dependency is needed but unavailable."""


class AtlasVault:
    def __init__(self, path: str | Path, config: AtlasVaultConfig | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self.config = config or AtlasVaultConfig.load(self.path)
        self.config.validate()
        try:
            self.metadata = MetadataStore(self.path, self.config.paths.metadata_dir)
        except (OSError, sqlite3.Error) as exc:
            raise AtlasVaultError(f"Could not open metadata store at {self.path}: {exc}") from exc

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str | None = None,
        library_type: str = "general",
        input_dir: str | Path | None = None,
        input_mode: str = "case",
        archive_sources: bool = True,
        duplicate_check: bool = True,
        preset: str = "custom",
        config: AtlasVaultConfig | None = None,
        overwrite: bool = False,
    ) -> "AtlasVault":
        library_path = Path(path).expanduser().resolve()
        if config is None:
            config = AtlasVaultConfig.default(name=name or library_path.name, library_type=library_type)
            config.apply_preset(preset)
            config.input.mode = input_mode
            config.input.input_dir = str(input_dir) if input_dir is not None else None
            config.input.archive_sources = False if input_mode == "sync" else archive_sources
            config.input.duplicate_check = duplicate_check
        config.validate()

        if (library_path / CONFIG_FILE).exists() and not overwrite:
            raise AtlasVaultError(f"Library already exists: {library_path}")
        if overwrite:
            cls._clear_managed_layout(library_path)

        try:
            config.save(library_path)
            library = cls(library_path, config)
            library.ensure_layout()

            manifest = Manifest.load_or_create(library_path)
            manifest.pipeline = config.to_dict()
            manifest.save(library_path)
            return library
        except (AtlasVaultError, OSError, sqlite3.Error) as exc:
            cleanup_message = ""
            try:
                cls._clear_managed_layout(library_path)
            except OSError as cleanup_exc:
                cleanup_message = f" Cleanup of partial files also failed: {cleanup_exc}"
            raise AtlasVaultError(
                f"Could not create library at {library_path}: {exc}.{cleanup_message}"
            ) from exc

    @staticmethod
    def _clear_managed_layout(library_path: Path) -> None:
        for relative in (
            CONFIG_FILE,
            "manifest.json",
            "sources",
            "parsed",
            "index",
            "metadata",
            "eval",
        ):
            target = library_path / relative
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    @classmethod
    def open(cls, path: str | Path) -> "AtlasVault":
        return cls(path)

    def ensure_layout(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        for folder in (
            self.config.paths.sources_dir,
            self.config.paths.parsed_dir,
            self.config.paths.index_dir,
            self.config.paths.metadata_dir,
            self.config.paths.eval_dir,
        ):
            (self.path / folder).mkdir(parents=True, exist_ok=True)

    def ingest(self, source: str | Path | None = None, archive_sources: bool | None = None) -> None:
        self.ensure_layout()
        source_path = self._resolve_input_source(source)
        archive = self._resolve_archive_sources(archive_sources)
        original_files = self._source_file_list(source_path)
        file_records = self._records_for_input_files(original_files, archive=False)
        removed_for_sync = False
        if self.config.input.mode == "sync":
            removed_for_sync = self._reconcile_sync_deletions(original_files)
        if self.config.input.duplicate_check and not removed_for_sync:
            original_files, file_records = self._filter_duplicate_inputs(original_files, file_records)
        if not original_files:
            if removed_for_sync:
                self.rebuild_index()
                return
            self._write_manifest()
            return
        if archive:
            ingest_root, input_files, file_records = self._archive_input_files(source_path, original_files)
        else:
            ingest_root = source_path.parent if source_path.is_file() else source_path
            input_files = original_files
        if removed_for_sync:
            self._clear_index()
        self._ingest_with_llamaindex(
            ingest_root,
            input_files=input_files,
            file_records=file_records,
        )
        self._write_manifest()

    def run(self) -> None:
        self.ingest()

    def search(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        index = self._load_llamaindex()
        retriever = index.as_retriever(similarity_top_k=top_k or self.config.retrieval.top_k)
        nodes = retriever.retrieve(query)
        return [self._source_from_node_with_score(node) for node in nodes]

    def ask(self, question: str, top_k: int | None = None, mode: str | None = None) -> AtlasAnswer:
        query_mode = mode or self.config.retrieval.mode
        if query_mode in {"comprehensive", "per_document"}:
            sources = self._comprehensive_sources(question, top_k=top_k, mode=query_mode)
            return self._synthesize_answer(question, sources)

        index = self._load_llamaindex()
        similarity_top_k = top_k or self._top_k_for_mode(query_mode)
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        sources = [self._source_from_node_with_score(node) for node in response.source_nodes]
        return AtlasAnswer(text=str(response), sources=sources, raw_response=response)

    def source_file(self, source: SourceChunk | str) -> Path:
        source_path = source.source_path if isinstance(source, SourceChunk) else source
        if not source_path:
            raise AtlasVaultError("Source does not contain a file path.")
        path = Path(source_path)
        if not path.is_absolute():
            path = self.path / path
        return path.resolve()

    def list_sources(self) -> list[dict[str, Any]]:
        return self.metadata.list_documents()

    def remove_source(
        self,
        source_path: str | Path,
        *,
        delete_raw: bool = False,
        rebuild_index: bool = True,
    ) -> None:
        target = normalize_source_key(source_path)
        removed_records = self.metadata.remove_document(target)
        if not removed_records:
            raise AtlasVaultError(f"No source found in manifest for: {source_path}")

        if delete_raw:
            for record in removed_records:
                self._delete_raw_source(record.get("path", ""))

        self._write_manifest()

        if rebuild_index:
            self.rebuild_index()

    def rebuild_index(self) -> None:
        input_files = [self._manifest_record_path(record) for record in self.metadata.list_documents()]
        input_files = [path for path in input_files if path.exists()]
        self._clear_index()
        if not input_files:
            self.metadata.clear_chunks()
            self._write_manifest()
            return
        self._ingest_with_llamaindex(self.path, input_files=input_files)
        self._write_manifest()

    def _resolve_input_source(self, source: str | Path | None) -> Path:
        raw_source = source if source is not None else self.config.input.input_dir
        if raw_source is None:
            raise AtlasVaultError("No source path provided and no input.input_dir configured.")
        return Path(raw_source).expanduser().resolve()

    def _resolve_archive_sources(self, archive_sources: bool | None) -> bool:
        if self.config.input.mode == "sync":
            return False
        if archive_sources is not None:
            return archive_sources
        return self.config.input.archive_sources

    def _filter_duplicate_inputs(
        self,
        input_files: list[Path],
        file_records: list[dict[str, Any]],
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        existing = {
            (record.get("sha256"), record.get("size"))
            for record in self.metadata.list_documents()
            if record.get("sha256")
        }
        kept_files: list[Path] = []
        kept_records: list[dict[str, Any]] = []
        for file_path, record in zip(input_files, file_records, strict=True):
            fingerprint = (record.get("sha256"), record.get("size"))
            if fingerprint in existing:
                continue
            kept_files.append(file_path)
            kept_records.append(record)
        return kept_files, kept_records

    def _reconcile_sync_deletions(self, input_files: list[Path]) -> bool:
        current_paths = {str(path.resolve()) for path in input_files}
        removed = False
        for record in self.metadata.list_documents():
            path = self._manifest_record_path(record).resolve()
            if str(path) not in current_paths:
                self.metadata.remove_document(record["path"])
                removed = True
        return removed

    def _top_k_for_mode(self, mode: str) -> int:
        if mode == "comprehensive":
            return max(self.config.retrieval.top_k, 50)
        if mode == "per_document":
            return max(self.config.retrieval.top_k, 100)
        return self.config.retrieval.top_k

    def _prepare_sources(
        self,
        source: Path,
        archive: bool,
    ) -> tuple[Path, list[Path], list[dict[str, Any]]]:
        if not source.exists():
            raise AtlasVaultError(f"Source path does not exist: {source}")
        input_files = self._source_file_list(source)
        if not archive:
            records = self._records_for_input_files(input_files, archive=False)
            ingest_root = source.parent if source.is_file() else source
            return ingest_root, input_files, records

        return self._archive_input_files(source, input_files)

    def _source_file_list(self, source: Path) -> list[Path]:
        if not source.exists():
            raise AtlasVaultError(f"Source path does not exist: {source}")
        return [source] if source.is_file() else list(iter_files(source))

    def _records_for_input_files(
        self,
        input_files: list[Path],
        *,
        archive: bool,
    ) -> list[dict[str, Any]]:
        if archive:
            return [
                file_record(file_path, file_path.relative_to(self.path).as_posix())
                for file_path in input_files
            ]
        return [
            file_record(file_path, str(file_path.resolve()))
            for file_path in input_files
        ]

    def _archive_input_files(
        self,
        source: Path,
        input_files: list[Path],
    ) -> tuple[Path, list[Path], list[dict[str, Any]]]:
        sources_dir = self.path / self.config.paths.sources_dir
        copied: list[Path] = []
        if source.is_file():
            destination = sources_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
        else:
            for file_path in input_files:
                relative = file_path.relative_to(source)
                destination = sources_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)
                copied.append(destination)

        records = self._records_for_input_files(copied, archive=True)
        return sources_dir, copied, records

    def _ingest_with_llamaindex(
        self,
        ingest_root: Path,
        input_files: list[Path] | None = None,
        file_records: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required for ingestion. Install with: pip install -e ".[rag]"'
            ) from exc

        self._configure_llamaindex_models()
        splitter = SentenceSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        documents = self._load_documents(ingest_root, input_files=input_files)
        nodes = self._parse_nodes(documents, fallback_splitter=splitter)
        nodes = self._normalize_node_metadata(nodes)
        storage_context = self._storage_context_for_ingest()
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        index.storage_context.persist(persist_dir=str(self._llamaindex_storage_dir()))
        if file_records is None:
            self._write_chunks_metadata(nodes)
        else:
            self.metadata.replace_documents_and_chunks(file_records, nodes)

    def _load_llamaindex(self):
        try:
            from llama_index.core import StorageContext, load_index_from_storage
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required for querying. Install with: pip install -e ".[rag]"'
            ) from exc

        self._configure_llamaindex_models()
        index_dir = self._llamaindex_storage_dir()
        if not index_dir.exists():
            raise AtlasVaultError(f"No index found at {index_dir}. Run atlasvault ingest first.")
        storage_context = self._storage_context_for_query()
        return load_index_from_storage(storage_context)

    def _load_documents(
        self,
        ingest_root: Path,
        input_files: list[Path] | None = None,
    ) -> list[object]:
        if self.config.parser.provider == "docling":
            try:
                from llama_index.readers.docling import DoclingReader
            except ImportError as exc:
                raise MissingDependencyError(
                    'Docling parsing is configured. Install with: pip install -e ".[docling]"'
                ) from exc

            reader = self._docling_reader(DoclingReader)
            documents: list[object] = []
            for file_path in input_files or list(iter_files(ingest_root)):
                extra_info = self._source_metadata(file_path)
                try:
                    documents.extend(reader.load_data(file_path=file_path, extra_info=extra_info))
                except TypeError:
                    documents.extend(reader.load_data(str(file_path), extra_info=extra_info))
            return documents

        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required for loading. Install with: pip install -e ".[rag]"'
            ) from exc
        if input_files:
            return SimpleDirectoryReader(input_files=[str(path) for path in input_files]).load_data()
        return SimpleDirectoryReader(str(ingest_root), recursive=True).load_data()

    def _docling_reader(self, docling_reader_cls):
        output_format = self.config.parser.output_format.lower()
        if output_format == "json":
            export_type = getattr(docling_reader_cls.ExportType, "JSON")
            return docling_reader_cls(export_type=export_type)
        if output_format == "markdown":
            export_type = getattr(docling_reader_cls.ExportType, "MARKDOWN", None)
            if export_type is not None:
                return docling_reader_cls(export_type=export_type)
            return docling_reader_cls()
        raise AtlasVaultError(f"Unsupported Docling output_format: {self.config.parser.output_format}")

    def _parse_nodes(self, documents: list[object], fallback_splitter: object) -> list[object]:
        if self.config.chunking.strategy == "docling":
            try:
                from llama_index.node_parser.docling import DoclingNodeParser
            except ImportError as exc:
                raise MissingDependencyError(
                    'Docling JSON parsing is configured. Install with: pip install -e ".[docling]"'
                ) from exc
            return DoclingNodeParser().get_nodes_from_documents(documents)
        return fallback_splitter.get_nodes_from_documents(documents)

    def _source_metadata(self, file_path: Path) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "file_name": file_path.name,
            "file_path": str(file_path),
        }
        try:
            metadata["source_path"] = file_path.relative_to(self.path).as_posix()
        except ValueError:
            metadata["source_path"] = str(file_path)
        return metadata

    def _clear_index(self) -> None:
        index_dir = self.path / self.config.paths.index_dir
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

    def _llamaindex_storage_dir(self) -> Path:
        return self.path / self.config.paths.index_dir / "llamaindex"

    def _chroma_storage_dir(self) -> Path:
        return self.path / self.config.paths.index_dir / "chroma"

    def _manifest_record_path(self, record: dict[str, Any]) -> Path:
        raw_path = Path(str(record.get("path", "")))
        if raw_path.is_absolute():
            return raw_path
        return self.path / raw_path

    def _delete_raw_source(self, raw_path: str) -> None:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.path / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self.path)
        except ValueError:
            raise AtlasVaultError(f"Refusing to delete source outside library: {resolved}")
        if resolved.exists() and resolved.is_file():
            resolved.unlink()

    def _storage_context_for_ingest(self):
        from llama_index.core import StorageContext

        provider = self.config.vector_store.provider
        if provider == "llamaindex_simple":
            return StorageContext.from_defaults()
        if provider == "chroma":
            return StorageContext.from_defaults(vector_store=self._chroma_vector_store())
        raise AtlasVaultError(f"Unsupported vector store provider: {provider}")

    def _storage_context_for_query(self):
        from llama_index.core import StorageContext

        provider = self.config.vector_store.provider
        if provider == "llamaindex_simple":
            return StorageContext.from_defaults(persist_dir=str(self._llamaindex_storage_dir()))
        if provider == "chroma":
            return StorageContext.from_defaults(
                persist_dir=str(self._llamaindex_storage_dir()),
                vector_store=self._chroma_vector_store(),
            )
        raise AtlasVaultError(f"Unsupported vector store provider: {provider}")

    def _chroma_vector_store(self):
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError as exc:
            raise MissingDependencyError(
                'Chroma is configured. Install with: pip install -e ".[chroma]"'
            ) from exc

        chroma_client = chromadb.PersistentClient(path=str(self._chroma_storage_dir()))
        collection = chroma_client.get_or_create_collection(self.config.vector_store.collection)
        return ChromaVectorStore(chroma_collection=collection)

    def _configure_llamaindex_models(self) -> None:
        try:
            from llama_index.core import Settings
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required. Install with: pip install -e ".[rag]"'
            ) from exc

        embedding = self.config.embedding
        if embedding.provider in {"llamaindex_default", "default", ""}:
            pass
        elif embedding.provider == "openai":
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
            except ImportError as exc:
                raise MissingDependencyError(
                    'OpenAI embeddings are configured. Install with: pip install -e ".[openai]"'
                ) from exc
            Settings.embed_model = OpenAIEmbedding(
                model=embedding.model,
                api_key=self._secret_from_env(embedding.api_key_env),
                api_base=embedding.base_url,
                embed_batch_size=embedding.batch_size,
            )
        elif embedding.provider == "huggingface":
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            except ImportError as exc:
                raise MissingDependencyError(
                    'HuggingFace embeddings are configured. Install with: pip install -e ".[local-models]"'
                ) from exc
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding.model,
                embed_batch_size=embedding.batch_size,
            )
        elif embedding.provider == "ollama":
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
            except ImportError as exc:
                raise MissingDependencyError(
                    'Ollama embeddings are configured. Install with: pip install -e ".[local-models]"'
                ) from exc
            Settings.embed_model = OllamaEmbedding(
                model_name=embedding.model,
                base_url=embedding.base_url,
            )
        else:
            raise AtlasVaultError(f"Unsupported embedding provider: {embedding.provider}")

        llm = self.config.llm
        if llm.provider in {"llamaindex_default", "default", ""}:
            return
        if llm.provider == "openai":
            try:
                from llama_index.llms.openai import OpenAI
            except ImportError as exc:
                raise MissingDependencyError(
                    'OpenAI LLM is configured. Install with: pip install -e ".[openai]"'
                ) from exc
            Settings.llm = OpenAI(
                model=llm.model,
                temperature=llm.temperature,
                max_tokens=llm.max_tokens,
                api_key=self._secret_from_env(llm.api_key_env),
                api_base=llm.base_url,
            )
            return
        if llm.provider == "ollama":
            try:
                from llama_index.llms.ollama import Ollama
            except ImportError as exc:
                raise MissingDependencyError(
                    'Ollama LLM is configured. Install with: pip install -e ".[local-models]"'
                ) from exc
            kwargs = {
                "model": llm.model,
                "temperature": llm.temperature,
                "request_timeout": llm.request_timeout,
            }
            if llm.base_url:
                kwargs["base_url"] = llm.base_url
            if llm.context_window:
                kwargs["context_window"] = llm.context_window
            Settings.llm = Ollama(**kwargs)
            return
        raise AtlasVaultError(f"Unsupported LLM provider: {llm.provider}")

    def _secret_from_env(self, env_name: str | None) -> str | None:
        if not env_name:
            return None
        value = os.environ.get(env_name)
        if not value:
            raise AtlasVaultError(f"Environment variable is not set: {env_name}")
        return value

    def _normalize_node_metadata(self, nodes: Iterable[object]) -> list[object]:
        normalized = []
        for index, node in enumerate(nodes):
            metadata = dict(getattr(node, "metadata", {}) or {})
            raw_path = metadata.get("file_path") or metadata.get("source")
            if raw_path:
                source_path = Path(str(raw_path))
                if source_path.is_absolute():
                    try:
                        metadata["source_path"] = source_path.relative_to(self.path).as_posix()
                    except ValueError:
                        metadata["source_path"] = str(source_path)
                else:
                    metadata["source_path"] = source_path.as_posix()
                metadata.setdefault("file_name", source_path.name)
            metadata.setdefault("chunk_index", index)
            setattr(node, "metadata", metadata)
            normalized.append(node)
        return normalized

    def _write_chunks_metadata(self, nodes: Iterable[object]) -> None:
        self.metadata.replace_chunks_for_nodes(nodes)

    def _write_manifest(self) -> None:
        manifest = Manifest.load_or_create(self.path)
        manifest.files = self.metadata.list_documents()
        manifest.pipeline = self.config.to_dict()
        manifest.save(self.path)

    def _comprehensive_sources(
        self,
        question: str,
        top_k: int | None,
        mode: str,
    ) -> list[SourceChunk]:
        chunk_records = self._chunk_records()
        total_chunks = len(chunk_records)
        document_count = len(
            {
                (record.get("metadata") or {}).get("source_path")
                or (record.get("metadata") or {}).get("file_path")
                or record.get("id")
                for record in chunk_records
            }
        )
        if total_chunks == 0:
            raise AtlasVaultError("No chunk metadata found. Run atlasvault ingest first.")

        if mode == "per_document" or self.config.library.type == "papers":
            retrieval_k = total_chunks
        else:
            retrieval_k = max(
                top_k or self.config.retrieval.top_k,
                document_count * self.config.retrieval.chunks_per_document,
                50,
            )
            retrieval_k = min(retrieval_k, total_chunks)

        candidates = self.search(question, top_k=retrieval_k)
        return self._cap_chunks_per_document(candidates)

    def _cap_chunks_per_document(self, chunks: list[SourceChunk]) -> list[SourceChunk]:
        per_document: dict[str, int] = {}
        selected: list[SourceChunk] = []
        limit = self.config.retrieval.chunks_per_document
        for chunk in chunks:
            key = chunk.source_path or chunk.file_name or chunk.id
            count = per_document.get(key, 0)
            if count >= limit:
                continue
            per_document[key] = count + 1
            selected.append(chunk)
        return selected

    def _synthesize_answer(self, question: str, sources: list[SourceChunk]) -> AtlasAnswer:
        if not sources:
            return AtlasAnswer(text="No relevant sources were found.", sources=[])

        self._configure_llamaindex_models()
        try:
            from llama_index.core import Settings
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required. Install with: pip install -e ".[rag]"'
            ) from exc

        summaries = self._summarize_sources_by_document(question, sources)
        summaries = self._context_limited_summaries(summaries)
        context_blocks = []
        for index, summary in enumerate(summaries, start=1):
            context_blocks.append(f"[{index}] {summary['location']}\n{summary['summary']}")

        prompt = (
            "Answer the question using only the per-document evidence summaries below. "
            "Cite claims with source numbers like [1] or [2]. If the sources do not contain "
            "the answer, say so.\n\n"
            f"Question: {question}\n\nEvidence summaries:\n" + "\n\n".join(context_blocks)
        )
        response = Settings.llm.complete(prompt)
        return AtlasAnswer(
            text=str(response),
            sources=[summary["source"] for summary in summaries],
            raw_response=response,
        )

    def _context_limited_summaries(self, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        char_count = 0
        for summary in summaries:
            if len(selected) >= self.config.retrieval.max_context_chunks:
                break
            next_size = len(summary["summary"])
            if selected and char_count + next_size > self.config.retrieval.max_context_chars:
                break
            selected.append(summary)
            char_count += next_size
        return selected

    def _summarize_sources_by_document(
        self,
        question: str,
        sources: list[SourceChunk],
    ) -> list[dict[str, Any]]:
        try:
            from llama_index.core import Settings
        except ImportError as exc:
            raise MissingDependencyError(
                'LlamaIndex is required. Install with: pip install -e ".[rag]"'
            ) from exc

        grouped: dict[str, list[SourceChunk]] = {}
        for source in sources:
            key = source.source_path or source.file_name or source.id
            grouped.setdefault(key, []).append(source)

        summaries: list[dict[str, Any]] = []
        for location, chunks in grouped.items():
            char_count = 0
            chunk_blocks: list[str] = []
            for chunk in chunks:
                remaining = self.config.retrieval.max_document_chars - char_count
                if remaining <= 0:
                    break
                text = chunk.text[:remaining]
                page = f", page {chunk.page}" if chunk.page is not None else ""
                chunk_blocks.append(f"{location}{page}\n{text}")
                char_count += len(text)
            prompt = (
                "Extract only evidence relevant to the question from this document. "
                "If there is no relevant evidence, answer exactly: NO_RELEVANT_EVIDENCE.\n\n"
                f"Question: {question}\n\nDocument evidence:\n" + "\n\n".join(chunk_blocks)
            )
            summary = str(Settings.llm.complete(prompt)).strip()
            if summary and summary != "NO_RELEVANT_EVIDENCE":
                summaries.append({"location": location, "summary": summary, "source": chunks[0]})
        if summaries:
            return summaries

        # Fall back to raw snippets if the LLM refuses every summary.
        fallback: list[dict[str, Any]] = []
        for index, source in enumerate(sources, start=1):
            location = source.file_name or source.source_path or "unknown source"
            page = f", page {source.page}" if source.page is not None else ""
            fallback.append(
                {
                    "location": f"{location}{page}",
                    "summary": source.text[:1000],
                    "source": source,
                }
            )
            if index >= 5:
                break
        return fallback

    def _chunk_records(self) -> list[dict[str, Any]]:
        return self.metadata.list_chunk_records()

    def _source_from_node_with_score(self, node_with_score: object) -> SourceChunk:
        node = getattr(node_with_score, "node", node_with_score)
        metadata = dict(getattr(node, "metadata", {}) or {})
        score = getattr(node_with_score, "score", None)
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        page = metadata.get("page_label") or metadata.get("page")
        return SourceChunk(
            id=str(getattr(node, "node_id", metadata.get("id", ""))),
            text=text,
            score=score,
            source_path=metadata.get("source_path") or metadata.get("file_path"),
            file_name=metadata.get("file_name"),
            page=int(page) if str(page).isdigit() else None,
            chunk_index=metadata.get("chunk_index"),
            title=metadata.get("title"),
            authors=metadata.get("authors", []) if isinstance(metadata.get("authors"), list) else [],
            year=metadata.get("year"),
            doi=metadata.get("doi"),
            metadata=metadata,
        )


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    ignored_dirs = {".git", ".venv", "__pycache__"}
    for child in path.rglob("*"):
        if child.is_dir():
            continue
        if any(part in ignored_dirs for part in child.parts):
            continue
        yield child


