from __future__ import annotations

from pathlib import Path
import argparse
import sys

from atlasvault.config import AtlasVaultConfig
from atlasvault.library import AtlasVault, AtlasVaultError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (AtlasVaultError, ValueError) as exc:
        print(f"atlasvault: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlasvault")
    subparsers = parser.add_subparsers(required=True)

    init = subparsers.add_parser("init-library", help="Create a portable knowledge library folder.")
    init.add_argument("path")
    init.add_argument("--name")
    init.add_argument("--type", default="general")
    init.add_argument(
        "--preset",
        default="custom",
        choices=["custom", "notes", "manuals", "papers"],
        help="Apply a recommended pipeline: notes, manuals, papers, or custom.",
    )
    init.add_argument("--input-dir")
    init.add_argument("--input-mode", default="case", choices=["case", "sync"])
    init.add_argument("--no-duplicate-check", action="store_true")
    init.add_argument("--parser", choices=["llamaindex", "docling"], help="Override the preset parser.")
    init.add_argument(
        "--parser-output-format",
        choices=["markdown", "json"],
        help="Docling export format. JSON is required for Docling structural chunking.",
    )
    init.add_argument(
        "--chunking-strategy",
        choices=["sentence", "docling"],
        help="sentence uses chunk-size/overlap; docling uses document structure.",
    )
    init.add_argument("--chunk-size", type=int, help="Target token size for sentence chunking.")
    init.add_argument("--chunk-overlap", type=int, help="Token overlap for sentence chunking.")
    init.add_argument(
        "--embedding-provider",
        default="llamaindex_default",
        choices=["llamaindex_default", "openai", "huggingface", "ollama"],
    )
    init.add_argument("--embedding-model", default="default")
    init.add_argument("--embedding-api-key-env")
    init.add_argument("--embedding-base-url")
    init.add_argument("--embedding-batch-size", type=int, default=32)
    init.add_argument(
        "--llm-provider",
        default="llamaindex_default",
        choices=["llamaindex_default", "openai", "ollama"],
    )
    init.add_argument("--llm-model", default="default")
    init.add_argument("--llm-api-key-env")
    init.add_argument("--llm-base-url")
    init.add_argument("--temperature", type=float, default=0.1)
    init.add_argument("--max-tokens", type=int)
    init.add_argument("--context-window", type=int)
    init.add_argument("--request-timeout", type=float, default=360.0)
    init.add_argument(
        "--vector-store",
        default="llamaindex_simple",
        choices=["llamaindex_simple", "chroma"],
    )
    init.add_argument(
        "--retrieval-mode",
        choices=["fast", "comprehensive", "per_document"],
    )
    init.add_argument("--reference-only", action="store_true")
    init.add_argument("--overwrite", action="store_true")
    init.set_defaults(func=cmd_init_library)

    ingest = subparsers.add_parser("ingest", help="Ingest files into a library.")
    ingest.add_argument("source", nargs="?")
    ingest.add_argument("--library", required=True)
    ingest.add_argument("--archive-sources", action="store_true", default=None)
    ingest.add_argument("--reference-only", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    run = subparsers.add_parser("run", help="Run the configured ingestion pipeline for a library.")
    run.add_argument("--library", required=True)
    run.set_defaults(func=cmd_run)

    ask = subparsers.add_parser("ask", help="Ask a question against a library.")
    ask.add_argument("question")
    ask.add_argument("--library", required=True)
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--mode", choices=["fast", "comprehensive", "per_document"])
    ask.set_defaults(func=cmd_ask)

    search = subparsers.add_parser("search", help="Retrieve source chunks without generating an answer.")
    search.add_argument("query")
    search.add_argument("--library", required=True)
    search.add_argument("--top-k", type=int)
    search.set_defaults(func=cmd_search)

    sources = subparsers.add_parser("sources", help="List source documents in a library.")
    sources.add_argument("--library", required=True)
    sources.set_defaults(func=cmd_sources)

    show_config = subparsers.add_parser("show-config", help="Print a library's atlasvault.toml.")
    show_config.add_argument("--library", required=True)
    show_config.set_defaults(func=cmd_show_config)

    remove = subparsers.add_parser("remove-source", help="Remove a source from metadata and rebuild the index.")
    remove.add_argument("source_path")
    remove.add_argument("--library", required=True)
    remove.add_argument("--delete-raw", action="store_true")
    remove.add_argument("--no-rebuild", action="store_true")
    remove.set_defaults(func=cmd_remove_source)

    return parser


def cmd_init_library(args: argparse.Namespace) -> int:
    library_path = Path(args.path).expanduser()
    config = AtlasVaultConfig.default(name=args.name or library_path.name, library_type=args.type)
    config.apply_preset(args.preset)
    config.input.mode = args.input_mode
    config.input.input_dir = args.input_dir
    config.input.archive_sources = False if args.input_mode == "sync" else not args.reference_only
    config.input.duplicate_check = not args.no_duplicate_check
    if args.parser is not None:
        config.parser.provider = args.parser
    if args.parser_output_format is not None:
        config.parser.output_format = args.parser_output_format
    if args.chunking_strategy is not None:
        config.chunking.strategy = args.chunking_strategy
    if args.chunk_size is not None:
        config.chunking.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunking.chunk_overlap = args.chunk_overlap
    config.embedding.provider = args.embedding_provider
    config.embedding.model = args.embedding_model
    config.embedding.api_key_env = args.embedding_api_key_env
    config.embedding.base_url = args.embedding_base_url
    config.embedding.batch_size = args.embedding_batch_size
    config.llm.provider = args.llm_provider
    config.llm.model = args.llm_model
    config.llm.api_key_env = args.llm_api_key_env
    config.llm.base_url = args.llm_base_url
    config.llm.temperature = args.temperature
    config.llm.max_tokens = args.max_tokens
    config.llm.context_window = args.context_window
    config.llm.request_timeout = args.request_timeout
    config.vector_store.provider = args.vector_store
    if args.retrieval_mode is not None:
        config.retrieval.mode = args.retrieval_mode
    config.validate()

    library = AtlasVault.create(
        args.path,
        config=config,
        overwrite=args.overwrite,
    )
    print(f"Created library: {library.path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    archive_sources = None
    if args.archive_sources:
        archive_sources = True
    if args.reference_only:
        archive_sources = False
    library.ingest(args.source, archive_sources=archive_sources)
    print(f"Ingested into: {library.path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    library.run()
    print(f"Pipeline completed for: {library.path}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    answer = library.ask(args.question, top_k=args.top_k, mode=args.mode)
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for index, source in enumerate(answer.sources, start=1):
            location = source.file_name or source.source_path or "unknown"
            page = f":page {source.page}" if source.page is not None else ""
            score = f" score={source.score:.3f}" if source.score is not None else ""
            print(f"[{index}] {location}{page}{score}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    chunks = library.search(args.query, top_k=args.top_k)
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.file_name or chunk.source_path or "unknown"
        page = f":page {chunk.page}" if chunk.page is not None else ""
        score = f" score={chunk.score:.3f}" if chunk.score is not None else ""
        print(f"[{index}] {location}{page}{score}")
        print(snippet(chunk.text))
        print()
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    for source in library.list_sources():
        print(source["path"])
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    path = Path(args.library) / "atlasvault.toml"
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_remove_source(args: argparse.Namespace) -> int:
    library = AtlasVault.open(args.library)
    library.remove_source(
        args.source_path,
        delete_raw=args.delete_raw,
        rebuild_index=not args.no_rebuild,
    )
    print(f"Removed source: {args.source_path}")
    return 0


def snippet(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())


