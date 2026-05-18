# atlasvault

`AtlasVault` manages portable local knowledge libraries. Each library is a folder containing its own
configuration, source-file archive, parsed artifacts, metadata, and vector index.

## Quick Start

```bash
python -m pip install -e ".[rag,docling,local-models,chroma,dev]"

atlasvault init-library D:/RAG/papers --name papers --type papers \
  --preset papers \
  --input-dir C:/Users/me/Papers \
  --input-mode case \
  --embedding-provider huggingface \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --embedding-batch-size 32 \
  --llm-provider ollama \
  --llm-model qwen2.5:14b \
  --temperature 0.1
atlasvault ingest --library D:/RAG/papers C:/Users/me/Papers
atlasvault run --library D:/RAG/papers
atlasvault ask --library D:/RAG/papers "Which papers discuss U-Net?"
```

Each library contains:

```text
atlasvault.toml
manifest.json
sources/
parsed/
index/
  llamaindex/
  chroma/
metadata/
  atlasvault.db
eval/
```

SQLite is the authoritative metadata store. `manifest.json` is kept as a lightweight summary
for portability checks, while `metadata/atlasvault.db` stores documents, chunks, hashes, page/source
metadata, and ingestion state.

For portability, use archived sources:

```bash
atlasvault ingest --library D:/RAG/papers C:/Users/me/Papers --archive-sources
```

Use reference-only mode if you do not want raw files copied into the portable library:

```bash
atlasvault ingest --library D:/RAG/papers C:/Users/me/Papers --reference-only
```

If the library was created with `--input-dir`, you can omit the source path:

```bash
atlasvault ingest --library D:/RAG/papers
atlasvault run --library D:/RAG/papers
```

Input modes:

```text
case
  One-off ingestion. Raw files can be archived into sources/.

sync
  Tracks an external input directory. Raw files are not copied into sources/.
  Missing files are removed from metadata on the next run.
  Unchanged files are skipped when duplicate_check is enabled.
```

Remove a source from metadata and rebuild the index:

```bash
atlasvault remove-source --library D:/RAG/papers sources/paper1.pdf
atlasvault remove-source --library D:/RAG/papers sources/paper1.pdf --delete-raw
```

List indexed source documents:

```bash
atlasvault sources --library D:/RAG/papers
```

## Presets And Chunking

Use presets for normal library creation:

```text
notes
  LlamaIndex file loading, sentence chunking, fast retrieval.
  Best for Markdown, text files, and Obsidian-style notes.

manuals
  Docling converts documents/PDFs to Markdown, then sentence chunking.
  Best for manuals where readable text flow matters more than layout structure.

papers
  Docling JSON parsing, Docling structural chunking, comprehensive retrieval.
  Best for PDFs where pages, sections, tables, and citations matter.

custom
  AtlasVault defaults, with every parser/chunking/retrieval option controlled manually.
```

Advanced equivalent for the papers preset:

```bash
atlasvault init-library D:/RAG/papers \
  --parser docling \
  --parser-output-format json \
  --chunking-strategy docling \
  --retrieval-mode comprehensive
```

Default text chunking uses LlamaIndex `SentenceSplitter`:

```text
chunk_size = 800
chunk_overlap = 120
```

That means each chunk is targeted around 800 tokens, with 120 tokens repeated between
neighboring chunks to preserve context across boundaries.

`chunking.strategy = "sentence"` uses `chunk_size` and `chunk_overlap`.
`chunking.strategy = "docling"` uses `DoclingNodeParser`, which chunks according to Docling
document elements such as headings, paragraphs, tables, and page-grounded layout metadata.
Docling chunking requires `parser.provider = "docling"` and `parser.output_format = "json"`.

`parser.output_format` only applies to Docling:

```text
markdown
  Docling exports readable Markdown, then AtlasVault sentence-chunks it.

json
  Docling exports structured document JSON. Use with chunking.strategy = "docling"
  when you want layout-aware chunks for papers.
```

The first implementation uses LlamaIndex when installed. OpenAI, Ollama, HuggingFace, and
Docling can be selected in `atlasvault.toml` or at `init-library` time.

Useful installs:

```bash
python -m pip install -e ".[rag,local-models,chroma,dev]"
python -m pip install -e ".[rag,docling,local-models,chroma,dev]"
python -m pip install -e ".[rag,openai,chroma,dev]"
```

Secrets should live in environment variables, not in the portable library folder:

```bash
atlasvault init-library D:/RAG/papers \
  --name papers \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --embedding-api-key-env OPENAI_API_KEY \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-api-key-env OPENAI_API_KEY
```


