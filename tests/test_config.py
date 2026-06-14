from atlasvault.config import AtlasVaultConfig
from atlasvault.library import AtlasVault
from atlasvault.library import AtlasVaultError
from atlasvault.manifest import Manifest
from atlasvault.metadata import MetadataStore
import pytest


def test_config_roundtrip(tmp_path):
    config = AtlasVaultConfig.default("papers", "papers")
    config.input.mode = "sync"
    config.input.input_dir = "D:/papers"
    config.embedding.provider = "huggingface"
    config.embedding.model = "BAAI/bge-base-en-v1.5"
    config.llm.max_tokens = None
    config.save(tmp_path)

    loaded = AtlasVaultConfig.load(tmp_path)

    assert loaded.library.name == "papers"
    assert loaded.library.type == "papers"
    assert loaded.input.mode == "sync"
    assert loaded.input.input_dir == "D:/papers"
    assert loaded.embedding.provider == "huggingface"
    assert loaded.embedding.model == "BAAI/bge-base-en-v1.5"
    assert loaded.llm.max_tokens is None


def test_papers_preset_sets_structured_docling_pipeline():
    config = AtlasVaultConfig.default("papers", "general")

    config.apply_preset("papers")

    assert config.library.preset == "papers"
    assert config.library.type == "papers"
    assert config.parser.provider == "docling"
    assert config.parser.output_format == "json"
    assert config.chunking.strategy == "docling"
    assert config.retrieval.mode == "comprehensive"


def test_docling_chunking_requires_docling_json():
    config = AtlasVaultConfig.default("notes")
    config.chunking.strategy = "docling"

    with pytest.raises(ValueError, match="chunking.strategy=docling"):
        config.validate()


def test_prepare_sources_archives_raw_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")

    library = AtlasVault.create(tmp_path / "library", name="notes")
    ingest_root, input_files, records = library._prepare_sources(input_dir, archive=True)

    assert ingest_root == library.path / "sources"
    assert len(input_files) == 1
    assert input_files[0] == library.path / "sources" / "note.txt"
    assert records[0]["path"] == "sources/note.txt"
    assert input_files[0].read_text(encoding="utf-8") == "alpha beta"


def test_sync_mode_forces_reference_only(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")

    library = AtlasVault.create(
        tmp_path / "library",
        name="notes",
        input_dir=input_dir,
        input_mode="sync",
        archive_sources=True,
    )

    assert library._resolve_archive_sources(None) is False


def test_prepare_sources_reference_only_single_file(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")

    library = AtlasVault.create(tmp_path / "library", name="notes")
    ingest_root, input_files, records = library._prepare_sources(source, archive=False)

    assert ingest_root == source.parent
    assert input_files == [source]
    assert records[0]["path"] == str(source.resolve())


def test_remove_source_updates_manifest_and_metadata_without_rebuild(tmp_path):
    library = AtlasVault.create(tmp_path / "library", name="notes")
    source = library.path / "sources" / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")
    records = [{"path": "sources/note.txt", "sha256": "abc", "size": 10, "modified_at": "now"}]
    for record in records:
        library.metadata.upsert_document(record)
    library._write_manifest()

    library.remove_source("sources/note.txt", delete_raw=False, rebuild_index=False)

    manifest = Manifest.load_or_create(library.path)
    assert "sources/note.txt" not in [record["path"] for record in manifest.files]
    assert source.exists()
    assert library.metadata.list_documents() == []


def test_create_overwrite_clears_managed_layout(tmp_path):
    library_path = tmp_path / "library"
    library = AtlasVault.create(library_path, name="notes")
    stale = library.path / "sources" / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    AtlasVault.create(library_path, name="notes", overwrite=True)

    assert not stale.exists()


def test_metadata_store_roundtrip(tmp_path):
    store = MetadataStore(tmp_path)
    store.upsert_document({"path": "sources/note.txt", "sha256": "abc", "size": 10})

    assert store.list_documents()[0]["path"] == "sources/note.txt"


def test_duplicate_filter_skips_existing_hash(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")
    library = AtlasVault.create(tmp_path / "library", name="notes")
    _, input_files, records = library._prepare_sources(source, archive=False)
    library.metadata.upsert_document(records[0])

    filtered_files, filtered_records = library._filter_duplicate_inputs(input_files, records)

    assert filtered_files == []
    assert filtered_records == []


def test_failed_ingest_does_not_mark_documents_as_ingested(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("alpha beta", encoding="utf-8")
    library = AtlasVault.create(tmp_path / "library", name="notes")

    def fail_ingest(*args, **kwargs):
        raise AtlasVaultError("boom")

    library._ingest_with_llamaindex = fail_ingest

    with pytest.raises(AtlasVaultError, match="boom"):
        library.ingest(source, archive_sources=False)

    assert library.metadata.list_documents() == []


def test_sync_reconcile_removes_missing_documents(tmp_path):
    library = AtlasVault.create(tmp_path / "library", name="notes", input_mode="sync")
    existing = tmp_path / "existing.txt"
    existing.write_text("alpha", encoding="utf-8")
    library.metadata.upsert_document({"path": str(existing), "sha256": "abc", "size": 5})
    library.metadata.upsert_document({"path": str(tmp_path / "missing.txt"), "sha256": "def", "size": 5})

    library._reconcile_sync_deletions([existing])

    paths = [record["path"] for record in library.metadata.list_documents()]
    assert str(existing) in paths
    assert str(tmp_path / "missing.txt") not in paths


def test_sync_deletion_reindexes_all_current_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    existing = input_dir / "existing.txt"
    added = input_dir / "added.txt"
    existing.write_text("alpha", encoding="utf-8")
    added.write_text("beta", encoding="utf-8")

    library = AtlasVault.create(
        tmp_path / "library",
        name="notes",
        input_dir=input_dir,
        input_mode="sync",
    )
    existing_record = library._records_for_input_files([existing], archive=False)[0]
    library.metadata.upsert_document(existing_record)
    library.metadata.upsert_document(
        {"path": str(input_dir / "missing.txt"), "sha256": "def", "size": 5}
    )

    captured = {}

    def capture_ingest(_ingest_root, *, input_files=None, file_records=None):
        captured["input_files"] = input_files
        captured["file_records"] = file_records
        for record in file_records:
            library.metadata.upsert_document(record)

    library._ingest_with_llamaindex = capture_ingest
    library.ingest()

    assert set(captured["input_files"]) == {existing, added}
    assert {record["path"] for record in captured["file_records"]} == {
        str(existing.resolve()),
        str(added.resolve()),
    }


