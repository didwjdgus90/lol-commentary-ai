from lol_commentary_backend.ingestion.multi_patch.models import (
    CorpusShardManifest,
    MultiPatchCorpusManifest,
)


def test_shard_manifest_uses_chunk_hash_as_corpus_hash() -> None:
    manifest = CorpusShardManifest(
        patch="26.17",
        locale="ko_KR",
        ddragon_version=("16.17.1"),
        raw_source_sha256=("a" * 64),
        normalized_count=10,
        resolved_count=10,
        resolved_entity_count=8,
        ambiguous_count=1,
        unresolved_count=1,
        document_count=10,
        chunk_count=12,
        max_chunk_chars=600,
        normalized_sha256=("b" * 64),
        resolved_sha256=("c" * 64),
        documents_sha256=("d" * 64),
        corpus_sha256=("e" * 64),
    )

    assert manifest.patch == "26.17"

    assert manifest.chunk_count == 12

    assert len(manifest.corpus_sha256) == 64


def test_aggregate_manifest_tracks_shards() -> None:
    manifest = MultiPatchCorpusManifest(
        start_patch="26.1",
        end_patch="26.17",
        locales=(
            "ko_KR",
            "en_US",
        ),
        patch_count=17,
        shard_count=34,
        total_document_count=100,
        total_chunk_count=120,
        patch_manifest_sha256=("a" * 64),
        version_mapping_sha256=("b" * 64),
        entity_catalog_sha256=("c" * 64),
        shards=(),
    )

    assert manifest.patch_count == 17

    assert manifest.shard_count == 34


def test_corpus_version_is_explicit() -> None:
    manifest = CorpusShardManifest(
        patch="26.1",
        locale="en_US",
        ddragon_version="16.1.1",
        raw_source_sha256="a" * 64,
        normalized_count=1,
        resolved_count=1,
        resolved_entity_count=1,
        ambiguous_count=0,
        unresolved_count=0,
        document_count=1,
        chunk_count=1,
        max_chunk_chars=600,
        normalized_sha256="b" * 64,
        resolved_sha256="c" * 64,
        documents_sha256="d" * 64,
        corpus_sha256="e" * 64,
    )

    assert manifest.corpus_version == "multi_patch_patch_notes_v1"
