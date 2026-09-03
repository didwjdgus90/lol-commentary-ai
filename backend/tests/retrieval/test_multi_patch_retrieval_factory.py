import pytest

from lol_commentary_backend.ingestion.multi_patch.models import (
    CorpusShardReference,
    MultiPatchCorpusManifest,
)
from lol_commentary_backend.retrieval.runtime.multi_factory import (
    select_multi_patch_shards,
)


def _shard(
    *,
    patch: str,
    locale: str,
    sha: str,
) -> CorpusShardReference:
    return CorpusShardReference(
        patch=patch,
        locale=locale,
        manifest_path=(f"data/{patch}/{locale}/manifest.json"),
        corpus_sha256=sha,
        chunk_count=10,
    )


def _manifest() -> MultiPatchCorpusManifest:
    return MultiPatchCorpusManifest(
        start_patch="26.1",
        end_patch="26.2",
        locales=(
            "ko_KR",
            "en_US",
        ),
        patch_count=2,
        shard_count=4,
        total_document_count=40,
        total_chunk_count=40,
        patch_manifest_sha256=("a" * 64),
        version_mapping_sha256=("b" * 64),
        entity_catalog_sha256=("c" * 64),
        shards=(
            _shard(
                patch="26.1",
                locale="ko_KR",
                sha="1" * 64,
            ),
            _shard(
                patch="26.1",
                locale="en_US",
                sha="2" * 64,
            ),
            _shard(
                patch="26.2",
                locale="ko_KR",
                sha="3" * 64,
            ),
            _shard(
                patch="26.2",
                locale="en_US",
                sha="4" * 64,
            ),
        ),
    )


def test_selects_korean_shards() -> None:
    selected = select_multi_patch_shards(
        _manifest(),
        locales=("ko_KR",),
    )

    assert len(selected) == 2

    assert all(shard.locale == "ko_KR" for shard in selected)


def test_selects_patch_scope() -> None:
    selected = select_multi_patch_shards(
        _manifest(),
        locales=(
            "ko_KR",
            "en_US",
        ),
        patches=("26.2",),
    )

    assert len(selected) == 2

    assert all(shard.patch == "26.2" for shard in selected)


def test_selects_single_patch_locale() -> None:
    selected = select_multi_patch_shards(
        _manifest(),
        locales=("en_US",),
        patches=("26.1",),
    )

    assert len(selected) == 1

    assert selected[0].patch == "26.1"

    assert selected[0].locale == "en_US"


def test_rejects_empty_locale_scope() -> None:
    with pytest.raises(
        ValueError,
        match="locales must not be empty",
    ):
        select_multi_patch_shards(
            _manifest(),
            locales=(),
        )
