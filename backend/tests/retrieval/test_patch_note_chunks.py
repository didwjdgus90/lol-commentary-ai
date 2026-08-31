from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.chunks.models import (
    ChunkStrategy,
)
from lol_commentary_backend.retrieval.chunks.patch_note_chunker import (
    build_patch_rag_chunks,
    build_patch_rag_chunks_for_documents,
)
from lol_commentary_backend.retrieval.documents.models import (
    PatchRagDocument,
    RagSourceType,
)

SOURCE_SHA = sha256(b"patch-source").hexdigest()


def _document(
    *,
    document_id_seed: str = "document-1",
    title: str = "E - 회전 베기",
    paragraphs: list[str] | None = None,
    changes: list[str] | None = None,
    retrieval_text: str | None = None,
    entity_type: PatchEntityType = PatchEntityType.CHAMPION,
    entity_name: str | None = "트린다미어",
    entity_id: str | None = "Tryndamere",
) -> PatchRagDocument:
    paragraphs = paragraphs or []
    changes = changes or []

    if retrieval_text is None:
        body = "\n".join([*paragraphs, *changes])
        retrieval_text = (
            f"26.1 패치 | 트린다미어 | {title}\n\n추가 패치 노트 > {title}\n\n{body or title}"
        )

    return PatchRagDocument(
        builder_version="0.1.0",
        document_id=sha256(document_id_seed.encode()).hexdigest(),
        content_sha256=sha256(retrieval_text.encode()).hexdigest(),
        source_type=RagSourceType.PATCH_NOTE,
        source_record_id=sha256(f"record:{document_id_seed}".encode()).hexdigest(),
        source_order=0,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        ddragon_version="16.1.1",
        section_kind="hotfix",
        entity_type=entity_type,
        entity_name=entity_name,
        entity_id=entity_id,
        entity_key=("23" if entity_id is not None else None),
        resolution_method=(
            ResolutionMethod.CONTEXT_EXACT if entity_id is not None else ResolutionMethod.UNRESOLVED
        ),
        entity_resolved=entity_id is not None,
        removed_from_target_map=False,
        target_map_id=None,
        heading_path=[
            "추가 패치 노트",
            title,
        ],
        title=title,
        paragraphs=paragraphs,
        changes=changes,
        retrieval_text=retrieval_text,
        context_method=("hotfix_structural_exact" if entity_id is not None else None),
        context_confidence=("high" if entity_id is not None else None),
        context_anchor_title=("트린다미어" if entity_id is not None else None),
    )


def test_small_document_remains_single_chunk() -> None:
    document = _document(changes=["기본 피해량: 70 ⇒ 80"])

    chunks = build_patch_rag_chunks(
        document,
        max_chars=600,
    )

    assert len(chunks) == 1
    assert chunks[0].text == document.retrieval_text
    assert chunks[0].strategy == ChunkStrategy.SINGLE_DOCUMENT
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_count == 1


def test_long_document_splits_on_semantic_blocks() -> None:
    changes = [f"변경 {index}: " + ("가" * 170) for index in range(6)]
    document = _document(changes=changes)

    chunks = build_patch_rag_chunks(
        document,
        max_chars=600,
    )

    assert len(chunks) > 1
    assert all(chunk.strategy == ChunkStrategy.SEMANTIC_SPLIT for chunk in chunks)
    assert all(chunk.char_count <= 600 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)


def test_semantic_blocks_are_not_duplicated_by_overlap() -> None:
    changes = [f"고유변경-{index}-" + ("나" * 180) for index in range(5)]
    document = _document(changes=changes)

    chunks = build_patch_rag_chunks(
        document,
        max_chars=500,
    )

    combined = "\n".join(chunk.text for chunk in chunks)

    for index in range(5):
        marker = f"고유변경-{index}-"
        assert combined.count(marker) == 1


def test_chunk_ids_are_deterministic() -> None:
    document = _document(changes=[f"변경 {index}: " + ("다" * 180) for index in range(5)])

    first = build_patch_rag_chunks(
        document,
        max_chars=500,
    )
    second = build_patch_rag_chunks(
        document,
        max_chars=500,
    )

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_metadata_inherits_parent_document() -> None:
    document = _document(changes=[f"변경 {index}: " + ("라" * 180) for index in range(5)])

    chunks = build_patch_rag_chunks(
        document,
        max_chars=500,
    )

    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.document_content_sha256 == document.content_sha256
        assert chunk.entity_id == document.entity_id
        assert chunk.patch == document.patch
        assert chunk.source_url == document.source_url


def test_unresolved_metadata_is_preserved() -> None:
    document = _document(
        title="확인된 문제",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        entity_id=None,
        changes=["관전 모드 관련 문제가 있습니다."],
    )

    chunk = build_patch_rag_chunks(document)[0]

    assert chunk.entity_type == PatchEntityType.UNKNOWN
    assert chunk.entity_resolved is False
    assert chunk.resolution_method == ResolutionMethod.UNRESOLVED


def test_batch_builder_preserves_all_documents() -> None:
    documents = [
        _document(
            document_id_seed="one",
            title="첫 번째",
        ),
        _document(
            document_id_seed="two",
            title="두 번째",
        ),
    ]

    chunks = build_patch_rag_chunks_for_documents(documents)

    assert {chunk.document_id for chunk in chunks} == {
        document.document_id for document in documents
    }


def test_rejects_unreasonably_small_limit() -> None:
    with pytest.raises(
        ValueError,
        match="max_chars",
    ):
        build_patch_rag_chunks(
            _document(),
            max_chars=100,
        )
