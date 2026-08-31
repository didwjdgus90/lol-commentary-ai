from hashlib import sha256

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
    ResolvedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.documents.patch_note_builder import (
    build_patch_rag_document,
    build_patch_rag_documents,
)

SOURCE_SHA = sha256(b"patch-source").hexdigest()
ENTITY_SHA = sha256(b"entity-source").hexdigest()


def _record(
    *,
    order: int = 0,
    title: str = "E - 회전 베기",
    entity_type: PatchEntityType = PatchEntityType.CHAMPION,
    entity_name: str | None = "트린다미어",
    entity_id: str | None = "Tryndamere",
    resolution_method: ResolutionMethod = ResolutionMethod.CONTEXT_EXACT,
    changes: list[str] | None = None,
) -> ResolvedPatchRecord:
    return ResolvedPatchRecord(
        record_id=sha256(f"record:{order}:{title}".encode()).hexdigest(),
        order=order,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        section_kind="hotfix",
        entity_type=entity_type,
        entity_name=entity_name,
        heading_path=["추가 패치 노트", title],
        title=title,
        paragraphs=[],
        changes=changes or ["기본 피해량 : 70 ⇒ 80"],
        content=title,
        ddragon_version="16.1.1",
        entity_id=entity_id,
        entity_key=(
            "23" if entity_type == PatchEntityType.CHAMPION and entity_id is not None else None
        ),
        resolution_method=resolution_method,
        resolution_candidate_count=(1 if entity_id is not None else 0),
        resolution_original_candidate_count=1,
        target_map_id=None,
        resolution_evidence_fields=[],
        context_method=(
            "hotfix_structural_exact"
            if resolution_method == ResolutionMethod.CONTEXT_EXACT
            else None
        ),
        context_confidence=(
            "high" if resolution_method == ResolutionMethod.CONTEXT_EXACT else None
        ),
        context_anchor_title=(
            "트린다미어" if resolution_method == ResolutionMethod.CONTEXT_EXACT else None
        ),
        context_evidence=[],
        entity_source_sha256=(ENTITY_SHA if entity_id is not None else None),
    )


def test_resolved_champion_metadata_is_preserved() -> None:
    document = build_patch_rag_document(_record())

    assert document.entity_type == PatchEntityType.CHAMPION
    assert document.entity_name == "트린다미어"
    assert document.entity_id == "Tryndamere"
    assert document.entity_resolved is True
    assert document.resolution_method == ResolutionMethod.CONTEXT_EXACT
    assert "트린다미어" in document.retrieval_text
    assert "기본 피해량 : 70 ⇒ 80" in document.retrieval_text


def test_unresolved_record_is_kept_in_rag_corpus() -> None:
    record = _record(
        title="확인된 문제",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        entity_id=None,
        resolution_method=ResolutionMethod.UNRESOLVED,
    )

    document = build_patch_rag_document(record)

    assert document.entity_type == PatchEntityType.UNKNOWN
    assert document.entity_resolved is False
    assert document.entity_id is None
    assert document.title == "확인된 문제"


def test_removed_item_lifecycle_is_preserved() -> None:
    record = _record(
        title="군단의 방패",
        entity_type=PatchEntityType.ITEM,
        entity_name="군단의 방패",
        entity_id=None,
        resolution_method=(ResolutionMethod.REMOVED_FROM_TARGET_MAP),
        changes=["게임에서 삭제되었습니다."],
    )

    document = build_patch_rag_document(record)

    assert document.removed_from_target_map is True
    assert document.entity_resolved is False
    assert "삭제되었습니다" in document.retrieval_text


def test_document_id_is_deterministic() -> None:
    record = _record()

    first = build_patch_rag_document(record)
    second = build_patch_rag_document(record)

    assert first.document_id == second.document_id
    assert first.content_sha256 == second.content_sha256


def test_resolution_change_changes_document_id() -> None:
    unresolved = _record(
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        entity_id=None,
        resolution_method=ResolutionMethod.UNRESOLVED,
    )
    resolved = unresolved.model_copy(
        update={
            "entity_type": PatchEntityType.CHAMPION,
            "entity_name": "트린다미어",
            "entity_id": "Tryndamere",
            "entity_key": "23",
            "resolution_method": (ResolutionMethod.CONTEXT_EXACT),
            "resolution_candidate_count": 1,
            "entity_source_sha256": ENTITY_SHA,
        }
    )

    unresolved_doc = build_patch_rag_document(unresolved)
    resolved_doc = build_patch_rag_document(resolved)

    assert unresolved_doc.document_id != resolved_doc.document_id


def test_batch_builder_preserves_order_and_count() -> None:
    records = [
        _record(order=0, title="첫 번째"),
        _record(order=1, title="두 번째"),
        _record(order=2, title="세 번째"),
    ]

    documents = build_patch_rag_documents(records)

    assert len(documents) == 3
    assert [document.source_order for document in documents] == [0, 1, 2]
