from datetime import UTC, datetime
from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_context.builder import (
    build_hotfix_champion_context_hints,
)
from lol_commentary_backend.ingestion.patch_note_context.models import (
    ContextConfidence,
    ContextMethod,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.normalizer import (
    normalize_patch_note_document,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchChange,
    PatchNoteDocument,
    PatchSection,
    SectionKind,
)

SOURCE_SHA = sha256(b"raw-html").hexdigest()
CHAMPION_SHA = sha256(b"champions").hexdigest()
ITEM_SHA = sha256(b"items").hexdigest()


def _champion(
    champion_id: str,
    key: str,
    name: str,
) -> CatalogEntity:
    return CatalogEntity(
        entity_type=PatchEntityType.CHAMPION,
        entity_id=champion_id,
        entity_key=key,
        name=name,
        source_sha256=CHAMPION_SHA,
    )


def _item(
    item_id: str,
    name: str,
) -> CatalogEntity:
    return CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id=item_id,
        entity_key=None,
        name=name,
        source_sha256=ITEM_SHA,
        map_ids=frozenset({"11"}),
    )


def _catalog() -> EntityCatalog:
    return EntityCatalog(
        ddragon_version="16.1.1",
        locale="ko_KR",
        champions_by_name={
            "아펠리오스": (
                _champion(
                    "Aphelios",
                    "523",
                    "아펠리오스",
                ),
            ),
            "트린다미어": (
                _champion(
                    "Tryndamere",
                    "23",
                    "트린다미어",
                ),
            ),
        },
        items_by_name={
            "정수 약탈자": (_item("3508", "정수 약탈자"),),
        },
    )


def _detail(
    title: str,
) -> PatchSection:
    return PatchSection(
        title=title,
        level=4,
        kind=SectionKind.HOTFIX,
        heading_path=["추가 패치 노트", title],
        changes=[PatchChange(raw_text="테스트 값: 1 ⇒ 2")],
    )


def _anchor(
    title: str,
) -> PatchSection:
    return PatchSection(
        title=title,
        level=3,
        kind=SectionKind.HOTFIX,
        heading_path=["추가 패치 노트", title],
    )


def _document(
    children: list[PatchSection],
) -> PatchNoteDocument:
    return PatchNoteDocument(
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        fetched_at=datetime(2026, 8, 26, tzinfo=UTC),
        source_sha256=SOURCE_SHA,
        source_size_bytes=100,
        collector_version="0.1.0",
        title="26.1 패치 노트",
        sections=[
            PatchSection(
                title="추가 패치 노트",
                level=2,
                kind=SectionKind.HOTFIX,
                heading_path=["추가 패치 노트"],
                children=children,
            )
        ],
    )


def test_builds_hint_for_conservative_champion_detail() -> None:
    document = _document(
        [
            _anchor("아펠리오스"),
            _detail("기본 지속 효과 - 암살자와 예언자"),
        ]
    )
    records = normalize_patch_note_document(document)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        _catalog(),
    )

    assert len(hints) == 1
    assert hints[0].champion_id == "Aphelios"
    assert hints[0].champion_key == "523"
    assert hints[0].context_method == ContextMethod.HOTFIX_STRUCTURAL_EXACT
    assert hints[0].context_confidence == ContextConfidence.HIGH


def test_next_champion_anchor_changes_context() -> None:
    document = _document(
        [
            _anchor("아펠리오스"),
            _detail("Q - 첫 번째 스킬"),
            _anchor("트린다미어"),
            _detail("E - 회전 베기"),
        ]
    )
    records = normalize_patch_note_document(document)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        _catalog(),
    )

    assert [hint.champion_name for hint in hints] == [
        "아펠리오스",
        "트린다미어",
    ]


def test_item_title_clears_champion_context() -> None:
    document = _document(
        [
            _anchor("트린다미어"),
            _detail("E - 회전 베기"),
            _detail("정수 약탈자"),
            _detail("Q - 가짜 후속 스킬"),
        ]
    )
    records = normalize_patch_note_document(document)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        _catalog(),
    )

    assert [hint.detail_title for hint in hints] == ["E - 회전 베기"]


def test_generic_h4_clears_champion_context() -> None:
    document = _document(
        [
            _anchor("트린다미어"),
            _detail("W - 조롱의 외침"),
            _detail("격전 일정 업데이트"),
            _detail("Q - 가짜 후속 스킬"),
        ]
    )
    records = normalize_patch_note_document(document)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        _catalog(),
    )

    assert [hint.detail_title for hint in hints] == ["W - 조롱의 외침"]


def test_repeated_detail_titles_map_to_distinct_records() -> None:
    document = _document(
        [
            _anchor("아펠리오스"),
            _detail("기본 능력치"),
            _anchor("트린다미어"),
            _detail("기본 능력치"),
        ]
    )
    records = normalize_patch_note_document(document)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        _catalog(),
    )

    assert len(hints) == 2
    assert hints[0].record_order != hints[1].record_order
    assert hints[0].record_id != hints[1].record_id
    assert hints[0].detail_title == "기본 능력치"
    assert hints[1].detail_title == "기본 능력치"


def test_rejects_normalized_source_mismatch() -> None:
    document = _document(
        [
            _anchor("아펠리오스"),
            _detail("Q - 첫 번째 스킬"),
        ]
    )
    records = normalize_patch_note_document(document)

    records[0] = records[0].model_copy(update={"source_sha256": "0" * 64})

    with pytest.raises(
        ValueError,
        match="source SHA-256",
    ):
        build_hotfix_champion_context_hints(
            document,
            records,
            _catalog(),
        )
