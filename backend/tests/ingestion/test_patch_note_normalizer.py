from datetime import UTC, datetime
from hashlib import sha256

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


def _document(
    sections: list[PatchSection],
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
        sections=sections,
    )


def test_normalizer_preserves_champion_entity_context() -> None:
    ability = PatchSection(
        title="만월총",
        level=4,
        kind=SectionKind.CHAMPION,
        heading_path=["챔피언", "아펠리오스", "만월총"],
        entity_name="아펠리오스",
        changes=[PatchChange(raw_text="표식 피해량: 20 ⇒ 18")],
    )

    champion = PatchSection(
        title="아펠리오스",
        level=3,
        kind=SectionKind.CHAMPION,
        heading_path=["챔피언", "아펠리오스"],
        entity_name="아펠리오스",
        paragraphs=["아펠리오스가 조정됩니다."],
        children=[ability],
    )

    records = normalize_patch_note_document(
        _document(
            [
                PatchSection(
                    title="챔피언",
                    level=2,
                    kind=SectionKind.CHAMPION,
                    heading_path=["챔피언"],
                    children=[champion],
                )
            ]
        )
    )

    assert len(records) == 2

    ability_record = records[1]
    assert ability_record.entity_type == PatchEntityType.CHAMPION
    assert ability_record.entity_name == "아펠리오스"
    assert ability_record.heading_path == [
        "챔피언",
        "아펠리오스",
        "만월총",
    ]
    assert "표식 피해량: 20 ⇒ 18" in ability_record.content


def test_normalizer_uses_item_title_as_entity_name() -> None:
    item = PatchSection(
        title="정수 약탈자",
        level=4,
        kind=SectionKind.ITEM,
        heading_path=["업데이트된 아이템", "정수 약탈자"],
        changes=[PatchChange(raw_text="가격: 3,100골드 ⇒ 3,000골드")],
    )

    records = normalize_patch_note_document(
        _document(
            [
                PatchSection(
                    title="업데이트된 아이템",
                    level=2,
                    kind=SectionKind.ITEM,
                    heading_path=["업데이트된 아이템"],
                    children=[item],
                )
            ]
        )
    )

    assert len(records) == 1
    assert records[0].entity_type == PatchEntityType.ITEM
    assert records[0].entity_name == "정수 약탈자"


def test_normalizer_does_not_guess_hotfix_entity_type() -> None:
    hotfix_detail = PatchSection(
        title="정수 약탈자",
        level=4,
        kind=SectionKind.HOTFIX,
        heading_path=["추가 패치 노트", "정수 약탈자"],
        changes=[PatchChange(raw_text="공격력: 60 ⇒ 55")],
    )

    records = normalize_patch_note_document(
        _document(
            [
                PatchSection(
                    title="추가 패치 노트",
                    level=2,
                    kind=SectionKind.HOTFIX,
                    heading_path=["추가 패치 노트"],
                    children=[hotfix_detail],
                )
            ]
        )
    )

    assert records[0].entity_type == PatchEntityType.UNKNOWN
    assert records[0].entity_name is None


def test_normalizer_assigns_deterministic_order_and_record_ids() -> None:
    first = PatchSection(
        title="첫 번째",
        level=2,
        kind=SectionKind.GENERAL,
        heading_path=["첫 번째"],
        paragraphs=["첫 내용"],
    )
    second = PatchSection(
        title="두 번째",
        level=2,
        kind=SectionKind.GENERAL,
        heading_path=["두 번째"],
        paragraphs=["두 번째 내용"],
    )

    document = _document([first, second])

    records_a = normalize_patch_note_document(document)
    records_b = normalize_patch_note_document(document)

    assert [record.order for record in records_a] == [0, 1]
    assert [record.record_id for record in records_a] == [record.record_id for record in records_b]


def test_normalizer_requires_source_sha256() -> None:
    document = _document([])
    document = document.model_copy(update={"source_sha256": None})

    try:
        normalize_patch_note_document(document)
    except ValueError as exc:
        assert "source_sha256" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
