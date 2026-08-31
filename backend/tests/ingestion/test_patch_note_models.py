from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchChange,
    PatchNoteDocument,
    PatchSection,
    SectionKind,
)


def test_patch_note_document_preserves_heading_hierarchy() -> None:
    change = PatchChange(
        raw_text="E 스킬 레벨당 물리 관통력: 5.5 ⇒ 4.5",
        subject="E 스킬 레벨당 물리 관통력",
        before="5.5",
        after="4.5",
    )

    ability_section = PatchSection(
        title="기본 지속 효과 - 암살자와 예언자",
        level=4,
        kind=SectionKind.CHAMPION,
        heading_path=[
            "추가 패치 노트",
            "아펠리오스",
            "기본 지속 효과 - 암살자와 예언자",
        ],
        entity_name="아펠리오스",
        changes=[change],
    )

    champion_section = PatchSection(
        title="아펠리오스",
        level=3,
        kind=SectionKind.CHAMPION,
        heading_path=["추가 패치 노트", "아펠리오스"],
        entity_name="아펠리오스",
        children=[ability_section],
    )

    hotfix_section = PatchSection(
        title="추가 패치 노트",
        level=2,
        kind=SectionKind.HOTFIX,
        heading_path=["추가 패치 노트"],
        children=[champion_section],
    )

    document = PatchNoteDocument(
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        fetched_at=datetime(2026, 8, 26, 2, 0, tzinfo=UTC),
        title="26.1 패치 노트",
        authors=["Riot Games"],
        sections=[hotfix_section],
    )

    parsed_change = document.sections[0].children[0].children[0].changes[0]

    assert parsed_change.subject == "E 스킬 레벨당 물리 관통력"
    assert parsed_change.before == "5.5"
    assert parsed_change.after == "4.5"


def test_patch_change_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PatchChange.model_validate(
            {
                "raw_text": "공격력이 증가합니다.",
                "unexpected_field": "should fail",
            }
        )


def test_patch_section_rejects_unsupported_heading_level() -> None:
    with pytest.raises(ValidationError):
        PatchSection.model_validate(
            {
                "title": "잘못된 heading",
                "level": 5,
            }
        )


def test_patch_change_requires_non_empty_raw_text() -> None:
    with pytest.raises(ValidationError):
        PatchChange.model_validate(
            {
                "raw_text": "",
            }
        )
