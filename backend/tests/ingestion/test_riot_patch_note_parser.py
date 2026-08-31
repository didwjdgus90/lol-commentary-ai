from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.patch_note_parser.models import (
    SectionKind,
)
from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_riot_patch_sections,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "riot_patch_26_1_structure.html"


def test_riot_parser_uses_patch_notes_container_sections() -> None:
    sections = parse_riot_patch_sections(FIXTURE_PATH.read_bytes())

    assert [section.title for section in sections] == [
        "추가 패치 노트",
        "챔피언",
        "업데이트된 아이템",
    ]

    assert sections[0].kind == SectionKind.HOTFIX
    assert sections[1].kind == SectionKind.CHAMPION
    assert sections[2].kind == SectionKind.ITEM


def test_riot_parser_nests_regular_champion_details_by_change_block() -> None:
    sections = parse_riot_patch_sections(FIXTURE_PATH.read_bytes())

    champions = sections[1]
    aphelios = champions.children[0]

    assert aphelios.title == "아펠리오스"
    assert aphelios.entity_name == "아펠리오스"
    assert [child.title for child in aphelios.children] == [
        "만월총",
        "R - 월광포화",
    ]

    assert aphelios.children[0].heading_path == [
        "챔피언",
        "아펠리오스",
        "만월총",
    ]
    assert aphelios.children[0].entity_name == "아펠리오스"


def test_riot_parser_does_not_guess_hotfix_h4_as_champion_children() -> None:
    sections = parse_riot_patch_sections(FIXTURE_PATH.read_bytes())

    hotfix = sections[0]

    assert [child.title for child in hotfix.children] == [
        "트린다미어",
        "E - 회전 베기",
        "정수 약탈자",
        "격전 일정 업데이트",
    ]

    tryndamere = hotfix.children[0]

    assert tryndamere.children == []
    assert hotfix.children[1].heading_path == [
        "추가 패치 노트",
        "E - 회전 베기",
    ]
    assert hotfix.children[2].heading_path == [
        "추가 패치 노트",
        "정수 약탈자",
    ]


def test_riot_parser_keeps_item_h4_under_item_section() -> None:
    sections = parse_riot_patch_sections(FIXTURE_PATH.read_bytes())

    updated_items = sections[2]
    essence_reaver = updated_items.children[0]

    assert essence_reaver.title == "정수 약탈자"
    assert essence_reaver.heading_path == [
        "업데이트된 아이템",
        "정수 약탈자",
    ]
    assert essence_reaver.changes[0].raw_text == ("가격: 3,100골드 ⇒ 3,000골드")


def test_riot_parser_requires_real_patch_container() -> None:
    with pytest.raises(ValueError, match="patch-notes-container"):
        parse_riot_patch_sections("<main><h2 id='patch-champions'>챔피언</h2></main>")
