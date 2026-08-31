from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_patch_sections,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "patch_26_1_excerpt.html"


def test_parse_patch_sections_preserves_heading_hierarchy() -> None:
    sections = parse_patch_sections(
        FIXTURE_PATH.read_bytes(),
        root_selector="#patch-notes",
    )

    assert [section.title for section in sections] == [
        "추가 패치 노트",
        "신규 아이템",
    ]

    hotfix = sections[0]
    aphelios = hotfix.children[0]
    passive = aphelios.children[0]

    assert aphelios.title == "아펠리오스"
    assert passive.title == "기본 지속 효과 - 암살자와 예언자"
    assert passive.heading_path == [
        "추가 패치 노트",
        "아펠리오스",
        "기본 지속 효과 - 암살자와 예언자",
    ]
    assert passive.changes[0].raw_text == ("E 스킬 레벨당 물리 관통력: 5.5 ⇒ 4.5")


def test_parse_patch_sections_attaches_text_to_current_section() -> None:
    sections = parse_patch_sections(
        FIXTURE_PATH.read_bytes(),
        root_selector="#patch-notes",
    )

    hotfix = sections[0]
    aphelios = hotfix.children[0]
    new_items = sections[1]
    test_item = new_items.children[0]

    assert hotfix.paragraphs == ["추가 밸런스 조정 내용입니다."]
    assert aphelios.paragraphs == ["아펠리오스의 관통력이 조정됩니다."]
    assert test_item.paragraphs == ["테스트를 위한 가상의 아이템 설명입니다."]
    assert test_item.changes[0].raw_text == "가격: 3,000골드"


def test_parse_patch_sections_ignores_content_before_first_h2() -> None:
    sections = parse_patch_sections(
        FIXTURE_PATH.read_bytes(),
        root_selector="#patch-notes",
    )

    all_paragraphs = [paragraph for section in sections for paragraph in section.paragraphs]

    assert "이 문단은 첫 H2 이전이라 section parser에서는 제외됩니다." not in all_paragraphs


def test_parse_patch_sections_requires_root() -> None:
    with pytest.raises(ValueError, match="Parser root not found"):
        parse_patch_sections(
            "<main><h2>패치 노트</h2></main>",
            root_selector="#patch-notes",
        )
