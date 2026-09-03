from lol_commentary_backend.ingestion.patch_note_parser.models import (
    SectionKind,
)
from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_riot_patch_sections,
)


def _html(
    heading: str,
    *,
    child: str = "Aatrox",
) -> str:
    return f"""
    <html>
      <body>
        <div id="patch-notes-container">
          <div class="header-primary">
            <h2>{heading}</h2>
          </div>

          <div class="patch-change-block">
            <h3>{child}</h3>
            <p>Example change.</p>
          </div>
        </div>
      </body>
    </html>
    """


def test_classifies_english_champions() -> None:
    sections = parse_riot_patch_sections(
        _html(
            "Champions",
            child="Aatrox",
        )
    )

    assert len(sections) == 1

    assert sections[0].kind == SectionKind.CHAMPION

    assert sections[0].children[0].entity_name == "Aatrox"


def test_classifies_english_items() -> None:
    sections = parse_riot_patch_sections(
        _html(
            "Items",
            child="Essence Reaver",
        )
    )

    assert sections[0].kind == SectionKind.ITEM


def test_classifies_mid_patch_updates() -> None:
    sections = parse_riot_patch_sections(_html("Mid-Patch Updates"))

    assert sections[0].kind == SectionKind.HOTFIX


def test_classifies_bugfixes() -> None:
    sections = parse_riot_patch_sections(_html("Bugfixes & Quality of Life Changes"))

    assert sections[0].kind == SectionKind.BUG_FIX


def test_classifies_systems() -> None:
    sections = parse_riot_patch_sections(_html("Systems"))

    assert sections[0].kind == SectionKind.SYSTEM


def test_classifies_arena_as_game_mode() -> None:
    sections = parse_riot_patch_sections(_html("Arena"))

    assert sections[0].kind == SectionKind.GAME_MODE
