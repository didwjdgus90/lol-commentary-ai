from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchSection,
)
from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_riot_patch_sections,
)


def _count_tree(sections: list[PatchSection]) -> int:
    count = 0
    stack = list(sections)

    while stack:
        section = stack.pop()
        count += 1
        stack.extend(section.children)

    return count


def _find_titles(
    sections: list[PatchSection],
    title: str,
) -> list[PatchSection]:
    matches: list[PatchSection] = []
    stack = list(sections)

    while stack:
        section = stack.pop()

        if section.title == title:
            matches.append(section)

        stack.extend(section.children)

    return matches


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    raw_html_path = (
        repository_root / "data" / "raw" / "patch_notes" / "26.1" / "ko_kr" / "page.html"
    )

    sections = parse_riot_patch_sections(raw_html_path.read_bytes())

    print(f"Top-level sections: {len(sections)}")
    print(f"Total parsed sections: {_count_tree(sections)}")
    print()

    print("Top-level titles:")
    for index, section in enumerate(sections, start=1):
        print(
            f"{index:02d}. {section.title} [{section.kind.value}] children={len(section.children)}"
        )

    print()

    for title in ("아펠리오스", "E - 회전 베기", "정수 약탈자"):
        matches = _find_titles(sections, title)

        print(f"{title!r}: {len(matches)} match(es)")
        for section in matches[:5]:
            print(f"  path={section.heading_path}")


if __name__ == "__main__":
    main()
