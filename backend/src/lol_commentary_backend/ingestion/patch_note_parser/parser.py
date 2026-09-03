from typing import Literal

from selectolax.lexbor import (
    LexborHTMLParser,
    LexborNode,
)

from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchChange,
    PatchSection,
    SectionKind,
)

type HeadingLevel = Literal[
    2,
    3,
    4,
]


_HEADING_LEVELS: dict[
    str,
    HeadingLevel,
] = {
    "h2": 2,
    "h3": 3,
    "h4": 4,
}


_RIOT_ROOT_SELECTOR = "#patch-notes-container"

_PATCH_CHANGE_BLOCK_CLASS = "patch-change-block"

_HEADER_PRIMARY_CLASS = "header-primary"


def _clean_text(
    node: LexborNode,
) -> str:
    return " ".join(
        node.text(
            separator=" ",
            strip=True,
        ).split()
    )


def _node_classes(
    node: LexborNode,
) -> set[str]:
    return set((node.attributes.get("class") or "").split())


def _has_ancestor_class(
    node: LexborNode,
    class_name: str,
) -> bool:
    current = node.parent

    while current is not None:
        if class_name in _node_classes(current):
            return True

        if current.attributes.get("id") == "patch-notes-container":
            return False

        current = current.parent

    return False


def _is_riot_top_level_heading(
    node: LexborNode,
) -> bool:
    if node.tag != "h2":
        return False

    node_id = node.attributes.get("id") or ""

    if node_id.startswith("patch-"):
        return True

    parent = node.parent

    return parent is not None and _HEADER_PRIMARY_CLASS in _node_classes(parent)


def _normalized_title(
    title: str,
) -> str:
    return " ".join(title.casefold().split())


def _classify_top_level_section(
    title: str,
) -> SectionKind:
    normalized = _normalized_title(title)

    if title == "추가 패치 노트" or "mid-patch" in normalized or "mid patch" in normalized:
        return SectionKind.HOTFIX

    if title == "챔피언" or normalized in {
        "champion",
        "champions",
    }:
        return SectionKind.CHAMPION

    if "아이템" in title or normalized in {
        "item",
        "items",
        "item changes",
        "new items",
        "returning items",
        "updated items",
    }:
        return SectionKind.ITEM

    if (
        "버그" in title
        or "bugfix" in normalized
        or "bug fix" in normalized
        or "quality of life" in normalized
        or "qol" in normalized
    ):
        return SectionKind.BUG_FIX

    if title == "시스템" or normalized in {
        "system",
        "systems",
        "system changes",
    }:
        return SectionKind.SYSTEM

    if (
        "게임 모드" in title
        or "무작위 총력전" in title
        or "game mode" in normalized
        or normalized == "arena"
        or normalized.startswith("arena ")
        or normalized == "aram"
        or normalized.startswith("aram ")
    ):
        return SectionKind.GAME_MODE

    return SectionKind.GENERAL


def parse_patch_sections(
    html: str | bytes,
    *,
    root_selector: str = "article",
) -> list[PatchSection]:
    tree = LexborHTMLParser(html)

    root = tree.css_first(root_selector)

    if root is None:
        raise ValueError(f"Parser root not found: {root_selector}")

    sections: list[PatchSection] = []

    stack: list[PatchSection] = []

    for node in root.css("h2, h3, h4, p, li"):
        text = _clean_text(node)

        if not text:
            continue

        level = _HEADING_LEVELS.get(node.tag)

        if level is not None:
            while stack and stack[-1].level >= level:
                stack.pop()

            section = PatchSection(
                title=text,
                level=level,
                heading_path=[item.title for item in stack] + [text],
            )

            if stack:
                stack[-1].children.append(section)
            else:
                sections.append(section)

            stack.append(section)

            continue

        if not stack:
            continue

        if node.tag == "p":
            stack[-1].paragraphs.append(text)

        elif node.tag == "li":
            stack[-1].changes.append(PatchChange(raw_text=text))

    return sections


def parse_riot_patch_sections(
    html: str | bytes,
) -> list[PatchSection]:
    tree = LexborHTMLParser(html)

    root = tree.css_first(_RIOT_ROOT_SELECTOR)

    if root is None:
        raise ValueError(f"Parser root not found: {_RIOT_ROOT_SELECTOR}")

    sections: list[PatchSection] = []

    current_top: PatchSection | None = None

    current_h3: PatchSection | None = None

    current_detail: PatchSection | None = None

    current_h3_in_change_block = False

    for node in root.css("h2, h3, h4, p, li"):
        text = _clean_text(node)

        if not text:
            continue

        if _is_riot_top_level_heading(node):
            current_top = PatchSection(
                title=text,
                level=2,
                kind=(_classify_top_level_section(text)),
                heading_path=[text],
            )

            sections.append(current_top)

            current_h3 = None
            current_detail = None
            current_h3_in_change_block = False

            continue

        if current_top is None:
            continue

        if node.tag == "h3":
            current_h3_in_change_block = _has_ancestor_class(
                node,
                _PATCH_CHANGE_BLOCK_CLASS,
            )

            current_h3 = PatchSection(
                title=text,
                level=3,
                kind=(current_top.kind),
                heading_path=[
                    current_top.title,
                    text,
                ],
                entity_name=(
                    text
                    if (current_h3_in_change_block and current_top.kind == SectionKind.CHAMPION)
                    else None
                ),
            )

            current_top.children.append(current_h3)

            current_detail = None

            continue

        if node.tag == "h4":
            h4_in_change_block = _has_ancestor_class(
                node,
                _PATCH_CHANGE_BLOCK_CLASS,
            )

            if current_h3 is not None and current_h3_in_change_block and h4_in_change_block:
                parent = current_h3

            else:
                parent = current_top

            current_detail = PatchSection(
                title=text,
                level=4,
                kind=current_top.kind,
                heading_path=[
                    *parent.heading_path,
                    text,
                ],
                entity_name=(current_h3.entity_name if (parent is current_h3) else None),
            )

            parent.children.append(current_detail)

            continue

        target = current_detail or current_h3 or current_top

        if node.tag == "p":
            target.paragraphs.append(text)

        elif node.tag == "li":
            target.changes.append(PatchChange(raw_text=text))

    return sections
