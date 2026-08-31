import sys
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

TARGET_HEADINGS = (
    "아펠리오스",
    "E - 회전 베기",
    "정수 약탈자",
    "격전 일정 업데이트",
    "패치 하이라이트",
    "포지션 퀘스트",
    "챔피언",
)

MAX_ANCESTOR_DEPTH = 10
MAX_HEADING_PREVIEW = 8
MAX_CLASS_LENGTH = 140


def _configure_utf8_output() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _clean_text(node: LexborNode) -> str:
    return " ".join(node.text(separator=" ", strip=True).split())


def _truncate(value: str | None, limit: int = MAX_CLASS_LENGTH) -> str:
    if not value:
        return "-"
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _describe(node: LexborNode) -> str:
    attributes = node.attributes
    return (
        f"<{node.tag}> "
        f"id={_truncate(attributes.get('id'))!r} "
        f"class={_truncate(attributes.get('class'))!r}"
    )


def _heading_preview(node: LexborNode) -> str:
    headings = node.css("h2, h3, h4")
    preview = [
        f"{heading.tag.upper()}:{_clean_text(heading)}"
        for heading in headings[:MAX_HEADING_PREVIEW]
        if _clean_text(heading)
    ]

    suffix = ""
    if len(headings) > MAX_HEADING_PREVIEW:
        suffix = f" ... (+{len(headings) - MAX_HEADING_PREVIEW})"

    return " | ".join(preview) + suffix


def _print_match(node: LexborNode, match_number: int) -> None:
    print(f"  MATCH #{match_number}: {_describe(node)}")
    print(f"  text={_clean_text(node)!r}")
    print("  ancestor chain:")

    current: LexborNode | None = node
    depth = 0

    while current is not None and depth <= MAX_ANCESTOR_DEPTH:
        heading_count = len(current.css("h2, h3, h4"))
        preview = _heading_preview(current)

        print(f"    {depth:02d} {_describe(current)} heading_count={heading_count}")

        if preview:
            print(f"       headings: {preview}")

        if current.tag == "main":
            break

        current = current.parent
        depth += 1

    print()


def main() -> None:
    _configure_utf8_output()

    repository_root = Path(__file__).resolve().parents[2]
    raw_html_path = (
        repository_root / "data" / "raw" / "patch_notes" / "26.1" / "ko_kr" / "page.html"
    )

    if not raw_html_path.is_file():
        raise FileNotFoundError(f"Raw patch note HTML not found: {raw_html_path}")

    tree = LexborHTMLParser(raw_html_path.read_bytes())
    headings = tree.css("h1, h2, h3, h4")

    print(f"Raw HTML: {raw_html_path}")
    print(f"Total headings: {len(headings)}")
    print()

    for target in TARGET_HEADINGS:
        matches = [node for node in headings if _clean_text(node) == target]

        print("=" * 88)
        print(f"TARGET: {target!r}")
        print(f"matches: {len(matches)}")
        print()

        for index, node in enumerate(matches, start=1):
            _print_match(node, index)


if __name__ == "__main__":
    main()
