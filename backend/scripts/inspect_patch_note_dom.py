from pathlib import Path

from selectolax.lexbor import LexborHTMLParser, LexborNode

MAX_HEADINGS = 120
MAX_ATTRIBUTE_LENGTH = 120

ROOT_CANDIDATES = (
    "article",
    "main",
    "[role='main']",
    "[class*='article']",
    "[class*='content']",
)


def _clean_text(node: LexborNode) -> str:
    return " ".join(node.text(separator=" ", strip=True).split())


def _truncate(value: str | None) -> str:
    if value is None:
        return "-"
    if len(value) <= MAX_ATTRIBUTE_LENGTH:
        return value
    return f"{value[:MAX_ATTRIBUTE_LENGTH]}..."


def _describe_node(node: LexborNode) -> str:
    attributes = node.attributes
    node_id = attributes.get("id")
    classes = attributes.get("class")

    return f"<{node.tag}> id={_truncate(node_id)!r} class={_truncate(classes)!r}"


def _print_document_summary(tree: LexborHTMLParser) -> None:
    title = tree.css_first("title")
    h1_nodes = tree.css("h1")

    print("=== DOCUMENT SUMMARY ===")
    print(f"title: {_clean_text(title) if title is not None else '-'}")
    print(f"h1 count: {len(h1_nodes)}")
    print(f"h2 count: {len(tree.css('h2'))}")
    print(f"h3 count: {len(tree.css('h3'))}")
    print(f"h4 count: {len(tree.css('h4'))}")
    print()


def _print_root_candidates(tree: LexborHTMLParser) -> None:
    print("=== ROOT CANDIDATES ===")

    for selector in ROOT_CANDIDATES:
        nodes = tree.css(selector)
        print(f"{selector!r}: {len(nodes)}")

        for index, node in enumerate(nodes[:5], start=1):
            heading_count = len(node.css("h1, h2, h3, h4"))
            text_preview = _clean_text(node)
            if len(text_preview) > 100:
                text_preview = f"{text_preview[:100]}..."

            print(
                f"  [{index}] {_describe_node(node)} headings={heading_count} text={text_preview!r}"
            )

    print()


def _print_heading_ancestor_chain(tree: LexborHTMLParser) -> None:
    heading = tree.css_first("h1, h2")

    print("=== FIRST HEADING ANCESTOR CHAIN ===")

    if heading is None:
        print("No h1/h2 found.")
        print()
        return

    current: LexborNode | None = heading
    depth = 0

    while current is not None and depth < 12:
        print(f"{depth:02d}: {_describe_node(current)}")
        current = current.parent
        depth += 1

    print()


def _print_headings(tree: LexborHTMLParser) -> None:
    print("=== HEADING ORDER ===")

    headings = tree.css("h1, h2, h3, h4")

    for index, node in enumerate(headings[:MAX_HEADINGS], start=1):
        print(f"{index:03d} {node.tag.upper():>2} {_clean_text(node)}")

    if len(headings) > MAX_HEADINGS:
        print(f"... truncated: showing {MAX_HEADINGS} of {len(headings)} headings")

    print()


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    raw_html_path = (
        repository_root / "data" / "raw" / "patch_notes" / "26.1" / "ko_kr" / "page.html"
    )

    if not raw_html_path.is_file():
        raise FileNotFoundError(f"Raw patch note HTML not found: {raw_html_path}")

    html = raw_html_path.read_bytes()
    tree = LexborHTMLParser(html)

    print(f"Raw HTML: {raw_html_path}")
    print(f"Size: {len(html):,} bytes")
    print()

    _print_document_summary(tree)
    _print_root_candidates(tree)
    _print_heading_ancestor_chain(tree)
    _print_headings(tree)


if __name__ == "__main__":
    main()
