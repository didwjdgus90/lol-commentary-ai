import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DDRAGON_VERSION = "16.1.1"
DDRAGON_LOCALE = "ko_KR"

TARGET_METHODS = {
    "ambiguous",
    "map_incompatible",
}

_TAG_RE = re.compile(r"<[^>]+>")
_NUMBER_RE = re.compile(
    r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?%?(?:x)?",
    re.IGNORECASE,
)


def _configure_utf8_output() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)

    if callable(reconfigure):
        reconfigure(
            encoding="utf-8",
            errors="replace",
        )


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            payload = json.loads(line)

            if isinstance(payload, dict):
                rows.append(payload)

    return rows


def _maps_true(
    raw_item: dict[str, Any],
) -> list[str]:
    raw_maps = raw_item.get("maps")

    if not isinstance(raw_maps, dict):
        return []

    return sorted(str(map_id) for map_id, enabled in raw_maps.items() if enabled is True)


def _strip_markup(value: object) -> str:
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    return " ".join(text.split())


def _normalize_number_token(token: str) -> str:
    return token.replace(",", "").casefold()


def _number_tokens(value: str) -> list[str]:
    return [_normalize_number_token(match.group(0)) for match in _NUMBER_RE.finditer(value)]


def _after_segments(
    changes: list[str],
) -> list[str]:
    segments: list[str] = []

    for change in changes:
        if "⇒" in change:
            segments.append(change.rsplit("⇒", maxsplit=1)[1].strip())
        elif "→" in change:
            segments.append(change.rsplit("→", maxsplit=1)[1].strip())

    return segments


def _after_number_tokens(
    changes: list[str],
) -> list[str]:
    tokens: list[str] = []

    for segment in _after_segments(changes):
        tokens.extend(_number_tokens(segment))

    return tokens


def _candidate_search_text(
    raw_item: dict[str, Any],
) -> str:
    parts: list[str] = []

    for key in (
        "description",
        "plaintext",
    ):
        value = raw_item.get(key)

        if value is not None:
            parts.append(_strip_markup(value))

    for key in (
        "stats",
        "gold",
        "effect",
    ):
        value = raw_item.get(key)

        if value is not None:
            parts.append(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    return " ".join(parts)


def _matched_tokens(
    expected_tokens: list[str],
    candidate_text: str,
) -> tuple[list[str], list[str]]:
    candidate_tokens = set(_number_tokens(candidate_text))

    matched: list[str] = []
    unmatched: list[str] = []

    for token in expected_tokens:
        if token in candidate_tokens:
            matched.append(token)
        else:
            unmatched.append(token)

    return matched, unmatched


def _preview(
    value: object,
    limit: int = 260,
) -> str:
    text = _strip_markup(value)

    if len(text) <= limit:
        return text

    return f"{text[:limit]}..."


def main() -> None:
    _configure_utf8_output()

    repository_root = Path(__file__).resolve().parents[2]

    item_path = (
        repository_root
        / "data"
        / "raw"
        / "data_dragon"
        / DDRAGON_VERSION
        / DDRAGON_LOCALE
        / "item.json"
    )

    resolved_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "resolved.jsonl"
    )

    if not item_path.is_file():
        raise FileNotFoundError(f"Data Dragon item.json not found: {item_path}")

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Resolved JSONL not found: {resolved_path}")

    item_payload = json.loads(item_path.read_text(encoding="utf-8"))
    item_data = item_payload.get("data")

    if not isinstance(item_data, dict):
        raise ValueError("Data Dragon item.json data must be an object")

    rows = _load_jsonl(resolved_path)

    target_rows = [row for row in rows if row.get("resolution_method") in TARGET_METHODS]

    items_by_name: dict[
        str,
        list[tuple[str, dict[str, Any]]],
    ] = defaultdict(list)

    for item_id, raw_item in item_data.items():
        if not isinstance(item_id, str):
            continue

        if not isinstance(raw_item, dict):
            continue

        name = raw_item.get("name")

        if isinstance(name, str):
            items_by_name[name].append((item_id, raw_item))

    print("=== PATCH EVIDENCE ↔ ITEM VARIANT PROBE ===")
    print(f"Data Dragon version: {item_payload.get('version')}")
    print(f"Resolved records: {len(rows)}")
    print(f"Target records (ambiguous + map_incompatible): {len(target_rows)}")
    print()

    for row in target_rows:
        title = str(row.get("title", ""))
        changes = [str(change) for change in row.get("changes", []) if isinstance(change, str)]

        target_map_id = row.get("target_map_id")
        candidates = items_by_name.get(title, [])

        if target_map_id is None:
            effective_candidates = candidates
        else:
            effective_candidates = [
                (item_id, raw_item)
                for item_id, raw_item in candidates
                if str(target_map_id) in _maps_true(raw_item)
            ]

        after_segments = _after_segments(changes)
        expected_tokens = _after_number_tokens(changes)

        print("=" * 100)
        print(f"ORDER: {row.get('order')} | TITLE: {title!r}")
        print(
            f"method={row.get('resolution_method')!r} "
            f"entity_type={row.get('entity_type')!r} "
            f"target_map_id={target_map_id!r}"
        )
        print(f"path={row.get('heading_path')!r}")
        print()

        print("Patch changes:")
        if changes:
            for change in changes:
                print(f"  - {change}")
        else:
            print("  (none)")

        print(f"After segments: {after_segments!r}")
        print(f"After numeric tokens: {expected_tokens!r}")
        print()

        print(
            f"Candidate counts: all={len(candidates)}, after_map_filter={len(effective_candidates)}"
        )

        if not effective_candidates:
            print("  No candidate survived the current map filter.")
            print()
            continue

        for item_id, raw_item in effective_candidates:
            candidate_text = _candidate_search_text(raw_item)
            matched, unmatched = _matched_tokens(
                expected_tokens,
                candidate_text,
            )

            denominator = len(expected_tokens)

            score = len(matched) / denominator if denominator else 0.0

            print()
            print(f"  ID {item_id}")
            print(f"    maps_true: {_maps_true(raw_item)}")
            print(f"    inStore: {raw_item.get('inStore')!r}")
            print(f"    gold: {raw_item.get('gold')!r}")
            print(f"    stats: {raw_item.get('stats')!r}")
            print(f"    after-token evidence: {len(matched)}/{denominator} ({score:.2f})")
            print(f"    matched: {matched!r}")
            print(f"    unmatched: {unmatched!r}")
            print(f"    plaintext: {_preview(raw_item.get('plaintext', ''))}")
            print(f"    description: {_preview(raw_item.get('description', ''))}")

        print()

    print("=" * 100)
    print("NOTE: token overlap is diagnostic evidence only.")
    print(
        "Do not automatically resolve an item solely because its numeric overlap score is higher."
    )


if __name__ == "__main__":
    main()
