import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AMBIGUOUS_METHOD = "ambiguous"
MAX_TEXT_LENGTH = 120


def _configure_utf8_output() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _truncate(value: object, limit: int = MAX_TEXT_LENGTH) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _true_map_ids(raw_item: dict[str, Any]) -> list[str]:
    maps = raw_item.get("maps")
    if not isinstance(maps, dict):
        return []

    return sorted(str(map_id) for map_id, enabled in maps.items() if enabled is True)


def _candidate_summary(
    item_id: str,
    raw_item: dict[str, Any],
) -> dict[str, object]:
    gold = raw_item.get("gold")
    gold_summary: dict[str, object] = {}

    if isinstance(gold, dict):
        gold_summary = {
            "base": gold.get("base"),
            "total": gold.get("total"),
            "purchasable": gold.get("purchasable"),
        }

    return {
        "id": item_id,
        "name": raw_item.get("name"),
        "maps_true": _true_map_ids(raw_item),
        "inStore": raw_item.get("inStore"),
        "hideFromAll": raw_item.get("hideFromAll"),
        "requiredChampion": raw_item.get("requiredChampion"),
        "requiredAlly": raw_item.get("requiredAlly"),
        "specialRecipe": raw_item.get("specialRecipe"),
        "tags": raw_item.get("tags"),
        "gold": gold_summary,
        "description": _truncate(raw_item.get("description", "")),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)

    return rows


def main() -> None:
    _configure_utf8_output()

    repository_root = Path(__file__).resolve().parents[2]

    item_path = repository_root / "data" / "raw" / "data_dragon" / "16.1.1" / "ko_KR" / "item.json"

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

    resolved_rows = _load_jsonl(resolved_path)
    ambiguous_rows = [
        row for row in resolved_rows if row.get("resolution_method") == AMBIGUOUS_METHOD
    ]

    items_by_name: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    for raw_item_id, raw_item in item_data.items():
        if not isinstance(raw_item_id, str):
            continue

        if not isinstance(raw_item, dict):
            continue

        name = raw_item.get("name")
        if isinstance(name, str):
            items_by_name[name].append((raw_item_id, raw_item))

    duplicate_item_names = {
        name: candidates for name, candidates in items_by_name.items() if len(candidates) > 1
    }

    print("=== DATA DRAGON ITEM VARIANT PROBE ===")
    print(f"Data Dragon version: {item_payload.get('version')}")
    print(f"Total item IDs: {len(item_data)}")
    print(f"Duplicate Korean item names: {len(duplicate_item_names)}")
    print(f"Resolved records: {len(resolved_rows)}")
    print(f"Ambiguous records: {len(ambiguous_rows)}")
    print()

    ambiguous_titles = Counter(str(row.get("title")) for row in ambiguous_rows)

    print("=== AMBIGUOUS TITLES ===")
    for title, count in ambiguous_titles.most_common():
        candidates = items_by_name.get(title, [])
        print(f"{title!r}: records={count}, item_candidates={len(candidates)}")
    print()

    for title in sorted(ambiguous_titles):
        print("=" * 96)
        print(f"TITLE: {title!r}")
        print()

        matching_rows = [row for row in ambiguous_rows if row.get("title") == title]

        print("Patch-note contexts:")
        for row in matching_rows:
            print(
                "  - "
                f"order={row.get('order')} "
                f"entity_type={row.get('entity_type')!r} "
                f"path={row.get('heading_path')!r}"
            )

        print()
        print("Data Dragon candidates:")

        candidates = items_by_name.get(title, [])

        if not candidates:
            print("  (no exact item-name candidate)")
            print()
            continue

        for item_id, raw_item in candidates:
            summary = _candidate_summary(
                item_id,
                raw_item,
            )

            print(f"  ID {summary['id']}")
            print(f"    name: {summary['name']!r}")
            print(f"    maps_true: {summary['maps_true']}")
            print(f"    inStore: {summary['inStore']!r}")
            print(f"    hideFromAll: {summary['hideFromAll']!r}")
            print(f"    requiredChampion: {summary['requiredChampion']!r}")
            print(f"    requiredAlly: {summary['requiredAlly']!r}")
            print(f"    specialRecipe: {summary['specialRecipe']!r}")
            print(f"    tags: {summary['tags']!r}")
            print(f"    gold: {summary['gold']!r}")
            print(f"    description: {summary['description']}")
        print()

    print("=" * 96)
    print("MAP-ID FREQUENCY AMONG AMBIGUOUS ITEM CANDIDATES")

    map_frequency: Counter[str] = Counter()

    for title in ambiguous_titles:
        for _, raw_item in items_by_name.get(title, []):
            map_frequency.update(_true_map_ids(raw_item))

    for map_id, count in map_frequency.most_common():
        print(f"map {map_id}: {count} candidate(s)")


if __name__ == "__main__":
    main()
