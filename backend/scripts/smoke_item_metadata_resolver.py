from __future__ import annotations

from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.item_metadata_resolver import (
    DDragonItemMetadataResolver,
)

DDRAGON_VERSION = "16.17.1"

ITEM_EVENT_TYPES = {
    "ITEM_PURCHASED",
    "ITEM_DESTROYED",
    "ITEM_SOLD",
    "ITEM_UNDO",
}

IMPORTANT_ITEM_IDS = (
    2055,
    3340,
    3865,
    3171,
    3172,
    3174,
    3869,
    3871,
    3876,
    3877,
)


def _event_type(
    event: NormalizedGameEvent,
) -> str:
    value = event.raw_event_type

    resolved = getattr(
        value,
        "value",
        value,
    )

    if not isinstance(
        resolved,
        str,
    ):
        raise TypeError("raw_event_type must resolve to string")

    return resolved


def _load_events(
    path: Path,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    result: list[NormalizedGameEvent] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            result.append(NormalizedGameEvent.model_validate_json(line))

        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

    return tuple(result)


def _event_item_ids(
    event: NormalizedGameEvent,
) -> tuple[
    int,
    ...,
]:
    result: set[int] = set()

    for value in (
        event.item_id,
        event.before_item_id,
        event.after_item_id,
    ):
        if value is not None and value > 0:
            result.add(value)

    return tuple(sorted(result))


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(repository_root),
        ddragon_version=(DDRAGON_VERSION),
    )

    match_directories = tuple(
        sorted(
            (
                path
                for path in normalized_root.iterdir()
                if (path.is_dir() and (path / "events.jsonl").is_file())
            ),
            key=lambda path: path.name,
        )
    )

    if len(match_directories) != 3:
        raise RuntimeError("Expected current 3-match baseline")

    item_event_count = 0

    observed_item_counts: Counter[int] = Counter()

    for directory in match_directories:
        events = _load_events(directory / "events.jsonl")

        match_item_events = 0

        for event in events:
            if _event_type(event) not in ITEM_EVENT_TYPES:
                continue

            item_event_count += 1
            match_item_events += 1

            for item_id in _event_item_ids(event):
                observed_item_counts[item_id] += 1

        print(f"MATCH={directory.name}")

        print(f"  item_events={match_item_events}")

    print()

    observed_ids = tuple(sorted(observed_item_counts))

    unresolved = tuple(item_id for item_id in observed_ids if (resolver.resolve(item_id) is None))

    print("=== DDRAGON CATALOG ===")

    print(f"ddragon_version={resolver.catalog_info.ddragon_version}")

    print(f"catalog_items={resolver.item_count}")

    print(f"ko_source_path={resolver.catalog_info.ko_source_path}")

    print(f"en_source_path={resolver.catalog_info.en_source_path}")

    print(f"catalog_sha256={resolver.catalog_info.catalog_sha256}")

    print()

    print("=== RIOT ITEM COVERAGE ===")

    print(f"item_events={item_event_count}")

    print(f"unique_observed_item_ids={len(observed_ids)}")

    print(f"resolved_item_ids={len(observed_ids) - len(unresolved)}")

    print(f"unresolved_item_ids={len(unresolved)}")

    if unresolved:
        print("unresolved=" + ", ".join(str(item_id) for item_id in unresolved))

    print()

    print("=== IMPORTANT ITEM METADATA ===")

    for item_id in IMPORTANT_ITEM_IDS:
        metadata = resolver.resolve(item_id)

        print(f"ITEM={item_id}")

        if metadata is None:
            print("  RESOLUTION=NOT_FOUND")

            continue

        print(f"  ko={metadata.name_ko}")

        print(f"  en={metadata.name_en}")

        print(f"  gold_total={metadata.gold_total}")

        print(f"  purchasable={metadata.purchasable}")

        print(f"  depth={metadata.depth}")

        print(f"  from={metadata.from_item_ids}")

        print(f"  into={metadata.into_item_ids}")

        print(f"  tags={metadata.tags}")

        print(f"  maps={metadata.map_ids}")

    print()

    print("=== TOP OBSERVED ITEMS ===")

    for (
        item_id,
        count,
    ) in observed_item_counts.most_common(20):
        metadata = resolver.resolve(item_id)

        name = metadata.name_ko if metadata is not None else "UNRESOLVED"

        print(f"  {item_id} {name}: {count}")

    print()

    if item_event_count != 1210:
        raise RuntimeError("Expected current 1210 item events")

    if unresolved:
        raise RuntimeError(
            "Data Dragon resolver "
            "does not cover all item IDs "
            "observed in current Riot "
            "timeline dataset"
        )

    for item_id in IMPORTANT_ITEM_IDS:
        if resolver.resolve(item_id) is None:
            raise RuntimeError(f"Important item ID could not be resolved: {item_id}")

    print("DDRAGON_ITEM_METADATA_RESOLVER_SMOKE=PASS")


if __name__ == "__main__":
    main()
