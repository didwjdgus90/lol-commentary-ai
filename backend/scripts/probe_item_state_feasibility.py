from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)

ITEM_EVENT_TYPES = (
    "ITEM_PURCHASED",
    "ITEM_DESTROYED",
    "ITEM_SOLD",
    "ITEM_UNDO",
)

EXPECTED_COUNTS = {
    "ITEM_PURCHASED": 633,
    "ITEM_DESTROYED": 531,
    "ITEM_SOLD": 24,
    "ITEM_UNDO": 22,
}

EXPECTED_TOTAL = 1210


def _compact_key(
    key: str,
) -> str:
    return "".join(character for character in key.casefold() if character.isalnum())


def _is_participant_key(
    key: str,
) -> bool:
    return "participant" in _compact_key(key)


def _is_item_key(
    key: str,
) -> bool:
    compact = _compact_key(key)

    return "item" in compact or compact in {
        "beforeid",
        "afterid",
    }


def _is_before_id_key(
    key: str,
) -> bool:
    compact = _compact_key(key)

    return compact == "beforeid" or ("before" in compact and compact.endswith("id"))


def _is_after_id_key(
    key: str,
) -> bool:
    compact = _compact_key(key)

    return compact == "afterid" or ("after" in compact and compact.endswith("id"))


def _positive_int_values(
    value: object,
) -> tuple[int, ...]:
    if isinstance(
        value,
        bool,
    ):
        return ()

    if isinstance(
        value,
        int,
    ):
        if value >= 0:
            return (value,)

        return ()

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        result: list[int] = []

        for item in value:
            if isinstance(
                item,
                bool,
            ):
                continue

            if (
                isinstance(
                    item,
                    int,
                )
                and item >= 0
            ):
                result.append(item)

        return tuple(result)

    return ()


def _participant_ids(
    payload: dict[
        str,
        Any,
    ],
) -> tuple[int, ...]:
    result: set[int] = set()

    for key, value in payload.items():
        if not _is_participant_key(key):
            continue

        for identifier in _positive_int_values(value):
            if identifier > 0:
                result.add(identifier)

    return tuple(sorted(result))


def _item_ids(
    payload: dict[
        str,
        Any,
    ],
) -> tuple[int, ...]:
    result: set[int] = set()

    for key, value in payload.items():
        if not _is_item_key(key):
            continue

        for identifier in _positive_int_values(value):
            result.add(identifier)

    return tuple(sorted(result))


def _before_ids(
    payload: dict[
        str,
        Any,
    ],
) -> tuple[int, ...]:
    result: set[int] = set()

    for key, value in payload.items():
        if not _is_before_id_key(key):
            continue

        result.update(_positive_int_values(value))

    return tuple(sorted(result))


def _after_ids(
    payload: dict[
        str,
        Any,
    ],
) -> tuple[int, ...]:
    result: set[int] = set()

    for key, value in payload.items():
        if not _is_after_id_key(key):
            continue

        result.update(_positive_int_values(value))

    return tuple(sorted(result))


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


def _safe_projection(
    payload: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    object,
]:
    result: dict[
        str,
        object,
    ] = {}

    for key, value in sorted(payload.items()):
        compact = _compact_key(key)

        include = _is_participant_key(key) or _is_item_key(key) or "gold" in compact

        if not include:
            continue

        if value is None or isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            result[key] = value

            continue

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            if all(
                isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                )
                or item is None
                for item in value
            ):
                result[key] = value

    return result


def _load_normalized_events(
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
            event = NormalizedGameEvent.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

        result.append(event)

    return tuple(result)


def _load_raw_item_events(
    path: Path,
) -> tuple[
    dict[str, Any],
    ...,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Timeline root must be a JSON object")

    info = payload.get("info")

    if not isinstance(
        info,
        dict,
    ):
        raise ValueError("Timeline does not contain info object")

    frames = info.get("frames")

    if not isinstance(
        frames,
        list,
    ):
        raise ValueError("Timeline info.frames must be a list")

    result: list[dict[str, Any]] = []

    for frame in frames:
        if not isinstance(
            frame,
            dict,
        ):
            continue

        events = frame.get("events")

        if not isinstance(
            events,
            list,
        ):
            continue

        for event in events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            event_type = event.get("type")

            if (
                isinstance(
                    event_type,
                    str,
                )
                and event_type in ITEM_EVENT_TYPES
            ):
                result.append(event)

    return tuple(result)


def _unmapped_fields(
    payload: dict[
        str,
        Any,
    ],
) -> tuple[str, ...]:
    value = payload.get("unmapped_fields")

    if value is None:
        return ()

    if not isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return ()

    return tuple(
        sorted(
            {
                item
                for item in value
                if isinstance(
                    item,
                    str,
                )
            }
        )
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    raw_root = repository_root / "data" / "raw" / "riot_api" / "matches"

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

    normalized_counts: Counter[str] = Counter()

    raw_counts: Counter[str] = Counter()

    normalized_key_coverage: dict[
        str,
        Counter[str],
    ] = {event_type: Counter() for event_type in ITEM_EVENT_TYPES}

    raw_key_coverage: dict[
        str,
        Counter[str],
    ] = {event_type: Counter() for event_type in ITEM_EVENT_TYPES}

    unmapped_coverage: dict[
        str,
        Counter[str],
    ] = {event_type: Counter() for event_type in ITEM_EVENT_TYPES}

    participant_coverage: Counter[str] = Counter()

    item_id_coverage: Counter[str] = Counter()

    undo_before_coverage = 0
    undo_after_coverage = 0
    undo_both_coverage = 0

    samples: dict[
        str,
        list[
            dict[
                str,
                object,
            ]
        ],
    ] = {event_type: [] for event_type in ITEM_EVENT_TYPES}

    print("=== ITEM STATE FEASIBILITY PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        normalized_events = _load_normalized_events(directory / "events.jsonl")

        raw_timeline_path = raw_root / directory.name / "timeline.json"

        if not raw_timeline_path.is_file():
            raise RuntimeError(f"Raw timeline not found: {raw_timeline_path}")

        raw_events = _load_raw_item_events(raw_timeline_path)

        match_normalized_counts: Counter[str] = Counter()

        match_raw_counts: Counter[str] = Counter()

        for event in normalized_events:
            event_type = _event_type(event)

            if event_type not in ITEM_EVENT_TYPES:
                continue

            payload: dict[
                str,
                Any,
            ] = event.model_dump(mode="python")

            normalized_counts[event_type] += 1

            match_normalized_counts[event_type] += 1

            for key, value in payload.items():
                if value is not None:
                    normalized_key_coverage[event_type][key] += 1

            for field_name in _unmapped_fields(payload):
                unmapped_coverage[event_type][field_name] += 1

            participant_ids = _participant_ids(payload)

            item_ids = _item_ids(payload)

            if participant_ids:
                participant_coverage[event_type] += 1

            if item_ids:
                item_id_coverage[event_type] += 1

            if event_type == "ITEM_UNDO":
                before_ids = _before_ids(payload)

                after_ids = _after_ids(payload)

                if before_ids:
                    undo_before_coverage += 1

                if after_ids:
                    undo_after_coverage += 1

                if before_ids and after_ids:
                    undo_both_coverage += 1

            if len(samples[event_type]) < 3:
                samples[event_type].append(_safe_projection(payload))

        for event in raw_events:
            event_type = event.get("type")

            if not isinstance(
                event_type,
                str,
            ):
                continue

            raw_counts[event_type] += 1

            match_raw_counts[event_type] += 1

            for key, value in event.items():
                if value is not None:
                    raw_key_coverage[event_type][key] += 1

        print(f"MATCH={directory.name}")

        for event_type in ITEM_EVENT_TYPES:
            print(
                f"  {event_type}: "
                f"normalized="
                f"{match_normalized_counts[event_type]} "
                f"raw="
                f"{match_raw_counts[event_type]}"
            )

        print()

    if sum(normalized_counts.values()) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Normalized item-event total changed from baseline: {sum(normalized_counts.values())}"
        )

    if sum(raw_counts.values()) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Raw item-event total changed from baseline: {sum(raw_counts.values())}"
        )

    for event_type, expected in EXPECTED_COUNTS.items():
        normalized_count = normalized_counts[event_type]

        raw_count = raw_counts[event_type]

        if normalized_count != expected:
            raise RuntimeError(f"{event_type} normalized count changed: {normalized_count}")

        if raw_count != expected:
            raise RuntimeError(f"{event_type} raw count changed: {raw_count}")

        if normalized_count != raw_count:
            raise RuntimeError(f"Raw/normalized count mismatch for {event_type}")

    print("=== EVENT COVERAGE ===")

    for event_type in ITEM_EVENT_TYPES:
        total = normalized_counts[event_type]

        print(f"{event_type}:")

        print(f"  total={total}")

        print(f"  participant_coverage={participant_coverage[event_type]}/{total}")

        print(f"  item_value_coverage={item_id_coverage[event_type]}/{total}")

    print()

    print("=== ITEM_UNDO COVERAGE ===")

    undo_total = normalized_counts["ITEM_UNDO"]

    print(f"before_id_coverage={undo_before_coverage}/{undo_total}")

    print(f"after_id_coverage={undo_after_coverage}/{undo_total}")

    print(f"before_and_after_coverage={undo_both_coverage}/{undo_total}")

    print()

    print("=== NORMALIZED POPULATED FIELDS ===")

    for event_type in ITEM_EVENT_TYPES:
        print(f"{event_type}:")

        for key, count in sorted(normalized_key_coverage[event_type].items()):
            print(f"  {key}: {count}")

    print()

    print("=== RAW FIELD COVERAGE ===")

    for event_type in ITEM_EVENT_TYPES:
        print(f"{event_type}:")

        for key, count in sorted(raw_key_coverage[event_type].items()):
            print(f"  {key}: {count}")

    print()

    print("=== NORMALIZED UNMAPPED FIELDS ===")

    for event_type in ITEM_EVENT_TYPES:
        print(f"{event_type}:")

        fields = unmapped_coverage[event_type]

        if not fields:
            print("  NONE")

            continue

        for key, count in sorted(fields.items()):
            print(f"  {key}: {count}")

    print()

    print("=== SAFE NORMALIZED SAMPLES ===")

    for event_type in ITEM_EVENT_TYPES:
        print(f"{event_type}:")

        for index, sample in enumerate(
            samples[event_type],
            start=1,
        ):
            print(f"  sample#{index}: {sample}")

    print()

    standard_ready = True

    for event_type in (
        "ITEM_PURCHASED",
        "ITEM_DESTROYED",
        "ITEM_SOLD",
    ):
        total = normalized_counts[event_type]

        if participant_coverage[event_type] != total:
            standard_ready = False

        if item_id_coverage[event_type] != total:
            standard_ready = False

    undo_ready = (
        participant_coverage["ITEM_UNDO"] == undo_total
        and undo_before_coverage == undo_total
        and undo_after_coverage == undo_total
    )

    inventory_ready = standard_ready and undo_ready

    print("=== FEASIBILITY DECISION ===")

    print("purchase/sell/destroy reconstruction ready: " + ("YES" if standard_ready else "NO"))

    print("undo reconstruction ready: " + ("YES" if undo_ready else "NO"))

    print("INVENTORY_RECONSTRUCTION_READY=" + ("YES" if inventory_ready else "NO"))

    print()

    if not inventory_ready:
        print("ACTION_REQUIRED=NORMALIZER_ENRICHMENT")

        print(
            "Normalized economy events "
            "do not yet preserve all "
            "information required for "
            "lossless inventory replay."
        )

    else:
        print("ACTION_REQUIRED=INVENTORY_STATE_BUILDER")

        print("Normalized economy events contain sufficient fields for inventory replay.")

    print()

    print("ITEM_STATE_FEASIBILITY_PROBE=PASS")


if __name__ == "__main__":
    main()
