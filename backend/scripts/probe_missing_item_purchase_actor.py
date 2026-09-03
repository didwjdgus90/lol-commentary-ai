from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)

TARGET_EVENT_TYPE = "ITEM_PURCHASED"

EXPECTED_PURCHASE_COUNT = 633
EXPECTED_MISSING_ACTOR_COUNT = 6


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


def _normalized_purchases(
    events: tuple[
        NormalizedGameEvent,
        ...,
    ],
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    return tuple(event for event in events if (event.raw_event_type == TARGET_EVENT_TYPE))


def _load_raw_purchases(
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
        raise ValueError("Timeline info object is missing")

    frames = info.get("frames")

    if not isinstance(
        frames,
        list,
    ):
        raise ValueError("Timeline info.frames must be a list")

    result: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(frames):
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

        for event_index, event in enumerate(events):
            if not isinstance(
                event,
                dict,
            ):
                continue

            event_type = event.get("type")

            if event_type != TARGET_EVENT_TYPE:
                continue

            enriched = dict(event)

            enriched["_frame_index"] = frame_index

            enriched["_event_index"] = event_index

            result.append(enriched)

    return tuple(result)


def _raw_coordinate(
    event: dict[
        str,
        Any,
    ],
) -> tuple[int, int]:
    frame_index = event.get("_frame_index")

    event_index = event.get("_event_index")

    if not isinstance(
        frame_index,
        int,
    ):
        raise ValueError("Raw frame index must be int")

    if not isinstance(
        event_index,
        int,
    ):
        raise ValueError("Raw event index must be int")

    return (
        frame_index,
        event_index,
    )


def _raw_participant_id(
    event: dict[
        str,
        Any,
    ],
) -> int:
    value = event.get("participantId")

    if not isinstance(
        value,
        int,
    ):
        raise ValueError("Raw participantId must be int")

    return value


def _raw_item_id(
    event: dict[
        str,
        Any,
    ],
) -> int:
    value = event.get("itemId")

    if not isinstance(
        value,
        int,
    ):
        raise ValueError("Raw itemId must be int")

    return value


def _raw_timestamp(
    event: dict[
        str,
        Any,
    ],
) -> int:
    value = event.get("timestamp")

    if not isinstance(
        value,
        int,
    ):
        raise ValueError("Raw timestamp must be int")

    return value


def _validate_aligned_event(
    *,
    normalized: NormalizedGameEvent,
    raw: dict[
        str,
        Any,
    ],
) -> None:
    if normalized.item_id is None:
        raise RuntimeError("Normalized purchase missing item_id")

    raw_item_id = _raw_item_id(raw)

    raw_timestamp = _raw_timestamp(raw)

    if normalized.item_id != raw_item_id:
        raise RuntimeError(
            "Aligned raw/normalized "
            "item mismatch: "
            f"frame={normalized.frame_index} "
            f"event={normalized.event_index} "
            f"normalized={normalized.item_id} "
            f"raw={raw_item_id}"
        )

    if normalized.timestamp_ms != raw_timestamp:
        raise RuntimeError(
            "Aligned raw/normalized "
            "timestamp mismatch: "
            f"frame={normalized.frame_index} "
            f"event={normalized.event_index} "
            f"normalized={normalized.timestamp_ms} "
            f"raw={raw_timestamp}"
        )


def _safe_raw_projection(
    event: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    int | str | None,
]:
    result: dict[
        str,
        int | str | None,
    ] = {}

    for key in (
        "type",
        "timestamp",
        "participantId",
        "itemId",
        "_frame_index",
        "_event_index",
    ):
        value = event.get(key)

        if value is None or isinstance(
            value,
            (
                int,
                str,
            ),
        ):
            result[key] = value

    return result


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

    total_raw = 0
    total_normalized = 0
    total_missing_actor = 0

    raw_participant_counts: Counter[int] = Counter()

    missing_participant_counts: Counter[int] = Counter()

    missing_item_counts: Counter[int] = Counter()

    missing_timestamp_counts: Counter[int] = Counter()

    missing_cases: list[
        tuple[
            str,
            dict[
                str,
                int | str | None,
            ],
        ]
    ] = []

    print("=== MISSING ITEM PURCHASE ACTOR ROOT-CAUSE PROBE V2 ===")

    print()

    for directory in match_directories:
        normalized_events = _load_normalized_events(directory / "events.jsonl")

        normalized_purchases = _normalized_purchases(normalized_events)

        raw_path = raw_root / directory.name / "timeline.json"

        if not raw_path.is_file():
            raise RuntimeError(f"Raw timeline missing: {raw_path}")

        raw_purchases = _load_raw_purchases(raw_path)

        if len(raw_purchases) != len(normalized_purchases):
            raise RuntimeError(
                "Raw/normalized purchase "
                "count mismatch for "
                f"{directory.name}: "
                f"raw={len(raw_purchases)} "
                "normalized="
                f"{len(normalized_purchases)}"
            )

        raw_by_coordinate: dict[
            tuple[int, int],
            dict[str, Any],
        ] = {}

        for raw_event in raw_purchases:
            coordinate = _raw_coordinate(raw_event)

            if coordinate in raw_by_coordinate:
                raise RuntimeError(f"Duplicate raw event coordinate: {directory.name} {coordinate}")

            raw_by_coordinate[coordinate] = raw_event

            raw_participant_counts[_raw_participant_id(raw_event)] += 1

        match_missing = 0

        for normalized_event in normalized_purchases:
            coordinate = (
                normalized_event.frame_index,
                normalized_event.event_index,
            )

            raw_event = raw_by_coordinate.get(coordinate)

            if raw_event is None:
                raise RuntimeError(
                    "Could not align "
                    "normalized purchase "
                    "to raw coordinate: "
                    f"match={directory.name} "
                    f"frame="
                    f"{normalized_event.frame_index} "
                    f"event="
                    f"{normalized_event.event_index}"
                )

            _validate_aligned_event(
                normalized=(normalized_event),
                raw=raw_event,
            )

            if normalized_event.actor_participant_id is not None:
                raw_participant_id = _raw_participant_id(raw_event)

                if normalized_event.actor_participant_id != raw_participant_id:
                    raise RuntimeError(
                        "Positive normalized actor "
                        "does not match raw "
                        "participantId: "
                        f"match={directory.name} "
                        f"frame="
                        f"{normalized_event.frame_index} "
                        f"event="
                        f"{normalized_event.event_index} "
                        f"normalized="
                        f"{normalized_event.actor_participant_id} "
                        f"raw={raw_participant_id}"
                    )

                continue

            raw_participant_id = _raw_participant_id(raw_event)

            raw_item_id = _raw_item_id(raw_event)

            raw_timestamp = _raw_timestamp(raw_event)

            missing_participant_counts[raw_participant_id] += 1

            missing_item_counts[raw_item_id] += 1

            missing_timestamp_counts[raw_timestamp] += 1

            missing_cases.append(
                (
                    directory.name,
                    _safe_raw_projection(raw_event),
                )
            )

            match_missing += 1

        total_raw += len(raw_purchases)

        total_normalized += len(normalized_purchases)

        total_missing_actor += match_missing

        print(f"MATCH={directory.name}")

        print(f"  raw_purchases={len(raw_purchases)}")

        print(f"  normalized_purchases={len(normalized_purchases)}")

        print(f"  missing_actor={match_missing}")

        print()

    if total_raw != EXPECTED_PURCHASE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PURCHASE_COUNT} raw purchases, got {total_raw}")

    if total_normalized != EXPECTED_PURCHASE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PURCHASE_COUNT} normalized purchases, got {total_normalized}"
        )

    if total_missing_actor != EXPECTED_MISSING_ACTOR_COUNT:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_MISSING_ACTOR_COUNT} "
            "missing actor purchases, "
            f"got {total_missing_actor}"
        )

    print("=== RAW participantId DISTRIBUTION ===")

    for participant_id, count in sorted(raw_participant_counts.items()):
        print(f"  {participant_id}: {count}")

    print()

    print("=== MISSING ACTOR RAW participantId ===")

    for participant_id, count in sorted(missing_participant_counts.items()):
        print(f"  {participant_id}: {count}")

    print()

    print("=== MISSING ACTOR ITEMS ===")

    for item_id, count in sorted(missing_item_counts.items()):
        print(f"  {item_id}: {count}")

    print()

    print("=== MISSING ACTOR TIMESTAMPS ===")

    for timestamp, count in sorted(missing_timestamp_counts.items()):
        print(f"  {timestamp}: {count}")

    print()

    print("=== EXACT MISSING CASES ===")

    for match_id, raw_event in missing_cases:
        print(f"MATCH={match_id}")

        print(f"  {raw_event}")

    print()

    missing_participant_ids = set(missing_participant_counts)

    missing_item_ids = set(missing_item_counts)

    missing_timestamps = set(missing_timestamp_counts)

    all_zero_participant = missing_participant_ids == {
        0,
    }

    all_positive_participant = bool(missing_participant_ids) and all(
        participant_id > 0 for participant_id in missing_participant_ids
    )

    all_item_3865 = missing_item_ids == {
        3865,
    }

    all_timestamp_zero = missing_timestamps == {
        0,
    }

    print("=== ROOT CAUSE SIGNALS ===")

    print("all_missing_raw_participant_zero=" + ("YES" if all_zero_participant else "NO"))

    print("all_missing_raw_participant_positive=" + ("YES" if all_positive_participant else "NO"))

    print("all_missing_item_3865=" + ("YES" if all_item_3865 else "NO"))

    print("all_missing_timestamp_zero=" + ("YES" if all_timestamp_zero else "NO"))

    print()

    if all_zero_participant:
        decision = "SYSTEM_OR_UNASSIGNED_EVENT"

        explanation = (
            "Raw Riot data itself uses "
            "participantId=0 for every "
            "missing normalized actor. "
            "The normalizer did not drop "
            "a valid player identity."
        )

    elif all_positive_participant:
        decision = "NORMALIZER_DATA_LOSS"

        explanation = (
            "Every missing normalized "
            "actor has a positive raw "
            "participantId. The raw "
            "player identity is being "
            "lost during normalization."
        )

    else:
        decision = "MIXED_OR_UNKNOWN"

        explanation = (
            "Missing actor events contain "
            "mixed participantId semantics. "
            "A single reconstruction rule "
            "cannot yet be safely applied."
        )

    print(f"DECISION={decision}")

    print(explanation)

    print()

    print("MISSING_ITEM_PURCHASE_ACTOR_PROBE=PASS")


if __name__ == "__main__":
    main()
