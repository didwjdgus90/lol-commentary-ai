from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
)
from lol_commentary_backend.intelligence.item_state_builder import (
    build_inventory_state,
)


def _load_match(
    path: Path,
) -> NormalizedMatch:
    return NormalizedMatch.model_validate_json(path.read_text(encoding="utf-8"))


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
            event = NormalizedGameEvent.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

        result.append(event)

    return tuple(result)


def _load_raw_final_items(
    path: Path,
) -> dict[
    int,
    tuple[int, ...],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Raw match root must be object")

    info = payload.get("info")

    if not isinstance(
        info,
        dict,
    ):
        raise ValueError("Raw match info missing")

    participants = info.get("participants")

    if not isinstance(
        participants,
        list,
    ):
        raise ValueError("Raw match participants must be list")

    result: dict[
        int,
        tuple[int, ...],
    ] = {}

    for participant in participants:
        if not isinstance(
            participant,
            dict,
        ):
            continue

        participant_id = participant.get("participantId")

        if not isinstance(
            participant_id,
            int,
        ):
            raise ValueError("Raw participantId must be int")

        item_ids: list[int] = []

        for index in range(7):
            value = participant.get(f"item{index}")

            if not isinstance(
                value,
                int,
            ):
                raise ValueError("Raw final item slot must be int")

            if value > 0:
                item_ids.append(value)

        result[participant_id] = tuple(sorted(item_ids))

    return result


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


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    raw_root = repository_root / "data" / "raw" / "riot_api" / "matches"

    match_directories = tuple(
        sorted(
            (
                path
                for path in normalized_root.iterdir()
                if (
                    path.is_dir()
                    and (path / "match.json").is_file()
                    and (path / "events.jsonl").is_file()
                )
            ),
            key=lambda path: path.name,
        )
    )

    if len(match_directories) != 3:
        raise RuntimeError("Expected current 3-match baseline")

    event_counts: Counter[str] = Counter()

    total_transitions = 0
    total_unassigned = 0
    total_anomalies = 0

    matched_final_inventories = 0
    mismatched_final_inventories = 0

    mismatch_details: list[
        tuple[
            str,
            int,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []

    unassigned_items: Counter[
        tuple[
            str,
            int | None,
            int,
        ]
    ] = Counter()

    print("=== INVENTORY STATE BUILDER SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        raw_match_path = raw_root / directory.name / "match.json"

        if not raw_match_path.is_file():
            raise RuntimeError(f"Raw match missing: {raw_match_path}")

        raw_final_items = _load_raw_final_items(raw_match_path)

        participant_ids = tuple(participant.participant_id for participant in match.participants)

        replay = build_inventory_state(
            match_id=(directory.name),
            participant_ids=(participant_ids),
            events=events,
            strict=False,
        )

        match_transition_count = sum(len(timeline.transitions) for timeline in replay.timelines)

        match_matched = 0
        match_mismatched = 0

        for event in events:
            event_type = _event_type(event)

            if event_type in {
                "ITEM_PURCHASED",
                "ITEM_DESTROYED",
                "ITEM_SOLD",
                "ITEM_UNDO",
            }:
                event_counts[event_type] += 1

        for unassigned in replay.unassigned_events:
            unassigned_items[
                (
                    unassigned.raw_event_type,
                    unassigned.item_id,
                    unassigned.timestamp_ms,
                )
            ] += 1

        for timeline in replay.timelines:
            expected = raw_final_items.get(timeline.participant_id)

            if expected is None:
                raise RuntimeError(
                    f"Raw final inventory missing participant {timeline.participant_id}"
                )

            actual = timeline.final_item_ids

            if actual == expected:
                matched_final_inventories += 1
                match_matched += 1

            else:
                mismatched_final_inventories += 1
                match_mismatched += 1

                mismatch_details.append(
                    (
                        directory.name,
                        timeline.participant_id,
                        actual,
                        expected,
                    )
                )

        total_transitions += match_transition_count

        total_unassigned += len(replay.unassigned_events)

        total_anomalies += len(replay.anomalies)

        print(f"MATCH={directory.name}")

        print(f"  transitions={match_transition_count}")

        print(f"  unassigned={len(replay.unassigned_events)}")

        print(f"  anomalies={len(replay.anomalies)}")

        print(f"  final_inventory_matched={match_matched}")

        print(f"  final_inventory_mismatched={match_mismatched}")

        print()

    print("=== EVENT COUNTS ===")

    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count}")

    print()

    print("=== UNASSIGNED EVENTS ===")

    for (
        event_type,
        item_id,
        timestamp_ms,
    ), count in sorted(
        unassigned_items.items(),
        key=lambda item: (
            item[0][0],
            (-1 if item[0][1] is None else item[0][1]),
            item[0][2],
        ),
    ):
        print(f"  {event_type} item={item_id} timestamp={timestamp_ms}: {count}")

    print()

    print("=== FINAL INVENTORY PARITY ===")

    print(f"matched_participants={matched_final_inventories}/30")

    print(f"mismatched_participants={mismatched_final_inventories}/30")

    if mismatch_details:
        print()

        print("MISMATCH DETAILS:")

        for (
            match_id,
            participant_id,
            actual,
            expected,
        ) in mismatch_details:
            print(f"  MATCH={match_id} participant={participant_id}")

            print(f"    replay={actual}")

            print(f"    riot_final={expected}")

    print()

    print("=== AGGREGATE ===")

    print(f"item_events={sum(event_counts.values())}")

    print(f"transitions={total_transitions}")

    print(f"unassigned_events={total_unassigned}")

    print(f"replay_anomalies={total_anomalies}")

    print()

    expected_event_counts = {
        "ITEM_DESTROYED": 531,
        "ITEM_PURCHASED": 633,
        "ITEM_SOLD": 24,
        "ITEM_UNDO": 22,
    }

    if dict(event_counts) != expected_event_counts:
        raise RuntimeError(f"Item event baseline changed unexpectedly: {dict(event_counts)}")

    if total_unassigned != 6:
        raise RuntimeError("Expected exactly six known unassigned Riot item events")

    if total_anomalies != 0:
        raise RuntimeError(f"Inventory replay produced {total_anomalies} anomaly(s)")

    if mismatched_final_inventories != 0:
        raise RuntimeError(
            "Final inventory replay "
            "does not yet match Riot "
            "Match-V5 final inventory "
            "for all participants"
        )

    if matched_final_inventories != 30:
        raise RuntimeError("Expected final inventory parity for 30 participants")

    print("INVENTORY_STATE_BUILDER_SMOKE=PASS")


if __name__ == "__main__":
    main()
