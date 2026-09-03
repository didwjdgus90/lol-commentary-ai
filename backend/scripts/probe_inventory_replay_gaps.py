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

ITEM_EVENT_TYPES = {
    "ITEM_PURCHASED",
    "ITEM_DESTROYED",
    "ITEM_SOLD",
    "ITEM_UNDO",
}


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
            result.append(NormalizedGameEvent.model_validate_json(line))

        except ValueError as exc:
            raise ValueError(f"Invalid event at {path}:{line_number}") from exc

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
        raise ValueError("Raw Match-V5 root must be object")

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
        raise ValueError("Raw participants must be list")

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
            raise ValueError("participantId must be int")

        items: list[int] = []

        for index in range(7):
            value = participant.get(f"item{index}")

            if not isinstance(
                value,
                int,
            ):
                raise ValueError("Final item slot must be int")

            if value > 0:
                items.append(value)

        result[participant_id] = tuple(sorted(items))

    return result


def _counter_difference(
    left: tuple[
        int,
        ...,
    ],
    right: tuple[
        int,
        ...,
    ],
) -> tuple[
    int,
    ...,
]:
    difference = Counter(left) - Counter(right)

    result: list[int] = []

    for item_id, count in sorted(difference.items()):
        result.extend([item_id] * count)

    return tuple(result)


def _event_item_ids(
    event: NormalizedGameEvent,
) -> tuple[int, ...]:
    result: list[int] = []

    if event.item_id is not None and event.item_id > 0:
        result.append(event.item_id)

    if event.before_item_id is not None and event.before_item_id > 0:
        result.append(event.before_item_id)

    if event.after_item_id is not None and event.after_item_id > 0:
        result.append(event.after_item_id)

    return tuple(result)


def _participant_item_events(
    *,
    events: tuple[
        NormalizedGameEvent,
        ...,
    ],
    participant_id: int,
    item_id: int,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    return tuple(
        event
        for event in events
        if (event.actor_participant_id == participant_id and item_id in _event_item_ids(event))
    )


def _event_projection(
    event: NormalizedGameEvent,
) -> dict[str, object]:
    return {
        "sequence": (event.sequence),
        "frame": (event.frame_index),
        "event": (event.event_index),
        "timestamp_ms": (event.timestamp_ms),
        "type": (_event_type(event)),
        "item_id": (event.item_id),
        "before_item_id": (event.before_item_id),
        "after_item_id": (event.after_item_id),
    }


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

    anomaly_type_counts: Counter[str] = Counter()

    anomaly_event_type_counts: Counter[str] = Counter()

    anomaly_item_counts: Counter[int] = Counter()

    extra_item_counts: Counter[int] = Counter()

    missing_item_counts: Counter[int] = Counter()

    missing_seen_in_timeline = 0
    missing_never_seen_in_timeline = 0

    extra_with_removal_event = 0
    extra_without_removal_event = 0

    participant_mismatches = 0
    matched_participants = 0

    mismatch_rows: list[
        tuple[
            str,
            int,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []

    missing_histories: list[
        tuple[
            str,
            int,
            int,
            tuple[
                dict[str, object],
                ...,
            ],
        ]
    ] = []

    extra_histories: list[
        tuple[
            str,
            int,
            int,
            tuple[
                dict[str, object],
                ...,
            ],
        ]
    ] = []

    print("=== INVENTORY REPLAY GAP ROOT-CAUSE PROBE ===")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        raw_final = _load_raw_final_items(raw_root / directory.name / "match.json")

        participant_ids = tuple(participant.participant_id for participant in match.participants)

        replay = build_inventory_state(
            match_id=directory.name,
            participant_ids=(participant_ids),
            events=events,
            strict=False,
        )

        item_events = tuple(event for event in events if (_event_type(event) in ITEM_EVENT_TYPES))

        print(f"MATCH={directory.name}")

        print(f"  item_events={len(item_events)}")

        print(f"  anomalies={len(replay.anomalies)}")

        print(f"  unassigned={len(replay.unassigned_events)}")

        for anomaly in replay.anomalies:
            anomaly_type_counts[anomaly.anomaly_type.value] += 1

            anomaly_event_type_counts[anomaly.raw_event_type] += 1

            if anomaly.item_id is not None:
                anomaly_item_counts[anomaly.item_id] += 1

        match_mismatches = 0

        for timeline in replay.timelines:
            participant_id = timeline.participant_id

            riot_final = raw_final.get(participant_id)

            if riot_final is None:
                raise RuntimeError(f"Missing Riot final inventory for participant={participant_id}")

            replay_final = timeline.final_item_ids

            if replay_final == riot_final:
                matched_participants += 1

                continue

            participant_mismatches += 1
            match_mismatches += 1

            extras = _counter_difference(
                replay_final,
                riot_final,
            )

            missing = _counter_difference(
                riot_final,
                replay_final,
            )

            mismatch_rows.append(
                (
                    directory.name,
                    participant_id,
                    extras,
                    missing,
                )
            )

            for item_id in extras:
                extra_item_counts[item_id] += 1

                history = _participant_item_events(
                    events=events,
                    participant_id=(participant_id),
                    item_id=item_id,
                )

                has_removal_event = any(
                    (
                        _event_type(event)
                        in {
                            "ITEM_DESTROYED",
                            "ITEM_SOLD",
                        }
                        and event.item_id == item_id
                    )
                    or (_event_type(event) == "ITEM_UNDO" and (event.before_item_id == item_id))
                    for event in history
                )

                if has_removal_event:
                    extra_with_removal_event += 1

                else:
                    extra_without_removal_event += 1

                extra_histories.append(
                    (
                        directory.name,
                        participant_id,
                        item_id,
                        tuple(_event_projection(event) for event in history),
                    )
                )

            for item_id in missing:
                missing_item_counts[item_id] += 1

                history = _participant_item_events(
                    events=events,
                    participant_id=(participant_id),
                    item_id=item_id,
                )

                if history:
                    missing_seen_in_timeline += 1

                else:
                    missing_never_seen_in_timeline += 1

                missing_histories.append(
                    (
                        directory.name,
                        participant_id,
                        item_id,
                        tuple(_event_projection(event) for event in history),
                    )
                )

        print(f"  final_mismatches={match_mismatches}")

        print()

    if matched_participants + participant_mismatches != 30:
        raise RuntimeError("Participant accounting must total 30")

    print("=== ANOMALY TYPES ===")

    for name, count in sorted(anomaly_type_counts.items()):
        print(f"  {name}: {count}")

    print()

    print("=== ANOMALY EVENT TYPES ===")

    for name, count in sorted(anomaly_event_type_counts.items()):
        print(f"  {name}: {count}")

    print()

    print("=== TOP ANOMALY ITEM IDS ===")

    for item_id, count in anomaly_item_counts.most_common(30):
        print(f"  {item_id}: {count}")

    print()

    print("=== FINAL PARITY GAPS ===")

    print(f"matched_participants={matched_participants}/30")

    print(f"mismatched_participants={participant_mismatches}/30")

    print()

    print("Replay EXTRA item IDs:")

    for item_id, count in extra_item_counts.most_common():
        print(f"  {item_id}: {count}")

    print()

    print("Riot-final MISSING item IDs:")

    for item_id, count in missing_item_counts.most_common():
        print(f"  {item_id}: {count}")

    print()

    print("=== GAP CLASSIFICATION ===")

    print(f"missing_seen_in_timeline={missing_seen_in_timeline}")

    print(f"missing_never_seen_in_timeline={missing_never_seen_in_timeline}")

    print(f"extra_with_removal_event={extra_with_removal_event}")

    print(f"extra_without_removal_event={extra_without_removal_event}")

    print()

    print("=== PARTICIPANT MISMATCH SUMMARY ===")

    for (
        match_id,
        participant_id,
        extras,
        missing,
    ) in mismatch_rows:
        print(f"MATCH={match_id} participant={participant_id}")

        print(f"  extra={extras}")

        print(f"  missing={missing}")

    print()

    print("=== MISSING ITEM TIMELINE HISTORIES ===")

    for (
        match_id,
        participant_id,
        item_id,
        history,
    ) in missing_histories:
        print(f"MATCH={match_id} participant={participant_id} item={item_id}")

        if not history:
            print("  NO_TIMELINE_EVENT")

            continue

        for event in history:
            print(f"  {event}")

    print()

    print("=== EXTRA ITEM TIMELINE HISTORIES ===")

    for (
        match_id,
        participant_id,
        item_id,
        history,
    ) in extra_histories:
        print(f"MATCH={match_id} participant={participant_id} item={item_id}")

        if not history:
            print("  NO_TIMELINE_EVENT")

            continue

        for event in history:
            print(f"  {event}")

    print()

    timeline_incomplete_signal = (
        missing_never_seen_in_timeline > 0 or extra_without_removal_event > 0
    )

    print("=== DECISION SIGNALS ===")

    print("timeline_incomplete_signal=" + ("YES" if timeline_incomplete_signal else "NO"))

    print("algorithm_only_explanation=" + ("NO" if timeline_incomplete_signal else "POSSIBLE"))

    print()

    print("NOTE:")

    print(
        "NO_TIMELINE_EVENT for an item "
        "present in Match-V5 final state "
        "is direct evidence that exact "
        "inventory reconstruction cannot "
        "come from ITEM_* replay alone."
    )

    print()

    print("INVENTORY_REPLAY_GAP_PROBE=PASS")


if __name__ == "__main__":
    main()
