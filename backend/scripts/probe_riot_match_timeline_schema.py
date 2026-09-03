from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect locally collected Riot "
            "Match-V5 and Timeline payload "
            "shapes without printing PUUIDs "
            "or Riot IDs."
        )
    )

    parser.add_argument(
        "--match-id",
        action="append",
        dest="match_ids",
        help=(
            "Optional match ID. "
            "May be supplied multiple times. "
            "If omitted, all collected matches "
            "are inspected."
        ),
    )

    return parser.parse_args()


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _as_dict(
    value: object,
) -> dict[str, Any] | None:
    if not isinstance(
        value,
        dict,
    ):
        return None

    return value


def _as_list(
    value: object,
) -> list[Any] | None:
    if not isinstance(
        value,
        list,
    ):
        return None

    return value


def _sorted_keys(
    value: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in value))


def _select_match_directories(
    *,
    matches_root: Path,
    requested_match_ids: (tuple[str, ...] | None),
) -> tuple[Path, ...]:
    if not matches_root.is_dir():
        raise FileNotFoundError(f"Collected Riot matches directory missing: {matches_root}")

    if requested_match_ids:
        directories = tuple(matches_root / match_id for match_id in requested_match_ids)

    else:
        directories = tuple(
            sorted(
                (path for path in matches_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
        )

    if not directories:
        raise ValueError("No Riot match directories were selected")

    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Match directory missing: {directory}")

        for filename in (
            "match.json",
            "timeline.json",
        ):
            path = directory / filename

            if not path.is_file():
                raise FileNotFoundError(f"Missing {filename}: {path}")

    return directories


def _extract_match_id(
    payload: dict[str, Any],
) -> str:
    metadata = _as_dict(payload.get("metadata"))

    if metadata is None:
        raise ValueError("Match metadata missing")

    match_id = metadata.get("matchId")

    if (
        not isinstance(
            match_id,
            str,
        )
        or not match_id
    ):
        raise ValueError("metadata.matchId missing")

    return match_id


def _print_participant_summary(
    participants: list[Any],
) -> None:
    print(f"Participants: {len(participants)}")

    common_keys: set[str] | None = None

    union_keys: set[str] = set()

    for raw_participant in participants:
        participant = _as_dict(raw_participant)

        if participant is None:
            continue

        keys = {str(key) for key in participant}

        union_keys.update(keys)

        if common_keys is None:
            common_keys = set(keys)

        else:
            common_keys.intersection_update(keys)

    print(f"Participant common key count: {len(common_keys or set())}")

    print(f"Participant union key count: {len(union_keys)}")

    important_fields = (
        "participantId",
        "teamId",
        "championId",
        "championName",
        "teamPosition",
        "individualPosition",
        "win",
    )

    print("Participant mapping:")

    for raw_participant in participants:
        participant = _as_dict(raw_participant)

        if participant is None:
            continue

        values = [
            (f"{field}={participant.get(field)!r}")
            for field in important_fields
            if field in participant
        ]

        print("  " + " ".join(values))

    print()


def _collect_timeline_statistics(
    timeline: dict[str, Any],
    *,
    event_counts: Counter[str],
    event_keys: dict[
        str,
        set[str],
    ],
    participant_frame_keys: (set[str]),
    frame_timestamps: list[int],
) -> tuple[int, int]:
    info = _as_dict(timeline.get("info"))

    if info is None:
        raise ValueError("Timeline info missing")

    frames = _as_list(info.get("frames"))

    if frames is None:
        raise ValueError("Timeline frames missing")

    total_events = 0

    for raw_frame in frames:
        frame = _as_dict(raw_frame)

        if frame is None:
            continue

        timestamp = frame.get("timestamp")

        if isinstance(
            timestamp,
            int,
        ):
            frame_timestamps.append(timestamp)

        frames_by_participant = _as_dict(frame.get("participantFrames"))

        if frames_by_participant is not None:
            for value in frames_by_participant.values():
                participant_frame = _as_dict(value)

                if participant_frame is None:
                    continue

                participant_frame_keys.update(str(key) for key in participant_frame)

        events = _as_list(frame.get("events"))

        if events is None:
            continue

        for raw_event in events:
            event = _as_dict(raw_event)

            if event is None:
                continue

            event_type = event.get("type")

            if (
                not isinstance(
                    event_type,
                    str,
                )
                or not event_type
            ):
                event_type = "<MISSING_TYPE>"

            event_counts[event_type] += 1

            event_keys[event_type].update(str(key) for key in event)

            total_events += 1

    return (
        len(frames),
        total_events,
    )


def _print_frame_interval_summary(
    timestamps: list[int],
) -> None:
    ordered = sorted(timestamps)

    if len(ordered) < 2:
        print("Frame interval samples: insufficient")

        return

    intervals = [
        current - previous for previous, current in pairwise(ordered) if current >= previous
    ]

    if not intervals:
        print("Frame interval samples: none")

        return

    counts = Counter(intervals)

    print("Frame interval ms:")

    for interval, count in counts.most_common(10):
        print(f"  {interval}: {count}")


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    matches_root = repository_root / "data" / "raw" / "riot_api" / "matches"

    requested = tuple(args.match_ids) if args.match_ids else None

    directories = _select_match_directories(
        matches_root=matches_root,
        requested_match_ids=(requested),
    )

    aggregate_event_counts: Counter[str] = Counter()

    aggregate_event_keys: dict[
        str,
        set[str],
    ] = defaultdict(set)

    participant_frame_keys: set[str] = set()

    frame_timestamps: list[int] = []

    total_frames = 0
    total_events = 0

    print("=== RIOT MATCH/TIMELINE SCHEMA PROBE ===")

    print(f"Selected matches: {len(directories)}")

    print()

    for directory in directories:
        match_path = directory / "match.json"

        timeline_path = directory / "timeline.json"

        match = _load_json_object(match_path)

        timeline = _load_json_object(timeline_path)

        match_id = _extract_match_id(match)

        timeline_match_id = _extract_match_id(timeline)

        if match_id != timeline_match_id:
            raise ValueError(f"Match/timeline ID mismatch: {match_id} != {timeline_match_id}")

        if match_id != directory.name:
            raise ValueError(f"Directory/match ID mismatch: {directory.name} != {match_id}")

        match_info = _as_dict(match.get("info"))

        if match_info is None:
            raise ValueError(f"Match info missing: {match_id}")

        participants = _as_list(match_info.get("participants"))

        if participants is None:
            raise ValueError(f"Match participants missing: {match_id}")

        timeline_info = _as_dict(timeline.get("info"))

        if timeline_info is None:
            raise ValueError(f"Timeline info missing: {match_id}")

        frames = _as_list(timeline_info.get("frames"))

        if frames is None:
            raise ValueError(f"Timeline frames missing: {match_id}")

        print("--------------------------------")

        print(f"MATCH={match_id}")

        print("Match top-level keys:")

        print("  " + ", ".join(_sorted_keys(match)))

        print(f"Match info key count: {len(match_info)}")

        game_mode = match_info.get("gameMode")

        queue_id = match_info.get("queueId")

        map_id = match_info.get("mapId")

        game_duration = match_info.get("gameDuration")

        game_version = match_info.get("gameVersion")

        print(f"gameMode={game_mode!r} queueId={queue_id!r} mapId={map_id!r}")

        print(f"gameDuration={game_duration!r}")

        print(f"gameVersion={game_version!r}")

        print()

        _print_participant_summary(participants)

        per_match_counts: Counter[str] = Counter()

        per_match_keys: dict[
            str,
            set[str],
        ] = defaultdict(set)

        frame_count, event_count = _collect_timeline_statistics(
            timeline,
            event_counts=(per_match_counts),
            event_keys=(per_match_keys),
            participant_frame_keys=(participant_frame_keys),
            frame_timestamps=(frame_timestamps),
        )

        total_frames += frame_count
        total_events += event_count

        aggregate_event_counts.update(per_match_counts)

        for (
            event_type,
            keys,
        ) in per_match_keys.items():
            aggregate_event_keys[event_type].update(keys)

        print(f"Timeline frames: {frame_count}")

        print(f"Timeline events: {event_count}")

        print("Event types:")

        for (
            event_type,
            count,
        ) in sorted(per_match_counts.items()):
            print(f"  {event_type}: {count}")

        print()

    print("================================")

    print("=== AGGREGATE SCHEMA ===")

    print(f"Matches: {len(directories)}")

    print(f"Frames: {total_frames}")

    print(f"Events: {total_events}")

    print()

    _print_frame_interval_summary(frame_timestamps)

    print()

    print("participantFrames union keys:")

    for key in sorted(participant_frame_keys):
        print(f"  {key}")

    print()

    print("Event schemas:")

    for event_type in sorted(aggregate_event_counts):
        print()

        print(f"[{event_type}] count={aggregate_event_counts[event_type]}")

        for key in sorted(aggregate_event_keys[event_type]):
            print(f"  {key}")

    print()

    print("RIOT_SCHEMA_PROBE=PASS")


if __name__ == "__main__":
    main()
