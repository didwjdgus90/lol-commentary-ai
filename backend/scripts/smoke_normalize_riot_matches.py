from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.normalizer import (
    normalize_match_timeline,
)


def _load(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    matches_root = repository_root / "data" / "raw" / "riot_api" / "matches"

    match_directories = tuple(
        sorted(
            (path for path in matches_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )

    if not match_directories:
        raise RuntimeError("No Riot raw matches found")

    total_participants = 0
    total_snapshots = 0
    total_events = 0

    categories: Counter[str] = Counter()

    raw_types: Counter[str] = Counter()

    unknown_types: Counter[str] = Counter()

    print("=== RIOT NORMALIZATION SMOKE ===")

    print(f"Selected matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load(directory / "match.json")

        timeline = _load(directory / "timeline.json")

        bundle = normalize_match_timeline(
            match=match,
            timeline=timeline,
        )

        if bundle.match.match_id != directory.name:
            raise RuntimeError("Normalized match ID does not match directory")

        participant_count = len(bundle.match.participants)

        snapshot_count = len(bundle.participant_frames)

        event_count = len(bundle.events)

        total_participants += participant_count

        total_snapshots += snapshot_count

        total_events += event_count

        for event in bundle.events:
            categories[event.category.value] += 1

            raw_types[event.raw_event_type] += 1

            if not event.known_event_type:
                unknown_types[event.raw_event_type] += 1

        print(f"MATCH={bundle.match.match_id}")

        print(f"  mode={bundle.match.game_mode}")

        print(f"  queue={bundle.match.queue_id}")

        print(f"  version={bundle.match.game_version}")

        print(f"  participants={participant_count}")

        print(f"  participant_frames={snapshot_count}")

        print(f"  events={event_count}")

        print()

    print("=== AGGREGATE ===")

    print(f"Participants: {total_participants}")

    print(f"Participant snapshots: {total_snapshots}")

    print(f"Events: {total_events}")

    print()

    print("Categories:")

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    print()

    print("Raw event types:")

    for event_type, count in sorted(raw_types.items()):
        print(f"  {event_type}: {count}")

    print()

    print("Unknown event types:")

    if unknown_types:
        for (
            event_type,
            count,
        ) in sorted(unknown_types.items()):
            print(f"  {event_type}: {count}")

    else:
        print("  none")

    if total_events <= 0:
        raise RuntimeError("No normalized events")

    print()

    print("RIOT_NORMALIZATION_SMOKE=PASS")


if __name__ == "__main__":
    main()
