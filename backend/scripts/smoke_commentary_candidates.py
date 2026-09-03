from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)


def _load_events(
    path: Path,
) -> tuple[NormalizedGameEvent, ...]:
    events: list[NormalizedGameEvent] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        events.append(NormalizedGameEvent.model_validate_json(line))

    return tuple(events)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    aggregate = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    expected_events = aggregate["total_event_count"]

    match_directories = tuple(
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )

    total_events = 0
    total_candidates = 0

    candidate_types: Counter[str] = Counter()

    bands: Counter[str] = Counter()

    print("=== COMMENTARY CANDIDATE SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        events = _load_events(directory / "events.jsonl")

        candidates = build_commentary_candidates(
            match_id=directory.name,
            events=events,
        )

        total_events += len(events)
        total_candidates += len(candidates)

        for candidate in candidates:
            candidate_types[candidate.candidate_type.value] += 1

            bands[candidate.salience_band.value] += 1

        top_candidates = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.salience_score,
                    candidate.timestamp_ms,
                    candidate.source_sequence,
                ),
            )[:10]
        )

        print(f"MATCH={directory.name}")

        print(f"  events={len(events)}")

        print(f"  candidates={len(candidates)}")

        print("  top:")

        for candidate in top_candidates:
            seconds = candidate.timestamp_ms / 1000

            print(
                "    "
                f"{seconds:7.1f}s "
                f"score={candidate.salience_score:3d} "
                f"type={candidate.candidate_type.value} "
                f"raw={candidate.raw_event_type}"
            )

        print()

    if total_events != expected_events:
        raise RuntimeError(f"Input event count mismatch: {total_events} != {expected_events}")

    print("=== AGGREGATE ===")

    print(f"Input events: {total_events}")

    print(f"Candidates: {total_candidates}")

    reduction = (1 - (total_candidates / total_events)) * 100

    print(f"Reduction: {reduction:.2f}%")

    print()

    print("Candidate types:")

    for name, count in sorted(candidate_types.items()):
        print(f"  {name}: {count}")

    print()

    print("Salience bands:")

    for name, count in sorted(bands.items()):
        print(f"  {name}: {count}")

    print()

    print("COMMENTARY_CANDIDATE_SMOKE=PASS")


if __name__ == "__main__":
    main()
