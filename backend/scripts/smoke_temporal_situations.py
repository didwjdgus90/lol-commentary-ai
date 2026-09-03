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
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidateType,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)

_SPECIAL_TYPES = {
    CommentaryCandidateType.FIRST_BLOOD,
    CommentaryCandidateType.MULTI_KILL,
    CommentaryCandidateType.ACE,
}


def _load_events(
    path: Path,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
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
    total_situations = 0

    total_primary = 0
    total_markers = 0

    orphan_markers = 0

    longest_duration_ms = 0
    largest_member_count = 0

    kinds: Counter[str] = Counter()

    print("=== TEMPORAL SITUATION SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        events = _load_events(directory / "events.jsonl")

        candidates = build_commentary_candidates(
            match_id=directory.name,
            events=events,
        )

        situations = build_temporal_situations(
            match_id=directory.name,
            candidates=candidates,
        )

        total_events += len(events)

        total_candidates += len(candidates)

        total_situations += len(situations)

        match_longest = 0
        match_largest = 0

        for situation in situations:
            primary_count = len(situation.primary_candidates)

            marker_count = len(situation.semantic_markers)

            member_count = primary_count + marker_count

            total_primary += primary_count

            total_markers += marker_count

            if primary_count == 0 and marker_count > 0:
                orphan_markers += marker_count

            kinds[situation.situation_kind.value] += 1

            match_longest = max(
                match_longest,
                situation.duration_ms,
            )

            match_largest = max(
                match_largest,
                member_count,
            )

        longest_duration_ms = max(
            longest_duration_ms,
            match_longest,
        )

        largest_member_count = max(
            largest_member_count,
            match_largest,
        )

        print(f"MATCH={directory.name}")

        print(f"  events={len(events)}")

        print(f"  candidates={len(candidates)}")

        print(f"  situations={len(situations)}")

        print(f"  longest_duration_ms={match_longest}")

        print(f"  largest_members={match_largest}")

        print()

    if total_events != expected_events:
        raise RuntimeError("Normalized event count mismatch")

    if total_candidates != 266:
        raise RuntimeError(f"Unexpected commentary candidate count: {total_candidates}")

    if total_primary + total_markers != total_candidates:
        raise RuntimeError("Candidate coverage mismatch")

    expected_markers = sum(
        1
        for directory in match_directories
        for candidate in build_commentary_candidates(
            match_id=directory.name,
            events=_load_events(directory / "events.jsonl"),
        )
        if (candidate.candidate_type in _SPECIAL_TYPES)
    )

    if total_markers != expected_markers:
        raise RuntimeError("Semantic marker count mismatch")

    print("=== AGGREGATE ===")

    print(f"Input events: {total_events}")

    print(f"Candidates: {total_candidates}")

    print(f"Situations: {total_situations}")

    situation_reduction = (1 - (total_situations / total_candidates)) * 100

    print(f"Candidate -> situation reduction: {situation_reduction:.2f}%")

    print(f"Primary candidates: {total_primary}")

    print(f"Semantic markers: {total_markers}")

    print(f"Orphan markers: {orphan_markers}")

    print(f"Longest situation ms: {longest_duration_ms}")

    print(f"Largest situation members: {largest_member_count}")

    print()

    print("Situation kinds:")

    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")

    if total_markers != 24:
        raise RuntimeError("Expected 24 semantic markers from probe")

    if orphan_markers != 0:
        raise RuntimeError("Observed special markers failed to attach")

    print()

    print("TEMPORAL_SITUATION_SMOKE=PASS")


if __name__ == "__main__":
    main()
