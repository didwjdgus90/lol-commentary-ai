from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.game_state_context import (
    build_situation_state_contexts,
)
from lol_commentary_backend.intelligence.game_state_models import (
    FrameAlignmentMode,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
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

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        result.append(NormalizedGameEvent.model_validate_json(line))

    return tuple(result)


def _load_snapshots(
    path: Path,
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    result: list[ParticipantFrameSnapshot] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        result.append(ParticipantFrameSnapshot.model_validate_json(line))

    return tuple(result)


def _participant_teams(
    match: NormalizedMatch,
) -> dict[int, int]:
    return {participant.participant_id: (participant.team_id) for participant in match.participants}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    match_directories = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if (path.is_dir() and (path / "match.json").is_file())
            ),
            key=lambda path: path.name,
        )
    )

    total_situations = 0
    total_contexts = 0

    alignment_modes: Counter[str] = Counter()

    frame_intervals: list[int] = []

    interval_gold_changes: list[int] = []

    lead_changes = 0

    print("=== GAME STATE CONTEXT SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        snapshots = _load_snapshots(directory / "participant_frames.jsonl")

        candidates = build_commentary_candidates(
            match_id=directory.name,
            events=events,
        )

        situations = build_temporal_situations(
            match_id=directory.name,
            candidates=candidates,
        )

        contexts = build_situation_state_contexts(
            match_id=directory.name,
            situations=situations,
            snapshots=snapshots,
            participant_teams=(_participant_teams(match)),
        )

        total_situations += len(situations)

        total_contexts += len(contexts)

        match_same = 0
        match_distinct = 0

        for context in contexts:
            alignment_modes[context.alignment_mode.value] += 1

            frame_intervals.append(context.frame_interval_ms)

            if context.alignment_mode == FrameAlignmentMode.SAME_FRAME:
                match_same += 1

            else:
                match_distinct += 1

            interval_change = context.interval_gold_diff_change_100_minus_200

            if interval_change is not None:
                interval_gold_changes.append(interval_change)

            if context.leading_team_changed:
                lead_changes += 1

        print(f"MATCH={directory.name}")

        print(f"  situations={len(situations)}")

        print(f"  contexts={len(contexts)}")

        print(f"  same_frame={match_same}")

        print(f"  distinct_frames={match_distinct}")

        print()

    if total_situations != 125:
        raise RuntimeError(f"Expected 125 situations, got {total_situations}")

    if total_contexts != 125:
        raise RuntimeError(f"Expected 125 contexts, got {total_contexts}")

    same_count = alignment_modes[FrameAlignmentMode.SAME_FRAME.value]

    distinct_count = alignment_modes[FrameAlignmentMode.DISTINCT_FRAMES.value]

    if same_count != 3:
        raise RuntimeError(f"Expected 3 same-frame contexts, got {same_count}")

    if distinct_count != 122:
        raise RuntimeError(f"Expected 122 distinct-frame contexts, got {distinct_count}")

    if len(interval_gold_changes) != 122:
        raise RuntimeError("Expected 122 interval gold-difference observations")

    absolute_gold_changes = [abs(value) for value in interval_gold_changes]

    print("=== AGGREGATE ===")

    print(f"Situations: {total_situations}")

    print(f"State contexts: {total_contexts}")

    print(f"Same frame: {same_count}")

    print(f"Distinct frames: {distinct_count}")

    print(f"Interval gold observations: {len(interval_gold_changes)}")

    print(f"Median frame interval ms: {int(median(frame_intervals))}")

    print(f"Median abs interval gold diff change: {int(median(absolute_gold_changes))}")

    print(f"Max abs interval gold diff change: {max(absolute_gold_changes)}")

    print(f"Leading-team changes: {lead_changes}")

    print()

    print("GAME_STATE_CONTEXT_SMOKE=PASS")


if __name__ == "__main__":
    main()
