from __future__ import annotations

from collections import Counter
from pathlib import Path

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
from lol_commentary_backend.intelligence.priority_builder import (
    build_situation_priorities,
)
from lol_commentary_backend.intelligence.priority_models import (
    PriorityReason,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
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


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


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

    total_priorities = 0

    tiers: Counter[str] = Counter()

    reasons: Counter[str] = Counter()

    macro_interval_ids: set[str] = set()

    macro_anchor_count = 0

    lead_flip_anchor_count = 0

    macro_p90_anchor_count = 0

    shared_context_count = 0

    print("=== SITUATION PRIORITY SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        snapshots = _load_snapshots(directory / "participant_frames.jsonl")

        candidates = build_commentary_candidates(
            match_id=(directory.name),
            events=events,
        )

        situations = tuple(
            sorted(
                build_temporal_situations(
                    match_id=(directory.name),
                    candidates=(candidates),
                ),
                key=_situation_sort_key,
            )
        )

        contexts = build_situation_state_contexts(
            match_id=(directory.name),
            situations=(situations),
            snapshots=snapshots,
            participant_teams=(_participant_teams(match)),
        )

        priorities = build_situation_priorities(
            match_id=(directory.name),
            situations=(situations),
            contexts=(contexts),
        )

        total_priorities += len(priorities)

        match_tiers: Counter[str] = Counter()

        for priority in priorities:
            tier = priority.priority_tier.value

            tiers[tier] += 1

            match_tiers[tier] += 1

            for reason in priority.reasons:
                reasons[reason.value] += 1

            macro_interval_ids.add(priority.macro_context.macro_interval_id)

            if priority.macro_context.shared_situation_count > 1:
                shared_context_count += 1

            if priority.is_macro_anchor:
                macro_anchor_count += 1

            if PriorityReason.MACRO_LEAD_FLIP in priority.reasons:
                lead_flip_anchor_count += 1

            if PriorityReason.MACRO_GOLD_P90 in priority.reasons:
                macro_p90_anchor_count += 1

        print(f"MATCH={directory.name}")

        print(f"  situations={len(situations)}")

        print(f"  priorities={len(priorities)}")

        for tier, count in sorted(match_tiers.items()):
            print(f"  {tier}={count}")

        print()

    if total_priorities != 125:
        raise RuntimeError(f"Expected 125 priorities, got {total_priorities}")

    if len(macro_interval_ids) != 77:
        raise RuntimeError(f"Expected 77 unique macro intervals, got {len(macro_interval_ids)}")

    if macro_anchor_count != 77:
        raise RuntimeError("Expected one macro anchor per interval")

    if lead_flip_anchor_count != 9:
        raise RuntimeError(
            f"Expected 9 independent lead-flip macro signals, got {lead_flip_anchor_count}"
        )

    if macro_p90_anchor_count != 8:
        raise RuntimeError(
            f"Expected 8 independent macro-p90 signals, got {macro_p90_anchor_count}"
        )

    print("=== AGGREGATE ===")

    print(f"Priorities: {total_priorities}")

    print(f"Unique macro intervals: {len(macro_interval_ids)}")

    print(f"Macro anchors: {macro_anchor_count}")

    print(f"Lead-flip macro reasons: {lead_flip_anchor_count}")

    print(f"Macro-p90 reasons: {macro_p90_anchor_count}")

    print(f"Situations referencing shared macro context: {shared_context_count}")

    print()

    print("Priority tiers:")

    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")

    print()

    print("Priority reasons:")

    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")

    print()

    print("SITUATION_PRIORITY_SMOKE=PASS")


if __name__ == "__main__":
    main()
