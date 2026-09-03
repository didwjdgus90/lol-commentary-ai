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
from lol_commentary_backend.intelligence.item_evidence_builder import (
    build_situation_item_evidence_contexts,
)
from lol_commentary_backend.intelligence.priority_builder import (
    build_situation_priorities,
)
from lol_commentary_backend.intelligence.record_builder import (
    build_commentary_intelligence_records,
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

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            result.append(NormalizedGameEvent.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

    return tuple(result)


def _load_snapshots(
    path: Path,
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    result: list[ParticipantFrameSnapshot] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            result.append(ParticipantFrameSnapshot.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid participant frame at {path}:{line_number}") from exc

    return tuple(result)


def _build_records(
    *,
    directory: Path,
) -> tuple[
    tuple[
        NormalizedGameEvent,
        ...,
    ],
    tuple,
]:
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
        participant_teams={
            participant.participant_id: (participant.team_id) for participant in match.participants
        },
    )

    priorities = build_situation_priorities(
        match_id=directory.name,
        situations=situations,
        contexts=contexts,
    )

    records = build_commentary_intelligence_records(
        match_id=directory.name,
        participants=tuple(match.participants),
        situations=situations,
        contexts=contexts,
        priorities=priorities,
    )

    return (
        events,
        records,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    match_directories = tuple(
        sorted(
            (
                path
                for path in normalized_root.iterdir()
                if (
                    path.is_dir()
                    and (path / "match.json").is_file()
                    and (path / "events.jsonl").is_file()
                    and (path / "participant_frames.jsonl").is_file()
                )
            ),
            key=lambda path: path.name,
        )
    )

    if len(match_directories) != 3:
        raise RuntimeError("Expected current 3-match baseline")

    total_records = 0
    total_contexts = 0
    total_participant_contexts = 0
    total_evidence = 0

    contexts_with_evidence = 0
    contexts_without_evidence = 0

    participant_contexts_with_evidence = 0

    action_counts: Counter[str] = Counter()

    item_counts: Counter[int] = Counter()

    max_evidence_per_participant = 0

    exact_inventory_claims_allowed = 0

    print("=== SITUATION ITEM EVIDENCE CONTEXT SMOKE ===")

    print()

    for directory in match_directories:
        events, records = _build_records(directory=directory)

        item_contexts = build_situation_item_evidence_contexts(
            records=records,
            events=events,
        )

        if len(item_contexts) != len(records):
            raise RuntimeError("Record/item-context coverage mismatch")

        match_evidence = 0
        match_with_evidence = 0

        for context in item_contexts:
            if context.exact_inventory_claim_allowed:
                exact_inventory_claims_allowed += 1

            context_evidence_count = 0

            for participant in context.participants:
                total_participant_contexts += 1

                evidence_count = len(participant.evidence)

                max_evidence_per_participant = max(
                    max_evidence_per_participant,
                    evidence_count,
                )

                if evidence_count:
                    participant_contexts_with_evidence += 1

                for evidence in participant.evidence:
                    if evidence.timestamp_ms > context.situation_start_timestamp_ms:
                        raise RuntimeError("Future item evidence leaked into situation")

                    if evidence.age_ms_at_situation_start < 0:
                        raise RuntimeError("Negative item evidence age")

                    action_counts[evidence.action.value] += 1

                    item_counts[evidence.item_id] += 1

                    total_evidence += 1
                    match_evidence += 1
                    context_evidence_count += 1

            if context_evidence_count:
                contexts_with_evidence += 1
                match_with_evidence += 1

            else:
                contexts_without_evidence += 1

        total_records += len(records)

        total_contexts += len(item_contexts)

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        print(f"  item_contexts={len(item_contexts)}")

        print(f"  contexts_with_evidence={match_with_evidence}")

        print(f"  evidence_links={match_evidence}")

        print()

    if total_records != 125:
        raise RuntimeError("Expected Step 44 baseline of 125 records")

    if total_contexts != 125:
        raise RuntimeError("Expected 125 item evidence contexts")

    # Step 44-4F-B에서 실제 검증된
    # referenced entity links와 동일해야 한다.
    if total_participant_contexts != 475:
        raise RuntimeError(
            f"Expected 475 participant item contexts, got {total_participant_contexts}"
        )

    if max_evidence_per_participant > 5:
        raise RuntimeError("Per-participant evidence limit exceeded")

    if exact_inventory_claims_allowed != 0:
        raise RuntimeError("Exact inventory claim must never be allowed")

    if contexts_with_evidence + contexts_without_evidence != 125:
        raise RuntimeError("Item evidence context accounting mismatch")

    print("=== AGGREGATE ===")

    print(f"records={total_records}")

    print(f"item_contexts={total_contexts}")

    print(f"participant_contexts={total_participant_contexts}")

    print(f"participant_contexts_with_evidence={participant_contexts_with_evidence}")

    print(f"contexts_with_evidence={contexts_with_evidence}")

    print(f"contexts_without_evidence={contexts_without_evidence}")

    print(f"evidence_links={total_evidence}")

    print(f"max_evidence_per_participant={max_evidence_per_participant}")

    print(f"exact_inventory_claims_allowed={exact_inventory_claims_allowed}")

    print()

    print("Evidence actions:")

    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")

    print()

    print("Top observed item IDs:")

    for item_id, count in item_counts.most_common(20):
        print(f"  {item_id}: {count}")

    print()

    print("EVIDENCE_AUTHORITY=confirmed_timeline_event_only")

    print("EXACT_INVENTORY_CLAIM=FORBIDDEN")

    print()

    print("SITUATION_ITEM_EVIDENCE_SMOKE=PASS")


if __name__ == "__main__":
    main()
