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
from lol_commentary_backend.intelligence.item_metadata_resolver import (
    DDragonItemMetadataResolver,
)
from lol_commentary_backend.intelligence.item_power_spike_builder import (
    build_item_power_spike_contexts,
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

DDRAGON_VERSION = "16.17.1"


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

    state_contexts = build_situation_state_contexts(
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
        contexts=state_contexts,
    )

    records = build_commentary_intelligence_records(
        match_id=directory.name,
        participants=tuple(match.participants),
        situations=situations,
        contexts=state_contexts,
        priorities=priorities,
    )

    return (
        events,
        records,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(repository_root),
        ddragon_version=(DDRAGON_VERSION),
    )

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
    total_item_contexts = 0
    total_power_contexts = 0

    total_evaluations = 0
    total_signals = 0

    contexts_with_signals = 0
    contexts_without_signals = 0

    tier_counts: Counter[str] = Counter()

    item_signal_counts: Counter[
        tuple[
            int,
            str,
        ]
    ] = Counter()

    score_counts: Counter[int] = Counter()

    utility_candidates = 0

    exact_inventory_claims_allowed = 0

    print("=== ITEM POWER SPIKE CONTEXT SMOKE ===")

    print()

    for directory in match_directories:
        events, records = _build_records(directory=directory)

        item_contexts = build_situation_item_evidence_contexts(
            records=records,
            events=events,
        )

        power_contexts = build_item_power_spike_contexts(
            item_contexts=(item_contexts),
            resolver=resolver,
            map_id=11,
            top_signal_limit=3,
        )

        if len(power_contexts) != len(records):
            raise RuntimeError("Power context coverage does not match records")

        match_evaluations = 0
        match_signals = 0
        match_with_signals = 0

        for context in power_contexts:
            if context.exact_inventory_claim_allowed:
                exact_inventory_claims_allowed += 1

            match_evaluations += len(context.evaluations)

            total_evaluations += len(context.evaluations)

            if context.top_signals:
                contexts_with_signals += 1
                match_with_signals += 1

            else:
                contexts_without_signals += 1

            for evaluation in context.evaluations:
                score_counts[evaluation.score] += 1

                if evaluation.commentary_candidate and (
                    "Consumable" in evaluation.tags
                    or "Trinket" in evaluation.tags
                    or "Vision" in evaluation.tags
                ):
                    utility_candidates += 1

            for signal in context.top_signals:
                match_signals += 1
                total_signals += 1

                tier_counts[signal.tier.value] += 1

                item_signal_counts[
                    (
                        signal.item_id,
                        signal.name_ko,
                    )
                ] += 1

        total_records += len(records)

        total_item_contexts += len(item_contexts)

        total_power_contexts += len(power_contexts)

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        print(f"  evaluations={match_evaluations}")

        print(f"  power_signals={match_signals}")

        print(f"  contexts_with_signals={match_with_signals}")

        print()

    print("=== AGGREGATE ===")

    print(f"records={total_records}")

    print(f"item_contexts={total_item_contexts}")

    print(f"power_contexts={total_power_contexts}")

    print(f"purchase_evaluations={total_evaluations}")

    print(f"top_signals={total_signals}")

    print(f"contexts_with_signals={contexts_with_signals}")

    print(f"contexts_without_signals={contexts_without_signals}")

    print(f"utility_candidates={utility_candidates}")

    print(f"exact_inventory_claims_allowed={exact_inventory_claims_allowed}")

    print()

    print("=== SIGNAL TIERS ===")

    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")

    print()

    print("=== TOP SIGNAL ITEMS ===")

    for (
        item_id,
        name,
    ), count in item_signal_counts.most_common(25):
        print(f"  {item_id} {name}: {count}")

    print()

    print("=== SCORE DISTRIBUTION ===")

    for score, count in sorted(score_counts.items()):
        print(f"  {score}: {count}")

    print()

    if total_records != 125:
        raise RuntimeError("Expected current baseline of 125 records")

    if total_item_contexts != 125:
        raise RuntimeError("Expected 125 item evidence contexts")

    if total_power_contexts != 125:
        raise RuntimeError("Expected 125 item power-spike contexts")

    if total_evaluations <= 0:
        raise RuntimeError("Expected actual purchase evaluations")

    if total_signals <= 0:
        raise RuntimeError("Expected at least one power-spike signal")

    if contexts_with_signals + contexts_without_signals != 125:
        raise RuntimeError("Power context accounting mismatch")

    if utility_candidates != 0:
        raise RuntimeError("Utility item leaked into power-spike candidates")

    if exact_inventory_claims_allowed != 0:
        raise RuntimeError("Exact inventory claim must remain forbidden")

    print("EVIDENCE_AUTHORITY=confirmed_timeline_event_plus_ddragon")

    print("EXACT_INVENTORY_CLAIM=FORBIDDEN")

    print()

    print("ITEM_POWER_SPIKE_CONTEXT_SMOKE=PASS")


if __name__ == "__main__":
    main()
