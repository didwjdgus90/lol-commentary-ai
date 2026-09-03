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
from lol_commentary_backend.intelligence.item_power_spike_selector import (
    select_item_power_spike_contexts,
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
            raise ValueError(f"Invalid event at {path}:{line_number}") from exc

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
            raise ValueError(f"Invalid frame at {path}:{line_number}") from exc

    return tuple(result)


def _build_match_pipeline(
    *,
    directory: Path,
    resolver: DDragonItemMetadataResolver,
):
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

    item_contexts = build_situation_item_evidence_contexts(
        records=records,
        events=events,
    )

    power_contexts = build_item_power_spike_contexts(
        item_contexts=item_contexts,
        resolver=resolver,
        map_id=11,
        top_signal_limit=3,
    )

    selections = select_item_power_spike_contexts(power_contexts=(power_contexts))

    return (
        records,
        power_contexts,
        selections,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    normalized_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(repository_root),
        ddragon_version=(DDRAGON_VERSION),
    )

    directories = tuple(
        sorted(
            (
                path
                for path in normalized_root.iterdir()
                if (path.is_dir() and (path / "match.json").is_file())
            ),
            key=lambda path: path.name,
        )
    )

    if len(directories) != 3:
        raise RuntimeError("Expected current 3-match baseline")

    total_records = 0

    raw_top_signal_exposures = 0

    selected_signal_count = 0

    contexts_with_selected = 0

    selected_sources: list[str] = []

    tiers: Counter[str] = Counter()

    shapes: Counter[str] = Counter()

    item_counts: Counter[
        tuple[
            int,
            str,
        ]
    ] = Counter()

    age_buckets: Counter[str] = Counter()

    exact_inventory_claims = 0

    print("=== ITEM POWER SPIKE SELECTION POLICY V2 SMOKE ===")

    print()

    for directory in directories:
        (
            records,
            power_contexts,
            selections,
        ) = _build_match_pipeline(
            directory=directory,
            resolver=resolver,
        )

        if len(records) != len(selections):
            raise RuntimeError("Selection coverage mismatch")

        match_raw = sum(len(context.top_signals) for context in power_contexts)

        match_selected = sum(len(context.selected_signals) for context in selections)

        match_with_selected = sum(1 for context in selections if context.selected_signals)

        total_records += len(records)

        raw_top_signal_exposures += match_raw

        selected_signal_count += match_selected

        contexts_with_selected += match_with_selected

        for context in selections:
            if context.exact_inventory_claim_allowed:
                exact_inventory_claims += 1

            for signal in context.selected_signals:
                selected_sources.append(signal.source_event_sha256)

                tiers[signal.tier.value] += 1

                shapes[signal.shape.value] += 1

                item_counts[
                    (
                        signal.item_id,
                        signal.name_ko,
                    )
                ] += 1

                age = signal.age_ms_at_situation_start

                if age <= 30_000:
                    bucket = "000-030s"

                elif age <= 60_000:
                    bucket = "031-060s"

                elif age <= 120_000:
                    bucket = "061-120s"

                else:
                    bucket = "121-180s"

                age_buckets[bucket] += 1

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        print(f"  raw_top_signal_exposures={match_raw}")

        print(f"  selected_signals={match_selected}")

        print(f"  contexts_with_selected={match_with_selected}")

        print()

    duplicate_selected_sources = len(selected_sources) - len(set(selected_sources))

    reduction_ratio = (
        1.0 - (selected_signal_count / raw_top_signal_exposures)
        if raw_top_signal_exposures
        else 0.0
    )

    high_count = tiers.get(
        "high",
        0,
    )

    high_ratio = (high_count / selected_signal_count) if selected_signal_count else 0.0

    print("=== AGGREGATE ===")

    print(f"records={total_records}")

    print(f"raw_top_signal_exposures={raw_top_signal_exposures}")

    print(f"selected_signals={selected_signal_count}")

    print(f"unique_selected_sources={len(set(selected_sources))}")

    print(f"duplicate_selected_sources={duplicate_selected_sources}")

    print(f"contexts_with_selected={contexts_with_selected}")

    print(f"selection_reduction_ratio={reduction_ratio:.4f}")

    print()

    print("=== SELECTED TIERS ===")

    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")

    print(f"selected_high_ratio={high_ratio:.4f}")

    print()

    print("=== SELECTED SHAPES ===")

    for shape, count in sorted(shapes.items()):
        print(f"  {shape}: {count}")

    print()

    print("=== SELECTED AGE BUCKETS ===")

    for bucket, count in sorted(age_buckets.items()):
        print(f"  {bucket}: {count}")

    print()

    print("=== TOP SELECTED ITEMS ===")

    for (
        item_id,
        name,
    ), count in item_counts.most_common(25):
        print(f"  {item_id} {name}: {count}")

    print()

    print(f"exact_inventory_claims_allowed={exact_inventory_claims}")

    if total_records != 125:
        raise RuntimeError("Expected current 125-record baseline")

    if raw_top_signal_exposures != 182:
        raise RuntimeError("Expected calibrated v1 baseline of 182 raw exposures")

    if selected_signal_count <= 0:
        raise RuntimeError("Selection removed all signals")

    if duplicate_selected_sources != 0:
        raise RuntimeError("Duplicate purchase source survived selection")

    if selected_signal_count >= raw_top_signal_exposures:
        raise RuntimeError("Selection did not reduce signal pressure")

    if exact_inventory_claims != 0:
        raise RuntimeError("Exact inventory claim must remain forbidden")

    print("EVIDENCE_AUTHORITY=confirmed_purchase_plus_ddragon_selected")

    print("EXACT_INVENTORY_CLAIM=FORBIDDEN")

    print()

    print("ITEM_POWER_SPIKE_SELECTION_V2_SMOKE=PASS")


if __name__ == "__main__":
    main()
