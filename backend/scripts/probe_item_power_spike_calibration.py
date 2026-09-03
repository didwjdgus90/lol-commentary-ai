from __future__ import annotations

from collections import Counter, defaultdict
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


def _age_bucket(
    age_ms: int,
) -> str:
    if age_ms <= 30_000:
        return "000-030s"

    if age_ms <= 60_000:
        return "031-060s"

    if age_ms <= 120_000:
        return "061-120s"

    if age_ms <= 180_000:
        return "121-180s"

    if age_ms <= 300_000:
        return "181-300s"

    return "300s+"


def _gold_bucket(
    gold_total: int,
) -> str:
    if gold_total < 1000:
        return "<1000"

    if gold_total < 1500:
        return "1000-1499"

    if gold_total < 2000:
        return "1500-1999"

    if gold_total < 2800:
        return "2000-2799"

    return "2800+"


def _item_shape(
    *,
    from_count: int,
    into_count: int,
) -> str:
    if from_count > 0 and into_count == 0:
        return "terminal_build"

    if from_count > 0 and into_count > 0:
        return "intermediate_build"

    if from_count == 0 and into_count > 0:
        return "starter_component"

    return "standalone_terminal"


def _percentile(
    values: list[int],
    fraction: float,
) -> int:
    if not values:
        return 0

    ordered = sorted(values)

    index = round((len(ordered) - 1) * fraction)

    return ordered[index]


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
    total_signals = 0

    signal_tiers: Counter[str] = Counter()

    age_buckets: Counter[str] = Counter()

    gold_buckets: Counter[str] = Counter()

    item_shapes: Counter[str] = Counter()

    tag_counts: Counter[str] = Counter()

    item_counts: Counter[
        tuple[
            int,
            str,
        ]
    ] = Counter()

    source_exposures: Counter[str] = Counter()

    source_item: dict[
        str,
        tuple[
            int,
            str,
        ],
    ] = {}

    source_ages: dict[
        str,
        list[int],
    ] = defaultdict(list)

    signal_ages: list[int] = []

    high_signal_ages: list[int] = []

    high_under_2000 = 0
    high_under_1500 = 0

    high_boots = 0

    intermediate_signals = 0
    starter_signals = 0
    standalone_signals = 0
    terminal_signals = 0

    older_than_300s = 0
    older_than_180s = 0

    print("=== ITEM POWER SPIKE CALIBRATION PROBE ===")

    print()

    for directory in match_directories:
        events, records = _build_records(directory=directory)

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

        match_signals = 0

        for context in power_contexts:
            for signal in context.top_signals:
                match_signals += 1
                total_signals += 1

                signal_tiers[signal.tier.value] += 1

                age_buckets[_age_bucket(signal.age_ms_at_situation_start)] += 1

                gold_buckets[_gold_bucket(signal.gold_total)] += 1

                signal_ages.append(signal.age_ms_at_situation_start)

                source_exposures[signal.source_event_sha256] += 1

                source_item[signal.source_event_sha256] = (
                    signal.item_id,
                    signal.name_ko,
                )

                source_ages[signal.source_event_sha256].append(signal.age_ms_at_situation_start)

                metadata = resolver.require(signal.item_id)

                shape = _item_shape(
                    from_count=len(metadata.from_item_ids),
                    into_count=len(metadata.into_item_ids),
                )

                item_shapes[shape] += 1

                if shape == "terminal_build":
                    terminal_signals += 1

                elif shape == "intermediate_build":
                    intermediate_signals += 1

                elif shape == "starter_component":
                    starter_signals += 1

                else:
                    standalone_signals += 1

                for tag in metadata.tags:
                    tag_counts[tag] += 1

                item_counts[
                    (
                        signal.item_id,
                        signal.name_ko,
                    )
                ] += 1

                if signal.age_ms_at_situation_start > 180_000:
                    older_than_180s += 1

                if signal.age_ms_at_situation_start > 300_000:
                    older_than_300s += 1

                if signal.tier.value == "high":
                    high_signal_ages.append(signal.age_ms_at_situation_start)

                    if signal.gold_total < 2000:
                        high_under_2000 += 1

                    if signal.gold_total < 1500:
                        high_under_1500 += 1

                    if "Boots" in metadata.tags:
                        high_boots += 1

        total_records += len(records)

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        print(f"  top_signal_exposures={match_signals}")

        print()

    unique_sources = len(source_exposures)

    duplicate_exposures = total_signals - unique_sources

    repeated_sources = {
        source_sha: count
        for (
            source_sha,
            count,
        ) in source_exposures.items()
        if count > 1
    }

    max_source_exposure = max(
        source_exposures.values(),
        default=0,
    )

    print("=== AGGREGATE ===")

    print(f"records={total_records}")

    print(f"top_signal_exposures={total_signals}")

    print(f"unique_purchase_sources={unique_sources}")

    print(f"duplicate_signal_exposures={duplicate_exposures}")

    print(f"repeated_purchase_sources={len(repeated_sources)}")

    print(f"max_exposures_per_purchase={max_source_exposure}")

    print()

    print("=== SIGNAL TIERS ===")

    for tier, count in sorted(signal_tiers.items()):
        print(f"  {tier}: {count}")

    print()

    print("=== AGE BUCKETS ===")

    for bucket, count in sorted(age_buckets.items()):
        print(f"  {bucket}: {count}")

    print()

    if signal_ages:
        print(f"signal_age_median_ms={int(median(signal_ages))}")

        print(f"signal_age_p75_ms={_percentile(signal_ages, 0.75)}")

        print(f"signal_age_p90_ms={_percentile(signal_ages, 0.90)}")

        print(f"signal_age_max_ms={max(signal_ages)}")

    print(f"signals_older_than_180s={older_than_180s}")

    print(f"signals_older_than_300s={older_than_300s}")

    print()

    print("=== ITEM SHAPES ===")

    for shape, count in sorted(item_shapes.items()):
        print(f"  {shape}: {count}")

    print()

    print(f"terminal_signals={terminal_signals}")

    print(f"intermediate_signals={intermediate_signals}")

    print(f"starter_signals={starter_signals}")

    print(f"standalone_signals={standalone_signals}")

    print()

    print("=== GOLD BUCKETS ===")

    for bucket, count in sorted(gold_buckets.items()):
        print(f"  {bucket}: {count}")

    print()

    print(f"high_under_2000={high_under_2000}")

    print(f"high_under_1500={high_under_1500}")

    print(f"high_boots={high_boots}")

    print()

    print("=== TOP SIGNAL TAGS ===")

    for tag, count in tag_counts.most_common(20):
        print(f"  {tag}: {count}")

    print()

    print("=== TOP SIGNAL ITEMS ===")

    for (
        item_id,
        name,
    ), count in item_counts.most_common(30):
        metadata = resolver.require(item_id)

        print(
            f"ITEM={item_id} "
            f"name={name} "
            f"count={count} "
            f"gold={metadata.gold_total} "
            f"depth={metadata.depth} "
            f"from={len(metadata.from_item_ids)} "
            f"into={len(metadata.into_item_ids)} "
            f"tags={metadata.tags}"
        )

    print()

    print("=== MOST REPEATED PURCHASE SOURCES ===")

    repeated_sorted = sorted(
        repeated_sources.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for (
        source_sha,
        count,
    ) in repeated_sorted[:20]:
        item_id, name = source_item[source_sha]

        ages = sorted(source_ages[source_sha])

        print(
            f"ITEM={item_id} "
            f"name={name} "
            f"exposures={count} "
            f"age_min={min(ages)} "
            f"age_max={max(ages)} "
            f"source={source_sha[:12]}"
        )

    print()

    duplicate_ratio = (duplicate_exposures / total_signals) if total_signals else 0.0

    intermediate_ratio = (intermediate_signals / total_signals) if total_signals else 0.0

    high_count = signal_tiers.get(
        "high",
        0,
    )

    high_ratio = (high_count / total_signals) if total_signals else 0.0

    print("=== DECISION SIGNALS ===")

    print(f"duplicate_exposure_ratio={duplicate_ratio:.4f}")

    print(f"intermediate_signal_ratio={intermediate_ratio:.4f}")

    print(f"high_signal_ratio={high_ratio:.4f}")

    print("repetition_pressure=" + ("HIGH" if duplicate_ratio >= 0.30 else "LOW"))

    print("intermediate_item_pressure=" + ("HIGH" if intermediate_ratio >= 0.20 else "LOW"))

    print("high_tier_pressure=" + ("HIGH" if high_ratio >= 0.70 else "LOW"))

    if total_records != 125:
        raise RuntimeError("Expected current 125-record baseline")

    if total_signals <= 0:
        raise RuntimeError("Expected power-spike signals")

    if unique_sources <= 0:
        raise RuntimeError("Expected unique purchase source events")

    print()

    print("ITEM_POWER_SPIKE_CALIBRATION_PROBE=PASS")


if __name__ == "__main__":
    main()
