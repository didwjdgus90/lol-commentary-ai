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
from lol_commentary_backend.rag.query_planner.intelligence_adapter import (
    build_rag_planner_input,
)
from lol_commentary_backend.rag.query_planner.models import (
    RAGQueryIntent,
)
from lol_commentary_backend.rag.query_planner.planner import (
    build_rag_query_plans,
)

PATCH = "26.17"

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

    selections = select_item_power_spike_contexts(power_contexts=(power_contexts))

    return (
        records,
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

    total_plans = 0

    total_queries = 0

    zero_query_plans = 0

    max_queries_per_plan = 0

    intent_counts: Counter[str] = Counter()

    priority_counts: Counter[str] = Counter()

    query_text_counts: Counter[str] = Counter()

    selected_item_keys: set[
        tuple[
            str,
            int,
        ]
    ] = set()

    item_query_keys: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    pii_hits: list[str] = []

    print("=== STEP 45-C RAG QUERY PLAN v1 SMOKE ===")

    print()

    for directory in directories:
        (
            records,
            selections,
        ) = _build_match_pipeline(
            directory=directory,
            resolver=resolver,
        )

        if len(records) != len(selections):
            raise RuntimeError("Record / item selection coverage mismatch")

        selection_by_record = {selection.record_id: selection for selection in selections}

        planner_inputs = []

        for record in records:
            selection = selection_by_record.get(record.record_id)

            if selection is None:
                raise RuntimeError("Missing item selection for record")

            for signal in selection.selected_signals:
                selected_item_keys.add(
                    (
                        record.record_id,
                        signal.item_id,
                    )
                )

            planner_inputs.append(
                build_rag_planner_input(
                    record=record,
                    item_selection=(selection),
                    patch=PATCH,
                    ddragon_version=(DDRAGON_VERSION),
                )
            )

        plans = build_rag_query_plans(tuple(planner_inputs))

        if len(plans) != len(records):
            raise RuntimeError("RAG plan coverage does not match records")

        match_queries = sum(plan.query_count for plan in plans)

        match_zero = sum(1 for plan in plans if plan.query_count == 0)

        total_records += len(records)

        total_plans += len(plans)

        total_queries += match_queries

        zero_query_plans += match_zero

        for plan in plans:
            priority_counts[plan.priority_tier.value] += 1

            max_queries_per_plan = max(
                max_queries_per_plan,
                plan.query_count,
            )

            seen_texts: set[str] = set()

            for query in plan.queries:
                intent_counts[query.intent.value] += 1

                query_text_counts[query.query_text] += 1

                normalized_text = query.query_text.casefold()

                if normalized_text in seen_texts:
                    raise RuntimeError("Duplicate query text inside one plan")

                seen_texts.add(normalized_text)

                if query.intent == RAGQueryIntent.ITEM_PATCH:
                    if query.subject_id is None:
                        raise RuntimeError("Item query missing subject ID")

                    item_query_keys.add(
                        (
                            plan.record_id,
                            query.subject_id,
                        )
                    )

            payload = plan.model_dump_json()

            for forbidden in (
                "puuid",
                "summonerid",
                "summoner_id",
                "gamename",
                "tagline",
                "game_name",
                "tag_line",
            ):
                if forbidden in payload.casefold():
                    pii_hits.append(f"{plan.plan_id}:{forbidden}")

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        print(f"  plans={len(plans)}")

        print(f"  queries={match_queries}")

        print(f"  zero_query_plans={match_zero}")

        print()

    mean_queries = total_queries / total_plans if total_plans else 0.0

    expected_item_query_keys = {
        (
            record_id,
            str(item_id),
        )
        for (
            record_id,
            item_id,
        ) in selected_item_keys
    }

    missing_item_queries = expected_item_query_keys - item_query_keys

    print("=== AGGREGATE ===")

    print(f"records={total_records}")

    print(f"plans={total_plans}")

    print(f"queries={total_queries}")

    print(f"mean_queries_per_plan={mean_queries:.2f}")

    print(f"max_queries_per_plan={max_queries_per_plan}")

    print(f"zero_query_plans={zero_query_plans}")

    print()

    print("=== QUERY INTENTS ===")

    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")

    print()

    print("=== PRIORITY COVERAGE ===")

    for priority, count in sorted(priority_counts.items()):
        print(f"  {priority}: {count}")

    print()

    print(f"selected_unique_record_items={len(expected_item_query_keys)}")

    print(f"item_query_keys={len(item_query_keys)}")

    print(f"missing_item_query_keys={len(missing_item_queries)}")

    print(f"pii_hits={len(pii_hits)}")

    print()

    print("=== MOST COMMON QUERY TEXTS ===")

    for text, count in query_text_counts.most_common(20):
        print(f"  {count:3d} {text}")

    print()

    if total_records != 125:
        raise RuntimeError("Expected current 125-record baseline")

    if total_plans != 125:
        raise RuntimeError("Expected exactly one RAG plan per record")

    if total_queries <= 0:
        raise RuntimeError("RAG planner produced no queries")

    if max_queries_per_plan > 6:
        raise RuntimeError("RAG plan exceeded query budget")

    if intent_counts[RAGQueryIntent.CHAMPION_PATCH.value] <= 0:
        raise RuntimeError("Expected champion patch queries")

    if intent_counts[RAGQueryIntent.ITEM_PATCH.value] <= 0:
        raise RuntimeError("Expected item patch queries")

    system_count = (
        intent_counts[RAGQueryIntent.OBJECTIVE_PATCH.value]
        + intent_counts[RAGQueryIntent.STRUCTURE_PATCH.value]
    )

    if system_count <= 0:
        raise RuntimeError("Expected objective or structure queries")

    if missing_item_queries:
        raise RuntimeError("Selected item signals were lost by query planner")

    if pii_hits:
        raise RuntimeError("PII-like fields leaked into RAG query plans")

    print("QUERY_BUDGET_MAX=6")

    print("PII_BOUNDARY=PASS")

    print("ITEM_SIGNAL_COVERAGE=PASS")

    print()

    print("RAG_QUERY_PLAN_V1_SMOKE=PASS")


if __name__ == "__main__":
    main()
