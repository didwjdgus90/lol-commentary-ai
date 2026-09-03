from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.ingestion.version_mapping.models import (
    MAPPING_RULE,
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
from lol_commentary_backend.intelligence.record_builder import (
    build_commentary_intelligence_records,
)
from lol_commentary_backend.intelligence.record_models import (
    CommentaryIntelligenceRecord,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)

EXPECTED_MAPPING_RULE = "season_2026_minor_alignment_v1"

PRIORITY_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

POLICIES = (
    "all_non_terminal",
    "medium_plus",
    "high_plus",
)

OBJECTIVE_CANDIDATE_TYPES = {
    "elite_monster",
    "dragon_soul",
}

STRUCTURE_CANDIDATE_TYPES = {
    "building",
}

TERMINAL_CANDIDATE_TYPES = {
    "game_end",
}


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
            event = NormalizedGameEvent.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

        result.append(event)

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
            snapshot = ParticipantFrameSnapshot.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"Invalid participant frame at {path}:{line_number}") from exc

        result.append(snapshot)

    return tuple(result)


def _game_version(
    match: NormalizedMatch,
) -> str:
    payload: dict[
        str,
        Any,
    ] = match.model_dump(mode="python")

    value = payload.get("game_version")

    if not isinstance(
        value,
        str,
    ):
        raise ValueError("NormalizedMatch does not contain a string game_version")

    normalized = value.strip()

    if not normalized:
        raise ValueError("game_version must not be empty")

    return normalized


def _probe_patch_from_game_version(
    game_version: str,
) -> str:
    parts = game_version.split(".")

    if len(parts) < 2:
        raise ValueError(f"Unexpected Riot game version: {game_version}")

    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Non-numeric Riot game version: {game_version}") from exc

    if MAPPING_RULE != EXPECTED_MAPPING_RULE:
        raise RuntimeError(f"Unexpected patch mapping rule: {MAPPING_RULE}")

    # Probe-only conversion for the already established
    # season_2026_minor_alignment_v1 contract:
    #
    # Riot game version 16.N.x
    #              ↓
    # LoL patch         26.N
    #
    # Production Query Planner will not duplicate this
    # conversion. It will receive resolved patch context
    # from the shared version-mapping boundary.
    if major != 16:
        raise RuntimeError(
            "Step 45-A probe only supports "
            "the established 2026 mapping "
            "for Riot game major version 16"
        )

    if minor <= 0:
        raise ValueError("Patch minor version must be positive")

    return f"26.{minor}"


def _candidate_type_value(
    candidate_type: object,
) -> str:
    value = getattr(
        candidate_type,
        "value",
        candidate_type,
    )

    if not isinstance(
        value,
        str,
    ):
        raise TypeError("Candidate type must resolve to a string")

    return value


def _candidate_types(
    record: CommentaryIntelligenceRecord,
) -> tuple[str, ...]:
    candidates = (
        *record.situation.primary_candidates,
        *record.situation.semantic_markers,
    )

    return tuple(
        sorted({_candidate_type_value(candidate.candidate_type) for candidate in candidates})
    )


def _is_terminal(
    candidate_types: tuple[
        str,
        ...,
    ],
) -> bool:
    return bool(TERMINAL_CANDIDATE_TYPES.intersection(candidate_types))


def _policy_allows(
    *,
    policy: str,
    priority: str,
    terminal: bool,
) -> bool:
    if terminal:
        return False

    rank = PRIORITY_RANK.get(priority)

    if rank is None:
        raise ValueError(f"Unknown priority tier: {priority}")

    if policy == "all_non_terminal":
        return True

    if policy == "medium_plus":
        return rank >= PRIORITY_RANK["medium"]

    if policy == "high_plus":
        return rank >= PRIORITY_RANK["high"]

    raise ValueError(f"Unknown policy: {policy}")


def _champion_queries(
    *,
    patch: str,
    record: CommentaryIntelligenceRecord,
) -> tuple[str, ...]:
    champion_names = tuple(
        sorted(
            {
                entity.champion_name.strip()
                for entity in record.referenced_entities
                if (entity.champion_name.strip())
            }
        )
    )

    return tuple((f"{patch} {champion_name} 변경 사항") for champion_name in champion_names)


def _system_queries(
    *,
    patch: str,
    candidate_types: tuple[
        str,
        ...,
    ],
) -> tuple[str, ...]:
    result: list[str] = []

    candidate_type_set = set(candidate_types)

    if candidate_type_set & OBJECTIVE_CANDIDATE_TYPES:
        result.append(f"{patch} 오브젝트 드래곤 바론 전령 변경 사항")

    if candidate_type_set & STRUCTURE_CANDIDATE_TYPES:
        result.append(f"{patch} 포탑 억제기 구조물 변경 사항")

    return tuple(result)


def _planned_queries(
    *,
    patch: str,
    record: CommentaryIntelligenceRecord,
) -> tuple[str, ...]:
    candidate_types = _candidate_types(record)

    ordered = (
        *_champion_queries(
            patch=patch,
            record=record,
        ),
        *_system_queries(
            patch=patch,
            candidate_types=(candidate_types),
        ),
    )

    return tuple(dict.fromkeys(ordered))


def _situation_kind(
    record: CommentaryIntelligenceRecord,
) -> str:
    value = getattr(
        record.situation.situation_kind,
        "value",
        record.situation.situation_kind,
    )

    if not isinstance(
        value,
        str,
    ):
        raise TypeError("Situation kind must resolve to a string")

    return value


def _priority_tier(
    record: CommentaryIntelligenceRecord,
) -> str:
    value = getattr(
        record.priority.priority_tier,
        "value",
        record.priority.priority_tier,
    )

    if not isinstance(
        value,
        str,
    ):
        raise TypeError("Priority tier must resolve to a string")

    return value


def _build_records(
    *,
    directory: Path,
) -> tuple[
    NormalizedMatch,
    tuple[
        CommentaryIntelligenceRecord,
        ...,
    ],
    int,
]:
    match = _load_match(directory / "match.json")

    events = _load_events(directory / "events.jsonl")

    snapshots = _load_snapshots(directory / "participant_frames.jsonl")

    candidates = build_commentary_candidates(
        match_id=(directory.name),
        events=events,
    )

    situations = build_temporal_situations(
        match_id=(directory.name),
        candidates=candidates,
    )

    contexts = build_situation_state_contexts(
        match_id=(directory.name),
        situations=situations,
        snapshots=snapshots,
        participant_teams={
            participant.participant_id: (participant.team_id) for participant in match.participants
        },
    )

    priorities = build_situation_priorities(
        match_id=(directory.name),
        situations=situations,
        contexts=contexts,
    )

    records = build_commentary_intelligence_records(
        match_id=(directory.name),
        participants=tuple(match.participants),
        situations=situations,
        contexts=contexts,
        priorities=priorities,
    )

    return (
        match,
        records,
        len(candidates),
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

    if not match_directories:
        raise RuntimeError("No normalized matches found")

    total_candidates = 0
    total_records = 0

    patch_counts: Counter[str] = Counter()

    game_version_counts: Counter[str] = Counter()

    priority_counts: Counter[str] = Counter()

    kind_counts: Counter[str] = Counter()

    candidate_type_counts: Counter[str] = Counter()

    entity_count_distribution: Counter[int] = Counter()

    records_with_entities = 0
    records_without_entities = 0

    terminal_records = 0
    non_terminal_without_entities = 0

    records_with_objective_context = 0
    records_with_structure_context = 0

    policy_record_counts: Counter[str] = Counter()

    policy_retrieval_records: Counter[str] = Counter()

    policy_query_exposures: Counter[str] = Counter()

    policy_champion_query_exposures: Counter[str] = Counter()

    policy_system_query_exposures: Counter[str] = Counter()

    policy_unique_queries: dict[
        str,
        set[str],
    ] = {policy: set() for policy in POLICIES}

    policy_queries_per_record: dict[
        str,
        list[int],
    ] = {policy: [] for policy in POLICIES}

    unique_champion_keys: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    query_examples: dict[
        str,
        list[
            tuple[
                str,
                str,
                str,
                tuple[
                    str,
                    ...,
                ],
            ]
        ],
    ] = {policy: [] for policy in POLICIES}

    print("=== RAG QUERY REQUIREMENTS PROBE ===")

    print(f"Version mapping rule: {MAPPING_RULE}")

    print()

    for directory in match_directories:
        (
            match,
            records,
            candidate_count,
        ) = _build_records(directory=directory)

        game_version = _game_version(match)

        patch = _probe_patch_from_game_version(game_version)

        game_version_counts[game_version] += 1

        patch_counts[patch] += 1

        total_candidates += candidate_count

        total_records += len(records)

        print(f"MATCH={directory.name}")

        print(f"  game_version={game_version}")

        print(f"  resolved_patch={patch}")

        print(f"  records={len(records)}")

        match_policy_counts: Counter[str] = Counter()

        for record in records:
            priority = _priority_tier(record)

            kind = _situation_kind(record)

            candidate_types = _candidate_types(record)

            terminal = _is_terminal(candidate_types)

            entity_count = len(record.referenced_entities)

            priority_counts[priority] += 1

            kind_counts[kind] += 1

            entity_count_distribution[entity_count] += 1

            for candidate_type in candidate_types:
                candidate_type_counts[candidate_type] += 1

            if entity_count:
                records_with_entities += 1
            else:
                records_without_entities += 1

            if terminal:
                terminal_records += 1

            if not terminal and entity_count == 0:
                non_terminal_without_entities += 1

            candidate_type_set = set(candidate_types)

            if candidate_type_set & OBJECTIVE_CANDIDATE_TYPES:
                records_with_objective_context += 1

            if candidate_type_set & STRUCTURE_CANDIDATE_TYPES:
                records_with_structure_context += 1

            for entity in record.referenced_entities:
                unique_champion_keys.add(
                    (
                        patch,
                        entity.champion_name,
                    )
                )

            champion_queries = _champion_queries(
                patch=patch,
                record=record,
            )

            system_queries = _system_queries(
                patch=patch,
                candidate_types=(candidate_types),
            )

            planned_queries = _planned_queries(
                patch=patch,
                record=record,
            )

            for policy in POLICIES:
                if not _policy_allows(
                    policy=policy,
                    priority=priority,
                    terminal=terminal,
                ):
                    continue

                policy_record_counts[policy] += 1

                match_policy_counts[policy] += 1

                if planned_queries:
                    policy_retrieval_records[policy] += 1

                policy_query_exposures[policy] += len(planned_queries)

                policy_champion_query_exposures[policy] += len(champion_queries)

                policy_system_query_exposures[policy] += len(system_queries)

                policy_unique_queries[policy].update(planned_queries)

                policy_queries_per_record[policy].append(len(planned_queries))

                if planned_queries and len(query_examples[policy]) < 8:
                    query_examples[policy].append(
                        (
                            priority,
                            kind,
                            record.situation_id[:12],
                            planned_queries,
                        )
                    )

        for policy in POLICIES:
            print(f"  {policy}={match_policy_counts[policy]}")

        print()

    if total_candidates != 266:
        raise RuntimeError(
            f"Current Step 44 baseline expected 266 candidates, got {total_candidates}"
        )

    if total_records != 125:
        raise RuntimeError(
            f"Current Step 44 baseline expected 125 intelligence records, got {total_records}"
        )

    if sum(priority_counts.values()) != total_records:
        raise RuntimeError("Priority coverage mismatch")

    if sum(entity_count_distribution.values()) != total_records:
        raise RuntimeError("Entity-count coverage mismatch")

    if records_with_entities + records_without_entities != total_records:
        raise RuntimeError("Entity coverage accounting mismatch")

    print("=== GAME VERSION / PATCH ===")

    for game_version, count in sorted(game_version_counts.items()):
        print(f"  {game_version}: {count} match(es)")

    print("Resolved patch scope:")

    for patch, count in sorted(patch_counts.items()):
        print(f"  {patch}: {count} match(es)")

    print()

    print("=== RECORD DISTRIBUTION ===")

    print(f"Candidates: {total_candidates}")

    print(f"Intelligence records: {total_records}")

    print(f"Records with entities: {records_with_entities}")

    print(f"Records without entities: {records_without_entities}")

    print(f"Terminal records: {terminal_records}")

    print(f"Non-terminal records without entities: {non_terminal_without_entities}")

    print(f"Objective-context records: {records_with_objective_context}")

    print(f"Structure-context records: {records_with_structure_context}")

    print(f"Unique patch/champion keys: {len(unique_champion_keys)}")

    print()

    print("Priority tiers:")

    for tier, count in sorted(priority_counts.items()):
        print(f"  {tier}: {count}")

    print()

    print("Situation kinds:")

    for kind, count in sorted(kind_counts.items()):
        print(f"  {kind}: {count}")

    print()

    print("Candidate types by record:")

    for (
        candidate_type,
        count,
    ) in sorted(candidate_type_counts.items()):
        print(f"  {candidate_type}: {count}")

    print()

    print("Referenced entity counts per record:")

    for entity_count, count in sorted(entity_count_distribution.items()):
        print(f"  {entity_count}: {count}")

    print()

    print("=== RETRIEVAL POLICY COMPARISON ===")

    for policy in POLICIES:
        allowed_records = policy_record_counts[policy]

        retrieval_records = policy_retrieval_records[policy]

        query_exposures = policy_query_exposures[policy]

        champion_exposures = policy_champion_query_exposures[policy]

        system_exposures = policy_system_query_exposures[policy]

        unique_queries = len(policy_unique_queries[policy])

        query_counts = policy_queries_per_record[policy]

        average_queries = mean(query_counts) if query_counts else 0.0

        max_queries = max(query_counts) if query_counts else 0

        print(f"POLICY={policy}")

        print(f"  eligible_records={allowed_records}")

        print(f"  retrieval_records={retrieval_records}")

        print(f"  one_combined_call_upper_bound={retrieval_records}")

        print(f"  separate_query_exposures={query_exposures}")

        print(f"  champion_query_exposures={champion_exposures}")

        print(f"  system_query_exposures={system_exposures}")

        print(f"  globally_unique_queries={unique_queries}")

        print(f"  mean_queries_per_eligible_record={average_queries:.2f}")

        print(f"  max_queries_per_eligible_record={max_queries}")

        print()

    print("=== QUERY EXAMPLES ===")

    for policy in POLICIES:
        print(f"POLICY={policy}")

        examples = query_examples[policy]

        if not examples:
            print("  NONE")
            print()
            continue

        for (
            priority,
            kind,
            situation_prefix,
            queries,
        ) in examples:
            print(f"  [{priority}/{kind}/{situation_prefix}]")

            for query in queries:
                print(f"    - {query}")

        print()

    print("=== CURRENT CONTRACT GAP ===")

    print("Champion context available: YES")

    print("Team/position context available: YES")

    print("Objective event context available: YES")

    print("Structure event context available: YES")

    print("Item/inventory context available in CommentaryIntelligenceRecord: NO")

    print()

    print("NOTE: Step 45-A does not execute RetrievalService.")

    print("It measures query demand before freezing the production planner policy.")

    print()

    print("RAG_QUERY_REQUIREMENTS_PROBE=PASS")


if __name__ == "__main__":
    main()
