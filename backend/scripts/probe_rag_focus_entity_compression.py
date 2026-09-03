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

TERMINAL_TYPES = {
    "game_end",
}

OBJECTIVE_TYPES = {
    "elite_monster",
    "dragon_soul",
}

STRUCTURE_TYPES = {
    "building",
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


def _enum_string(
    value: object,
) -> str:
    resolved = getattr(
        value,
        "value",
        value,
    )

    if not isinstance(
        resolved,
        str,
    ):
        raise TypeError("Expected enum/string value")

    return resolved


def _candidate_type(
    candidate: object,
) -> str:
    value = getattr(
        candidate,
        "candidate_type",
        None,
    )

    return _enum_string(value)


def _candidate_types(
    record: CommentaryIntelligenceRecord,
) -> tuple[str, ...]:
    candidates = (
        *record.situation.primary_candidates,
        *record.situation.semantic_markers,
    )

    return tuple(sorted({_candidate_type(candidate) for candidate in candidates}))


def _priority(
    record: CommentaryIntelligenceRecord,
) -> str:
    return _enum_string(record.priority.priority_tier)


def _is_terminal(
    candidate_types: tuple[
        str,
        ...,
    ],
) -> bool:
    return bool(TERMINAL_TYPES.intersection(candidate_types))


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
        raise ValueError(f"Unknown priority: {priority}")

    if policy == "all_non_terminal":
        return True

    if policy == "medium_plus":
        return rank >= PRIORITY_RANK["medium"]

    if policy == "high_plus":
        return rank >= PRIORITY_RANK["high"]

    raise ValueError(f"Unknown policy: {policy}")


def _participant_ids_from_value(
    value: object,
) -> set[int]:
    result: set[int] = set()

    if isinstance(
        value,
        bool,
    ):
        return result

    if isinstance(
        value,
        int,
    ):
        if value > 0:
            result.add(value)

        return result

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        for item in value:
            if isinstance(
                item,
                bool,
            ):
                continue

            if (
                isinstance(
                    item,
                    int,
                )
                and item > 0
            ):
                result.add(item)

    return result


def _participant_role(
    key: str,
) -> str:
    lowered = key.casefold()

    if "assist" in lowered:
        return "assist"

    if "target" in lowered or "victim" in lowered:
        return "target"

    if "actor" in lowered or "killer" in lowered:
        return "actor"

    return "other"


def _candidate_participant_roles(
    candidate: object,
) -> dict[
    str,
    set[int],
]:
    model_dump = getattr(
        candidate,
        "model_dump",
        None,
    )

    if not callable(model_dump):
        raise TypeError("Candidate does not support model_dump")

    payload: dict[
        str,
        Any,
    ] = model_dump(mode="python")

    result: dict[
        str,
        set[int],
    ] = {
        "actor": set(),
        "target": set(),
        "assist": set(),
        "other": set(),
    }

    for key, value in payload.items():
        if "participant" not in key.casefold():
            continue

        participant_ids = _participant_ids_from_value(value)

        if not participant_ids:
            continue

        role = _participant_role(key)

        result[role].update(participant_ids)

    return result


def _record_role_sets(
    record: CommentaryIntelligenceRecord,
) -> dict[
    str,
    set[int],
]:
    result: dict[
        str,
        set[int],
    ] = {
        "actor": set(),
        "target": set(),
        "assist": set(),
        "other": set(),
    }

    candidates = (
        *record.situation.primary_candidates,
        *record.situation.semantic_markers,
    )

    for candidate in candidates:
        roles = _candidate_participant_roles(candidate)

        for role in result:
            result[role].update(roles[role])

    return result


def _focus_participant_ids(
    *,
    role_sets: dict[
        str,
        set[int],
    ],
) -> set[int]:
    # Production candidate for the next step:
    #
    # Direct actor + direct target are preferred.
    #
    # Assist/other references are NOT automatically
    # promoted into RAG queries because they can cause
    # severe query expansion during team fights.
    return role_sets["actor"] | role_sets["target"]


def _entity_name_map(
    record: CommentaryIntelligenceRecord,
) -> dict[int, str]:
    return {entity.participant_id: (entity.champion_name) for entity in record.referenced_entities}


def _champion_names(
    *,
    participant_ids: set[int],
    entity_names: dict[
        int,
        str,
    ],
) -> tuple[str, ...]:
    unresolved = participant_ids - set(entity_names)

    if unresolved:
        raise RuntimeError(
            f"Participant references missing from intelligence record: {sorted(unresolved)}"
        )

    return tuple(sorted({entity_names[participant_id] for participant_id in participant_ids}))


def _system_query_count(
    candidate_types: tuple[
        str,
        ...,
    ],
) -> int:
    candidate_type_set = set(candidate_types)

    result = 0

    if candidate_type_set & OBJECTIVE_TYPES:
        result += 1

    if candidate_type_set & STRUCTURE_TYPES:
        result += 1

    return result


def _system_query_labels(
    candidate_types: tuple[
        str,
        ...,
    ],
) -> tuple[str, ...]:
    candidate_type_set = set(candidate_types)

    result: list[str] = []

    if candidate_type_set & OBJECTIVE_TYPES:
        result.append("system:objective")

    if candidate_type_set & STRUCTURE_TYPES:
        result.append("system:structure")

    return tuple(result)


def _build_records(
    directory: Path,
) -> tuple[
    CommentaryIntelligenceRecord,
    ...,
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

    return build_commentary_intelligence_records(
        match_id=(directory.name),
        participants=tuple(match.participants),
        situations=situations,
        contexts=contexts,
        priorities=priorities,
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
        raise RuntimeError("Expected current 3-match Step 44 baseline")

    total_records = 0

    role_reference_exposures: Counter[str] = Counter()

    role_records: Counter[str] = Counter()

    participant_key_counts: Counter[str] = Counter()

    participant_keys_by_type: dict[
        str,
        Counter[str],
    ] = {}

    policy_eligible: Counter[str] = Counter()

    policy_naive_queries: Counter[str] = Counter()

    policy_focus_queries: Counter[str] = Counter()

    policy_naive_champion_queries: Counter[str] = Counter()

    policy_focus_champion_queries: Counter[str] = Counter()

    policy_system_queries: Counter[str] = Counter()

    policy_naive_per_record: dict[
        str,
        list[int],
    ] = {policy: [] for policy in POLICIES}

    policy_focus_per_record: dict[
        str,
        list[int],
    ] = {policy: [] for policy in POLICIES}

    policy_unique_naive: dict[
        str,
        set[str],
    ] = {policy: set() for policy in POLICIES}

    policy_unique_focus: dict[
        str,
        set[str],
    ] = {policy: set() for policy in POLICIES}

    focus_zero_records: Counter[str] = Counter()

    focus_examples: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []

    print("=== RAG FOCUS ENTITY COMPRESSION PROBE ===")

    print()

    for directory in match_directories:
        records = _build_records(directory)

        total_records += len(records)

        print(f"MATCH={directory.name}")

        print(f"  records={len(records)}")

        for record in records:
            candidate_types = _candidate_types(record)

            terminal = _is_terminal(candidate_types)

            priority = _priority(record)

            role_sets = _record_role_sets(record)

            entity_names = _entity_name_map(record)

            all_entity_ids = set(entity_names)

            focus_ids = _focus_participant_ids(role_sets=role_sets)

            if not focus_ids.issubset(all_entity_ids):
                raise RuntimeError(
                    "Focus participant is not present in intelligence record entities"
                )

            for role, refs in role_sets.items():
                role_reference_exposures[role] += len(refs)

                if refs:
                    role_records[role] += 1

            candidates = (
                *record.situation.primary_candidates,
                *record.situation.semantic_markers,
            )

            for candidate in candidates:
                candidate_type = _candidate_type(candidate)

                model_dump = getattr(
                    candidate,
                    "model_dump",
                    None,
                )

                if not callable(model_dump):
                    raise TypeError("Candidate does not support model_dump")

                payload: dict[
                    str,
                    Any,
                ] = model_dump(mode="python")

                if candidate_type not in participant_keys_by_type:
                    participant_keys_by_type[candidate_type] = Counter()

                for key, value in payload.items():
                    if "participant" not in key.casefold():
                        continue

                    refs = _participant_ids_from_value(value)

                    if not refs:
                        continue

                    participant_key_counts[key] += 1

                    participant_keys_by_type[candidate_type][key] += 1

            all_names = _champion_names(
                participant_ids=(all_entity_ids),
                entity_names=(entity_names),
            )

            focus_names = _champion_names(
                participant_ids=(focus_ids),
                entity_names=(entity_names),
            )

            assist_names = _champion_names(
                participant_ids=(role_sets["assist"] & all_entity_ids),
                entity_names=(entity_names),
            )

            system_count = _system_query_count(candidate_types)

            system_labels = _system_query_labels(candidate_types)

            for policy in POLICIES:
                if not _policy_allows(
                    policy=policy,
                    priority=priority,
                    terminal=terminal,
                ):
                    continue

                policy_eligible[policy] += 1

                naive_champion_count = len(all_names)

                focus_champion_count = len(focus_names)

                naive_total = naive_champion_count + system_count

                focus_total = focus_champion_count + system_count

                policy_naive_queries[policy] += naive_total

                policy_focus_queries[policy] += focus_total

                policy_naive_champion_queries[policy] += naive_champion_count

                policy_focus_champion_queries[policy] += focus_champion_count

                policy_system_queries[policy] += system_count

                policy_naive_per_record[policy].append(naive_total)

                policy_focus_per_record[policy].append(focus_total)

                for champion in all_names:
                    policy_unique_naive[policy].add(f"champion:{champion}")

                for champion in focus_names:
                    policy_unique_focus[policy].add(f"champion:{champion}")

                policy_unique_naive[policy].update(system_labels)

                policy_unique_focus[policy].update(system_labels)

                if focus_total == 0:
                    focus_zero_records[policy] += 1

            if len(focus_examples) < 12 and (len(all_names) >= 4 or len(assist_names) >= 2):
                focus_examples.append(
                    (
                        priority,
                        record.situation.situation_id[:12],
                        all_names,
                        focus_names,
                        assist_names,
                    )
                )

        print()

    if total_records != 125:
        raise RuntimeError(f"Expected 125 Step 44 intelligence records, got {total_records}")

    print("=== PARTICIPANT ROLE REFERENCES ===")

    for role in (
        "actor",
        "target",
        "assist",
        "other",
    ):
        print(f"{role}:")

        print(f"  records={role_records[role]}")

        print(f"  unique_ref_exposures={role_reference_exposures[role]}")

    print()

    print("=== PARTICIPANT FIELD COVERAGE ===")

    for key, count in sorted(participant_key_counts.items()):
        print(f"  {key}: {count}")

    print()

    print("=== PARTICIPANT FIELDS BY CANDIDATE TYPE ===")

    for candidate_type in sorted(participant_keys_by_type):
        print(f"{candidate_type}:")

        for key, count in sorted(participant_keys_by_type[candidate_type].items()):
            print(f"  {key}: {count}")

    print()

    print("=== QUERY COMPRESSION ===")

    for policy in POLICIES:
        eligible = policy_eligible[policy]

        naive_total = policy_naive_queries[policy]

        focus_total = policy_focus_queries[policy]

        naive_champion = policy_naive_champion_queries[policy]

        focus_champion = policy_focus_champion_queries[policy]

        system_queries = policy_system_queries[policy]

        naive_counts = policy_naive_per_record[policy]

        focus_counts = policy_focus_per_record[policy]

        reduction = 0.0

        if naive_total:
            reduction = (1.0 - (focus_total / naive_total)) * 100.0

        naive_mean = mean(naive_counts) if naive_counts else 0.0

        focus_mean = mean(focus_counts) if focus_counts else 0.0

        naive_max = max(naive_counts) if naive_counts else 0

        focus_max = max(focus_counts) if focus_counts else 0

        print(f"POLICY={policy}")

        print(f"  eligible_records={eligible}")

        print(f"  naive_total_queries={naive_total}")

        print(f"  focus_total_queries={focus_total}")

        print(f"  query_reduction={reduction:.2f}%")

        print(f"  naive_champion_queries={naive_champion}")

        print(f"  focus_champion_queries={focus_champion}")

        print(f"  system_queries={system_queries}")

        print(f"  naive_mean_per_record={naive_mean:.2f}")

        print(f"  focus_mean_per_record={focus_mean:.2f}")

        print(f"  naive_max_per_record={naive_max}")

        print(f"  focus_max_per_record={focus_max}")

        print(f"  naive_unique_targets={len(policy_unique_naive[policy])}")

        print(f"  focus_unique_targets={len(policy_unique_focus[policy])}")

        print(f"  zero_query_records={focus_zero_records[policy]}")

        print()

    print("=== FOCUS EXAMPLES ===")

    for (
        priority,
        situation_id,
        all_names,
        focus_names,
        assist_names,
    ) in focus_examples:
        print(f"[{priority}/{situation_id}]")

        print("  all=" + ", ".join(all_names))

        print("  focus=" + (", ".join(focus_names) if focus_names else "NONE"))

        print("  assists=" + (", ".join(assist_names) if assist_names else "NONE"))

        print()

    print("NOTE:")

    print("This probe does not freeze the production focus policy.")

    print(
        "It measures whether direct "
        "actor/target selection can "
        "reduce retrieval noise before "
        "RAG QueryPlan v1 is implemented."
    )

    print()

    print("RAG_FOCUS_ENTITY_COMPRESSION_PROBE=PASS")


if __name__ == "__main__":
    main()
