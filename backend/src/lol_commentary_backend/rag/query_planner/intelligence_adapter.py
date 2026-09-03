from __future__ import annotations

from enum import Enum

from lol_commentary_backend.intelligence.item_power_spike_selection_models import (
    ItemPowerSpikeSelectionContext,
)
from lol_commentary_backend.intelligence.record_models import (
    CommentaryIntelligenceRecord,
)
from lol_commentary_backend.intelligence.situation_models import (
    CommentaryCandidate,
)
from lol_commentary_backend.rag.query_planner.models import (
    RAGFocusChampion,
    RAGFocusRole,
    RAGPlannerInput,
    RAGPriorityTier,
    RAGSelectedItem,
)

MAX_FOCUS_CHAMPIONS = 2


_OBJECTIVE_RAW_EVENT_TYPES = {
    "ELITE_MONSTER_KILL",
    "DRAGON_SOUL_GIVEN",
}


_STRUCTURE_RAW_EVENT_TYPES = {
    "BUILDING_KILL",
}


def _enum_string(
    value: str | Enum,
) -> str:
    if isinstance(
        value,
        str,
    ):
        return value

    raw_value = value.value

    if not isinstance(
        raw_value,
        str,
    ):
        raise TypeError("Enum value must be string")

    return raw_value


def _candidate_sort_key(
    candidate: CommentaryCandidate,
) -> tuple[
    int,
    int,
    int,
    str,
]:
    return (
        -candidate.salience_score,
        candidate.timestamp_ms,
        candidate.source_sequence,
        candidate.candidate_id,
    )


def _focus_champions(
    record: CommentaryIntelligenceRecord,
) -> tuple[
    RAGFocusChampion,
    ...,
]:
    entities_by_participant = {
        entity.participant_id: entity for entity in record.referenced_entities
    }

    ordered_candidates = sorted(
        record.situation.primary_candidates,
        key=_candidate_sort_key,
    )

    result: list[RAGFocusChampion] = []

    seen_participants: set[int] = set()

    def add_participant(
        *,
        participant_id: int | None,
        role: RAGFocusRole,
    ) -> None:
        if participant_id is None:
            return

        if participant_id <= 0:
            return

        if participant_id in seen_participants:
            return

        entity = entities_by_participant.get(participant_id)

        if entity is None:
            raise ValueError(
                "Primary candidate participant "
                "has no referenced entity: "
                f"participant_id={participant_id}"
            )

        result.append(
            RAGFocusChampion(
                participant_id=(entity.participant_id),
                champion_id=(entity.champion_id),
                champion_name=(entity.champion_name),
                role=role,
            )
        )

        seen_participants.add(participant_id)

    for candidate in ordered_candidates:
        add_participant(
            participant_id=(candidate.actor_participant_id),
            role=RAGFocusRole.ACTOR,
        )

        if len(result) >= MAX_FOCUS_CHAMPIONS:
            break

        add_participant(
            participant_id=(candidate.target_participant_id),
            role=RAGFocusRole.TARGET,
        )

        if len(result) >= MAX_FOCUS_CHAMPIONS:
            break

    return tuple(result)


def _is_objective_candidate(
    candidate: CommentaryCandidate,
) -> bool:
    if candidate.raw_event_type in _OBJECTIVE_RAW_EVENT_TYPES:
        return True

    if candidate.monster_type is not None:
        return True

    if candidate.monster_sub_type is not None:
        return True

    if candidate.objective_name is not None:
        return True

    return False


def _is_structure_candidate(
    candidate: CommentaryCandidate,
) -> bool:
    if candidate.raw_event_type in _STRUCTURE_RAW_EVENT_TYPES:
        return True

    if candidate.building_type is not None:
        return True

    if candidate.tower_type is not None:
        return True

    return False


def _context_flags(
    record: CommentaryIntelligenceRecord,
) -> tuple[
    bool,
    bool,
]:
    objective_context = any(
        _is_objective_candidate(candidate) for candidate in record.situation.primary_candidates
    )

    structure_context = any(
        _is_structure_candidate(candidate) for candidate in record.situation.primary_candidates
    )

    return (
        objective_context,
        structure_context,
    )


def _priority_tier(
    record: CommentaryIntelligenceRecord,
) -> RAGPriorityTier:
    value = _enum_string(record.priority.priority_tier)

    return RAGPriorityTier(value)


def _situation_kind(
    record: CommentaryIntelligenceRecord,
) -> str:
    return _enum_string(record.situation.situation_kind)


def _selected_items(
    item_selection: ItemPowerSpikeSelectionContext,
) -> tuple[
    RAGSelectedItem,
    ...,
]:
    return tuple(
        RAGSelectedItem(
            participant_id=(signal.participant_id),
            item_id=(signal.item_id),
            item_name=(signal.name_ko),
            score=signal.score,
            tier=signal.tier.value,
            age_ms_at_situation_start=(signal.age_ms_at_situation_start),
            source_event_sha256=(signal.source_event_sha256),
        )
        for signal in item_selection.selected_signals
    )


def build_rag_planner_input(
    *,
    record: CommentaryIntelligenceRecord,
    item_selection: ItemPowerSpikeSelectionContext,
    patch: str,
    ddragon_version: str,
) -> RAGPlannerInput:
    clean_patch = patch.strip()

    clean_ddragon_version = ddragon_version.strip()

    if not clean_patch:
        raise ValueError("patch must not be empty")

    if not clean_ddragon_version:
        raise ValueError("ddragon_version must not be empty")

    if item_selection.record_id != record.record_id:
        raise ValueError("Item selection record ID does not match intelligence record")

    if item_selection.match_id != record.match_id:
        raise ValueError("Item selection match ID does not match intelligence record")

    if item_selection.situation_id != record.situation_id:
        raise ValueError("Item selection situation ID does not match intelligence record")

    if item_selection.exact_inventory_claim_allowed:
        raise ValueError("Item selection must forbid exact inventory claims")

    (
        objective_context,
        structure_context,
    ) = _context_flags(record)

    return RAGPlannerInput(
        record_id=(record.record_id),
        match_id=(record.match_id),
        situation_id=(record.situation_id),
        patch=clean_patch,
        ddragon_version=(clean_ddragon_version),
        priority_tier=(_priority_tier(record)),
        situation_kind=(_situation_kind(record)),
        focus_champions=(_focus_champions(record)),
        selected_items=(_selected_items(item_selection)),
        objective_context=(objective_context),
        structure_context=(structure_context),
    )
