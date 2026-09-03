from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    ParticipantIdentity,
)
from lol_commentary_backend.intelligence.game_state_models import (
    SituationStateContext,
)
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
)
from lol_commentary_backend.intelligence.priority_models import (
    SituationPriority,
)
from lol_commentary_backend.intelligence.record_models import (
    COMMENTARY_INTELLIGENCE_RECORD_VERSION,
    CommentaryIntelligenceRecord,
    CommentaryParticipantEntity,
    CommentaryRecordProvenance,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def _record_sort_key(
    record: CommentaryIntelligenceRecord,
) -> tuple[int, int, str]:
    return (
        record.start_timestamp_ms,
        record.end_timestamp_ms,
        record.record_id,
    )


def _entity_sort_key(
    entity: CommentaryParticipantEntity,
) -> int:
    return entity.participant_id


def _candidate_sort_key(
    candidate: CommentaryCandidate,
) -> tuple[int, int, str]:
    return (
        candidate.timestamp_ms,
        candidate.source_sequence,
        candidate.candidate_id,
    )


def _required_position(
    *,
    value: str | None,
    field_name: str,
    participant_id: int,
) -> str:
    if value is None:
        raise ValueError(f"{field_name} is missing for participant_id={participant_id}")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} is empty for participant_id={participant_id}")

    return normalized


def _candidate_participant_refs(
    candidate: CommentaryCandidate,
) -> set[int]:
    payload: dict[
        str,
        Any,
    ] = candidate.model_dump(mode="python")

    references: set[int] = set()

    for key, value in payload.items():
        if "participant" not in key.casefold():
            continue

        if isinstance(
            value,
            bool,
        ):
            continue

        if isinstance(
            value,
            int,
        ):
            if value > 0:
                references.add(value)

            continue

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
                    references.add(item)

    return references


def _all_candidates(
    situation: TemporalSituation,
) -> tuple[
    CommentaryCandidate,
    ...,
]:
    return tuple(
        sorted(
            (
                *situation.primary_candidates,
                *situation.semantic_markers,
            ),
            key=_candidate_sort_key,
        )
    )


def _participant_map(
    participants: tuple[
        ParticipantIdentity,
        ...,
    ],
) -> dict[
    int,
    ParticipantIdentity,
]:
    result: dict[
        int,
        ParticipantIdentity,
    ] = {}

    for participant in participants:
        participant_id = participant.participant_id

        if participant_id in result:
            raise ValueError(f"Duplicate participant_id: {participant_id}")

        result[participant_id] = participant

    return result


def _build_entities(
    *,
    situation: TemporalSituation,
    participants_by_id: dict[
        int,
        ParticipantIdentity,
    ],
) -> tuple[
    CommentaryParticipantEntity,
    ...,
]:
    participant_ids: set[int] = set()

    for candidate in _all_candidates(situation):
        participant_ids.update(_candidate_participant_refs(candidate))

    entities: list[CommentaryParticipantEntity] = []

    for participant_id in sorted(participant_ids):
        participant = participants_by_id.get(participant_id)

        if participant is None:
            raise ValueError(f"Situation references unknown participant_id: {participant_id}")

        entities.append(
            CommentaryParticipantEntity(
                participant_id=(participant.participant_id),
                team_id=(participant.team_id),
                champion_id=(participant.champion_id),
                champion_name=(participant.champion_name),
                individual_position=(
                    _required_position(
                        value=(participant.individual_position),
                        field_name=("individual_position"),
                        participant_id=(participant.participant_id),
                    )
                ),
                team_position=(
                    _required_position(
                        value=(participant.team_position),
                        field_name=("team_position"),
                        participant_id=(participant.participant_id),
                    )
                ),
            )
        )

    return tuple(
        sorted(
            entities,
            key=_entity_sort_key,
        )
    )


def _ordered_unique(
    values: tuple[
        str,
        ...,
    ],
) -> tuple[
    str,
    ...,
]:
    seen: set[str] = set()

    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)

        result.append(value)

    return tuple(result)


def _build_provenance(
    *,
    situation: TemporalSituation,
    priority: SituationPriority,
) -> CommentaryRecordProvenance:
    candidates = _all_candidates(situation)

    if not candidates:
        raise ValueError("Situation must contain at least one candidate")

    candidate_ids = _ordered_unique(tuple(candidate.candidate_id for candidate in candidates))

    source_event_sha256s = _ordered_unique(
        tuple(candidate.source_event_sha256 for candidate in candidates)
    )

    return CommentaryRecordProvenance(
        candidate_ids=(candidate_ids),
        source_event_sha256s=(source_event_sha256s),
        macro_interval_id=(priority.macro_context.macro_interval_id),
    )


def _record_identity_payload(
    *,
    situation: TemporalSituation,
    entities: tuple[
        CommentaryParticipantEntity,
        ...,
    ],
    game_state: SituationStateContext,
    priority: SituationPriority,
    provenance: CommentaryRecordProvenance,
) -> dict[str, object]:
    return {
        "record_version": (COMMENTARY_INTELLIGENCE_RECORD_VERSION),
        "match_id": (situation.match_id),
        "situation_id": (situation.situation_id),
        "start_timestamp_ms": (situation.start_timestamp_ms),
        "end_timestamp_ms": (situation.end_timestamp_ms),
        "candidate_ids": list(provenance.candidate_ids),
        "source_event_sha256s": list(provenance.source_event_sha256s),
        "entities": [
            {
                "participant_id": (entity.participant_id),
                "team_id": (entity.team_id),
                "champion_id": (entity.champion_id),
                "champion_name": (entity.champion_name),
                "individual_position": (entity.individual_position),
                "team_position": (entity.team_position),
            }
            for entity in entities
        ],
        "macro_interval_id": (provenance.macro_interval_id),
        "before_frame_index": (game_state.before_state.frame_index),
        "after_frame_index": (game_state.after_state.frame_index),
        "before_gold_diff": (game_state.before_gold_diff_100_minus_200),
        "after_gold_diff": (game_state.after_gold_diff_100_minus_200),
        "priority_tier": (priority.priority_tier.value),
        "priority_reasons": [reason.value for reason in priority.reasons],
    }


def _record_id(
    *,
    situation: TemporalSituation,
    entities: tuple[
        CommentaryParticipantEntity,
        ...,
    ],
    game_state: SituationStateContext,
    priority: SituationPriority,
    provenance: CommentaryRecordProvenance,
) -> str:
    payload = _record_identity_payload(
        situation=situation,
        entities=entities,
        game_state=game_state,
        priority=priority,
        provenance=provenance,
    )

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return sha256(canonical.encode("utf-8")).hexdigest()


def _validate_cross_contract(
    *,
    situation: TemporalSituation,
    game_state: SituationStateContext,
    priority: SituationPriority,
) -> None:
    if game_state.situation_id != situation.situation_id:
        raise ValueError("Situation/game-state ID mismatch")

    if priority.situation_id != situation.situation_id:
        raise ValueError("Situation/priority ID mismatch")

    if game_state.match_id != situation.match_id:
        raise ValueError("Situation/game-state match_id mismatch")

    if priority.match_id != situation.match_id:
        raise ValueError("Situation/priority match_id mismatch")

    if game_state.situation_start_timestamp_ms != situation.start_timestamp_ms:
        raise ValueError("Situation/game-state start timestamp mismatch")

    if game_state.situation_end_timestamp_ms != situation.end_timestamp_ms:
        raise ValueError("Situation/game-state end timestamp mismatch")

    if priority.event_salience_score != situation.max_salience_score:
        raise ValueError("Situation/priority salience mismatch")

    expected_member_count = len(situation.primary_candidates) + len(situation.semantic_markers)

    if priority.situation_member_count != expected_member_count:
        raise ValueError("Situation/priority member-count mismatch")

    if priority.semantic_marker_count != len(situation.semantic_markers):
        raise ValueError("Situation/priority semantic-marker mismatch")

    macro_context = priority.macro_context

    if macro_context.before_frame_index != game_state.before_state.frame_index:
        raise ValueError("Priority/game-state before-frame mismatch")

    if macro_context.after_frame_index != game_state.after_state.frame_index:
        raise ValueError("Priority/game-state after-frame mismatch")

    if macro_context.before_gold_diff_100_minus_200 != game_state.before_gold_diff_100_minus_200:
        raise ValueError("Priority/game-state before-gold mismatch")

    if macro_context.after_gold_diff_100_minus_200 != game_state.after_gold_diff_100_minus_200:
        raise ValueError("Priority/game-state after-gold mismatch")

    if (
        macro_context.interval_gold_diff_change_100_minus_200
        != game_state.interval_gold_diff_change_100_minus_200
    ):
        raise ValueError("Priority/game-state interval-gold mismatch")

    if macro_context.leading_team_changed != game_state.leading_team_changed:
        raise ValueError("Priority/game-state lead-change mismatch")


def build_commentary_intelligence_records(
    *,
    match_id: str,
    participants: tuple[
        ParticipantIdentity,
        ...,
    ],
    situations: tuple[
        TemporalSituation,
        ...,
    ],
    contexts: tuple[
        SituationStateContext,
        ...,
    ],
    priorities: tuple[
        SituationPriority,
        ...,
    ],
) -> tuple[
    CommentaryIntelligenceRecord,
    ...,
]:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    participants_by_id = _participant_map(participants)

    if not participants_by_id:
        raise ValueError("participants must not be empty")

    ordered_situations = tuple(
        sorted(
            situations,
            key=_situation_sort_key,
        )
    )

    situations_by_id = {situation.situation_id: situation for situation in ordered_situations}

    if len(situations_by_id) != len(ordered_situations):
        raise ValueError("Duplicate situation_id")

    contexts_by_id = {context.situation_id: context for context in contexts}

    if len(contexts_by_id) != len(contexts):
        raise ValueError("Duplicate game-state situation_id")

    priorities_by_id = {priority.situation_id: priority for priority in priorities}

    if len(priorities_by_id) != len(priorities):
        raise ValueError("Duplicate priority situation_id")

    expected_ids = set(situations_by_id)

    if set(contexts_by_id) != expected_ids:
        raise ValueError("Situation/game-state coverage mismatch")

    if set(priorities_by_id) != expected_ids:
        raise ValueError("Situation/priority coverage mismatch")

    records: list[CommentaryIntelligenceRecord] = []

    for situation in ordered_situations:
        if situation.match_id != clean_match_id:
            raise ValueError("Situation match_id does not match")

        game_state = contexts_by_id[situation.situation_id]

        priority = priorities_by_id[situation.situation_id]

        _validate_cross_contract(
            situation=situation,
            game_state=game_state,
            priority=priority,
        )

        entities = _build_entities(
            situation=situation,
            participants_by_id=(participants_by_id),
        )

        provenance = _build_provenance(
            situation=situation,
            priority=priority,
        )

        record_identifier = _record_id(
            situation=situation,
            entities=entities,
            game_state=game_state,
            priority=priority,
            provenance=provenance,
        )

        records.append(
            CommentaryIntelligenceRecord(
                record_id=(record_identifier),
                match_id=(clean_match_id),
                situation_id=(situation.situation_id),
                start_timestamp_ms=(situation.start_timestamp_ms),
                end_timestamp_ms=(situation.end_timestamp_ms),
                situation=situation,
                referenced_entities=(entities),
                game_state=(game_state),
                priority=(priority),
                provenance=(provenance),
            )
        )

    result = tuple(
        sorted(
            records,
            key=_record_sort_key,
        )
    )

    if len(result) != len(ordered_situations):
        raise RuntimeError("Commentary record coverage mismatch")

    record_ids = [record.record_id for record in result]

    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate commentary record_id")

    return result
