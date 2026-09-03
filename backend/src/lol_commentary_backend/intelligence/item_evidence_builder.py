from __future__ import annotations

import json
from hashlib import sha256

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.item_evidence_models import (
    ITEM_EVIDENCE_CONTEXT_VERSION,
    ConfirmedItemEvidence,
    ItemEvidenceAction,
    ParticipantItemEvidence,
    SituationItemEvidenceContext,
)
from lol_commentary_backend.intelligence.record_models import (
    CommentaryIntelligenceRecord,
)

DEFAULT_MAX_ITEM_EVENTS_PER_PARTICIPANT = 5

ITEM_PURCHASED = "ITEM_PURCHASED"
ITEM_DESTROYED = "ITEM_DESTROYED"
ITEM_SOLD = "ITEM_SOLD"
ITEM_UNDO = "ITEM_UNDO"

ITEM_EVENT_TYPES = {
    ITEM_PURCHASED,
    ITEM_DESTROYED,
    ITEM_SOLD,
    ITEM_UNDO,
}


def _raw_event_type(
    event: NormalizedGameEvent,
) -> str:
    value = event.raw_event_type

    resolved = getattr(
        value,
        "value",
        value,
    )

    if not isinstance(
        resolved,
        str,
    ):
        raise TypeError("raw_event_type must resolve to string")

    return resolved


def _event_sort_key(
    event: NormalizedGameEvent,
) -> tuple[int, int, int]:
    return (
        event.sequence,
        event.frame_index,
        event.event_index,
    )


def _evidence_sort_key(
    evidence: ConfirmedItemEvidence,
) -> tuple[int, int, int, str]:
    return (
        evidence.sequence,
        evidence.frame_index,
        evidence.event_index,
        evidence.action.value,
    )


def _participant_sort_key(
    context: ParticipantItemEvidence,
) -> int:
    return context.participant_id


def _build_evidence(
    *,
    event: NormalizedGameEvent,
    action: ItemEvidenceAction,
    participant_id: int,
    item_id: int,
    situation_start_timestamp_ms: int,
) -> ConfirmedItemEvidence:
    if item_id <= 0:
        raise ValueError("item_id must be positive")

    if event.timestamp_ms > situation_start_timestamp_ms:
        raise ValueError("Cannot use future item event for situation item evidence")

    return ConfirmedItemEvidence(
        action=action,
        participant_id=participant_id,
        item_id=item_id,
        sequence=event.sequence,
        frame_index=event.frame_index,
        event_index=event.event_index,
        timestamp_ms=event.timestamp_ms,
        age_ms_at_situation_start=(situation_start_timestamp_ms - event.timestamp_ms),
        source_event_sha256=(event.source_event_sha256),
    )


def _standard_event_evidence(
    *,
    event: NormalizedGameEvent,
    participant_id: int,
    situation_start_timestamp_ms: int,
) -> tuple[
    ConfirmedItemEvidence,
    ...,
]:
    raw_event_type = _raw_event_type(event)

    item_id = event.item_id

    if item_id is None:
        return ()

    if item_id <= 0:
        return ()

    if raw_event_type == ITEM_PURCHASED:
        action = ItemEvidenceAction.PURCHASED

    elif raw_event_type == ITEM_DESTROYED:
        action = ItemEvidenceAction.DESTROYED

    elif raw_event_type == ITEM_SOLD:
        action = ItemEvidenceAction.SOLD

    else:
        return ()

    return (
        _build_evidence(
            event=event,
            action=action,
            participant_id=(participant_id),
            item_id=item_id,
            situation_start_timestamp_ms=(situation_start_timestamp_ms),
        ),
    )


def _undo_event_evidence(
    *,
    event: NormalizedGameEvent,
    participant_id: int,
    situation_start_timestamp_ms: int,
) -> tuple[
    ConfirmedItemEvidence,
    ...,
]:
    result: list[ConfirmedItemEvidence] = []

    before_item_id = event.before_item_id

    after_item_id = event.after_item_id

    if before_item_id is not None and before_item_id > 0:
        result.append(
            _build_evidence(
                event=event,
                action=(ItemEvidenceAction.UNDO_REMOVE),
                participant_id=(participant_id),
                item_id=(before_item_id),
                situation_start_timestamp_ms=(situation_start_timestamp_ms),
            )
        )

    if after_item_id is not None and after_item_id > 0:
        result.append(
            _build_evidence(
                event=event,
                action=(ItemEvidenceAction.UNDO_RESTORE),
                participant_id=(participant_id),
                item_id=(after_item_id),
                situation_start_timestamp_ms=(situation_start_timestamp_ms),
            )
        )

    return tuple(result)


def _event_evidence(
    *,
    event: NormalizedGameEvent,
    participant_id: int,
    situation_start_timestamp_ms: int,
) -> tuple[
    ConfirmedItemEvidence,
    ...,
]:
    raw_event_type = _raw_event_type(event)

    if raw_event_type == ITEM_UNDO:
        return _undo_event_evidence(
            event=event,
            participant_id=(participant_id),
            situation_start_timestamp_ms=(situation_start_timestamp_ms),
        )

    return _standard_event_evidence(
        event=event,
        participant_id=(participant_id),
        situation_start_timestamp_ms=(situation_start_timestamp_ms),
    )


def _context_identity_payload(
    *,
    record: CommentaryIntelligenceRecord,
    max_events_per_participant: int,
    participants: tuple[
        ParticipantItemEvidence,
        ...,
    ],
) -> dict[str, object]:
    return {
        "context_version": (ITEM_EVIDENCE_CONTEXT_VERSION),
        "match_id": (record.match_id),
        "record_id": (record.record_id),
        "situation_id": (record.situation_id),
        "situation_start_timestamp_ms": (record.start_timestamp_ms),
        "max_events_per_participant": (max_events_per_participant),
        "participants": [
            {
                "participant_id": (participant.participant_id),
                "evidence": [
                    {
                        "action": (evidence.action.value),
                        "item_id": (evidence.item_id),
                        "sequence": (evidence.sequence),
                        "source_event_sha256": (evidence.source_event_sha256),
                    }
                    for evidence in participant.evidence
                ],
            }
            for participant in participants
        ],
    }


def _context_id(
    *,
    record: CommentaryIntelligenceRecord,
    max_events_per_participant: int,
    participants: tuple[
        ParticipantItemEvidence,
        ...,
    ],
) -> str:
    payload = _context_identity_payload(
        record=record,
        max_events_per_participant=(max_events_per_participant),
        participants=participants,
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


def build_situation_item_evidence_context(
    *,
    record: CommentaryIntelligenceRecord,
    events: tuple[
        NormalizedGameEvent,
        ...,
    ],
    max_events_per_participant: int = (DEFAULT_MAX_ITEM_EVENTS_PER_PARTICIPANT),
) -> SituationItemEvidenceContext:
    if max_events_per_participant <= 0:
        raise ValueError("max_events_per_participant must be positive")

    referenced_participant_ids = tuple(
        sorted({entity.participant_id for entity in record.referenced_entities})
    )

    evidence_by_participant: dict[
        int,
        list[ConfirmedItemEvidence],
    ] = {participant_id: [] for participant_id in referenced_participant_ids}

    relevant_participant_ids = set(referenced_participant_ids)

    ordered_events = tuple(
        sorted(
            events,
            key=_event_sort_key,
        )
    )

    for event in ordered_events:
        raw_event_type = _raw_event_type(event)

        if raw_event_type not in ITEM_EVENT_TYPES:
            continue

        if event.timestamp_ms > record.start_timestamp_ms:
            continue

        participant_id = event.actor_participant_id

        # Riot가 participantId=0으로 제공한
        # system/unassigned item event는
        # 특정 플레이어 evidence로 추측 배정하지 않는다.
        if participant_id is None:
            continue

        if participant_id not in relevant_participant_ids:
            continue

        evidence_by_participant[participant_id].extend(
            _event_evidence(
                event=event,
                participant_id=(participant_id),
                situation_start_timestamp_ms=(record.start_timestamp_ms),
            )
        )

    participant_contexts: list[ParticipantItemEvidence] = []

    for participant_id in referenced_participant_ids:
        ordered = tuple(
            sorted(
                evidence_by_participant[participant_id],
                key=_evidence_sort_key,
            )
        )

        recent = ordered[-max_events_per_participant:]

        participant_contexts.append(
            ParticipantItemEvidence(
                participant_id=(participant_id),
                evidence=recent,
            )
        )

    participants = tuple(
        sorted(
            participant_contexts,
            key=_participant_sort_key,
        )
    )

    context_identifier = _context_id(
        record=record,
        max_events_per_participant=(max_events_per_participant),
        participants=participants,
    )

    return SituationItemEvidenceContext(
        context_id=(context_identifier),
        match_id=(record.match_id),
        record_id=(record.record_id),
        situation_id=(record.situation_id),
        situation_start_timestamp_ms=(record.start_timestamp_ms),
        max_events_per_participant=(max_events_per_participant),
        participants=participants,
    )


def build_situation_item_evidence_contexts(
    *,
    records: tuple[
        CommentaryIntelligenceRecord,
        ...,
    ],
    events: tuple[
        NormalizedGameEvent,
        ...,
    ],
    max_events_per_participant: int = (DEFAULT_MAX_ITEM_EVENTS_PER_PARTICIPANT),
) -> tuple[
    SituationItemEvidenceContext,
    ...,
]:
    result = tuple(
        build_situation_item_evidence_context(
            record=record,
            events=events,
            max_events_per_participant=(max_events_per_participant),
        )
        for record in records
    )

    context_ids = [context.context_id for context in result]

    if len(context_ids) != len(set(context_ids)):
        raise RuntimeError("Duplicate situation item evidence context_id")

    return result
