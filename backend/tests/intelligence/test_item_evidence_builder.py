from __future__ import annotations

import pytest

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.item_evidence_builder import (
    build_situation_item_evidence_context,
)
from lol_commentary_backend.intelligence.item_evidence_models import (
    ItemEvidenceAction,
)
from lol_commentary_backend.intelligence.record_models import (
    CommentaryIntelligenceRecord,
    CommentaryParticipantEntity,
)


def _entity(
    participant_id: int,
) -> CommentaryParticipantEntity:
    return CommentaryParticipantEntity.model_construct(
        schema_version=1,
        participant_id=(participant_id),
        team_id=100,
        champion_id=1,
        champion_name="TestChampion",
        individual_position="TOP",
        team_position="TOP",
    )


def _record(
    *,
    start_timestamp_ms: int = 1000,
    participant_ids: tuple[
        int,
        ...,
    ] = (1,),
) -> CommentaryIntelligenceRecord:
    return CommentaryIntelligenceRecord.model_construct(
        schema_version=1,
        record_version=("commentary_intelligence_record_v1"),
        record_id=("a" * 64),
        match_id="KR_1",
        situation_id=("b" * 64),
        start_timestamp_ms=(start_timestamp_ms),
        end_timestamp_ms=(start_timestamp_ms),
        referenced_entities=tuple(_entity(participant_id) for participant_id in participant_ids),
    )


def _event(
    *,
    sequence: int,
    timestamp_ms: int,
    raw_event_type: str,
    participant_id: int | None,
    item_id: int | None = None,
    before_item_id: int | None = None,
    after_item_id: int | None = None,
) -> NormalizedGameEvent:
    return NormalizedGameEvent.model_construct(
        schema_version=1,
        sequence=sequence,
        frame_index=0,
        event_index=sequence,
        timestamp_ms=(timestamp_ms),
        raw_event_type=(raw_event_type),
        source_event_sha256=(f"{sequence + 1:064x}"),
        actor_participant_id=(participant_id),
        item_id=item_id,
        before_item_id=(before_item_id),
        after_item_id=(after_item_id),
    )


def test_purchase_is_confirmed_evidence() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(
            _event(
                sequence=0,
                timestamp_ms=900,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=3071,
            ),
        ),
    )

    evidence = context.participants[0].evidence

    assert len(evidence) == 1

    assert evidence[0].action == ItemEvidenceAction.PURCHASED

    assert evidence[0].item_id == 3071


def test_destroy_is_confirmed_evidence() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(
            _event(
                sequence=0,
                timestamp_ms=900,
                raw_event_type=("ITEM_DESTROYED"),
                participant_id=1,
                item_id=2003,
            ),
        ),
    )

    assert context.participants[0].evidence[0].action == ItemEvidenceAction.DESTROYED


def test_sell_is_confirmed_evidence() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(
            _event(
                sequence=0,
                timestamp_ms=900,
                raw_event_type=("ITEM_SOLD"),
                participant_id=1,
                item_id=1001,
            ),
        ),
    )

    assert context.participants[0].evidence[0].action == ItemEvidenceAction.SOLD


def test_undo_expands_to_remove_and_restore() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(
            _event(
                sequence=0,
                timestamp_ms=900,
                raw_event_type=("ITEM_UNDO"),
                participant_id=1,
                before_item_id=2003,
                after_item_id=1001,
            ),
        ),
    )

    evidence = context.participants[0].evidence

    assert tuple(item.action for item in evidence) == (
        ItemEvidenceAction.UNDO_REMOVE,
        ItemEvidenceAction.UNDO_RESTORE,
    )


def test_future_event_is_not_used() -> None:
    context = build_situation_item_evidence_context(
        record=_record(start_timestamp_ms=1000),
        events=(
            _event(
                sequence=0,
                timestamp_ms=1001,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=3071,
            ),
        ),
    )

    assert context.participants[0].evidence == ()


def test_unassigned_event_is_not_guessed() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(
            _event(
                sequence=0,
                timestamp_ms=0,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=None,
                item_id=3865,
            ),
        ),
    )

    assert context.participants[0].evidence == ()


def test_unreferenced_participant_is_excluded() -> None:
    context = build_situation_item_evidence_context(
        record=_record(participant_ids=(1,)),
        events=(
            _event(
                sequence=0,
                timestamp_ms=900,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=2,
                item_id=3071,
            ),
        ),
    )

    assert context.participants[0].participant_id == 1

    assert context.participants[0].evidence == ()


def test_only_latest_n_events_are_kept() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=tuple(
            _event(
                sequence=index,
                timestamp_ms=(100 + index),
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=(1000 + index),
            )
            for index in range(6)
        ),
        max_events_per_participant=3,
    )

    assert tuple(evidence.item_id for evidence in context.participants[0].evidence) == (
        1003,
        1004,
        1005,
    )


def test_context_id_is_deterministic() -> None:
    record = _record()

    events = (
        _event(
            sequence=0,
            timestamp_ms=900,
            raw_event_type=("ITEM_PURCHASED"),
            participant_id=1,
            item_id=3071,
        ),
    )

    first = build_situation_item_evidence_context(
        record=record,
        events=events,
    )

    second = build_situation_item_evidence_context(
        record=record,
        events=events,
    )

    assert first.context_id == second.context_id


def test_exact_inventory_claim_is_forbidden() -> None:
    context = build_situation_item_evidence_context(
        record=_record(),
        events=(),
    )

    assert context.evidence_authority == "confirmed_timeline_event_only"

    assert context.exact_inventory_claim_allowed is False


def test_invalid_event_limit_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("max_events_per_participant"),
    ):
        build_situation_item_evidence_context(
            record=_record(),
            events=(),
            max_events_per_participant=0,
        )
