from __future__ import annotations

import pytest

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.item_state_builder import (
    build_inventory_state,
    inventory_snapshot_at_or_before,
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


def test_purchase_adds_item() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=1001,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == (1001,)


def test_duplicate_items_are_preserved() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
            _event(
                sequence=1,
                timestamp_ms=200,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == (
        2003,
        2003,
    )


def test_destroy_removes_one_item() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
            _event(
                sequence=1,
                timestamp_ms=200,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
            _event(
                sequence=2,
                timestamp_ms=300,
                raw_event_type=("ITEM_DESTROYED"),
                participant_id=1,
                item_id=2003,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == (2003,)


def test_sell_removes_item() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=1001,
            ),
            _event(
                sequence=1,
                timestamp_ms=200,
                raw_event_type=("ITEM_SOLD"),
                participant_id=1,
                item_id=1001,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == ()


def test_undo_purchase_removes_before_item() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
            _event(
                sequence=1,
                timestamp_ms=200,
                raw_event_type=("ITEM_UNDO"),
                participant_id=1,
                before_item_id=2003,
                after_item_id=0,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == ()


def test_undo_can_restore_after_item() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_UNDO"),
                participant_id=1,
                before_item_id=0,
                after_item_id=3363,
            ),
        ),
    )

    assert replay.timelines[0].final_item_ids == (3363,)


def test_unassigned_item_event_is_audited() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
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

    assert len(replay.unassigned_events) == 1

    assert replay.timelines[0].final_item_ids == ()


def test_missing_inventory_item_creates_anomaly() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_DESTROYED"),
                participant_id=1,
                item_id=2003,
            ),
        ),
        strict=False,
    )

    assert len(replay.anomalies) == 1

    assert replay.anomalies[0].anomaly_type.value == "item_not_present"


def test_strict_replay_rejects_anomaly() -> None:
    with pytest.raises(
        ValueError,
        match=("Inventory replay anomaly"),
    ):
        build_inventory_state(
            match_id="KR_1",
            participant_ids=(1,),
            events=(
                _event(
                    sequence=0,
                    timestamp_ms=100,
                    raw_event_type=("ITEM_DESTROYED"),
                    participant_id=1,
                    item_id=2003,
                ),
            ),
        )


def test_inventory_snapshot_uses_time_boundary() -> None:
    replay = build_inventory_state(
        match_id="KR_1",
        participant_ids=(1,),
        events=(
            _event(
                sequence=0,
                timestamp_ms=100,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=1001,
            ),
            _event(
                sequence=1,
                timestamp_ms=300,
                raw_event_type=("ITEM_PURCHASED"),
                participant_id=1,
                item_id=2003,
            ),
        ),
    )

    snapshot = inventory_snapshot_at_or_before(
        replay=replay,
        participant_id=1,
        timestamp_ms=200,
    )

    assert snapshot.item_ids == (1001,)

    assert snapshot.applied_transition_sequence == 0
