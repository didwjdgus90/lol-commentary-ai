from __future__ import annotations

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.item_state_models import (
    InventoryReplayAnomaly,
    InventoryReplayAnomalyType,
    InventoryReplayResult,
    InventorySnapshot,
    InventoryTransition,
    InventoryTransitionType,
    ParticipantInventoryTimeline,
    UnassignedItemEvent,
)

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


def _event_sort_key(
    event: NormalizedGameEvent,
) -> tuple[int, int, int]:
    return (
        event.sequence,
        event.frame_index,
        event.event_index,
    )


def _transition_sort_key(
    transition: InventoryTransition,
) -> tuple[int, int, int]:
    return (
        transition.sequence,
        transition.frame_index,
        transition.event_index,
    )


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


def _sorted_items(
    item_ids: list[int],
) -> tuple[int, ...]:
    return tuple(sorted(item_ids))


def _append_anomaly(
    *,
    anomalies: list[InventoryReplayAnomaly],
    anomaly_type: InventoryReplayAnomalyType,
    event: NormalizedGameEvent,
    participant_id: int | None,
    item_id: int | None,
) -> None:
    anomalies.append(
        InventoryReplayAnomaly(
            anomaly_type=(anomaly_type),
            participant_id=(participant_id),
            item_id=item_id,
            sequence=(event.sequence),
            frame_index=(event.frame_index),
            event_index=(event.event_index),
            timestamp_ms=(event.timestamp_ms),
            raw_event_type=(_raw_event_type(event)),
            source_event_sha256=(event.source_event_sha256),
        )
    )


def _remove_one(
    *,
    inventory: list[int],
    item_id: int,
    participant_id: int,
    event: NormalizedGameEvent,
    anomalies: list[InventoryReplayAnomaly],
) -> bool:
    if item_id == 0:
        return False

    try:
        inventory.remove(item_id)

    except ValueError:
        _append_anomaly(
            anomalies=anomalies,
            anomaly_type=(InventoryReplayAnomalyType.ITEM_NOT_PRESENT),
            event=event,
            participant_id=(participant_id),
            item_id=item_id,
        )

        return False

    return True


def _add_item(
    *,
    inventory: list[int],
    item_id: int,
) -> bool:
    if item_id == 0:
        return False

    inventory.append(item_id)

    return True


def _unassigned_event(
    event: NormalizedGameEvent,
) -> UnassignedItemEvent:
    return UnassignedItemEvent(
        raw_event_type=(_raw_event_type(event)),
        sequence=(event.sequence),
        frame_index=(event.frame_index),
        event_index=(event.event_index),
        timestamp_ms=(event.timestamp_ms),
        item_id=(event.item_id),
        before_item_id=(event.before_item_id),
        after_item_id=(event.after_item_id),
        source_event_sha256=(event.source_event_sha256),
    )


def _apply_standard_item_event(
    *,
    event: NormalizedGameEvent,
    transition_type: InventoryTransitionType,
    inventory: list[int],
    participant_id: int,
    anomalies: list[InventoryReplayAnomaly],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
]:
    item_id = event.item_id

    if item_id is None:
        _append_anomaly(
            anomalies=anomalies,
            anomaly_type=(InventoryReplayAnomalyType.MISSING_ITEM_ID),
            event=event,
            participant_id=(participant_id),
            item_id=None,
        )

        return (
            (),
            (),
        )

    if transition_type == InventoryTransitionType.PURCHASE:
        added = _add_item(
            inventory=inventory,
            item_id=item_id,
        )

        return (
            (),
            (item_id,) if added else (),
        )

    removed = _remove_one(
        inventory=inventory,
        item_id=item_id,
        participant_id=(participant_id),
        event=event,
        anomalies=anomalies,
    )

    return (
        (item_id,) if removed else (),
        (),
    )


def _apply_undo(
    *,
    event: NormalizedGameEvent,
    inventory: list[int],
    participant_id: int,
    anomalies: list[InventoryReplayAnomaly],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
]:
    before_item_id = event.before_item_id

    after_item_id = event.after_item_id

    if before_item_id is None or after_item_id is None:
        _append_anomaly(
            anomalies=anomalies,
            anomaly_type=(InventoryReplayAnomalyType.MISSING_UNDO_ITEM_IDS),
            event=event,
            participant_id=(participant_id),
            item_id=None,
        )

        return (
            (),
            (),
        )

    removed_items: list[int] = []
    added_items: list[int] = []

    if before_item_id > 0:
        removed = _remove_one(
            inventory=inventory,
            item_id=before_item_id,
            participant_id=(participant_id),
            event=event,
            anomalies=anomalies,
        )

        if removed:
            removed_items.append(before_item_id)

    if after_item_id > 0:
        added = _add_item(
            inventory=inventory,
            item_id=after_item_id,
        )

        if added:
            added_items.append(after_item_id)

    return (
        tuple(removed_items),
        tuple(added_items),
    )


def _transition_type(
    raw_event_type: str,
) -> InventoryTransitionType:
    if raw_event_type == ITEM_PURCHASED:
        return InventoryTransitionType.PURCHASE

    if raw_event_type == ITEM_DESTROYED:
        return InventoryTransitionType.DESTROY

    if raw_event_type == ITEM_SOLD:
        return InventoryTransitionType.SELL

    if raw_event_type == ITEM_UNDO:
        return InventoryTransitionType.UNDO

    raise ValueError(f"Unsupported item event type: {raw_event_type}")


def build_inventory_state(
    *,
    match_id: str,
    participant_ids: tuple[
        int,
        ...,
    ],
    events: tuple[
        NormalizedGameEvent,
        ...,
    ],
    strict: bool = True,
) -> InventoryReplayResult:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    if not participant_ids:
        raise ValueError("participant_ids must not be empty")

    if any(participant_id <= 0 for participant_id in participant_ids):
        raise ValueError("participant_ids must be positive")

    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("participant_ids must be unique")

    known_participants = set(participant_ids)

    inventories: dict[
        int,
        list[int],
    ] = {participant_id: [] for participant_id in participant_ids}

    transitions: dict[
        int,
        list[InventoryTransition],
    ] = {participant_id: [] for participant_id in participant_ids}

    unassigned_events: list[UnassignedItemEvent] = []

    anomalies: list[InventoryReplayAnomaly] = []

    item_events = tuple(
        sorted(
            (event for event in events if (_raw_event_type(event) in ITEM_EVENT_TYPES)),
            key=_event_sort_key,
        )
    )

    sequences = [event.sequence for event in item_events]

    if len(sequences) != len(set(sequences)):
        raise ValueError("Item event sequences must be unique")

    for event in item_events:
        raw_event_type = _raw_event_type(event)

        participant_id = event.actor_participant_id

        if participant_id is None:
            unassigned_events.append(_unassigned_event(event))

            continue

        if participant_id not in known_participants:
            _append_anomaly(
                anomalies=anomalies,
                anomaly_type=(InventoryReplayAnomalyType.UNKNOWN_PARTICIPANT),
                event=event,
                participant_id=(participant_id),
                item_id=(event.item_id),
            )

            continue

        inventory = inventories[participant_id]

        before_items = _sorted_items(inventory)

        transition_type = _transition_type(raw_event_type)

        if transition_type == InventoryTransitionType.UNDO:
            (
                removed_items,
                added_items,
            ) = _apply_undo(
                event=event,
                inventory=inventory,
                participant_id=(participant_id),
                anomalies=anomalies,
            )

        else:
            (
                removed_items,
                added_items,
            ) = _apply_standard_item_event(
                event=event,
                transition_type=(transition_type),
                inventory=inventory,
                participant_id=(participant_id),
                anomalies=anomalies,
            )

        after_items = _sorted_items(inventory)

        transitions[participant_id].append(
            InventoryTransition(
                transition_type=(transition_type),
                sequence=(event.sequence),
                frame_index=(event.frame_index),
                event_index=(event.event_index),
                timestamp_ms=(event.timestamp_ms),
                participant_id=(participant_id),
                source_event_sha256=(event.source_event_sha256),
                before_item_ids=(before_items),
                removed_item_ids=(removed_items),
                added_item_ids=(added_items),
                after_item_ids=(after_items),
            )
        )

    timelines = tuple(
        ParticipantInventoryTimeline(
            participant_id=(participant_id),
            transitions=tuple(
                sorted(
                    transitions[participant_id],
                    key=(_transition_sort_key),
                )
            ),
            final_item_ids=(_sorted_items(inventories[participant_id])),
        )
        for participant_id in sorted(participant_ids)
    )

    result = InventoryReplayResult(
        match_id=clean_match_id,
        timelines=timelines,
        unassigned_events=tuple(unassigned_events),
        anomalies=tuple(anomalies),
    )

    if strict and result.anomalies:
        first = result.anomalies[0]

        raise ValueError(
            "Inventory replay anomaly: "
            f"{first.anomaly_type.value} "
            f"participant="
            f"{first.participant_id} "
            f"item={first.item_id} "
            f"sequence={first.sequence}"
        )

    return result


def inventory_snapshot_at_or_before(
    *,
    replay: InventoryReplayResult,
    participant_id: int,
    timestamp_ms: int,
) -> InventorySnapshot:
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must not be negative")

    timeline = next(
        (current for current in replay.timelines if (current.participant_id == participant_id)),
        None,
    )

    if timeline is None:
        raise ValueError(f"Unknown participant_id: {participant_id}")

    item_ids: tuple[
        int,
        ...,
    ] = ()

    applied_sequence: int | None = None

    for transition in timeline.transitions:
        if transition.timestamp_ms > timestamp_ms:
            break

        item_ids = transition.after_item_ids

        applied_sequence = transition.sequence

    return InventorySnapshot(
        participant_id=(participant_id),
        timestamp_ms=(timestamp_ms),
        applied_transition_sequence=(applied_sequence),
        item_ids=item_ids,
    )
