from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

INVENTORY_STATE_VERSION = "inventory_state_v1"


class InventoryTransitionType(StrEnum):
    PURCHASE = "purchase"
    DESTROY = "destroy"
    SELL = "sell"
    UNDO = "undo"


class InventoryReplayAnomalyType(StrEnum):
    UNKNOWN_PARTICIPANT = "unknown_participant"
    MISSING_ITEM_ID = "missing_item_id"
    MISSING_UNDO_ITEM_IDS = "missing_undo_item_ids"
    ITEM_NOT_PRESENT = "item_not_present"


class InventoryTransition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    transition_type: InventoryTransitionType

    sequence: int = Field(
        ge=0,
    )

    frame_index: int = Field(
        ge=0,
    )

    event_index: int = Field(
        ge=0,
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    participant_id: int = Field(
        ge=1,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    before_item_ids: tuple[
        int,
        ...,
    ]

    removed_item_ids: tuple[
        int,
        ...,
    ]

    added_item_ids: tuple[
        int,
        ...,
    ]

    after_item_ids: tuple[
        int,
        ...,
    ]


class UnassignedItemEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    raw_event_type: str = Field(
        min_length=1,
    )

    sequence: int = Field(
        ge=0,
    )

    frame_index: int = Field(
        ge=0,
    )

    event_index: int = Field(
        ge=0,
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    item_id: int | None = Field(
        default=None,
        ge=0,
    )

    before_item_id: int | None = Field(
        default=None,
        ge=0,
    )

    after_item_id: int | None = Field(
        default=None,
        ge=0,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class InventoryReplayAnomaly(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    anomaly_type: InventoryReplayAnomalyType

    participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    item_id: int | None = Field(
        default=None,
        ge=0,
    )

    sequence: int = Field(
        ge=0,
    )

    frame_index: int = Field(
        ge=0,
    )

    event_index: int = Field(
        ge=0,
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    raw_event_type: str = Field(
        min_length=1,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class ParticipantInventoryTimeline(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
    )

    transitions: tuple[
        InventoryTransition,
        ...,
    ]

    final_item_ids: tuple[
        int,
        ...,
    ]


class InventorySnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    applied_transition_sequence: int | None = Field(
        default=None,
        ge=0,
    )

    item_ids: tuple[
        int,
        ...,
    ]


class InventoryReplayResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    inventory_version: Literal["inventory_state_v1"] = INVENTORY_STATE_VERSION

    match_id: str = Field(
        min_length=1,
    )

    timelines: tuple[
        ParticipantInventoryTimeline,
        ...,
    ]

    unassigned_events: tuple[
        UnassignedItemEvent,
        ...,
    ]

    anomalies: tuple[
        InventoryReplayAnomaly,
        ...,
    ]
