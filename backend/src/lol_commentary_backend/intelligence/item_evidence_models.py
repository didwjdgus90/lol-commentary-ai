from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ITEM_EVIDENCE_CONTEXT_VERSION = "situation_item_evidence_v1"


class ItemEvidenceAction(StrEnum):
    PURCHASED = "purchased"
    DESTROYED = "destroyed"
    SOLD = "sold"
    UNDO_REMOVE = "undo_remove"
    UNDO_RESTORE = "undo_restore"


class ConfirmedItemEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    action: ItemEvidenceAction

    participant_id: int = Field(
        ge=1,
    )

    item_id: int = Field(
        ge=1,
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

    age_ms_at_situation_start: int = Field(
        ge=0,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class ParticipantItemEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
    )

    evidence: tuple[
        ConfirmedItemEvidence,
        ...,
    ]


class SituationItemEvidenceContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    context_version: Literal["situation_item_evidence_v1"] = ITEM_EVIDENCE_CONTEXT_VERSION

    context_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    situation_start_timestamp_ms: int = Field(
        ge=0,
    )

    max_events_per_participant: int = Field(
        ge=1,
    )

    participants: tuple[
        ParticipantItemEvidence,
        ...,
    ]

    evidence_authority: Literal["confirmed_timeline_event_only"] = "confirmed_timeline_event_only"

    exact_inventory_claim_allowed: Literal[False] = False
