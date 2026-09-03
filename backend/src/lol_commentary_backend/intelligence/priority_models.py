from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SITUATION_PRIORITY_VERSION = "situation_priority_v1"

MACRO_GOLD_P90_THRESHOLD = 1_793


class SituationPriorityTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PriorityReason(StrEnum):
    TERMINAL_EVENT = "terminal_event"

    EXCEPTIONAL_EVENT_SALIENCE = "exceptional_event_salience"

    HIGH_EVENT_SALIENCE = "high_event_salience"

    MULTI_EVENT_SITUATION = "multi_event_situation"

    LARGE_MULTI_EVENT_SITUATION = "large_multi_event_situation"

    SEMANTIC_MARKER_PRESENT = "semantic_marker_present"

    MACRO_INTERVAL_ANCHOR = "macro_interval_anchor"

    MACRO_LEAD_FLIP = "macro_lead_flip"

    MACRO_GOLD_P90 = "macro_gold_p90"

    ROUTINE_EVENT = "routine_event"


class MacroContextReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    interpretation: Literal["shared_macro_context_only"] = "shared_macro_context_only"

    macro_interval_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    before_frame_index: int = Field(
        ge=0,
    )

    after_frame_index: int = Field(
        ge=0,
    )

    before_timestamp_ms: int = Field(
        ge=0,
    )

    after_timestamp_ms: int = Field(
        ge=0,
    )

    frame_interval_ms: int = Field(
        ge=0,
    )

    before_gold_diff_100_minus_200: int

    after_gold_diff_100_minus_200: int

    interval_gold_diff_change_100_minus_200: int | None

    leading_team_changed: bool

    large_macro_gold_change: bool

    shared_situation_count: int = Field(
        ge=1,
    )

    anchor_situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class SituationPriority(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    priority_version: Literal["situation_priority_v1"] = "situation_priority_v1"

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    priority_tier: SituationPriorityTier

    event_salience_score: int = Field(
        ge=0,
        le=100,
    )

    situation_member_count: int = Field(
        ge=1,
    )

    semantic_marker_count: int = Field(
        ge=0,
    )

    macro_context: MacroContextReference

    is_macro_anchor: bool

    reasons: tuple[
        PriorityReason,
        ...,
    ]
