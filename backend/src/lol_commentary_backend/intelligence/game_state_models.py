from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GAME_STATE_CONTEXT_VERSION = "game_state_context_v1"


class FrameAlignmentMode(StrEnum):
    SAME_FRAME = "same_frame"
    DISTINCT_FRAMES = "distinct_frames"


class TeamMacroState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    team_id: int

    gold: int = Field(
        ge=0,
    )

    xp: int = Field(
        ge=0,
    )

    levels: int = Field(
        ge=0,
    )

    lane_cs: int = Field(
        ge=0,
    )

    jungle_cs: int = Field(
        ge=0,
    )


class GameStateSnapshot(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    frame_index: int = Field(
        ge=0,
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    teams: tuple[
        TeamMacroState,
        ...,
    ]


class SituationStateContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    context_version: Literal["game_state_context_v1"] = "game_state_context_v1"

    interpretation: Literal["macro_context_only"] = "macro_context_only"

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    situation_start_timestamp_ms: int = Field(
        ge=0,
    )

    situation_end_timestamp_ms: int = Field(
        ge=0,
    )

    before_state: GameStateSnapshot
    after_state: GameStateSnapshot

    alignment_mode: FrameAlignmentMode

    start_to_before_frame_ms: int = Field(
        ge=0,
    )

    end_to_after_frame_ms: int = Field(
        ge=0,
    )

    frame_interval_ms: int = Field(
        ge=0,
    )

    before_gold_diff_100_minus_200: int
    after_gold_diff_100_minus_200: int

    interval_gold_diff_change_100_minus_200: int | None

    before_leading_team_id: int | None
    after_leading_team_id: int | None

    leading_team_changed: bool
