from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

RIOT_NORMALIZER_VERSION = "riot_match_normalizer_v1"


class EventCategory(StrEnum):
    COMBAT = "combat"
    OBJECTIVE = "objective"
    ECONOMY = "economy"
    VISION = "vision"
    PROGRESSION = "progression"
    SYSTEM = "system"


class MapPosition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    x: int
    y: int


class ParticipantIdentity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    participant_id: int = Field(
        ge=1,
    )

    team_id: int = Field(
        ge=0,
    )

    champion_id: int = Field(
        ge=0,
    )

    champion_name: str = Field(
        min_length=1,
    )

    team_position: str | None = None
    individual_position: str | None = None

    win: bool


class ParticipantFrameSnapshot(BaseModel):
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

    participant_id: int = Field(
        ge=1,
    )

    level: int | None = Field(
        default=None,
        ge=0,
    )

    current_gold: int | None = Field(
        default=None,
        ge=0,
    )

    total_gold: int | None = Field(
        default=None,
        ge=0,
    )

    xp: int | None = Field(
        default=None,
        ge=0,
    )

    minions_killed: int | None = Field(
        default=None,
        ge=0,
    )

    jungle_minions_killed: int | None = Field(
        default=None,
        ge=0,
    )

    gold_per_second: int | None = Field(
        default=None,
        ge=0,
    )

    time_enemy_spent_controlled: int | float | None = Field(
        default=None,
        ge=0,
    )

    position: MapPosition | None = None


class NormalizedMatch(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    normalizer_version: str = Field(
        min_length=1,
    )

    match_id: str = Field(
        min_length=1,
    )

    game_mode: str = Field(
        min_length=1,
    )

    queue_id: int = Field(
        ge=0,
    )

    map_id: int = Field(
        ge=0,
    )

    game_duration_seconds: int = Field(
        ge=0,
    )

    game_version: str = Field(
        min_length=1,
    )

    source_match_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    source_timeline_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    participants: tuple[
        ParticipantIdentity,
        ...,
    ]


class NormalizedGameEvent(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

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

    category: EventCategory

    known_event_type: bool

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    actor_participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    target_participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    assisting_participant_ids: tuple[
        int,
        ...,
    ] = ()

    team_id: int | None = None
    killer_team_id: int | None = None
    winning_team_id: int | None = None

    position: MapPosition | None = None

    bounty: int | None = None
    shutdown_bounty: int | None = None

    kill_streak_length: int | None = None
    multi_kill_length: int | None = None
    kill_type: str | None = None

    item_id: int | None = None
    before_item_id: int | None = None
    after_item_id: int | None = None
    gold_gain: int | None = None

    level: int | None = None
    skill_slot: int | None = None
    level_up_type: str | None = None

    monster_type: str | None = None
    monster_sub_type: str | None = None

    building_type: str | None = None
    tower_type: str | None = None
    lane_type: str | None = None

    ward_type: str | None = None

    objective_name: str | None = None

    actual_start_time: int | None = None
    game_id: int | None = None
    real_timestamp: int | None = None

    unmapped_fields: tuple[
        str,
        ...,
    ] = ()


class NormalizedMatchBundle(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    match: NormalizedMatch

    participant_frames: tuple[
        ParticipantFrameSnapshot,
        ...,
    ]

    events: tuple[
        NormalizedGameEvent,
        ...,
    ]
