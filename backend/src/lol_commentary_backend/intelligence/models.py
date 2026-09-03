from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    MapPosition,
)

COMMENTARY_CANDIDATE_VERSION = "commentary_candidate_v1"


class CommentaryCandidateType(StrEnum):
    CHAMPION_KILL = "champion_kill"
    FIRST_BLOOD = "first_blood"
    MULTI_KILL = "multi_kill"
    ACE = "ace"
    ELITE_MONSTER = "elite_monster"
    BUILDING = "building"
    DRAGON_SOUL = "dragon_soul"
    GAME_END = "game_end"


class SalienceBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CommentaryCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    candidate_version: Literal["commentary_candidate_v1"] = "commentary_candidate_v1"

    candidate_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    source_sequence: int = Field(
        ge=0,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    timestamp_ms: int = Field(
        ge=0,
    )

    raw_event_type: str = Field(
        min_length=1,
    )

    candidate_type: CommentaryCandidateType

    salience_score: int = Field(
        ge=0,
        le=100,
    )

    salience_band: SalienceBand

    reasons: tuple[str, ...]

    actor_participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    target_participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    assisting_participant_ids: tuple[int, ...] = ()

    team_id: int | None = None
    killer_team_id: int | None = None
    winning_team_id: int | None = None

    position: MapPosition | None = None

    shutdown_bounty: int | None = None

    multi_kill_length: int | None = None
    kill_type: str | None = None

    monster_type: str | None = None
    monster_sub_type: str | None = None

    building_type: str | None = None
    tower_type: str | None = None
    lane_type: str | None = None

    objective_name: str | None = None
