from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
)

TEMPORAL_SITUATION_VERSION = "temporal_situation_v1"


class SituationKind(StrEnum):
    COMBAT = "combat"
    OBJECTIVE = "objective"
    PUSH = "push"
    TERMINAL = "terminal"
    MIXED = "mixed"
    SPECIAL_ONLY = "special_only"


class TemporalSituation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    situation_version: Literal["temporal_situation_v1"] = "temporal_situation_v1"

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    start_timestamp_ms: int = Field(
        ge=0,
    )

    end_timestamp_ms: int = Field(
        ge=0,
    )

    duration_ms: int = Field(
        ge=0,
    )

    situation_kind: SituationKind

    max_salience_score: int = Field(
        ge=0,
        le=100,
    )

    primary_candidates: tuple[
        CommentaryCandidate,
        ...,
    ]

    semantic_markers: tuple[
        CommentaryCandidate,
        ...,
    ]
