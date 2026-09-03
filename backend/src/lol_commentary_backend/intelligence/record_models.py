from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.intelligence.game_state_models import (
    SituationStateContext,
)
from lol_commentary_backend.intelligence.priority_models import (
    SituationPriority,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)

COMMENTARY_INTELLIGENCE_RECORD_VERSION = "commentary_intelligence_record_v1"


class CommentaryParticipantEntity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
    )

    team_id: int = Field(
        ge=1,
    )

    champion_id: int = Field(
        ge=0,
    )

    champion_name: str = Field(
        min_length=1,
    )

    individual_position: str = Field(
        min_length=1,
    )

    team_position: str = Field(
        min_length=1,
    )


class CommentaryRecordProvenance(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    event_interpretation: Literal["normalized_event_facts"] = "normalized_event_facts"

    macro_interpretation: Literal["shared_macro_context_only"] = "shared_macro_context_only"

    causal_policy: Literal["do_not_attribute_macro_interval_delta_to_single_situation"] = (
        "do_not_attribute_macro_interval_delta_to_single_situation"
    )

    contains_player_pii: Literal[False] = False

    candidate_ids: tuple[
        str,
        ...,
    ]

    source_event_sha256s: tuple[
        str,
        ...,
    ]

    macro_interval_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class CommentaryIntelligenceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    record_version: Literal["commentary_intelligence_record_v1"] = (
        "commentary_intelligence_record_v1"
    )

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    start_timestamp_ms: int = Field(
        ge=0,
    )

    end_timestamp_ms: int = Field(
        ge=0,
    )

    situation: TemporalSituation

    referenced_entities: tuple[
        CommentaryParticipantEntity,
        ...,
    ]

    game_state: SituationStateContext

    priority: SituationPriority

    provenance: CommentaryRecordProvenance
