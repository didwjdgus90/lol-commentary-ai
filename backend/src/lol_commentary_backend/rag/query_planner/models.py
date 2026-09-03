from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RAG_QUERY_PLAN_VERSION = "rag_query_plan_v1"

RAG_QUERY_PLANNER_POLICY_VERSION = "representative_actor_target_item_system_v1"


class RAGPriorityTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RAGFocusRole(StrEnum):
    ACTOR = "actor"
    TARGET = "target"


class RAGQueryIntent(StrEnum):
    CHAMPION_PATCH = "champion_patch"
    ITEM_PATCH = "item_patch"
    OBJECTIVE_PATCH = "objective_patch"
    STRUCTURE_PATCH = "structure_patch"


class RAGFocusChampion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    participant_id: int = Field(
        ge=1,
    )

    champion_id: int = Field(
        ge=1,
    )

    champion_name: str = Field(
        min_length=1,
    )

    role: RAGFocusRole


class RAGSelectedItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    participant_id: int = Field(
        ge=1,
    )

    item_id: int = Field(
        ge=1,
    )

    item_name: str = Field(
        min_length=1,
    )

    score: int = Field(
        ge=0,
        le=100,
    )

    tier: Literal[
        "medium",
        "high",
    ]

    age_ms_at_situation_start: int = Field(
        ge=0,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class RAGPlannerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    priority_tier: RAGPriorityTier

    situation_kind: str = Field(
        min_length=1,
    )

    focus_champions: tuple[
        RAGFocusChampion,
        ...,
    ] = ()

    selected_items: tuple[
        RAGSelectedItem,
        ...,
    ] = ()

    objective_context: bool = False

    structure_context: bool = False


class RAGRetrievalQuery(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    query_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    intent: RAGQueryIntent

    query_text: str = Field(
        min_length=1,
    )

    subject_key: str = Field(
        min_length=1,
    )

    subject_id: str | None = None

    subject_name: str | None = None

    participant_id: int | None = Field(
        default=None,
        ge=1,
    )

    focus_role: RAGFocusRole | None = None

    source_event_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class RAGQueryPlan(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    plan_version: Literal["rag_query_plan_v1"] = RAG_QUERY_PLAN_VERSION

    planner_policy_version: Literal["representative_actor_target_item_system_v1"] = (
        RAG_QUERY_PLANNER_POLICY_VERSION
    )

    plan_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
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

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    priority_tier: RAGPriorityTier

    situation_kind: str = Field(
        min_length=1,
    )

    queries: tuple[
        RAGRetrievalQuery,
        ...,
    ]

    @property
    def query_count(
        self,
    ) -> int:
        return len(self.queries)
