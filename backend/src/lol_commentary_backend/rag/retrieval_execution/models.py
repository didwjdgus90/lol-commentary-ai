from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.rag.query_planner.models import (
    RAGPriorityTier,
    RAGQueryIntent,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
    RetrievalMode,
    RetrievalStrategy,
)

RAG_RETRIEVAL_EXECUTION_VERSION = "rag_retrieval_execution_v1"


class RAGQueryEvidenceResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    query_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
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

    intent: RAGQueryIntent

    query_text: str = Field(
        min_length=1,
    )

    subject_key: str = Field(
        min_length=1,
    )

    subject_id: str | None = None

    subject_name: str | None = None

    cache_reused: bool

    expanded_query: str = Field(
        min_length=1,
    )

    requested_mode: RetrievalMode

    strategy_used: RetrievalStrategy

    fallback_used: bool

    fallback_reason: str | None = None

    source_top_n: int = Field(
        gt=0,
    )

    source_retrieval_elapsed_ms: float = Field(
        ge=0.0,
    )

    hits: tuple[
        RetrievalHit,
        ...,
    ]


class RAGPlanEvidenceResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

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

    priority_tier: RAGPriorityTier

    situation_kind: str = Field(
        min_length=1,
    )

    query_results: tuple[
        RAGQueryEvidenceResult,
        ...,
    ]


class RAGBatchEvidenceResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    execution_version: Literal["rag_retrieval_execution_v1"] = RAG_RETRIEVAL_EXECUTION_VERSION

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    requested_mode: RetrievalMode

    top_k: int = Field(
        gt=0,
    )

    plan_count: int = Field(
        ge=1,
    )

    query_reference_count: int = Field(
        ge=0,
    )

    unique_query_execution_count: int = Field(
        ge=0,
    )

    cache_hit_count: int = Field(
        ge=0,
    )

    fallback_execution_count: int = Field(
        ge=0,
    )

    zero_hit_execution_count: int = Field(
        ge=0,
    )

    total_hit_references: int = Field(
        ge=0,
    )

    wall_elapsed_ms: float = Field(
        ge=0.0,
    )

    plans: tuple[
        RAGPlanEvidenceResult,
        ...,
    ]
