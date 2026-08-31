from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FailureCategory(StrEnum):
    SOLVED_TOP1 = "solved_top1"
    RERANKABLE_TOP10 = "rerankable_top10"
    CANDIDATE_RECALL_FAILURE = "candidate_recall_failure"


class EngineRankDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str = Field(min_length=1)
    first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    hit_at_1: bool
    hit_at_5: bool
    relevant_in_top_10: bool

    top_1_title: str | None = None
    top_1_entity_name: str | None = None


class QueryFailureDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(pattern=r"^q[0-9]{3}$")
    query: str = Field(min_length=1)

    language: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)

    category: FailureCategory

    best_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    best_engines: list[str]

    engine_diagnostics: list[EngineRankDiagnostic]


class FusionEffect(StrEnum):
    PRESERVED_SUCCESS = "preserved_success"
    NEW_TOP1_SUCCESS = "new_top1_success"
    LOST_SOURCE_TOP1 = "lost_source_top1"
    STILL_MISS_TOP1 = "still_miss_top1"


class FusionQueryDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(pattern=r"^q[0-9]{3}$")
    query: str = Field(min_length=1)

    dense_hit_at_1: bool
    sparse_hit_at_1: bool
    hybrid_hit_at_1: bool

    dense_first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    sparse_first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    hybrid_first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )

    effect: FusionEffect


class RetrievalDiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    report_version: str = Field(min_length=1)

    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: int = Field(ge=0)

    solved_top1: list[str]
    rerankable_top10: list[str]
    candidate_recall_failures: list[str]

    query_diagnostics: list[QueryFailureDiagnostic]
    fusion_diagnostics: list[FusionQueryDiagnostic]
