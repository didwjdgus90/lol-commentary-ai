from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMode(StrEnum):
    AUTO = "auto"
    PRIMARY = "primary"
    FALLBACK = "fallback"


class RetrievalStrategy(StrEnum):
    BGE_ALIAS_RRF = "rrf_bge_m3_alias_bm25"
    ALIAS_BM25 = "alias_bm25"


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int = Field(ge=1)
    score: float


class FusedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float

    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)

    dense_score: float | None = None
    sparse_score: float | None = None


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    section_kind: str = Field(min_length=1)

    title: str = Field(min_length=1)
    entity_name: str | None = None
    heading_path: list[str] = Field(min_length=1)

    text: str = Field(min_length=1)

    score: float
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    sparse_score: float | None = None


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    query: str = Field(min_length=1)
    expanded_query: str = Field(min_length=1)

    requested_mode: RetrievalMode
    strategy_used: RetrievalStrategy

    fallback_used: bool
    fallback_reason: str | None = None

    top_k: int = Field(gt=0)
    source_top_n: int = Field(gt=0)

    elapsed_ms: float = Field(ge=0.0)

    hits: list[RetrievalHit]
