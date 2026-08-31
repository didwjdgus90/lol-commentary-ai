from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalStrategy,
)


class ContextEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")

    source_rank: int = Field(ge=1)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    patch: str = Field(min_length=1)

    locale: str = Field(min_length=1)

    source_url: str = Field(min_length=1)

    section_kind: str = Field(min_length=1)

    title: str = Field(min_length=1)

    entity_name: str | None = None

    heading_path: tuple[str, ...] = Field(min_length=1)

    text: str = Field(min_length=1)

    retrieval_score: float

    dense_rank: int | None = Field(
        default=None,
        ge=1,
    )

    sparse_rank: int | None = Field(
        default=None,
        ge=1,
    )


class ContextBundle(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    query: str = Field(min_length=1)

    expanded_query: str = Field(min_length=1)

    requested_mode: RetrievalMode
    strategy_used: RetrievalStrategy

    fallback_used: bool
    fallback_reason: str | None = None

    retrieval_elapsed_ms: float = Field(ge=0.0)

    retrieval_top_k: int = Field(gt=0)

    source_top_n: int = Field(gt=0)

    selected_count: int = Field(ge=0)

    dropped_duplicate_count: int = Field(ge=0)

    dropped_limit_count: int = Field(ge=0)

    dropped_relevance_count: int = Field(
        default=0,
        ge=0,
    )

    selection_policy: Literal[
        "unfiltered",
        "entity_title_match_v1",
    ] = "unfiltered"

    selection_fallback_used: bool = False

    evidence: tuple[
        ContextEvidence,
        ...,
    ]
