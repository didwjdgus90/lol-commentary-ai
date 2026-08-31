from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalMetricSummary,
    RetrievalQueryResult,
)


class HybridRetrievalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    experiment_version: str = Field(min_length=1)

    hybrid_key: str = Field(min_length=1)
    dense_engine: str = Field(min_length=1)
    sparse_engine: str = Field(min_length=1)

    fusion_method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(gt=0)
    source_top_n: int = Field(gt=0)

    corpus_chunks: int = Field(ge=0)
    evaluation_cases: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    metrics: RetrievalMetricSummary
    by_language: dict[str, RetrievalMetricSummary]
    by_query_type: dict[str, RetrievalMetricSummary]
    by_difficulty: dict[str, RetrievalMetricSummary]

    fusion_total_seconds: float = Field(ge=0.0)
    fusion_query_mean_ms: float = Field(ge=0.0)

    query_results: list[RetrievalQueryResult]
