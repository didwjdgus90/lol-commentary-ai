from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: int = Field(ge=0)

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)

    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    title: str = Field(min_length=1)
    entity_name: str | None = None

    score: float
    relevant: bool


class RetrievalQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(pattern=r"^q[0-9]{3}$")
    query: str = Field(min_length=1)

    language: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)

    relevant_chunk_ids: list[str] = Field(min_length=1)

    first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )

    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    recall_at_5: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)

    top_results: list[RetrievedChunk]


class BenchmarkTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_load_seconds: float = Field(ge=0.0)
    corpus_index_seconds: float = Field(ge=0.0)
    query_total_seconds: float = Field(ge=0.0)
    query_mean_ms: float = Field(ge=0.0)


class BenchmarkMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_rss_before_mb: float = Field(ge=0.0)
    process_rss_after_load_mb: float = Field(ge=0.0)
    process_rss_after_index_mb: float = Field(ge=0.0)

    cuda_peak_allocated_mb: float | None = Field(
        default=None,
        ge=0.0,
    )


class RetrievalBenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    benchmark_version: str = Field(min_length=1)

    engine_key: str = Field(min_length=1)
    engine_type: Literal["dense", "bm25"]

    model_id: str | None = None
    device: str = Field(min_length=1)

    corpus_chunks: int = Field(ge=0)
    evaluation_cases: int = Field(ge=0)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    embedding_dimension: int | None = Field(
        default=None,
        gt=0,
    )

    metrics: RetrievalMetricSummary

    by_language: dict[str, RetrievalMetricSummary]
    by_query_type: dict[str, RetrievalMetricSummary]
    by_difficulty: dict[str, RetrievalMetricSummary]

    timing: BenchmarkTiming
    memory: BenchmarkMemory

    query_results: list[RetrievalQueryResult]
