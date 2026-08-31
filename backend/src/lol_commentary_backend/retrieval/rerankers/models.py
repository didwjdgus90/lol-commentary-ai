from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalMetricSummary,
    RetrievalQueryResult,
)


class RerankerArchitecture(StrEnum):
    ENCODER = "encoder"
    CAUSAL_LM = "causal_lm"


class RerankerCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    key: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    architecture: RerankerArchitecture
    max_length: int = Field(gt=0)

    trust_remote_code: bool = False
    license_name: str = Field(min_length=1)


GTE_MULTILINGUAL_RERANKER_BASE = RerankerCandidate(
    key="gte_multilingual_reranker_base",
    model_id=("Alibaba-NLP/gte-multilingual-reranker-base"),
    architecture=RerankerArchitecture.ENCODER,
    max_length=1024,
    trust_remote_code=True,
    license_name="apache-2.0",
)

BGE_RERANKER_V2_M3 = RerankerCandidate(
    key="bge_reranker_v2_m3",
    model_id="BAAI/bge-reranker-v2-m3",
    architecture=RerankerArchitecture.ENCODER,
    max_length=1024,
    trust_remote_code=False,
    license_name="apache-2.0",
)

QWEN3_RERANKER_06B = RerankerCandidate(
    key="qwen3_reranker_06b",
    model_id="Qwen/Qwen3-Reranker-0.6B",
    architecture=RerankerArchitecture.CAUSAL_LM,
    max_length=1024,
    trust_remote_code=False,
    license_name="apache-2.0",
)

RERANKER_CANDIDATES = (
    GTE_MULTILINGUAL_RERANKER_BASE,
    BGE_RERANKER_V2_M3,
    QWEN3_RERANKER_06B,
)


class CandidateSourceCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_key: str = Field(min_length=1)
    top_n: int = Field(gt=0)

    cases: int = Field(ge=0)
    full_gold_cases: int = Field(ge=0)
    coverage_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    missing_query_ids: list[str]


class CandidateSourceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    selection_version: str = Field(min_length=1)

    source_key: str = Field(min_length=1)
    top_n: int = Field(gt=0)

    coverage_rate: float = Field(
        ge=0.0,
        le=1.0,
    )
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    reason: str = Field(min_length=1)


class RerankerTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_load_seconds: float = Field(ge=0.0)
    inference_total_seconds: float = Field(ge=0.0)
    query_mean_ms: float = Field(ge=0.0)
    pair_mean_ms: float = Field(ge=0.0)


class RerankerMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rss_before_mb: float = Field(ge=0.0)
    rss_after_load_mb: float = Field(ge=0.0)
    rss_after_inference_mb: float = Field(ge=0.0)


class RerankerBenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    benchmark_version: str = Field(min_length=1)

    reranker_key: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    device: str = Field(min_length=1)

    candidate_source: str = Field(min_length=1)
    candidate_top_n: int = Field(gt=0)

    corpus_chunks: int = Field(ge=0)
    evaluation_cases: int = Field(ge=0)
    pairs_scored: int = Field(ge=0)

    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    metrics: RetrievalMetricSummary
    by_language: dict[str, RetrievalMetricSummary]
    by_query_type: dict[str, RetrievalMetricSummary]
    by_difficulty: dict[str, RetrievalMetricSummary]

    timing: RerankerTiming
    memory: RerankerMemory

    query_results: list[RetrievalQueryResult]
