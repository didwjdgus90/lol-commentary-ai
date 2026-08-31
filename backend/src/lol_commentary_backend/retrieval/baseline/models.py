from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalBaselineRole(StrEnum):
    PRIMARY = "primary"
    FAST_FALLBACK = "fast_fallback"


class RetrievalBaselineCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    role: RetrievalBaselineRole

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)

    online_dense_required: bool
    reranker_required: bool

    notes: list[str]


class RetrievalBaselineDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    decision_version: str = Field(min_length=1)

    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_cases: int = Field(gt=0)

    primary: RetrievalBaselineCandidate
    fast_fallback: RetrievalBaselineCandidate

    rejected_rerankers: list[str]

    sentence_transformers_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)

    decision_rationale: list[str]


class ReproducibilityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    check_version: str = Field(min_length=1)

    engine_key: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    reference_hit_at_1: float
    reproduced_hit_at_1: float

    reference_hit_at_3: float
    reproduced_hit_at_3: float

    reference_recall_at_5: float
    reproduced_recall_at_5: float

    reference_mrr: float
    reproduced_mrr: float

    metrics_match: bool

    sentence_transformers_version: str
    transformers_version: str
