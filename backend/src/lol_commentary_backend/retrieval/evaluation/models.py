from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueryLanguage(StrEnum):
    KO = "ko"
    EN = "en"
    MIXED = "mixed"


class QueryType(StrEnum):
    CHAMPION_ABILITY = "champion_ability"
    ITEM_CHANGE = "item_change"
    NUMERIC_CHANGE = "numeric_change"
    LIFECYCLE = "lifecycle"
    HOTFIX = "hotfix"
    SYSTEM = "system"
    BUG = "bug"


class QueryDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TargetGranularity(StrEnum):
    DOCUMENT = "document"
    ANSWER_CHUNK = "answer_chunk"


class RetrievalTargetSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: str = Field(min_length=1)
    title: str = Field(min_length=1)

    entity_name: str | None = None
    section_kind: str | None = None

    required_text_contains: str | None = None


class RetrievalEvalSeed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    query_id: str = Field(pattern=r"^q[0-9]{3}$")
    query: str = Field(min_length=1)

    language: QueryLanguage
    query_type: QueryType
    difficulty: QueryDifficulty

    selector: RetrievalTargetSelector
    rationale: str = Field(min_length=1)


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_version: str = Field(min_length=1)

    query_id: str = Field(pattern=r"^q[0-9]{3}$")
    query: str = Field(min_length=1)

    language: QueryLanguage
    query_type: QueryType
    difficulty: QueryDifficulty

    target_granularity: TargetGranularity

    relevant_document_ids: list[str] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    relevant_source_record_ids: list[str] = Field(min_length=1)

    selector: RetrievalTargetSelector
    rationale: str = Field(min_length=1)

    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
