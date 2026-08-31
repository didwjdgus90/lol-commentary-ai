from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    key: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    embedding_dimension: int = Field(gt=0)
    declared_max_tokens: int = Field(gt=0)
    experiment_max_tokens: int = Field(gt=0)

    license_name: str = Field(min_length=1)

    query_mode: Literal[
        "qwen_instruct",
        "e5_instruct",
        "plain",
    ]

    experiment_enabled: bool = True


QWEN3_EMBEDDING_06B = EmbeddingCandidate(
    key="qwen3_embedding_06b",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    embedding_dimension=1024,
    declared_max_tokens=32768,
    experiment_max_tokens=8192,
    license_name="apache-2.0",
    query_mode="qwen_instruct",
)

BGE_M3 = EmbeddingCandidate(
    key="bge_m3",
    model_id="BAAI/bge-m3",
    embedding_dimension=1024,
    declared_max_tokens=8192,
    experiment_max_tokens=8192,
    license_name="mit",
    query_mode="plain",
)

MULTILINGUAL_E5_LARGE_INSTRUCT = EmbeddingCandidate(
    key="multilingual_e5_large_instruct",
    model_id="intfloat/multilingual-e5-large-instruct",
    embedding_dimension=1024,
    declared_max_tokens=512,
    experiment_max_tokens=512,
    license_name="mit",
    query_mode="e5_instruct",
)

EMBEDDING_CANDIDATES = (
    QWEN3_EMBEDDING_06B,
    BGE_M3,
    MULTILINGUAL_E5_LARGE_INSTRUCT,
)
