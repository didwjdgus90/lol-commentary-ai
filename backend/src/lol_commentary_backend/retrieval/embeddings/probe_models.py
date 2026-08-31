from pydantic import BaseModel, ConfigDict, Field


class TokenLengthStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    minimum: int = Field(ge=0)
    median: int = Field(ge=0)
    p95: int = Field(ge=0)
    maximum: int = Field(ge=0)


class TokenizerProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_key: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    corpus_chunks: int = Field(ge=0)

    declared_max_tokens: int = Field(gt=0)
    experiment_max_tokens: int = Field(gt=0)
    tokenizer_model_max_length: int = Field(gt=0)

    chunk_stats: TokenLengthStats
    query_stats: TokenLengthStats

    chunks_over_256: int = Field(ge=0)
    chunks_over_512: int = Field(ge=0)
    chunks_over_1024: int = Field(ge=0)
    chunks_over_8192: int = Field(ge=0)

    chunks_over_experiment_limit: int = Field(ge=0)

    compatible_with_current_corpus: bool
