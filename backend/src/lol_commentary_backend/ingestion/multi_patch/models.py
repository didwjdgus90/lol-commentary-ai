from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

MULTI_PATCH_CORPUS_VERSION = "multi_patch_patch_notes_v1"


class CorpusShardManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    corpus_version: Literal["multi_patch_patch_notes_v1"] = MULTI_PATCH_CORPUS_VERSION

    patch: str = Field(
        min_length=1,
    )

    locale: str = Field(
        min_length=1,
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    raw_source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    normalized_count: int = Field(
        ge=0,
    )

    resolved_count: int = Field(
        ge=0,
    )

    resolved_entity_count: int = Field(
        ge=0,
    )

    ambiguous_count: int = Field(
        ge=0,
    )

    unresolved_count: int = Field(
        ge=0,
    )

    document_count: int = Field(
        ge=0,
    )

    chunk_count: int = Field(
        ge=0,
    )

    max_chunk_chars: int = Field(
        ge=200,
    )

    normalized_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    resolved_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    documents_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class CorpusShardReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    patch: str

    locale: str

    manifest_path: str = Field(
        min_length=1,
    )

    corpus_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    chunk_count: int = Field(
        ge=0,
    )


class MultiPatchCorpusManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    corpus_version: Literal["multi_patch_patch_notes_v1"] = MULTI_PATCH_CORPUS_VERSION

    start_patch: str

    end_patch: str

    locales: tuple[
        str,
        ...,
    ]

    patch_count: int = Field(
        ge=1,
    )

    shard_count: int = Field(
        ge=1,
    )

    total_document_count: int = Field(
        ge=0,
    )

    total_chunk_count: int = Field(
        ge=0,
    )

    patch_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    version_mapping_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    entity_catalog_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    shards: tuple[
        CorpusShardReference,
        ...,
    ]
