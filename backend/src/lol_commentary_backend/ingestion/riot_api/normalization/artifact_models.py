from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

NORMALIZED_DATASET_VERSION = "normalized_v1"


class NormalizedArtifactType(StrEnum):
    MATCH = "match"
    PARTICIPANT_FRAMES = "participant_frames"
    EVENTS = "events"


class NormalizedArtifactFile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_type: NormalizedArtifactType

    file_path: str = Field(
        min_length=1,
    )

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    size_bytes: int = Field(
        gt=0,
    )

    record_count: int = Field(
        ge=0,
    )


class NormalizedMatchArtifactManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    dataset_version: Literal["normalized_v1"] = "normalized_v1"

    normalizer_version: str = Field(
        min_length=1,
    )

    match_id: str = Field(
        min_length=1,
    )

    source_match_file_path: str = Field(
        min_length=1,
    )

    source_timeline_file_path: str = Field(
        min_length=1,
    )

    source_match_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    source_timeline_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    participant_count: int = Field(
        ge=0,
    )

    participant_frame_count: int = Field(
        ge=0,
    )

    event_count: int = Field(
        ge=0,
    )

    files: tuple[
        NormalizedArtifactFile,
        ...,
    ]


class NormalizedMatchReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    match_id: str = Field(
        min_length=1,
    )

    manifest_path: str = Field(
        min_length=1,
    )

    manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    participant_count: int = Field(
        ge=0,
    )

    participant_frame_count: int = Field(
        ge=0,
    )

    event_count: int = Field(
        ge=0,
    )


class NormalizedDatasetManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    dataset_version: Literal["normalized_v1"] = "normalized_v1"

    normalizer_version: str = Field(
        min_length=1,
    )

    match_count: int = Field(
        ge=0,
    )

    total_participant_count: int = Field(
        ge=0,
    )

    total_participant_frame_count: int = Field(
        ge=0,
    )

    total_event_count: int = Field(
        ge=0,
    )

    matches: tuple[
        NormalizedMatchReference,
        ...,
    ]


class NormalizedDatasetBuildResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    built_match_count: int = Field(
        ge=0,
    )

    reused_match_count: int = Field(
        ge=0,
    )

    aggregate_manifest_changed: bool

    manifest: NormalizedDatasetManifest
