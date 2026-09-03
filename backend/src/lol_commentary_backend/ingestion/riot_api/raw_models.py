from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

RIOT_RAW_COLLECTOR_VERSION = "riot_match_raw_collector_v1"


class RiotRawResourceType(StrEnum):
    ACCOUNT = "account"
    MATCH = "match"
    TIMELINE = "timeline"


class RiotRawManifestRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    collector_version: str = Field(
        min_length=1,
    )

    resource_key: str = Field(
        min_length=1,
    )

    resource_type: RiotRawResourceType

    routing_region: Literal["asia"] = "asia"

    source_endpoint: str = Field(
        min_length=1,
    )

    fetched_at: datetime

    file_path: str = Field(
        min_length=1,
    )

    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    size_bytes: int = Field(
        gt=0,
    )

    account_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str | None = None


class RiotRawCollectionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    requested_match_count: int = Field(
        ge=1,
        le=100,
    )

    returned_match_count: int = Field(
        ge=0,
    )

    downloaded_resources: int = Field(
        ge=0,
    )

    reused_resources: int = Field(
        ge=0,
    )

    manifest_record_count: int = Field(
        ge=0,
    )

    match_ids: tuple[
        str,
        ...,
    ]
