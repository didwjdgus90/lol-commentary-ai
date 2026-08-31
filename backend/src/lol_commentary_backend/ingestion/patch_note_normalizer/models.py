from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class PatchEntityType(StrEnum):
    CHAMPION = "champion"
    ITEM = "item"
    UNKNOWN = "unknown"


class NormalizedPatchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order: int = Field(ge=0)

    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)

    source_url: HttpUrl
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    section_kind: str = Field(min_length=1)
    entity_type: PatchEntityType = PatchEntityType.UNKNOWN
    entity_name: str | None = None

    heading_path: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)

    paragraphs: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)
    content: str = Field(min_length=1)
