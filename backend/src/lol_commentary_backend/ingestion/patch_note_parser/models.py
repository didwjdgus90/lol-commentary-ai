from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SectionKind(StrEnum):
    GENERAL = "general"
    HOTFIX = "hotfix"
    CHAMPION = "champion"
    ITEM = "item"
    SYSTEM = "system"
    GAME_MODE = "game_mode"
    BUG_FIX = "bug_fix"
    OTHER = "other"


class PatchChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(min_length=1)
    subject: str | None = None
    before: str | None = None
    after: str | None = None


class PatchSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    level: Literal[2, 3, 4]
    kind: SectionKind = SectionKind.OTHER
    heading_path: list[str] = Field(default_factory=list)

    entity_name: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    changes: list[PatchChange] = Field(default_factory=list)
    children: list[PatchSection] = Field(default_factory=list)


class PatchNoteDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)

    source_url: HttpUrl
    fetched_at: datetime
    published_at: datetime | None = None

    source_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_size_bytes: int | None = Field(default=None, gt=0)
    collector_version: str | None = None

    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)

    sections: list[PatchSection] = Field(default_factory=list)
