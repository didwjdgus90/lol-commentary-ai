from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)


class RagSourceType(StrEnum):
    PATCH_NOTE = "patch_note"


class PatchRagDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    builder_version: str = Field(min_length=1)

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_type: Literal[RagSourceType.PATCH_NOTE] = RagSourceType.PATCH_NOTE
    source_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_order: int = Field(ge=0)

    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)

    source_url: HttpUrl
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ddragon_version: str = Field(min_length=1)

    section_kind: str = Field(min_length=1)

    entity_type: PatchEntityType
    entity_name: str | None = None
    entity_id: str | None = None
    entity_key: str | None = None

    resolution_method: ResolutionMethod
    entity_resolved: bool
    removed_from_target_map: bool

    target_map_id: str | None = None

    heading_path: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)

    paragraphs: list[str] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list)

    retrieval_text: str = Field(min_length=1)

    context_method: str | None = None
    context_confidence: str | None = None
    context_anchor_title: str | None = None
