from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)


class ChunkStrategy(StrEnum):
    SINGLE_DOCUMENT = "single_document"
    SEMANTIC_SPLIT = "semantic_split"


class PatchRagChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    chunker_version: str = Field(min_length=1)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_record_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    chunk_index: int = Field(ge=0)
    chunk_count: int = Field(ge=1)
    char_count: int = Field(ge=1)
    strategy: ChunkStrategy

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
    entity_resolved: bool

    resolution_method: ResolutionMethod
    removed_from_target_map: bool
    target_map_id: str | None = None

    heading_path: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)

    text: str = Field(min_length=1)
