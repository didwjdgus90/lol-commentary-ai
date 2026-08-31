from datetime import datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from selectolax.lexbor import LexborHTMLParser

from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
)
from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_riot_patch_sections,
)


class RawPatchNoteMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    collector_version: str = Field(min_length=1)

    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)
    source_url: HttpUrl
    fetched_at: datetime

    status_code: int = Field(ge=200, lt=300)
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


def load_raw_patch_note_metadata(
    metadata_path: Path,
) -> RawPatchNoteMetadata:
    return RawPatchNoteMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))


def _extract_document_title(html: bytes) -> str:
    tree = LexborHTMLParser(html)
    heading = tree.css_first("h1")

    if heading is None:
        raise ValueError("Patch note h1 title not found")

    title = " ".join(heading.text(separator=" ", strip=True).split())

    if not title:
        raise ValueError("Patch note h1 title is empty")

    return title


def _verify_raw_integrity(
    html: bytes,
    metadata: RawPatchNoteMetadata,
) -> None:
    actual_size = len(html)
    if actual_size != metadata.size_bytes:
        raise ValueError(
            f"Raw HTML size mismatch: metadata={metadata.size_bytes}, actual={actual_size}"
        )

    actual_sha256 = sha256(html).hexdigest()
    if actual_sha256 != metadata.sha256:
        raise ValueError(
            f"Raw HTML SHA-256 mismatch: metadata={metadata.sha256}, actual={actual_sha256}"
        )


def build_patch_note_document(
    html: bytes,
    metadata: RawPatchNoteMetadata,
) -> PatchNoteDocument:
    _verify_raw_integrity(html, metadata)

    return PatchNoteDocument(
        patch=metadata.patch,
        locale=metadata.locale,
        source_url=metadata.source_url,
        fetched_at=metadata.fetched_at,
        source_sha256=metadata.sha256,
        source_size_bytes=metadata.size_bytes,
        collector_version=metadata.collector_version,
        title=_extract_document_title(html),
        sections=parse_riot_patch_sections(html),
    )


def save_patch_note_document(
    document: PatchNoteDocument,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
