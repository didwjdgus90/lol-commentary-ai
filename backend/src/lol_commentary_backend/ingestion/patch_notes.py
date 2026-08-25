import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
DEFAULT_HEADERS = {
    "User-Agent": "LoLCommentaryAI/0.1 (educational data collector)",
}
SCHEMA_VERSION = 1
COLLECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class PatchNoteTarget:
    patch: str
    locale: str
    source_url: str


@dataclass(frozen=True)
class RawPatchNote:
    target: PatchNoteTarget
    fetched_at: datetime
    status_code: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    content: bytes
    sha256: str


def fetch_patch_note(
    target: PatchNoteTarget,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RawPatchNote:
    with httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        transport=transport,
    ) as client:
        response = client.get(target.source_url)
        response.raise_for_status()

    content = response.content

    return RawPatchNote(
        target=target,
        fetched_at=datetime.now(UTC),
        status_code=response.status_code,
        content_type=response.headers.get("content-type"),
        etag=response.headers.get("etag"),
        last_modified=response.headers.get("last-modified"),
        content=content,
        sha256=sha256(content).hexdigest(),
    )


def save_raw_patch_note(
    raw_patch_note: RawPatchNote,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "page.html"
    metadata_path = output_dir / "metadata.json"

    html_path.write_bytes(raw_patch_note.content)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "patch": raw_patch_note.target.patch,
        "locale": raw_patch_note.target.locale,
        "source_url": raw_patch_note.target.source_url,
        "fetched_at": raw_patch_note.fetched_at.isoformat(),
        "status_code": raw_patch_note.status_code,
        "content_type": raw_patch_note.content_type,
        "etag": raw_patch_note.etag,
        "last_modified": raw_patch_note.last_modified,
        "sha256": raw_patch_note.sha256,
        "size_bytes": len(raw_patch_note.content),
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return html_path, metadata_path
