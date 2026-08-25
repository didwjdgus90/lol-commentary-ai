import json
from datetime import UTC, datetime
from hashlib import sha256

import httpx
import pytest

from lol_commentary_backend.ingestion.patch_notes import (
    PatchNoteTarget,
    RawPatchNote,
    fetch_patch_note,
    save_raw_patch_note,
)

TARGET = PatchNoteTarget(
    patch="26.1",
    locale="ko_kr",
    source_url="https://example.test/patch-26-1-notes/",
)


def test_fetch_patch_note_collects_content_and_metadata() -> None:
    html = b"<html><body>Patch 26.1</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["user-agent"].startswith("LoLCommentaryAI/")
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "etag": '"test-etag"',
            },
            content=html,
            request=request,
        )

    raw_patch_note = fetch_patch_note(
        TARGET,
        transport=httpx.MockTransport(handler),
    )

    assert raw_patch_note.status_code == 200
    assert raw_patch_note.content == html
    assert raw_patch_note.etag == '"test-etag"'
    assert raw_patch_note.sha256 == sha256(html).hexdigest()


def test_fetch_patch_note_raises_for_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        fetch_patch_note(
            TARGET,
            transport=httpx.MockTransport(handler),
        )


def test_save_raw_patch_note_writes_html_and_metadata(tmp_path) -> None:
    html = b"<html><body>Patch 26.1</body></html>"
    digest = sha256(html).hexdigest()

    raw_patch_note = RawPatchNote(
        target=TARGET,
        fetched_at=datetime(2026, 1, 7, 19, 0, tzinfo=UTC),
        status_code=200,
        content_type="text/html; charset=utf-8",
        etag='"test-etag"',
        last_modified=None,
        content=html,
        sha256=digest,
    )

    html_path, metadata_path = save_raw_patch_note(raw_patch_note, tmp_path)

    assert html_path.read_bytes() == html

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["patch"] == "26.1"
    assert metadata["locale"] == "ko_kr"
    assert metadata["sha256"] == digest
    assert metadata["size_bytes"] == len(html)
