from hashlib import sha256
from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.patch_note_parser.document import (
    RawPatchNoteMetadata,
    build_patch_note_document,
    save_patch_note_document,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
)

HTML = """
<!doctype html>
<html lang="ko">
  <body>
    <main>
      <h1>26.1 패치 노트</h1>

      <div id="patch-notes-container">
        <header class="header-primary">
          <h2 id="patch-champions">챔피언</h2>
        </header>

        <div class="content-border">
          <div class="patch-change-block white-stone accent-before">
            <div>
              <h3 id="patch-aphelios" class="change-title">아펠리오스</h3>
              <h4 class="change-detail-title ability-title">만월총</h4>
              <ul>
                <li>표식 피해량: 20 ⇒ 18</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </main>
  </body>
</html>
""".encode()


def _metadata(html: bytes = HTML) -> RawPatchNoteMetadata:
    return RawPatchNoteMetadata(
        schema_version=1,
        collector_version="0.1.0",
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        fetched_at="2026-08-26T02:00:00+00:00",
        status_code=200,
        content_type="text/html; charset=utf-8",
        etag=None,
        last_modified=None,
        sha256=sha256(html).hexdigest(),
        size_bytes=len(html),
    )


def test_build_patch_note_document_combines_raw_metadata_and_parser() -> None:
    document = build_patch_note_document(HTML, _metadata())

    assert document.patch == "26.1"
    assert document.locale == "ko_kr"
    assert document.title == "26.1 패치 노트"
    assert document.source_sha256 == sha256(HTML).hexdigest()
    assert document.source_size_bytes == len(HTML)
    assert document.collector_version == "0.1.0"

    champions = document.sections[0]
    aphelios = champions.children[0]
    ability = aphelios.children[0]

    assert ability.heading_path == [
        "챔피언",
        "아펠리오스",
        "만월총",
    ]


def test_build_patch_note_document_rejects_size_mismatch() -> None:
    metadata = _metadata().model_copy(update={"size_bytes": len(HTML) + 1})

    with pytest.raises(ValueError, match="size mismatch"):
        build_patch_note_document(HTML, metadata)


def test_build_patch_note_document_rejects_sha256_mismatch() -> None:
    metadata = _metadata().model_copy(update={"sha256": "0" * 64})

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_patch_note_document(HTML, metadata)


def test_saved_document_can_be_validated_again(
    tmp_path: Path,
) -> None:
    document = build_patch_note_document(HTML, _metadata())
    output_path = save_patch_note_document(
        document,
        tmp_path / "parsed.json",
    )

    restored = PatchNoteDocument.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert restored == document
