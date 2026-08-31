from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_parser.document import (
    build_patch_note_document,
    load_raw_patch_note_metadata,
    save_patch_note_document,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    raw_dir = repository_root / "data" / "raw" / "patch_notes" / "26.1" / "ko_kr"

    html_path = raw_dir / "page.html"
    metadata_path = raw_dir / "metadata.json"

    output_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "parsed.json"
    )

    if not html_path.is_file():
        raise FileNotFoundError(f"Raw HTML not found: {html_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Raw metadata not found: {metadata_path}")

    html = html_path.read_bytes()
    metadata = load_raw_patch_note_metadata(metadata_path)

    document = build_patch_note_document(html, metadata)
    saved_path = save_patch_note_document(document, output_path)

    print(f"Parsed JSON saved: {saved_path}")
    print(f"Patch: {document.patch}")
    print(f"Locale: {document.locale}")
    print(f"Title: {document.title}")
    print(f"Source SHA-256: {document.source_sha256}")
    print(f"Source size: {document.source_size_bytes:,} bytes")
    print(f"Top-level sections: {len(document.sections)}")


if __name__ == "__main__":
    main()
