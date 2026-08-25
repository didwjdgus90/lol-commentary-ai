from pathlib import Path

from lol_commentary_backend.ingestion.patch_notes import (
    PatchNoteTarget,
    fetch_patch_note,
    save_raw_patch_note,
)

TARGET = PatchNoteTarget(
    patch="26.1",
    locale="ko_kr",
    source_url="https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/",
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    output_dir = repository_root / "data" / "raw" / "patch_notes" / TARGET.patch / TARGET.locale

    raw_patch_note = fetch_patch_note(TARGET)

    html_path, metadata_path = save_raw_patch_note(
        raw_patch_note,
        output_dir,
    )

    print(f"HTML saved: {html_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"SHA-256: {raw_patch_note.sha256}")


if __name__ == "__main__":
    main()
