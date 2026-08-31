import json
from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_normalizer.normalizer import (
    normalize_patch_note_document,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    parsed_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "parsed.json"
    )

    output_path = parsed_path.with_name("normalized.jsonl")

    if not parsed_path.is_file():
        raise FileNotFoundError(f"Parsed patch note not found: {parsed_path}")

    document = PatchNoteDocument.model_validate_json(parsed_path.read_text(encoding="utf-8"))

    records = normalize_patch_note_document(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            payload = record.model_dump(mode="json")
            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    print(f"Normalized JSONL saved: {output_path}")
    print(f"Records: {len(records)}")

    champion_records = sum(record.entity_type.value == "champion" for record in records)
    item_records = sum(record.entity_type.value == "item" for record in records)
    unknown_records = sum(record.entity_type.value == "unknown" for record in records)

    print(f"Champion records: {champion_records}")
    print(f"Item records: {item_records}")
    print(f"Unknown records: {unknown_records}")


if __name__ == "__main__":
    main()
