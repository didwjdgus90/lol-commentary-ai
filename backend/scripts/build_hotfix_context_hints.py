import json
from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_context.builder import (
    build_hotfix_champion_context_hints,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    load_entity_catalog,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
)

DDRAGON_VERSION = "16.1.1"
DDRAGON_LOCALE = "ko_KR"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    parsed_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "parsed.json"
    )

    normalized_path = parsed_path.with_name("normalized.jsonl")

    output_path = parsed_path.with_name("hotfix_context_hints.jsonl")

    catalog_dir = (
        repository_root / "data" / "raw" / "data_dragon" / DDRAGON_VERSION / DDRAGON_LOCALE
    )

    document = PatchNoteDocument.model_validate_json(parsed_path.read_text(encoding="utf-8"))

    records: list[NormalizedPatchRecord] = []

    with normalized_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(NormalizedPatchRecord.model_validate_json(line))

    catalog = load_entity_catalog(catalog_dir)

    hints = build_hotfix_champion_context_hints(
        document,
        records,
        catalog,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for hint in hints:
            file.write(
                json.dumps(
                    hint.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    champion_counts = Counter(hint.champion_name for hint in hints)

    print(f"Hotfix context hints saved: {output_path}")
    print(f"Hints: {len(hints)}")
    print()

    print("Hints by champion:")

    for champion_name, count in champion_counts.items():
        print(f"  {champion_name}: {count}")

    print()
    print("Hint records:")

    for hint in hints:
        print(
            f"  order={hint.record_order} "
            f"champion={hint.champion_name!r} "
            f"detail={hint.detail_title!r} "
            f"method={hint.context_method.value}"
        )


if __name__ == "__main__":
    main()
