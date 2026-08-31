import json
from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_context.resolver import (
    apply_hotfix_champion_context_hints,
    load_hotfix_context_hints,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    load_entity_catalog,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.resolver import (
    resolve_patch_records,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)

DDRAGON_VERSION = "16.1.1"
DDRAGON_LOCALE = "ko_KR"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    processed_dir = repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr"

    normalized_path = processed_dir / "normalized.jsonl"
    hints_path = processed_dir / "hotfix_context_hints.jsonl"
    output_path = processed_dir / "resolved.jsonl"

    catalog_dir = (
        repository_root / "data" / "raw" / "data_dragon" / DDRAGON_VERSION / DDRAGON_LOCALE
    )

    catalog = load_entity_catalog(catalog_dir)

    normalized_records: list[NormalizedPatchRecord] = []

    with normalized_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            normalized_records.append(NormalizedPatchRecord.model_validate_json(line))

    base_resolved_records = resolve_patch_records(
        normalized_records,
        catalog,
    )

    hints = load_hotfix_context_hints(hints_path)

    resolved_records = apply_hotfix_champion_context_hints(
        base_resolved_records,
        hints,
        catalog,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in resolved_records:
            file.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    methods = Counter(record.resolution_method for record in resolved_records)

    resolved_champions = sum(
        record.entity_type == PatchEntityType.CHAMPION and record.entity_id is not None
        for record in resolved_records
    )
    resolved_items = sum(
        record.entity_type == PatchEntityType.ITEM and record.entity_id is not None
        for record in resolved_records
    )

    print(f"Resolved JSONL saved: {output_path}")
    print(f"Data Dragon version: {catalog.ddragon_version}")
    print(f"Context hints loaded: {len(hints)}")
    print(f"Records: {len(resolved_records)}")
    print(f"Resolved champions: {resolved_champions}")
    print(f"Resolved items: {resolved_items}")

    for method in ResolutionMethod:
        print(f"{method.value}: {methods[method]}")


if __name__ == "__main__":
    main()
