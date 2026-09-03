from pathlib import Path

from lol_commentary_backend.retrieval.aliases.multi_catalog import (
    build_multi_patch_alias_catalog,
    load_canonical_entities,
    save_multi_patch_alias_catalog,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    entities_path = repository_root / "data" / "processed" / "entities" / "v1" / "entities.jsonl"

    output_path = (
        repository_root
        / "data"
        / "processed"
        / "retrieval"
        / "entity_aliases"
        / "multi_patch_v1"
        / "ko_en_aliases.json"
    )

    entities = load_canonical_entities(entities_path)

    catalog = build_multi_patch_alias_catalog(entities)

    save_multi_patch_alias_catalog(
        catalog=catalog,
        path=output_path,
    )

    entity_keys = {
        (
            alias.entity_type.value,
            alias.entity_key,
        )
        for alias in catalog.aliases
    }

    print("=== MULTI-PATCH BILINGUAL ALIAS CATALOG ===")

    print(f"Builder: {catalog.builder_version}")

    print(f"Version scope: {catalog.ddragon_version}")

    print(f"Canonical entities: {len(entity_keys)}")

    print(f"Alias records: {len(catalog.aliases)}")

    print(f"Source hashes: {len(catalog.source_hashes)}")

    print(f"Output: {output_path}")

    if not catalog.aliases:
        raise RuntimeError("Alias catalog is empty")

    print()

    print("MULTI_PATCH_ALIAS_CATALOG=PASS")


if __name__ == "__main__":
    main()
