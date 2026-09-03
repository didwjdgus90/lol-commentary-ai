from pathlib import Path

from lol_commentary_backend.ingestion.entity_catalog.builder import (
    build_alias_index,
    build_canonical_entities,
    collect_entity_observations,
    discover_ddragon_versions,
)
from lol_commentary_backend.ingestion.entity_catalog.io import (
    save_entity_catalog,
)
from lol_commentary_backend.ingestion.entity_catalog.models import (
    EntityType,
)

LOCALES = (
    "ko_KR",
    "en_US",
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    ddragon_root = repository_root / "data" / "raw" / "ddragon"

    output_dir = repository_root / "data" / "processed" / "entities" / "v1"

    versions = discover_ddragon_versions(ddragon_root)

    print("=== ENTITY CATALOG BUILD ===")

    print(f"Data Dragon versions: {len(versions)}")

    print(f"First version: {versions[0]}")

    print(f"Last version: {versions[-1]}")

    print(f"Locales: {LOCALES}")

    observations = collect_entity_observations(
        repository_root=(repository_root),
        versions=versions,
        locales=LOCALES,
    )

    entities = build_canonical_entities(observations)

    aliases = build_alias_index(entities)

    (
        entities_path,
        aliases_path,
        metadata_path,
        metadata,
    ) = save_entity_catalog(
        output_dir=output_dir,
        entities=entities,
        aliases=aliases,
    )

    ambiguous_aliases = [alias for alias in aliases if len(alias.entity_uids) > 1]

    champion_count = sum(entity.entity_type == EntityType.CHAMPION for entity in entities)

    item_count = sum(entity.entity_type == EntityType.ITEM for entity in entities)

    print()
    print(f"Observations: {len(observations)}")

    print(f"Entities: {len(entities)}")

    print(f"Champions: {champion_count}")

    print(f"Items: {item_count}")

    print(f"Aliases: {len(aliases)}")

    print(f"Ambiguous aliases: {len(ambiguous_aliases)}")

    print()

    print(f"Entities file: {entities_path}")

    print(f"Aliases file: {aliases_path}")

    print(f"Metadata file: {metadata_path}")

    print(f"Entities SHA256: {metadata.entities_sha256}")

    print(f"Aliases SHA256: {metadata.aliases_sha256}")

    if metadata.entity_count != len(entities):
        raise RuntimeError("Entity count mismatch")

    if metadata.alias_count != len(aliases):
        raise RuntimeError("Alias count mismatch")

    print()

    print("ENTITY_CATALOG_BUILD=PASS")


if __name__ == "__main__":
    main()
