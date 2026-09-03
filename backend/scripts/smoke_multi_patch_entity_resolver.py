from pathlib import Path

from lol_commentary_backend.ingestion.entity_catalog.models import (
    EntityType,
)
from lol_commentary_backend.ingestion.entity_catalog.resolver import (
    EntityCatalogResolver,
    ResolutionStatus,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    resolver = EntityCatalogResolver.from_files(
        entities_path=(
            repository_root / "data" / "processed" / "entities" / "v1" / "entities.jsonl"
        ),
        aliases_path=(repository_root / "data" / "processed" / "entities" / "v1" / "aliases.jsonl"),
        mappings_path=(
            repository_root
            / "data"
            / "processed"
            / "version_mapping"
            / "v1"
            / "patch_ddragon.jsonl"
        ),
    )

    print("=== MULTI-PATCH ENTITY RESOLVER SMOKE ===")

    ko = resolver.resolve(
        "아트록스",
        patch="26.17",
        entity_type=(EntityType.CHAMPION),
    )

    en = resolver.resolve(
        "Aatrox",
        patch="26.17",
        entity_type=(EntityType.CHAMPION),
    )

    essence = resolver.resolve(
        "정수 약탈자",
        patch="26.17",
        entity_type=(EntityType.ITEM),
        map_id="11",
    )

    arena = resolver.resolve(
        "정수 약탈자",
        patch="26.17",
        entity_type=(EntityType.ITEM),
        map_id="30",
    )

    for label, result in (
        ("KO_CHAMPION", ko),
        ("EN_CHAMPION", en),
        ("SR_ITEM", essence),
        ("ARENA_ITEM", arena),
    ):
        print()
        print(f"[{label}]")
        print(f"Query: {result.query}")
        print(f"Patch: {result.patch}")
        print(f"Data Dragon: {result.ddragon_version}")
        print(f"Status: {result.status}")
        print(f"Candidates: {result.candidate_entity_uids}")
        print(f"Selected: {result.selected_entity_uid}")

    if ko.status != ResolutionStatus.RESOLVED:
        raise RuntimeError("KO champion resolution failed")

    if en.selected_entity_uid != ko.selected_entity_uid:
        raise RuntimeError("KO/EN champion alias parity failed")

    if essence.status != ResolutionStatus.RESOLVED:
        raise RuntimeError("Summoner's Rift item resolution failed")

    if essence.selected_entity_uid != "item:3508":
        raise RuntimeError("Unexpected standard Essence Reaver entity")

    print()
    print("CHAMPION_ALIAS_PARITY=PASS")

    print("MAP_AWARE_ITEM_RESOLUTION=PASS")

    print("MULTI_PATCH_ENTITY_RESOLVER_SMOKE=PASS")


if __name__ == "__main__":
    main()
