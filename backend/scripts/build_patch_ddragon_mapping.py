from pathlib import Path

from lol_commentary_backend.ingestion.version_mapping.builder import (
    build_patch_ddragon_mappings,
    save_patch_ddragon_mapping,
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    patch_manifest = repository_root / "data" / "manifests" / "patch_notes" / "raw_sources.jsonl"

    ddragon_manifest = repository_root / "data" / "manifests" / "ddragon" / "raw_sources.jsonl"

    output_dir = repository_root / "data" / "processed" / "version_mapping" / "v1"

    mappings = build_patch_ddragon_mappings(
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
        start_patch="26.1",
        end_patch="26.17",
    )

    (
        mappings_path,
        metadata_path,
        metadata,
    ) = save_patch_ddragon_mapping(
        output_dir=output_dir,
        mappings=mappings,
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
    )

    print("=== PATCH -> DDRAGON MAPPING ===")

    for mapping in mappings:
        print(f"{mapping.patch} -> {mapping.ddragon_version}")

    print()

    print(f"Mapping count: {metadata.mapping_count}")

    print(f"First patch: {metadata.first_patch}")

    print(f"Last patch: {metadata.last_patch}")

    print(f"Mapping file: {mappings_path}")

    print(f"Metadata file: {metadata_path}")

    print(f"Mapping SHA256: {metadata.mappings_sha256}")

    if metadata.mapping_count != 17:
        raise RuntimeError("Expected exactly 17 patch mappings")

    print()

    print("PATCH_DDRAGON_MAPPING_BUILD=PASS")


if __name__ == "__main__":
    main()
