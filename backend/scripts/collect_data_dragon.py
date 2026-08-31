from pathlib import Path

from lol_commentary_backend.ingestion.data_dragon import (
    DataDragonTarget,
    fetch_data_dragon_bundle,
    save_raw_data_dragon_bundle,
)

TARGET = DataDragonTarget(
    version="16.1.1",
    locale="ko_KR",
)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    output_dir = repository_root / "data" / "raw" / "data_dragon" / TARGET.version / TARGET.locale

    bundle = fetch_data_dragon_bundle(TARGET)

    (
        versions_path,
        champion_path,
        item_path,
        metadata_path,
    ) = save_raw_data_dragon_bundle(
        bundle,
        output_dir,
    )

    print(f"Data Dragon version: {TARGET.version}")
    print(f"Locale: {TARGET.locale}")
    print(f"Versions saved: {versions_path}")
    print(f"Champions saved: {champion_path}")
    print(f"Items saved: {item_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Champion SHA-256: {bundle.champion.sha256}")
    print(f"Item SHA-256: {bundle.item.sha256}")


if __name__ == "__main__":
    main()
