import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from lol_commentary_backend.retrieval.aliases.builder import (
    build_bilingual_alias_catalog,
)


def _download(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": ("lol-commentary-ai/step37-alias-builder")},
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        return response.read()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        default="16.1.1",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[2]

    raw_root = project_root / "data" / "raw" / "data_dragon_aliases" / args.version

    processed_dir = (
        project_root / "data" / "processed" / "retrieval" / "entity_aliases" / args.version
    )

    locales = (
        "ko_KR",
        "en_US",
    )
    entity_files = (
        "champion.json",
        "item.json",
    )

    downloaded: dict[
        tuple[str, str],
        bytes,
    ] = {}

    for locale in locales:
        locale_dir = raw_root / locale
        locale_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename in entity_files:
            url = f"https://ddragon.leagueoflegends.com/cdn/{args.version}/data/{locale}/{filename}"
            target = locale_dir / filename

            print(f"Downloading: {url}")

            data = _download(url)

            target.write_bytes(data)

            downloaded[(locale, filename)] = data

            print(f"  saved: {target} ({len(data)} bytes)")

    catalog = build_bilingual_alias_catalog(
        ddragon_version=args.version,
        ko_locale="ko_KR",
        en_locale="en_US",
        ko_champion_bytes=downloaded[("ko_KR", "champion.json")],
        en_champion_bytes=downloaded[("en_US", "champion.json")],
        ko_item_bytes=downloaded[("ko_KR", "item.json")],
        en_item_bytes=downloaded[("en_US", "item.json")],
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = processed_dir / "ko_en_aliases.json"

    output_path.write_text(
        json.dumps(
            catalog.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    champions = sum(alias.entity_type.value == "champion" for alias in catalog.aliases)
    items = sum(alias.entity_type.value == "item" for alias in catalog.aliases)

    print()
    print("=== BILINGUAL ENTITY ALIAS CATALOG ===")
    print(f"Data Dragon version: {catalog.ddragon_version}")
    print(f"Champion aliases: {champions}")
    print(f"Item aliases: {items}")
    print(f"Total aliases: {len(catalog.aliases)}")

    examples = [
        alias
        for alias in catalog.aliases
        if (
            alias.en_name
            in {
                "Tryndamere",
                "Aphelios",
                "Essence Reaver",
            }
        )
    ]

    print()
    print("Focused aliases:")

    for alias in examples:
        print(
            f"  {alias.en_name} <-> {alias.ko_name} ({alias.entity_type.value}:{alias.entity_key})"
        )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
