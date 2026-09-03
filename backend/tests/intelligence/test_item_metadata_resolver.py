from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_commentary_backend.intelligence.item_metadata_resolver import (
    DDragonItemMetadataResolver,
)

VERSION = "16.17.1"


def _item(
    *,
    name: str,
    plaintext: str,
    gold_total: int,
    from_ids: list[str] | None = None,
    into_ids: list[str] | None = None,
    depth: int | None = None,
) -> dict[str, object]:
    result: dict[
        str,
        object,
    ] = {
        "name": name,
        "plaintext": plaintext,
        "gold": {
            "base": gold_total,
            "purchasable": True,
            "total": gold_total,
            "sell": int(gold_total * 0.7),
        },
        "tags": [
            "Boots",
        ],
        "maps": {
            "11": True,
            "12": False,
        },
    }

    if from_ids is not None:
        result["from"] = from_ids

    if into_ids is not None:
        result["into"] = into_ids

    if depth is not None:
        result["depth"] = depth

    return result


def _write_locale(
    *,
    root: Path,
    locale: str,
    version: str = VERSION,
    item_ids: tuple[
        str,
        ...,
    ] = (
        "1001",
        "3006",
    ),
    blank_name_ids: tuple[
        str,
        ...,
    ] = (),
) -> None:
    names = {
        "ko_KR": {
            "1001": "장화",
            "3006": "광전사의 군화",
            "2008": "",
        },
        "en_US": {
            "1001": "Boots",
            "3006": ("Berserker's Greaves"),
            "2008": "",
        },
    }

    data: dict[
        str,
        object,
    ] = {}

    for item_id in item_ids:
        if item_id == "1001":
            data[item_id] = _item(
                name=names[locale][item_id],
                plaintext="Move faster.",
                gold_total=300,
                into_ids=[
                    "3006",
                ],
                depth=1,
            )

        elif item_id == "3006":
            data[item_id] = _item(
                name=names[locale][item_id],
                plaintext=("Attack speed boots."),
                gold_total=1100,
                from_ids=[
                    "1001",
                ],
                depth=2,
            )

        elif item_id == "2008":
            data[item_id] = _item(
                name="",
                plaintext="",
                gold_total=60,
            )

        else:
            raise ValueError(f"Unsupported test item ID: {item_id}")

    for item_id in blank_name_ids:
        value = data.get(item_id)

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(f"blank_name_ids contains unknown item: {item_id}")

        value["name"] = ""

    path = root / "data" / "raw" / "ddragon" / VERSION / locale / "item" / "item.json"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "type": "item",
        "version": version,
        "data": data,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_catalog(
    root: Path,
) -> None:
    _write_locale(
        root=root,
        locale="ko_KR",
    )

    _write_locale(
        root=root,
        locale="en_US",
    )


def test_loads_bilingual_item_names(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    item = resolver.require(1001)

    assert item.name_ko == "장화"

    assert item.name_en == "Boots"


def test_uses_collector_path_contract(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    assert resolver.catalog_info.ko_source_path == ("data/raw/ddragon/16.17.1/ko_KR/item/item.json")

    assert resolver.catalog_info.en_source_path == ("data/raw/ddragon/16.17.1/en_US/item/item.json")


def test_parses_gold_and_components(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    boots = resolver.require(1001)

    upgrade = resolver.require(3006)

    assert boots.gold_total == 300

    assert boots.into_item_ids == (3006,)

    assert upgrade.from_item_ids == (1001,)


def test_parses_tags_maps_and_depth(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    item = resolver.require(3006)

    assert item.tags == ("Boots",)

    assert item.map_ids == (11,)

    assert item.depth == 2


def test_unknown_item_returns_none(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    assert resolver.resolve(999999) is None


def test_require_unknown_item_raises(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    with pytest.raises(
        KeyError,
        match="Unknown or unusable",
    ):
        resolver.require(999999)


def test_version_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    _write_locale(
        root=tmp_path,
        locale="ko_KR",
        version="16.16.1",
    )

    _write_locale(
        root=tmp_path,
        locale="en_US",
    )

    with pytest.raises(
        ValueError,
        match="version mismatch",
    ):
        (
            DDragonItemMetadataResolver.from_repository(
                repository_root=(tmp_path),
                ddragon_version=VERSION,
            )
        )


def test_locale_item_id_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    _write_locale(
        root=tmp_path,
        locale="ko_KR",
        item_ids=(
            "1001",
            "3006",
        ),
    )

    _write_locale(
        root=tmp_path,
        locale="en_US",
        item_ids=("1001",),
    )

    with pytest.raises(
        ValueError,
        match="item ID sets differ",
    ):
        (
            DDragonItemMetadataResolver.from_repository(
                repository_root=(tmp_path),
                ddragon_version=VERSION,
            )
        )


def test_catalog_lineage_hashes_exist(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    info = resolver.catalog_info

    assert len(info.ko_source_sha256) == 64

    assert len(info.en_source_sha256) == 64

    assert len(info.catalog_sha256) == 64


def test_catalog_count_matches_resolver(
    tmp_path: Path,
) -> None:
    _write_catalog(tmp_path)

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    assert resolver.item_count == 2

    assert resolver.catalog_info.raw_item_count == 2

    assert resolver.catalog_info.item_count == 2

    assert resolver.catalog_info.skipped_item_count == 0


def test_bilingual_blank_name_item_is_skipped(
    tmp_path: Path,
) -> None:
    item_ids = (
        "1001",
        "3006",
        "2008",
    )

    _write_locale(
        root=tmp_path,
        locale="ko_KR",
        item_ids=item_ids,
    )

    _write_locale(
        root=tmp_path,
        locale="en_US",
        item_ids=item_ids,
    )

    resolver = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    assert resolver.item_count == 2

    assert resolver.resolve(2008) is None

    assert resolver.is_skipped(2008) is True

    assert resolver.catalog_info.raw_item_count == 3

    assert resolver.catalog_info.skipped_item_count == 1

    assert resolver.catalog_info.skipped_item_ids == (2008,)


def test_one_sided_blank_name_is_rejected(
    tmp_path: Path,
) -> None:
    item_ids = (
        "1001",
        "3006",
    )

    _write_locale(
        root=tmp_path,
        locale="ko_KR",
        item_ids=item_ids,
        blank_name_ids=("1001",),
    )

    _write_locale(
        root=tmp_path,
        locale="en_US",
        item_ids=item_ids,
    )

    with pytest.raises(
        ValueError,
        match=("name availability differs"),
    ):
        (
            DDragonItemMetadataResolver.from_repository(
                repository_root=(tmp_path),
                ddragon_version=VERSION,
            )
        )


def test_skipped_item_accounting_is_deterministic(
    tmp_path: Path,
) -> None:
    item_ids = (
        "1001",
        "3006",
        "2008",
    )

    _write_locale(
        root=tmp_path,
        locale="ko_KR",
        item_ids=item_ids,
    )

    _write_locale(
        root=tmp_path,
        locale="en_US",
        item_ids=item_ids,
    )

    first = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    second = DDragonItemMetadataResolver.from_repository(
        repository_root=(tmp_path),
        ddragon_version=VERSION,
    )

    assert first.catalog_info == second.catalog_info

    assert first.catalog_info.raw_item_count == (
        first.catalog_info.item_count + first.catalog_info.skipped_item_count
    )
