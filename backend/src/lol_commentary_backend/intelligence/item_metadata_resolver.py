from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from lol_commentary_backend.intelligence.item_metadata_models import (
    DDRAGON_ITEM_CATALOG_POLICY_VERSION,
    DDragonItemCatalogInfo,
    DDragonItemMetadata,
)

KO_LOCALE = "ko_KR"
EN_LOCALE = "en_US"

ITEM_RELATIVE_PATH = Path("item") / "item.json"


def _file_sha256(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _relative_posix(
    *,
    path: Path,
    repository_root: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def _item_path(
    *,
    repository_root: Path,
    ddragon_version: str,
    locale: str,
) -> Path:
    return (
        repository_root / "data" / "raw" / "ddragon" / ddragon_version / locale / ITEM_RELATIVE_PATH
    )


def _read_payload(
    *,
    path: Path,
    expected_version: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Data Dragon item snapshot not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Data Dragon item root must be object")

    version = payload.get("version")

    if version != expected_version:
        raise ValueError(
            f"Data Dragon item version mismatch: expected={expected_version} actual={version}"
        )

    data = payload.get("data")

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError("Data Dragon item payload must contain data object")

    return payload


def _item_records(
    payload: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[
        str,
        Any,
    ],
]:
    data = payload.get("data")

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError("Data Dragon item data must be object")

    result: dict[
        str,
        dict[
            str,
            Any,
        ],
    ] = {}

    for key, value in data.items():
        if not isinstance(
            key,
            str,
        ):
            raise ValueError("Data Dragon item key must be string")

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(f"Data Dragon item record must be object: {key}")

        result[key] = value

    return result


def _normalized_name(
    *,
    record: dict[
        str,
        Any,
    ],
    item_id: int,
    locale: str,
) -> str:
    value = record.get("name")

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(f"Item {item_id} {locale} field name must be string")

    return value.strip()


def _required_string(
    *,
    record: dict[
        str,
        Any,
    ],
    key: str,
    item_id: int,
) -> str:
    value = record.get(key)

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(f"Item {item_id} field {key} must be string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"Item {item_id} field {key} must not be empty")

    return normalized


def _optional_string(
    value: object,
) -> str | None:
    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError("Optional item string must be string")

    normalized = value.strip()

    if not normalized:
        return None

    return normalized


def _plaintext(
    record: dict[
        str,
        Any,
    ],
) -> str:
    value = record.get(
        "plaintext",
        "",
    )

    if value is None:
        return ""

    if not isinstance(
        value,
        str,
    ):
        raise ValueError("Item plaintext must be string")

    return value.strip()


def _nonnegative_int(
    *,
    value: object,
    field_name: str,
    item_id: int,
) -> int:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value < 0
    ):
        raise ValueError(f"Item {item_id} {field_name} must be non-negative integer")

    return value


def _optional_nonnegative_int(
    *,
    value: object,
    field_name: str,
    item_id: int,
) -> int | None:
    if value is None:
        return None

    return _nonnegative_int(
        value=value,
        field_name=field_name,
        item_id=item_id,
    )


def _optional_positive_int(
    *,
    value: object,
    field_name: str,
    item_id: int,
) -> int | None:
    if value is None:
        return None

    parsed = _nonnegative_int(
        value=value,
        field_name=field_name,
        item_id=item_id,
    )

    if parsed <= 0:
        raise ValueError(f"Item {item_id} {field_name} must be positive integer")

    return parsed


def _boolean(
    *,
    value: object,
    field_name: str,
    item_id: int,
    default: bool,
) -> bool:
    if value is None:
        return default

    if not isinstance(
        value,
        bool,
    ):
        raise ValueError(f"Item {item_id} {field_name} must be bool")

    return value


def _string_tuple(
    value: object,
) -> tuple[
    str,
    ...,
]:
    if value is None:
        return ()

    if not isinstance(
        value,
        list,
    ):
        raise ValueError("Expected item string list")

    result: list[str] = []

    for item in value:
        if not isinstance(
            item,
            str,
        ):
            raise ValueError("Expected string in item list")

        normalized = item.strip()

        if normalized:
            result.append(normalized)

    return tuple(sorted(set(result)))


def _item_id_tuple(
    value: object,
) -> tuple[
    int,
    ...,
]:
    if value is None:
        return ()

    if not isinstance(
        value,
        list,
    ):
        raise ValueError("Expected Data Dragon item ID list")

    result: set[int] = set()

    for raw_value in value:
        if not isinstance(
            raw_value,
            str,
        ):
            raise ValueError("Data Dragon related item ID must be string")

        try:
            item_id = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid related item ID: {raw_value}") from exc

        if item_id <= 0:
            raise ValueError("Related item ID must be positive")

        result.add(item_id)

    return tuple(sorted(result))


def _map_ids(
    value: object,
) -> tuple[
    int,
    ...,
]:
    if value is None:
        return ()

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError("Data Dragon item maps must be object")

    result: set[int] = set()

    for raw_map_id, enabled in value.items():
        if not isinstance(
            raw_map_id,
            str,
        ):
            raise ValueError("Data Dragon map ID must be string")

        if not isinstance(
            enabled,
            bool,
        ):
            raise ValueError("Data Dragon map flag must be bool")

        if not enabled:
            continue

        try:
            map_id = int(raw_map_id)
        except ValueError as exc:
            raise ValueError(f"Invalid Data Dragon map ID: {raw_map_id}") from exc

        result.add(map_id)

    return tuple(sorted(result))


def _gold_fields(
    *,
    record: dict[
        str,
        Any,
    ],
    item_id: int,
) -> tuple[
    int,
    int,
    int,
    bool,
]:
    gold = record.get("gold")

    if not isinstance(
        gold,
        dict,
    ):
        raise ValueError(f"Item {item_id} gold must be object")

    base = _nonnegative_int(
        value=gold.get("base"),
        field_name="gold.base",
        item_id=item_id,
    )

    total = _nonnegative_int(
        value=gold.get("total"),
        field_name="gold.total",
        item_id=item_id,
    )

    sell = _nonnegative_int(
        value=gold.get("sell"),
        field_name="gold.sell",
        item_id=item_id,
    )

    purchasable = _boolean(
        value=gold.get("purchasable"),
        field_name=("gold.purchasable"),
        item_id=item_id,
        default=False,
    )

    return (
        base,
        total,
        sell,
        purchasable,
    )


def _build_metadata(
    *,
    ddragon_version: str,
    item_id: int,
    ko_record: dict[
        str,
        Any,
    ],
    en_record: dict[
        str,
        Any,
    ],
) -> DDragonItemMetadata:
    (
        gold_base,
        gold_total,
        gold_sell,
        purchasable,
    ) = _gold_fields(
        record=ko_record,
        item_id=item_id,
    )

    (
        en_gold_base,
        en_gold_total,
        en_gold_sell,
        en_purchasable,
    ) = _gold_fields(
        record=en_record,
        item_id=item_id,
    )

    if (
        gold_base,
        gold_total,
        gold_sell,
        purchasable,
    ) != (
        en_gold_base,
        en_gold_total,
        en_gold_sell,
        en_purchasable,
    ):
        raise ValueError(f"Localized item gold metadata differs for item {item_id}")

    ko_from = _item_id_tuple(ko_record.get("from"))

    en_from = _item_id_tuple(en_record.get("from"))

    ko_into = _item_id_tuple(ko_record.get("into"))

    en_into = _item_id_tuple(en_record.get("into"))

    if ko_from != en_from:
        raise ValueError(f"Localized item from-list differs for {item_id}")

    if ko_into != en_into:
        raise ValueError(f"Localized item into-list differs for {item_id}")

    ko_tags = _string_tuple(ko_record.get("tags"))

    en_tags = _string_tuple(en_record.get("tags"))

    if ko_tags != en_tags:
        raise ValueError(f"Localized item tags differ for {item_id}")

    ko_maps = _map_ids(ko_record.get("maps"))

    en_maps = _map_ids(en_record.get("maps"))

    if ko_maps != en_maps:
        raise ValueError(f"Localized item maps differ for {item_id}")

    depth = _optional_positive_int(
        value=ko_record.get("depth"),
        field_name="depth",
        item_id=item_id,
    )

    en_depth = _optional_positive_int(
        value=en_record.get("depth"),
        field_name="depth",
        item_id=item_id,
    )

    if depth != en_depth:
        raise ValueError(f"Localized item depth differs for {item_id}")

    special_recipe = _optional_nonnegative_int(
        value=ko_record.get("specialRecipe"),
        field_name="specialRecipe",
        item_id=item_id,
    )

    en_special_recipe = _optional_nonnegative_int(
        value=en_record.get("specialRecipe"),
        field_name="specialRecipe",
        item_id=item_id,
    )

    if special_recipe != en_special_recipe:
        raise ValueError(f"Localized specialRecipe differs for {item_id}")

    consumed = _boolean(
        value=ko_record.get("consumed"),
        field_name="consumed",
        item_id=item_id,
        default=False,
    )

    en_consumed = _boolean(
        value=en_record.get("consumed"),
        field_name="consumed",
        item_id=item_id,
        default=False,
    )

    consume_on_full = _boolean(
        value=ko_record.get("consumeOnFull"),
        field_name="consumeOnFull",
        item_id=item_id,
        default=False,
    )

    en_consume_on_full = _boolean(
        value=en_record.get("consumeOnFull"),
        field_name="consumeOnFull",
        item_id=item_id,
        default=False,
    )

    if consumed != en_consumed or consume_on_full != en_consume_on_full:
        raise ValueError(f"Localized consumption metadata differs for {item_id}")

    required_champion = _optional_string(ko_record.get("requiredChampion"))

    en_required_champion = _optional_string(en_record.get("requiredChampion"))

    required_ally = _optional_string(ko_record.get("requiredAlly"))

    en_required_ally = _optional_string(en_record.get("requiredAlly"))

    if required_champion != en_required_champion:
        raise ValueError(f"Localized requiredChampion differs for {item_id}")

    if required_ally != en_required_ally:
        raise ValueError(f"Localized requiredAlly differs for {item_id}")

    return DDragonItemMetadata(
        ddragon_version=(ddragon_version),
        item_id=item_id,
        name_ko=_required_string(
            record=ko_record,
            key="name",
            item_id=item_id,
        ),
        name_en=_required_string(
            record=en_record,
            key="name",
            item_id=item_id,
        ),
        plaintext_ko=(_plaintext(ko_record)),
        plaintext_en=(_plaintext(en_record)),
        gold_base=gold_base,
        gold_total=gold_total,
        gold_sell=gold_sell,
        purchasable=purchasable,
        tags=ko_tags,
        from_item_ids=ko_from,
        into_item_ids=ko_into,
        map_ids=ko_maps,
        depth=depth,
        consumed=consumed,
        consume_on_full=(consume_on_full),
        special_recipe=(special_recipe),
        required_champion=(required_champion),
        required_ally=(required_ally),
    )


class DDragonItemMetadataResolver:
    def __init__(
        self,
        *,
        catalog_info: DDragonItemCatalogInfo,
        items: tuple[
            DDragonItemMetadata,
            ...,
        ],
    ) -> None:
        if not items:
            raise ValueError("Item metadata catalog must not be empty")

        by_id: dict[
            int,
            DDragonItemMetadata,
        ] = {}

        for item in items:
            if item.item_id in by_id:
                raise ValueError(f"Duplicate item ID: {item.item_id}")

            by_id[item.item_id] = item

        if len(by_id) != catalog_info.item_count:
            raise ValueError("Catalog item count does not match metadata")

        if catalog_info.raw_item_count != (
            catalog_info.item_count + catalog_info.skipped_item_count
        ):
            raise ValueError("Raw item accounting does not match usable plus skipped counts")

        if catalog_info.skipped_item_count != len(catalog_info.skipped_item_ids):
            raise ValueError("Skipped item count does not match skipped IDs")

        if set(by_id) & set(catalog_info.skipped_item_ids):
            raise ValueError("Skipped item ID must not exist in usable catalog")

        self.catalog_info = catalog_info

        self._items_by_id = by_id

    @classmethod
    def from_repository(
        cls,
        *,
        repository_root: Path,
        ddragon_version: str,
    ) -> DDragonItemMetadataResolver:
        clean_version = ddragon_version.strip()

        if not clean_version:
            raise ValueError("ddragon_version must not be empty")

        ko_path = _item_path(
            repository_root=(repository_root),
            ddragon_version=(clean_version),
            locale=KO_LOCALE,
        )

        en_path = _item_path(
            repository_root=(repository_root),
            ddragon_version=(clean_version),
            locale=EN_LOCALE,
        )

        ko_payload = _read_payload(
            path=ko_path,
            expected_version=(clean_version),
        )

        en_payload = _read_payload(
            path=en_path,
            expected_version=(clean_version),
        )

        ko_records = _item_records(ko_payload)

        en_records = _item_records(en_payload)

        ko_ids = set(ko_records)

        en_ids = set(en_records)

        if ko_ids != en_ids:
            only_ko = sorted(ko_ids - en_ids)

            only_en = sorted(en_ids - ko_ids)

            raise ValueError(
                "Localized Data Dragon "
                "item ID sets differ: "
                f"only_ko={only_ko[:10]} "
                f"only_en={only_en[:10]}"
            )

        items: list[DDragonItemMetadata] = []

        skipped_item_ids: list[int] = []

        for raw_item_id in sorted(
            ko_ids,
            key=int,
        ):
            try:
                item_id = int(raw_item_id)
            except ValueError as exc:
                raise ValueError(f"Data Dragon item ID must be numeric: {raw_item_id}") from exc

            if item_id <= 0:
                raise ValueError("Data Dragon item ID must be positive")

            ko_record = ko_records[raw_item_id]

            en_record = en_records[raw_item_id]

            ko_name = _normalized_name(
                record=ko_record,
                item_id=item_id,
                locale=KO_LOCALE,
            )

            en_name = _normalized_name(
                record=en_record,
                item_id=item_id,
                locale=EN_LOCALE,
            )

            ko_blank = not ko_name
            en_blank = not en_name

            if ko_blank != en_blank:
                raise ValueError(
                    "Localized item name "
                    "availability differs "
                    f"for item {item_id}: "
                    f"ko_blank={ko_blank} "
                    f"en_blank={en_blank}"
                )

            if ko_blank and en_blank:
                skipped_item_ids.append(item_id)

                continue

            items.append(
                _build_metadata(
                    ddragon_version=(clean_version),
                    item_id=item_id,
                    ko_record=ko_record,
                    en_record=en_record,
                )
            )

        ko_sha256 = _file_sha256(ko_path)

        en_sha256 = _file_sha256(en_path)

        catalog_sha256 = sha256(
            (
                DDRAGON_ITEM_CATALOG_POLICY_VERSION
                + "\0"
                + clean_version
                + "\0"
                + ko_sha256
                + "\0"
                + en_sha256
            ).encode("utf-8")
        ).hexdigest()

        skipped_item_ids_tuple = tuple(sorted(skipped_item_ids))

        catalog_info = DDragonItemCatalogInfo(
            ddragon_version=(clean_version),
            raw_item_count=len(ko_ids),
            item_count=len(items),
            skipped_item_count=len(skipped_item_ids_tuple),
            skipped_item_ids=(skipped_item_ids_tuple),
            ko_source_path=(
                _relative_posix(
                    path=ko_path,
                    repository_root=(repository_root),
                )
            ),
            en_source_path=(
                _relative_posix(
                    path=en_path,
                    repository_root=(repository_root),
                )
            ),
            ko_source_sha256=(ko_sha256),
            en_source_sha256=(en_sha256),
            catalog_sha256=(catalog_sha256),
        )

        return cls(
            catalog_info=(catalog_info),
            items=tuple(items),
        )

    def resolve(
        self,
        item_id: int,
    ) -> DDragonItemMetadata | None:
        if item_id <= 0:
            return None

        return self._items_by_id.get(item_id)

    def require(
        self,
        item_id: int,
    ) -> DDragonItemMetadata:
        result = self.resolve(item_id)

        if result is None:
            raise KeyError(f"Unknown or unusable Data Dragon item ID: {item_id}")

        return result

    def is_skipped(
        self,
        item_id: int,
    ) -> bool:
        return item_id in self.catalog_info.skipped_item_ids

    @property
    def item_count(
        self,
    ) -> int:
        return len(self._items_by_id)
