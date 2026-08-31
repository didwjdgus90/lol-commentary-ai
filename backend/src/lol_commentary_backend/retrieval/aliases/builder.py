import json
from hashlib import sha256

from lol_commentary_backend.retrieval.aliases.models import (
    AliasEntityType,
    AliasSourceHash,
    BilingualAliasCatalog,
    BilingualEntityAlias,
)

ALIAS_BUILDER_VERSION = "0.1.0"


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _decode_json(data: bytes) -> dict[str, object]:
    payload = json.loads(data.decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Data Dragon payload must be a JSON object")

    return payload


def _data_mapping(
    payload: dict[str, object],
) -> dict[str, dict[str, object]]:
    raw_data = payload.get("data")

    if not isinstance(raw_data, dict):
        raise ValueError("Data Dragon payload has no data object")

    result: dict[str, dict[str, object]] = {}

    for key, value in raw_data.items():
        if isinstance(key, str) and isinstance(value, dict):
            result[key] = value

    return result


def _champion_aliases(
    ko_payload: dict[str, object],
    en_payload: dict[str, object],
) -> list[BilingualEntityAlias]:
    ko_by_key: dict[str, str] = {}
    en_by_key: dict[str, str] = {}

    for record in _data_mapping(ko_payload).values():
        key = record.get("key")
        name = record.get("name")

        if isinstance(key, str) and isinstance(name, str):
            ko_by_key[key] = name

    for record in _data_mapping(en_payload).values():
        key = record.get("key")
        name = record.get("name")

        if isinstance(key, str) and isinstance(name, str):
            en_by_key[key] = name

    shared_keys = sorted(
        set(ko_by_key) & set(en_by_key),
        key=lambda value: int(value),
    )

    return [
        BilingualEntityAlias(
            entity_type=AliasEntityType.CHAMPION,
            entity_key=key,
            ko_name=ko_by_key[key],
            en_name=en_by_key[key],
        )
        for key in shared_keys
    ]


def _item_aliases(
    ko_payload: dict[str, object],
    en_payload: dict[str, object],
) -> list[BilingualEntityAlias]:
    ko_data = _data_mapping(ko_payload)
    en_data = _data_mapping(en_payload)

    shared_ids = sorted(
        set(ko_data) & set(en_data),
        key=int,
    )

    aliases: list[BilingualEntityAlias] = []

    for item_id in shared_ids:
        ko_name = ko_data[item_id].get("name")
        en_name = en_data[item_id].get("name")

        if (
            not isinstance(ko_name, str)
            or not isinstance(en_name, str)
            or not ko_name.strip()
            or not en_name.strip()
        ):
            continue

        aliases.append(
            BilingualEntityAlias(
                entity_type=AliasEntityType.ITEM,
                entity_key=item_id,
                ko_name=ko_name.strip(),
                en_name=en_name.strip(),
            )
        )

    return aliases


def build_bilingual_alias_catalog(
    *,
    ddragon_version: str,
    ko_locale: str,
    en_locale: str,
    ko_champion_bytes: bytes,
    en_champion_bytes: bytes,
    ko_item_bytes: bytes,
    en_item_bytes: bytes,
) -> BilingualAliasCatalog:
    ko_champion = _decode_json(ko_champion_bytes)
    en_champion = _decode_json(en_champion_bytes)
    ko_item = _decode_json(ko_item_bytes)
    en_item = _decode_json(en_item_bytes)

    aliases = [
        *_champion_aliases(
            ko_champion,
            en_champion,
        ),
        *_item_aliases(
            ko_item,
            en_item,
        ),
    ]

    return BilingualAliasCatalog(
        builder_version=ALIAS_BUILDER_VERSION,
        ddragon_version=ddragon_version,
        ko_locale=ko_locale,
        en_locale=en_locale,
        source_hashes=[
            AliasSourceHash(
                locale=ko_locale,
                entity_type=AliasEntityType.CHAMPION,
                sha256=sha256_bytes(ko_champion_bytes),
            ),
            AliasSourceHash(
                locale=en_locale,
                entity_type=AliasEntityType.CHAMPION,
                sha256=sha256_bytes(en_champion_bytes),
            ),
            AliasSourceHash(
                locale=ko_locale,
                entity_type=AliasEntityType.ITEM,
                sha256=sha256_bytes(ko_item_bytes),
            ),
            AliasSourceHash(
                locale=en_locale,
                entity_type=AliasEntityType.ITEM,
                sha256=sha256_bytes(en_item_bytes),
            ),
        ],
        aliases=aliases,
    )
