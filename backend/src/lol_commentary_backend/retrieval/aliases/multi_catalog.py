from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from lol_commentary_backend.ingestion.entity_catalog.models import (
    CanonicalEntity,
    EntityType,
)
from lol_commentary_backend.retrieval.aliases.models import (
    AliasEntityType,
    AliasSourceHash,
    BilingualAliasCatalog,
    BilingualEntityAlias,
)

MULTI_PATCH_ALIAS_BUILDER_VERSION = "multi_patch_alias_catalog_v1"


def _version_key(
    version: str,
) -> tuple[
    int,
    int,
    int,
]:
    parts = version.split(".")

    if len(parts) != 3:
        raise ValueError(f"Invalid Data Dragon version: {version}")

    return (
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
    )


def _alias_entity_type(
    entity_type: EntityType,
) -> AliasEntityType:
    if entity_type == EntityType.CHAMPION:
        return AliasEntityType.CHAMPION

    if entity_type == EntityType.ITEM:
        return AliasEntityType.ITEM

    raise ValueError(f"Unsupported entity type: {entity_type}")


def load_canonical_entities(
    path: Path,
) -> tuple[
    CanonicalEntity,
    ...,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Entity catalog missing: {path}")

    entities = tuple(
        CanonicalEntity.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    if not entities:
        raise ValueError("Entity catalog is empty")

    return entities


def _composite_sha256(
    hashes: set[str],
) -> str:
    if not hashes:
        raise ValueError("Source hash set must not be empty")

    payload = "\n".join(sorted(hashes)).encode("utf-8")

    return sha256(payload).hexdigest()


def build_multi_patch_alias_catalog(
    entities: tuple[
        CanonicalEntity,
        ...,
    ],
) -> BilingualAliasCatalog:
    if not entities:
        raise ValueError("entities must not be empty")

    alias_records: dict[
        tuple[
            AliasEntityType,
            str,
            str,
            str,
        ],
        BilingualEntityAlias,
    ] = {}

    versions: set[str] = set()

    source_hashes: dict[
        tuple[
            str,
            AliasEntityType,
        ],
        set[str],
    ] = defaultdict(set)

    for entity in entities:
        alias_type = _alias_entity_type(entity.entity_type)

        by_version: dict[
            str,
            dict[str, str],
        ] = defaultdict(dict)

        for observation in entity.observations:
            versions.add(observation.ddragon_version)

            if observation.locale not in {
                "ko_KR",
                "en_US",
            }:
                continue

            by_version[observation.ddragon_version][observation.locale] = observation.name

            source_hashes[
                (
                    observation.locale,
                    alias_type,
                )
            ].add(observation.source_sha256)

        for version in sorted(
            by_version,
            key=_version_key,
        ):
            names = by_version[version]

            ko_name = names.get("ko_KR")

            en_name = names.get("en_US")

            if not ko_name or not en_name:
                continue

            key = (
                alias_type,
                entity.riot_key,
                ko_name.casefold(),
                en_name.casefold(),
            )

            alias_records[key] = BilingualEntityAlias(
                entity_type=(alias_type),
                entity_key=(entity.riot_key),
                ko_name=ko_name,
                en_name=en_name,
            )

    if not alias_records:
        raise ValueError("No bilingual aliases could be built")

    if not versions:
        raise ValueError("No Data Dragon versions were observed")

    ordered_versions = tuple(
        sorted(
            versions,
            key=_version_key,
        )
    )

    ordered_aliases = sorted(
        alias_records.values(),
        key=lambda alias: (
            alias.entity_type.value,
            alias.entity_key,
            alias.ko_name.casefold(),
            alias.en_name.casefold(),
        ),
    )

    ordered_source_hashes = [
        AliasSourceHash(
            locale=locale,
            entity_type=entity_type,
            sha256=_composite_sha256(hashes),
        )
        for (
            locale,
            entity_type,
        ), hashes in sorted(
            source_hashes.items(),
            key=lambda item: (
                item[0][0],
                item[0][1].value,
            ),
        )
    ]

    return BilingualAliasCatalog(
        builder_version=(MULTI_PATCH_ALIAS_BUILDER_VERSION),
        ddragon_version=(f"{ordered_versions[0]}..{ordered_versions[-1]}"),
        ko_locale="ko_KR",
        en_locale="en_US",
        source_hashes=(ordered_source_hashes),
        aliases=ordered_aliases,
    )


def save_multi_patch_alias_catalog(
    *,
    catalog: BilingualAliasCatalog,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        json.dumps(
            catalog.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(path)
