from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.entity_catalog.models import (
    AliasEntry,
    CanonicalEntity,
    EntityObservation,
    EntityType,
)

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def normalize_alias(
    value: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )

    return " ".join(normalized.split()).casefold()


def _version_key(
    version: str,
) -> tuple[
    int,
    int,
    int,
]:
    if _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"Invalid Data Dragon version: {version}")

    major, minor, build = version.split(".")

    return (
        int(major),
        int(minor),
        int(build),
    )


def discover_ddragon_versions(
    ddragon_root: Path,
) -> tuple[str, ...]:
    if not ddragon_root.is_dir():
        raise FileNotFoundError(f"Data Dragon root not found: {ddragon_root}")

    versions = [
        path.name
        for path in (ddragon_root.iterdir())
        if (path.is_dir() and _VERSION_PATTERN.fullmatch(path.name))
    ]

    if not versions:
        raise ValueError("No versioned Data Dragon directories found")

    return tuple(
        sorted(
            versions,
            key=_version_key,
        )
    )


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_payload_version(
    *,
    payload: dict[str, Any],
    expected_version: str,
    path: Path,
) -> None:
    raw_version = payload.get("version")

    if raw_version is None:
        return

    if raw_version != expected_version:
        raise ValueError(
            "Data Dragon payload "
            "version mismatch: "
            f"{path} expected="
            f"{expected_version} "
            f"actual={raw_version}"
        )


def _relative_posix(
    *,
    path: Path,
    repository_root: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def load_champion_observations(
    *,
    repository_root: Path,
    ddragon_root: Path,
    version: str,
    locale: str,
) -> tuple[
    EntityObservation,
    ...,
]:
    path = ddragon_root / version / locale / "champion" / "champion.json"

    payload = _load_json(path)

    _validate_payload_version(
        payload=payload,
        expected_version=version,
        path=path,
    )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(f"Champion payload has no data object: {path}")

    source_sha256 = _sha256_file(path)

    observations: list[EntityObservation] = []

    for raw in data.values():
        if not isinstance(
            raw,
            dict,
        ):
            continue

        riot_id = raw.get("id")
        riot_key = raw.get("key")
        name = raw.get("name")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                riot_id,
                riot_key,
                name,
            )
        ):
            raise ValueError(f"Champion entry is incomplete: {path}")

        observations.append(
            EntityObservation(
                entity_type=(EntityType.CHAMPION),
                entity_uid=(f"champion:{riot_key}"),
                riot_key=riot_key,
                riot_id=riot_id,
                name=name,
                locale=locale,
                ddragon_version=(version),
                source_file=(
                    _relative_posix(
                        path=path,
                        repository_root=(repository_root),
                    )
                ),
                source_sha256=(source_sha256),
            )
        )

    return tuple(observations)


def _item_map_ids(
    raw: dict[str, Any],
) -> tuple[str, ...]:
    maps = raw.get("maps")

    if not isinstance(
        maps,
        dict,
    ):
        return ()

    result = [str(map_id) for map_id, enabled in maps.items() if enabled is True]

    return tuple(sorted(result))


def load_item_observations(
    *,
    repository_root: Path,
    ddragon_root: Path,
    version: str,
    locale: str,
) -> tuple[
    EntityObservation,
    ...,
]:
    path = ddragon_root / version / locale / "item" / "item.json"

    payload = _load_json(path)

    _validate_payload_version(
        payload=payload,
        expected_version=version,
        path=path,
    )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(f"Item payload has no data object: {path}")

    source_sha256 = _sha256_file(path)

    observations: list[EntityObservation] = []

    for item_id, raw in data.items():
        if not isinstance(
            item_id,
            str,
        ) or not isinstance(
            raw,
            dict,
        ):
            continue

        name = raw.get("name")

        if not isinstance(name, str) or not name.strip():
            continue

        observations.append(
            EntityObservation(
                entity_type=(EntityType.ITEM),
                entity_uid=(f"item:{item_id}"),
                riot_key=item_id,
                riot_id=None,
                name=name,
                locale=locale,
                ddragon_version=(version),
                source_file=(
                    _relative_posix(
                        path=path,
                        repository_root=(repository_root),
                    )
                ),
                source_sha256=(source_sha256),
                map_ids=(_item_map_ids(raw)),
            )
        )

    return tuple(observations)


def collect_entity_observations(
    *,
    repository_root: Path,
    versions: tuple[
        str,
        ...,
    ],
    locales: tuple[
        str,
        ...,
    ],
) -> tuple[
    EntityObservation,
    ...,
]:
    ddragon_root = repository_root / "data" / "raw" / "ddragon"

    observations: list[EntityObservation] = []

    for version in versions:
        for locale in locales:
            observations.extend(
                load_champion_observations(
                    repository_root=(repository_root),
                    ddragon_root=(ddragon_root),
                    version=version,
                    locale=locale,
                )
            )

            observations.extend(
                load_item_observations(
                    repository_root=(repository_root),
                    ddragon_root=(ddragon_root),
                    version=version,
                    locale=locale,
                )
            )

    if not observations:
        raise ValueError("No entity observations were created")

    return tuple(observations)


def _unique_sorted(
    values: list[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda value: (
                normalize_alias(value),
                value,
            ),
        )
    )


def build_canonical_entities(
    observations: tuple[
        EntityObservation,
        ...,
    ],
) -> tuple[
    CanonicalEntity,
    ...,
]:
    grouped: dict[
        str,
        list[EntityObservation],
    ] = defaultdict(list)

    for observation in observations:
        grouped[observation.entity_uid].append(observation)

    entities: list[CanonicalEntity] = []

    for entity_uid in sorted(grouped):
        rows = grouped[entity_uid]

        entity_types = {row.entity_type for row in rows}

        riot_keys = {row.riot_key for row in rows}

        riot_ids = {row.riot_id for row in rows if row.riot_id}

        if len(entity_types) != 1:
            raise ValueError(f"Entity type conflict for {entity_uid}")

        if len(riot_keys) != 1:
            raise ValueError(f"Riot key conflict for {entity_uid}")

        entity_type = next(iter(entity_types))

        if entity_type == EntityType.CHAMPION and len(riot_ids) != 1:
            raise ValueError(f"Champion Riot ID conflict for {entity_uid}: {sorted(riot_ids)}")

        riot_id = next(iter(riot_ids)) if riot_ids else None

        aliases = [row.name for row in rows]

        if riot_id is not None:
            aliases.append(riot_id)

        ordered_observations = tuple(
            sorted(
                rows,
                key=lambda row: (
                    _version_key(row.ddragon_version),
                    row.locale,
                    row.name,
                ),
            )
        )

        entities.append(
            CanonicalEntity(
                entity_uid=entity_uid,
                entity_type=(entity_type),
                riot_key=next(iter(riot_keys)),
                riot_id=riot_id,
                aliases=(_unique_sorted(aliases)),
                locales=(tuple(sorted({row.locale for row in rows}))),
                ddragon_versions=(
                    tuple(
                        sorted(
                            {row.ddragon_version for row in rows},
                            key=_version_key,
                        )
                    )
                ),
                map_ids=(tuple(sorted({map_id for row in rows for map_id in row.map_ids}))),
                observations=(ordered_observations),
            )
        )

    return tuple(entities)


def build_alias_index(
    entities: tuple[
        CanonicalEntity,
        ...,
    ],
) -> tuple[
    AliasEntry,
    ...,
]:
    entity_ids_by_alias: dict[
        str,
        set[str],
    ] = defaultdict(set)

    display_values: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for entity in entities:
        for alias in entity.aliases:
            normalized = normalize_alias(alias)

            if not normalized:
                continue

            entity_ids_by_alias[normalized].add(entity.entity_uid)

            display_values[normalized].add(alias)

    return tuple(
        AliasEntry(
            normalized_alias=(normalized),
            display_aliases=tuple(sorted(display_values[normalized])),
            entity_uids=tuple(sorted(entity_ids_by_alias[normalized])),
        )
        for normalized in sorted(entity_ids_by_alias)
    )
