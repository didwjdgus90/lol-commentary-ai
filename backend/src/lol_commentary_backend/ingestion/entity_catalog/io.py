from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lol_commentary_backend.ingestion.entity_catalog.models import (
    AliasEntry,
    CanonicalEntity,
    EntityCatalogMetadata,
    EntityType,
)


def _write_jsonl(
    *,
    path: Path,
    models: tuple,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        for model in models
    ]

    payload = "\n".join(lines) + "\n" if lines else ""

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(path)


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_entity_catalog(
    *,
    output_dir: Path,
    entities: tuple[
        CanonicalEntity,
        ...,
    ],
    aliases: tuple[
        AliasEntry,
        ...,
    ],
) -> tuple[
    Path,
    Path,
    Path,
    EntityCatalogMetadata,
]:
    if not entities:
        raise ValueError("entities must not be empty")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    entities_path = output_dir / "entities.jsonl"

    aliases_path = output_dir / "aliases.jsonl"

    metadata_path = output_dir / "metadata.json"

    _write_jsonl(
        path=entities_path,
        models=entities,
    )

    _write_jsonl(
        path=aliases_path,
        models=aliases,
    )

    versions = tuple(
        sorted(
            {version for entity in entities for version in entity.ddragon_versions},
            key=lambda value: tuple(int(part) for part in value.split(".")),
        )
    )

    locales = tuple(sorted({locale for entity in entities for locale in entity.locales}))

    champion_count = sum(entity.entity_type == EntityType.CHAMPION for entity in entities)

    item_count = sum(entity.entity_type == EntityType.ITEM for entity in entities)

    observation_count = sum(len(entity.observations) for entity in entities)

    metadata = EntityCatalogMetadata(
        ddragon_versions=versions,
        locales=locales,
        entity_count=len(entities),
        champion_count=(champion_count),
        item_count=item_count,
        alias_count=len(aliases),
        observation_count=(observation_count),
        entities_sha256=(_sha256_file(entities_path)),
        aliases_sha256=(_sha256_file(aliases_path)),
    )

    temporary = metadata_path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(
            metadata.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(metadata_path)

    return (
        entities_path,
        aliases_path,
        metadata_path,
        metadata,
    )
