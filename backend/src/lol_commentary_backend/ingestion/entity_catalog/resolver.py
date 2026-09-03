from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from lol_commentary_backend.ingestion.entity_catalog.builder import (
    normalize_alias,
)
from lol_commentary_backend.ingestion.entity_catalog.models import (
    AliasEntry,
    CanonicalEntity,
    EntityType,
)
from lol_commentary_backend.ingestion.version_mapping.models import (
    PatchDataDragonMapping,
)


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


class EntityResolutionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: int = 1

    query: str

    normalized_query: str

    status: ResolutionStatus

    matched_alias: str | None = None

    patch: str | None = None

    ddragon_version: str | None = None

    entity_type: EntityType | None = None

    map_id: str | None = None

    candidate_entity_uids: tuple[
        str,
        ...,
    ]

    selected_entity_uid: str | None = None


class EntityCatalogResolver:
    def __init__(
        self,
        *,
        entities: tuple[
            CanonicalEntity,
            ...,
        ],
        aliases: tuple[
            AliasEntry,
            ...,
        ],
        mappings: tuple[
            PatchDataDragonMapping,
            ...,
        ],
    ) -> None:
        self._entities = {entity.entity_uid: entity for entity in entities}

        self._aliases = {alias.normalized_alias: alias for alias in aliases}

        self._mappings = {mapping.patch: mapping for mapping in mappings}

    @classmethod
    def from_files(
        cls,
        *,
        entities_path: Path,
        aliases_path: Path,
        mappings_path: Path,
    ) -> EntityCatalogResolver:
        entities = tuple(
            CanonicalEntity.model_validate_json(line)
            for line in (entities_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        )

        aliases = tuple(
            AliasEntry.model_validate_json(line)
            for line in (aliases_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        )

        mappings = tuple(
            PatchDataDragonMapping.model_validate_json(line)
            for line in (mappings_path.read_text(encoding="utf-8").splitlines())
            if line.strip()
        )

        if not entities:
            raise ValueError("Entity catalog is empty")

        if not aliases:
            raise ValueError("Alias index is empty")

        if not mappings:
            raise ValueError("Version mapping is empty")

        return cls(
            entities=entities,
            aliases=aliases,
            mappings=mappings,
        )

    def resolve(
        self,
        query: str,
        *,
        patch: str | None = None,
        entity_type: (EntityType | None) = None,
        map_id: str | None = None,
    ) -> EntityResolutionResult:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("query must not be empty")

        normalized_query = normalize_alias(cleaned_query)

        alias = self._aliases.get(normalized_query)

        if alias is None:
            return EntityResolutionResult(
                query=cleaned_query,
                normalized_query=(normalized_query),
                status=(ResolutionStatus.NOT_FOUND),
                patch=patch,
                entity_type=(entity_type),
                map_id=map_id,
                candidate_entity_uids=(),
            )

        ddragon_version: str | None = None

        if patch is not None:
            mapping = self._mappings.get(patch)

            if mapping is None:
                raise ValueError(f"No Data Dragon mapping for patch: {patch}")

            ddragon_version = mapping.ddragon_version

        candidates = [
            self._entities[entity_uid]
            for entity_uid in (alias.entity_uids)
            if (entity_uid in self._entities)
        ]

        if entity_type is not None:
            candidates = [entity for entity in candidates if (entity.entity_type == entity_type)]

        if ddragon_version is not None:
            candidates = [
                entity for entity in candidates if (ddragon_version in entity.ddragon_versions)
            ]

        if map_id is not None:
            filtered: list[CanonicalEntity] = []

            for entity in candidates:
                if entity.entity_type != EntityType.ITEM:
                    filtered.append(entity)
                    continue

                if map_id in (entity.map_ids):
                    filtered.append(entity)

            candidates = filtered

        candidate_uids = tuple(sorted(entity.entity_uid for entity in candidates))

        if len(candidate_uids) == 1:
            return EntityResolutionResult(
                query=cleaned_query,
                normalized_query=(normalized_query),
                status=(ResolutionStatus.RESOLVED),
                matched_alias=(alias.normalized_alias),
                patch=patch,
                ddragon_version=(ddragon_version),
                entity_type=(entity_type),
                map_id=map_id,
                candidate_entity_uids=(candidate_uids),
                selected_entity_uid=(candidate_uids[0]),
            )

        status = ResolutionStatus.AMBIGUOUS if candidate_uids else ResolutionStatus.NOT_FOUND

        return EntityResolutionResult(
            query=cleaned_query,
            normalized_query=(normalized_query),
            status=status,
            matched_alias=(alias.normalized_alias),
            patch=patch,
            ddragon_version=(ddragon_version),
            entity_type=(entity_type),
            map_id=map_id,
            candidate_entity_uids=(candidate_uids),
        )
