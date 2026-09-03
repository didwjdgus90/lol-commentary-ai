from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ENTITY_CATALOG_VERSION = "entity_catalog_v1"


class EntityType(StrEnum):
    CHAMPION = "champion"
    ITEM = "item"


class EntityObservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    entity_type: EntityType

    entity_uid: str = Field(
        min_length=1,
    )

    riot_key: str = Field(
        min_length=1,
    )

    riot_id: str | None = None

    name: str = Field(
        min_length=1,
    )

    locale: str = Field(
        min_length=1,
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    source_file: str = Field(
        min_length=1,
    )

    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    map_ids: tuple[str, ...] = ()


class CanonicalEntity(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    catalog_version: Literal["entity_catalog_v1"] = ENTITY_CATALOG_VERSION

    entity_uid: str = Field(
        min_length=1,
    )

    entity_type: EntityType

    riot_key: str = Field(
        min_length=1,
    )

    riot_id: str | None = None

    aliases: tuple[
        str,
        ...,
    ]

    locales: tuple[
        str,
        ...,
    ]

    ddragon_versions: tuple[
        str,
        ...,
    ]

    map_ids: tuple[
        str,
        ...,
    ] = ()

    observations: tuple[
        EntityObservation,
        ...,
    ]


class AliasEntry(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    normalized_alias: str = Field(
        min_length=1,
    )

    display_aliases: tuple[
        str,
        ...,
    ]

    entity_uids: tuple[
        str,
        ...,
    ]


class EntityCatalogMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    catalog_version: Literal["entity_catalog_v1"] = ENTITY_CATALOG_VERSION

    ddragon_versions: tuple[
        str,
        ...,
    ]

    locales: tuple[
        str,
        ...,
    ]

    entity_count: int = Field(
        ge=0,
    )

    champion_count: int = Field(
        ge=0,
    )

    item_count: int = Field(
        ge=0,
    )

    alias_count: int = Field(
        ge=0,
    )

    observation_count: int = Field(
        ge=0,
    )

    entities_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    aliases_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
