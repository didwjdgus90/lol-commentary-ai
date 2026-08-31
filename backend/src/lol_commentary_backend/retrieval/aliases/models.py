from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AliasEntityType(StrEnum):
    CHAMPION = "champion"
    ITEM = "item"


class AliasSourceHash(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: str = Field(min_length=1)
    entity_type: AliasEntityType
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BilingualEntityAlias(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: AliasEntityType
    entity_key: str = Field(min_length=1)

    ko_name: str = Field(min_length=1)
    en_name: str = Field(min_length=1)


class BilingualAliasCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    builder_version: str = Field(min_length=1)

    ddragon_version: str = Field(min_length=1)
    ko_locale: str = Field(min_length=1)
    en_locale: str = Field(min_length=1)

    source_hashes: list[AliasSourceHash]
    aliases: list[BilingualEntityAlias]


class QueryExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(min_length=1)
    expanded_query: str = Field(min_length=1)

    matched_entity_keys: list[str]
    added_aliases: list[str]

    changed: bool
