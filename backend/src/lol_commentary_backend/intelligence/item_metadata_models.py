from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DDRAGON_ITEM_METADATA_VERSION = "ddragon_item_metadata_v1"

DDRAGON_ITEM_CATALOG_POLICY_VERSION = "ddragon_item_catalog_policy_v2_skip_bilingual_blank_names"


class DDragonItemMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    metadata_version: Literal["ddragon_item_metadata_v1"] = DDRAGON_ITEM_METADATA_VERSION

    ddragon_version: str = Field(
        min_length=1,
    )

    item_id: int = Field(
        ge=1,
    )

    name_ko: str = Field(
        min_length=1,
    )

    name_en: str = Field(
        min_length=1,
    )

    plaintext_ko: str = ""

    plaintext_en: str = ""

    gold_base: int = Field(
        ge=0,
    )

    gold_total: int = Field(
        ge=0,
    )

    gold_sell: int = Field(
        ge=0,
    )

    purchasable: bool

    tags: tuple[
        str,
        ...,
    ]

    from_item_ids: tuple[
        int,
        ...,
    ]

    into_item_ids: tuple[
        int,
        ...,
    ]

    map_ids: tuple[
        int,
        ...,
    ]

    depth: int | None = Field(
        default=None,
        ge=1,
    )

    consumed: bool = False

    consume_on_full: bool = False

    special_recipe: int | None = Field(
        default=None,
        ge=0,
    )

    required_champion: str | None = None

    required_ally: str | None = None


class DDragonItemCatalogInfo(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    catalog_policy_version: Literal["ddragon_item_catalog_policy_v2_skip_bilingual_blank_names"] = (
        DDRAGON_ITEM_CATALOG_POLICY_VERSION
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    raw_item_count: int = Field(
        ge=1,
    )

    item_count: int = Field(
        ge=1,
    )

    skipped_item_count: int = Field(
        ge=0,
    )

    skipped_item_ids: tuple[
        int,
        ...,
    ]

    ko_source_path: str = Field(
        min_length=1,
    )

    en_source_path: str = Field(
        min_length=1,
    )

    ko_source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    en_source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    catalog_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
