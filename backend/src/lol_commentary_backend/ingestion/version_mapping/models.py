from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VERSION_MAPPING_VERSION = "patch_ddragon_mapping_v1"

MAPPING_RULE = "season_2026_minor_alignment_v1"


class PatchDataDragonMapping(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    mapping_version: Literal["patch_ddragon_mapping_v1"] = VERSION_MAPPING_VERSION

    mapping_rule: Literal["season_2026_minor_alignment_v1"] = MAPPING_RULE

    patch: str = Field(
        pattern=r"^26\.[1-9]\d?$",
    )

    ddragon_version: str = Field(
        pattern=r"^16\.\d+\.\d+$",
    )

    patch_locales: tuple[
        str,
        ...,
    ]

    ddragon_locales: tuple[
        str,
        ...,
    ]


class VersionMappingMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    mapping_version: Literal["patch_ddragon_mapping_v1"] = VERSION_MAPPING_VERSION

    mapping_rule: Literal["season_2026_minor_alignment_v1"] = MAPPING_RULE

    mapping_count: int = Field(
        ge=0,
    )

    first_patch: str = Field(
        min_length=1,
    )

    last_patch: str = Field(
        min_length=1,
    )

    patch_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    ddragon_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    mappings_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
