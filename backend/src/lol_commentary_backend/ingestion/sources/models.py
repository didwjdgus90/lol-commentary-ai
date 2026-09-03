from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceFamily(StrEnum):
    PATCH_NOTES = "patch_notes"
    DDRAGON = "ddragon"
    GAME_CONSTANTS = "game_constants"
    RIOT_API = "riot_api"
    LIVE_CLIENT = "live_client"
    REPLAY = "replay"


class SourceAuth(StrEnum):
    PUBLIC = "public"
    RIOT_API_KEY = "riot_api_key"
    LOCAL_GAME_CLIENT = "local_game_client"


class SourceFormat(StrEnum):
    HTML = "html"
    JSON = "json"


class SourcePurpose(StrEnum):
    RAG = "rag"
    ENTITY = "entity"
    MATCH = "match"
    LIVE = "live"
    REPLAY = "replay"


class DataSourceSpec(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_]*$",
    )

    name: str = Field(
        min_length=1,
    )

    family: SourceFamily

    auth: SourceAuth

    response_format: SourceFormat

    url_template: str = Field(
        min_length=1,
    )

    purposes: tuple[
        SourcePurpose,
        ...,
    ] = Field(
        min_length=1,
    )

    storage_subdir: str = Field(
        min_length=1,
    )

    locales: tuple[str, ...] = ()

    versioned: bool = False

    probe_enabled: bool = False

    runtime_only: bool = False

    @field_validator(
        "url_template",
    )
    @classmethod
    def validate_url_template(
        cls,
        value: str,
    ) -> str:
        if not value.startswith("https://"):
            raise ValueError("url_template must use HTTPS")

        return value

    @field_validator(
        "storage_subdir",
    )
    @classmethod
    def validate_storage_subdir(
        cls,
        value: str,
    ) -> str:
        normalized = value.replace("\\", "/").strip("/")

        if not normalized or ".." in normalized.split("/"):
            raise ValueError("storage_subdir must be a safe relative path")

        return normalized
