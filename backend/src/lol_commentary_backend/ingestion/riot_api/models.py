from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class RiotAccount(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        populate_by_name=True,
    )

    puuid: str = Field(
        min_length=1,
    )

    game_name: str = Field(
        alias="gameName",
        min_length=1,
    )

    tag_line: str = Field(
        alias="tagLine",
        min_length=1,
    )


class RiotMatchIds(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    puuid: str = Field(
        min_length=1,
    )

    start: int = Field(
        ge=0,
    )

    count: int = Field(
        ge=1,
        le=100,
    )

    match_ids: tuple[
        str,
        ...,
    ]
