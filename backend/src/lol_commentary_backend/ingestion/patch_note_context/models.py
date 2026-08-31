from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContextMethod(StrEnum):
    HOTFIX_STRUCTURAL_EXACT = "hotfix_structural_exact"


class ContextConfidence(StrEnum):
    HIGH = "high"


class HotfixChampionContextHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1

    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_order: int = Field(ge=0)

    patch: str = Field(min_length=1)
    locale: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    champion_name: str = Field(min_length=1)
    champion_id: str = Field(min_length=1)
    champion_key: str = Field(min_length=1)

    anchor_title: str = Field(min_length=1)
    detail_title: str = Field(min_length=1)

    context_method: ContextMethod
    context_confidence: ContextConfidence

    evidence: list[str] = Field(min_length=1)
