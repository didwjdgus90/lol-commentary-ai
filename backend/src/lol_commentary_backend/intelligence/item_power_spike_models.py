from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.intelligence.item_evidence_models import (
    ItemEvidenceAction,
)

ITEM_POWER_SPIKE_CONTEXT_VERSION = "item_power_spike_context_v1"

ITEM_POWER_SPIKE_POLICY_VERSION = "item_power_spike_policy_v1"


class ItemPowerSpikeTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ItemPowerSpikeReason(StrEnum):
    RECENT_PURCHASE = "recent_purchase"
    FRESH_PURCHASE = "fresh_purchase"

    BUILT_FROM_COMPONENTS = "built_from_components"

    MULTI_COMPONENT_BUILD = "multi_component_build"

    TERMINAL_UPGRADE_PATH = "terminal_upgrade_path"

    HIGH_DEPTH = "high_depth"

    HIGH_TOTAL_GOLD = "high_total_gold"

    MEDIUM_TOTAL_GOLD = "medium_total_gold"

    UTILITY_ITEM = "utility_item"

    NOT_PURCHASABLE = "not_purchasable"

    WRONG_MAP = "wrong_map"


class ItemPowerSpikeEvaluation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
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

    action: ItemEvidenceAction

    timestamp_ms: int = Field(
        ge=0,
    )

    age_ms_at_situation_start: int = Field(
        ge=0,
    )

    gold_total: int = Field(
        ge=0,
    )

    depth: int | None = Field(
        default=None,
        ge=1,
    )

    from_item_ids: tuple[
        int,
        ...,
    ]

    into_item_ids: tuple[
        int,
        ...,
    ]

    tags: tuple[
        str,
        ...,
    ]

    score: int = Field(
        ge=0,
        le=100,
    )

    tier: ItemPowerSpikeTier

    commentary_candidate: bool

    reasons: tuple[
        ItemPowerSpikeReason,
        ...,
    ]

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class ItemPowerSpikeSignal(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    participant_id: int = Field(
        ge=1,
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

    score: int = Field(
        ge=0,
        le=100,
    )

    tier: ItemPowerSpikeTier

    age_ms_at_situation_start: int = Field(
        ge=0,
    )

    gold_total: int = Field(
        ge=0,
    )

    reasons: tuple[
        ItemPowerSpikeReason,
        ...,
    ]

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class ItemPowerSpikeContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    context_version: Literal["item_power_spike_context_v1"] = ITEM_POWER_SPIKE_CONTEXT_VERSION

    policy_version: Literal["item_power_spike_policy_v1"] = ITEM_POWER_SPIKE_POLICY_VERSION

    context_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    ddragon_version: str = Field(
        min_length=1,
    )

    ddragon_catalog_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    evaluations: tuple[
        ItemPowerSpikeEvaluation,
        ...,
    ]

    top_signals: tuple[
        ItemPowerSpikeSignal,
        ...,
    ]

    evidence_authority: Literal["confirmed_timeline_event_plus_ddragon"] = (
        "confirmed_timeline_event_plus_ddragon"
    )

    exact_inventory_claim_allowed: Literal[False] = False
