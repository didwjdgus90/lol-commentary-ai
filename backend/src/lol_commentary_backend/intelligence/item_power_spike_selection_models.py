from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ITEM_POWER_SPIKE_SELECTION_VERSION = "item_power_spike_selection_v1"

ITEM_POWER_SPIKE_SELECTION_POLICY_VERSION = "item_power_spike_selection_policy_v2"


class SelectedItemPowerSpikeTier(StrEnum):
    MEDIUM = "medium"
    HIGH = "high"


class SelectedItemShape(StrEnum):
    TERMINAL_BUILD = "terminal_build"

    INTERMEDIATE_BUILD = "intermediate_build"


class SelectedItemPowerSpikeSignal(BaseModel):
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

    tier: SelectedItemPowerSpikeTier

    shape: SelectedItemShape

    age_ms_at_situation_start: int = Field(
        ge=0,
    )

    gold_total: int = Field(
        ge=0,
    )

    source_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


class ItemPowerSpikeSelectionContext(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    selection_version: Literal["item_power_spike_selection_v1"] = ITEM_POWER_SPIKE_SELECTION_VERSION

    policy_version: Literal["item_power_spike_selection_policy_v2"] = (
        ITEM_POWER_SPIKE_SELECTION_POLICY_VERSION
    )

    selection_id: str = Field(
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

    source_power_context_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    eligible_candidate_count: int = Field(
        ge=0,
    )

    selected_signals: tuple[
        SelectedItemPowerSpikeSignal,
        ...,
    ]

    evidence_authority: Literal["confirmed_purchase_plus_ddragon_selected"] = (
        "confirmed_purchase_plus_ddragon_selected"
    )

    exact_inventory_claim_allowed: Literal[False] = False
