from __future__ import annotations

import json
from hashlib import sha256

from lol_commentary_backend.intelligence.item_evidence_models import (
    ItemEvidenceAction,
    SituationItemEvidenceContext,
)
from lol_commentary_backend.intelligence.item_metadata_models import (
    DDragonItemMetadata,
)
from lol_commentary_backend.intelligence.item_metadata_resolver import (
    DDragonItemMetadataResolver,
)
from lol_commentary_backend.intelligence.item_power_spike_models import (
    ITEM_POWER_SPIKE_CONTEXT_VERSION,
    ITEM_POWER_SPIKE_POLICY_VERSION,
    ItemPowerSpikeContext,
    ItemPowerSpikeEvaluation,
    ItemPowerSpikeReason,
    ItemPowerSpikeSignal,
    ItemPowerSpikeTier,
)

DEFAULT_MAP_ID = 11

DEFAULT_TOP_SIGNAL_LIMIT = 3

COMMENTARY_CANDIDATE_MIN_SCORE = 50

HIGH_TIER_MIN_SCORE = 75

MEDIUM_TIER_MIN_SCORE = 50

UTILITY_TAGS = {
    "Consumable",
    "Trinket",
    "Vision",
}


def _clamp_score(
    value: int,
) -> int:
    return max(
        0,
        min(
            100,
            value,
        ),
    )


def _tier_for_score(
    score: int,
) -> ItemPowerSpikeTier:
    if score >= HIGH_TIER_MIN_SCORE:
        return ItemPowerSpikeTier.HIGH

    if score >= MEDIUM_TIER_MIN_SCORE:
        return ItemPowerSpikeTier.MEDIUM

    return ItemPowerSpikeTier.LOW


def _utility_item(
    metadata: DDragonItemMetadata,
) -> bool:
    return bool(set(metadata.tags) & UTILITY_TAGS)


def _map_available(
    *,
    metadata: DDragonItemMetadata,
    map_id: int,
) -> bool:
    if not metadata.map_ids:
        return True

    return map_id in metadata.map_ids


def _evaluate_score(
    *,
    metadata: DDragonItemMetadata,
    age_ms: int,
    map_id: int,
) -> tuple[
    int,
    bool,
    tuple[
        ItemPowerSpikeReason,
        ...,
    ],
]:
    score = 0

    reasons: list[ItemPowerSpikeReason] = []

    utility_item = _utility_item(metadata)

    map_available = _map_available(
        metadata=metadata,
        map_id=map_id,
    )

    if metadata.from_item_ids:
        score += 20

        reasons.append(ItemPowerSpikeReason.BUILT_FROM_COMPONENTS)

    if len(metadata.from_item_ids) >= 2:
        score += 10

        reasons.append(ItemPowerSpikeReason.MULTI_COMPONENT_BUILD)

    if not metadata.into_item_ids:
        score += 15

        reasons.append(ItemPowerSpikeReason.TERMINAL_UPGRADE_PATH)

    if metadata.depth is not None and metadata.depth >= 3:
        score += 10

        reasons.append(ItemPowerSpikeReason.HIGH_DEPTH)

    if metadata.gold_total >= 2800:
        score += 20

        reasons.append(ItemPowerSpikeReason.HIGH_TOTAL_GOLD)

    elif metadata.gold_total >= 2000:
        score += 14

        reasons.append(ItemPowerSpikeReason.MEDIUM_TOTAL_GOLD)

    elif metadata.gold_total >= 1200:
        score += 8

        reasons.append(ItemPowerSpikeReason.MEDIUM_TOTAL_GOLD)

    if age_ms <= 120_000:
        score += 15

        reasons.append(ItemPowerSpikeReason.FRESH_PURCHASE)

    elif age_ms <= 300_000:
        score += 8

        reasons.append(ItemPowerSpikeReason.RECENT_PURCHASE)

    if utility_item:
        reasons.append(ItemPowerSpikeReason.UTILITY_ITEM)

    if not metadata.purchasable:
        reasons.append(ItemPowerSpikeReason.NOT_PURCHASABLE)

    if not map_available:
        reasons.append(ItemPowerSpikeReason.WRONG_MAP)

    normalized_score = _clamp_score(score)

    commentary_candidate = (
        normalized_score >= COMMENTARY_CANDIDATE_MIN_SCORE
        and not utility_item
        and metadata.purchasable
        and map_available
    )

    return (
        normalized_score,
        commentary_candidate,
        tuple(reasons),
    )


def _evaluation_sort_key(
    evaluation: ItemPowerSpikeEvaluation,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    return (
        -evaluation.score,
        evaluation.age_ms_at_situation_start,
        evaluation.participant_id,
        evaluation.item_id,
    )


def _signal_sort_key(
    signal: ItemPowerSpikeSignal,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    return (
        -signal.score,
        signal.age_ms_at_situation_start,
        signal.participant_id,
        signal.item_id,
    )


def _context_id(
    *,
    item_context: SituationItemEvidenceContext,
    resolver: DDragonItemMetadataResolver,
    evaluations: tuple[
        ItemPowerSpikeEvaluation,
        ...,
    ],
    top_signals: tuple[
        ItemPowerSpikeSignal,
        ...,
    ],
) -> str:
    payload = {
        "context_version": (ITEM_POWER_SPIKE_CONTEXT_VERSION),
        "policy_version": (ITEM_POWER_SPIKE_POLICY_VERSION),
        "item_evidence_context_id": (item_context.context_id),
        "ddragon_version": (resolver.catalog_info.ddragon_version),
        "ddragon_catalog_sha256": (resolver.catalog_info.catalog_sha256),
        "evaluations": [
            {
                "participant_id": (evaluation.participant_id),
                "item_id": (evaluation.item_id),
                "source_event_sha256": (evaluation.source_event_sha256),
                "score": (evaluation.score),
                "candidate": (evaluation.commentary_candidate),
            }
            for evaluation in evaluations
        ],
        "top_signals": [
            {
                "participant_id": (signal.participant_id),
                "item_id": (signal.item_id),
                "score": signal.score,
                "source_event_sha256": (signal.source_event_sha256),
            }
            for signal in top_signals
        ],
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return sha256(canonical.encode("utf-8")).hexdigest()


def build_item_power_spike_context(
    *,
    item_context: SituationItemEvidenceContext,
    resolver: DDragonItemMetadataResolver,
    map_id: int = DEFAULT_MAP_ID,
    top_signal_limit: int = (DEFAULT_TOP_SIGNAL_LIMIT),
) -> ItemPowerSpikeContext:
    if map_id <= 0:
        raise ValueError("map_id must be positive")

    if top_signal_limit <= 0:
        raise ValueError("top_signal_limit must be positive")

    if item_context.exact_inventory_claim_allowed:
        raise ValueError("Item evidence context must forbid exact inventory claims")

    evaluations: list[ItemPowerSpikeEvaluation] = []

    for participant_context in item_context.participants:
        for evidence in participant_context.evidence:
            if evidence.action != ItemEvidenceAction.PURCHASED:
                continue

            metadata = resolver.resolve(evidence.item_id)

            if metadata is None:
                raise ValueError(
                    "Observed purchased "
                    "item has no usable "
                    "Data Dragon metadata: "
                    f"{evidence.item_id}"
                )

            (
                score,
                commentary_candidate,
                reasons,
            ) = _evaluate_score(
                metadata=metadata,
                age_ms=(evidence.age_ms_at_situation_start),
                map_id=map_id,
            )

            evaluations.append(
                ItemPowerSpikeEvaluation(
                    participant_id=(evidence.participant_id),
                    item_id=(evidence.item_id),
                    name_ko=(metadata.name_ko),
                    name_en=(metadata.name_en),
                    action=(evidence.action),
                    timestamp_ms=(evidence.timestamp_ms),
                    age_ms_at_situation_start=(evidence.age_ms_at_situation_start),
                    gold_total=(metadata.gold_total),
                    depth=(metadata.depth),
                    from_item_ids=(metadata.from_item_ids),
                    into_item_ids=(metadata.into_item_ids),
                    tags=(metadata.tags),
                    score=score,
                    tier=(_tier_for_score(score)),
                    commentary_candidate=(commentary_candidate),
                    reasons=reasons,
                    source_event_sha256=(evidence.source_event_sha256),
                )
            )

    ordered_evaluations = tuple(
        sorted(
            evaluations,
            key=_evaluation_sort_key,
        )
    )

    best_by_participant_item: dict[
        tuple[int, int],
        ItemPowerSpikeEvaluation,
    ] = {}

    for evaluation in ordered_evaluations:
        if not (evaluation.commentary_candidate):
            continue

        key = (
            evaluation.participant_id,
            evaluation.item_id,
        )

        if key not in best_by_participant_item:
            best_by_participant_item[key] = evaluation

    signals = [
        ItemPowerSpikeSignal(
            participant_id=(evaluation.participant_id),
            item_id=(evaluation.item_id),
            name_ko=(evaluation.name_ko),
            name_en=(evaluation.name_en),
            score=(evaluation.score),
            tier=(evaluation.tier),
            age_ms_at_situation_start=(evaluation.age_ms_at_situation_start),
            gold_total=(evaluation.gold_total),
            reasons=(evaluation.reasons),
            source_event_sha256=(evaluation.source_event_sha256),
        )
        for evaluation in best_by_participant_item.values()
    ]

    top_signals = tuple(
        sorted(
            signals,
            key=_signal_sort_key,
        )[:top_signal_limit]
    )

    context_identifier = _context_id(
        item_context=item_context,
        resolver=resolver,
        evaluations=(ordered_evaluations),
        top_signals=top_signals,
    )

    return ItemPowerSpikeContext(
        context_id=(context_identifier),
        match_id=(item_context.match_id),
        record_id=(item_context.record_id),
        situation_id=(item_context.situation_id),
        ddragon_version=(resolver.catalog_info.ddragon_version),
        ddragon_catalog_sha256=(resolver.catalog_info.catalog_sha256),
        evaluations=(ordered_evaluations),
        top_signals=top_signals,
    )


def build_item_power_spike_contexts(
    *,
    item_contexts: tuple[
        SituationItemEvidenceContext,
        ...,
    ],
    resolver: DDragonItemMetadataResolver,
    map_id: int = DEFAULT_MAP_ID,
    top_signal_limit: int = (DEFAULT_TOP_SIGNAL_LIMIT),
) -> tuple[
    ItemPowerSpikeContext,
    ...,
]:
    result = tuple(
        build_item_power_spike_context(
            item_context=(item_context),
            resolver=resolver,
            map_id=map_id,
            top_signal_limit=(top_signal_limit),
        )
        for item_context in item_contexts
    )

    context_ids = [context.context_id for context in result]

    if len(context_ids) != len(set(context_ids)):
        raise RuntimeError("Duplicate item power-spike context ID")

    return result
