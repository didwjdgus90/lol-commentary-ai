from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from lol_commentary_backend.intelligence.item_power_spike_models import (
    ItemPowerSpikeContext,
    ItemPowerSpikeEvaluation,
)
from lol_commentary_backend.intelligence.item_power_spike_selection_models import (
    ITEM_POWER_SPIKE_SELECTION_POLICY_VERSION,
    ITEM_POWER_SPIKE_SELECTION_VERSION,
    ItemPowerSpikeSelectionContext,
    SelectedItemPowerSpikeSignal,
    SelectedItemPowerSpikeTier,
    SelectedItemShape,
)

MAX_SIGNAL_AGE_MS = 180_000

MIN_SELECTION_SCORE = 60

HIGH_SELECTION_SCORE = 85

MAX_SIGNALS_PER_CONTEXT = 2

MIN_INTERMEDIATE_GOLD = 2_000

MIN_INTERMEDIATE_DEPTH = 3


@dataclass(
    frozen=True,
    slots=True,
)
class _SelectionCandidate:
    context: ItemPowerSpikeContext

    evaluation: ItemPowerSpikeEvaluation

    shape: SelectedItemShape

    tier: SelectedItemPowerSpikeTier


def _shape(
    evaluation: ItemPowerSpikeEvaluation,
) -> SelectedItemShape | None:
    has_from = bool(evaluation.from_item_ids)

    has_into = bool(evaluation.into_item_ids)

    if has_from and not has_into:
        return SelectedItemShape.TERMINAL_BUILD

    if has_from and has_into:
        return SelectedItemShape.INTERMEDIATE_BUILD

    return None


def _tier(
    score: int,
) -> SelectedItemPowerSpikeTier:
    if score >= HIGH_SELECTION_SCORE:
        return SelectedItemPowerSpikeTier.HIGH

    return SelectedItemPowerSpikeTier.MEDIUM


def _eligible(
    *,
    context: ItemPowerSpikeContext,
    evaluation: ItemPowerSpikeEvaluation,
) -> _SelectionCandidate | None:
    if not (evaluation.commentary_candidate):
        return None

    if evaluation.age_ms_at_situation_start > MAX_SIGNAL_AGE_MS:
        return None

    if evaluation.score < MIN_SELECTION_SCORE:
        return None

    shape = _shape(evaluation)

    if shape is None:
        return None

    if shape == SelectedItemShape.INTERMEDIATE_BUILD:
        if evaluation.gold_total < MIN_INTERMEDIATE_GOLD:
            return None

        if evaluation.depth is None or evaluation.depth < MIN_INTERMEDIATE_DEPTH:
            return None

    return _SelectionCandidate(
        context=context,
        evaluation=evaluation,
        shape=shape,
        tier=_tier(evaluation.score),
    )


def _source_owner_sort_key(
    candidate: _SelectionCandidate,
) -> tuple[
    int,
    str,
    str,
]:
    return (
        candidate.evaluation.age_ms_at_situation_start,
        candidate.context.record_id,
        candidate.context.context_id,
    )


def _context_signal_sort_key(
    candidate: _SelectionCandidate,
) -> tuple[
    int,
    int,
    int,
    int,
    str,
]:
    return (
        -candidate.evaluation.score,
        candidate.evaluation.age_ms_at_situation_start,
        candidate.evaluation.participant_id,
        candidate.evaluation.item_id,
        candidate.evaluation.source_event_sha256,
    )


def _selection_id(
    *,
    source_context: ItemPowerSpikeContext,
    eligible_candidate_count: int,
    selected_signals: tuple[
        SelectedItemPowerSpikeSignal,
        ...,
    ],
) -> str:
    payload = {
        "selection_version": (ITEM_POWER_SPIKE_SELECTION_VERSION),
        "policy_version": (ITEM_POWER_SPIKE_SELECTION_POLICY_VERSION),
        "source_power_context_id": (source_context.context_id),
        "eligible_candidate_count": (eligible_candidate_count),
        "selected_signals": [
            {
                "participant_id": (signal.participant_id),
                "item_id": (signal.item_id),
                "score": (signal.score),
                "tier": (signal.tier.value),
                "shape": (signal.shape.value),
                "age_ms": (signal.age_ms_at_situation_start),
                "source_event_sha256": (signal.source_event_sha256),
            }
            for signal in selected_signals
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


def _to_signal(
    candidate: _SelectionCandidate,
) -> SelectedItemPowerSpikeSignal:
    evaluation = candidate.evaluation

    return SelectedItemPowerSpikeSignal(
        participant_id=(evaluation.participant_id),
        item_id=(evaluation.item_id),
        name_ko=(evaluation.name_ko),
        name_en=(evaluation.name_en),
        score=(evaluation.score),
        tier=(candidate.tier),
        shape=(candidate.shape),
        age_ms_at_situation_start=(evaluation.age_ms_at_situation_start),
        gold_total=(evaluation.gold_total),
        source_event_sha256=(evaluation.source_event_sha256),
    )


def select_item_power_spike_contexts(
    *,
    power_contexts: tuple[
        ItemPowerSpikeContext,
        ...,
    ],
    max_signals_per_context: int = (MAX_SIGNALS_PER_CONTEXT),
) -> tuple[
    ItemPowerSpikeSelectionContext,
    ...,
]:
    if max_signals_per_context <= 0:
        raise ValueError("max_signals_per_context must be positive")

    if not power_contexts:
        return ()

    context_ids = [context.context_id for context in power_contexts]

    if len(context_ids) != len(set(context_ids)):
        raise ValueError("Duplicate source power context ID")

    matches = {context.match_id for context in power_contexts}

    if len(matches) != 1:
        raise ValueError("Selection batch must contain one match only")

    all_candidates: list[_SelectionCandidate] = []

    eligible_count_by_context: dict[
        str,
        int,
    ] = {context.context_id: 0 for context in power_contexts}

    for context in power_contexts:
        if context.exact_inventory_claim_allowed:
            raise ValueError("Power context must forbid exact inventory claims")

        for evaluation in context.evaluations:
            candidate = _eligible(
                context=context,
                evaluation=evaluation,
            )

            if candidate is None:
                continue

            all_candidates.append(candidate)

            eligible_count_by_context[context.context_id] += 1

    candidates_by_source: dict[
        str,
        list[_SelectionCandidate],
    ] = {}

    for candidate in all_candidates:
        source_sha = candidate.evaluation.source_event_sha256

        candidates_by_source.setdefault(
            source_sha,
            [],
        ).append(candidate)

    owned_candidates: list[_SelectionCandidate] = []

    for candidates in candidates_by_source.values():
        owner = min(
            candidates,
            key=_source_owner_sort_key,
        )

        owned_candidates.append(owner)

    owned_by_context: dict[
        str,
        list[_SelectionCandidate],
    ] = {context.context_id: [] for context in power_contexts}

    for candidate in owned_candidates:
        owned_by_context[candidate.context.context_id].append(candidate)

    result: list[ItemPowerSpikeSelectionContext] = []

    for context in power_contexts:
        candidates = sorted(
            owned_by_context[context.context_id],
            key=_context_signal_sort_key,
        )

        best_by_participant_item: dict[
            tuple[
                int,
                int,
            ],
            _SelectionCandidate,
        ] = {}

        for candidate in candidates:
            key = (
                candidate.evaluation.participant_id,
                candidate.evaluation.item_id,
            )

            if key not in (best_by_participant_item):
                best_by_participant_item[key] = candidate

        deduplicated = sorted(
            best_by_participant_item.values(),
            key=_context_signal_sort_key,
        )

        selected_candidates = tuple(deduplicated[:max_signals_per_context])

        selected_signals = tuple(_to_signal(candidate) for candidate in selected_candidates)

        result.append(
            ItemPowerSpikeSelectionContext(
                selection_id=(
                    _selection_id(
                        source_context=context,
                        eligible_candidate_count=(eligible_count_by_context[context.context_id]),
                        selected_signals=(selected_signals),
                    )
                ),
                match_id=(context.match_id),
                record_id=(context.record_id),
                situation_id=(context.situation_id),
                source_power_context_id=(context.context_id),
                eligible_candidate_count=(eligible_count_by_context[context.context_id]),
                selected_signals=(selected_signals),
            )
        )

    selection_ids = [context.selection_id for context in result]

    if len(selection_ids) != len(set(selection_ids)):
        raise RuntimeError("Duplicate item power selection ID")

    selected_sources = [
        signal.source_event_sha256 for context in result for signal in context.selected_signals
    ]

    if len(selected_sources) != len(set(selected_sources)):
        raise RuntimeError("Duplicate purchase source escaped global selection")

    return tuple(result)
