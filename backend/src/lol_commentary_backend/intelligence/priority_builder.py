from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

from lol_commentary_backend.intelligence.game_state_models import (
    SituationStateContext,
)
from lol_commentary_backend.intelligence.priority_models import (
    MACRO_GOLD_P90_THRESHOLD,
    MacroContextReference,
    PriorityReason,
    SituationPriority,
    SituationPriorityTier,
)
from lol_commentary_backend.intelligence.situation_models import (
    SituationKind,
    TemporalSituation,
)


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def _priority_sort_key(
    priority: SituationPriority,
) -> tuple[str, str]:
    return (
        priority.match_id,
        priority.situation_id,
    )


def _member_count(
    situation: TemporalSituation,
) -> int:
    return len(situation.primary_candidates) + len(situation.semantic_markers)


def _anchor_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, int, int, str]:
    return (
        situation.max_salience_score,
        _member_count(situation),
        len(situation.semantic_markers),
        -situation.start_timestamp_ms,
        situation.situation_id,
    )


def _macro_interval_id(
    *,
    match_id: str,
    before_frame_index: int,
    after_frame_index: int,
) -> str:
    identity = f"{match_id}|{before_frame_index}|{after_frame_index}"

    return sha256(identity.encode("utf-8")).hexdigest()


def _context_key(
    context: SituationStateContext,
) -> tuple[str, int, int]:
    return (
        context.match_id,
        context.before_state.frame_index,
        context.after_state.frame_index,
    )


def _large_macro_change(
    *,
    context: SituationStateContext,
    threshold: int,
) -> bool:
    change = context.interval_gold_diff_change_100_minus_200

    if change is None:
        return False

    return abs(change) >= threshold


def _priority_tier(
    *,
    situation: TemporalSituation,
    is_macro_anchor: bool,
    macro_lead_flip: bool,
    macro_gold_p90: bool,
) -> SituationPriorityTier:
    member_count = _member_count(situation)

    if situation.situation_kind == SituationKind.TERMINAL:
        return SituationPriorityTier.CRITICAL

    if situation.max_salience_score >= 95:
        return SituationPriorityTier.CRITICAL

    if is_macro_anchor and macro_lead_flip and macro_gold_p90:
        return SituationPriorityTier.CRITICAL

    if situation.max_salience_score >= 80:
        return SituationPriorityTier.HIGH

    if is_macro_anchor and (macro_lead_flip or macro_gold_p90):
        return SituationPriorityTier.HIGH

    if member_count >= 5:
        return SituationPriorityTier.HIGH

    if situation.max_salience_score >= 60:
        return SituationPriorityTier.MEDIUM

    if member_count >= 2:
        return SituationPriorityTier.MEDIUM

    if situation.semantic_markers:
        return SituationPriorityTier.MEDIUM

    return SituationPriorityTier.LOW


def _priority_reasons(
    *,
    situation: TemporalSituation,
    is_macro_anchor: bool,
    macro_lead_flip: bool,
    macro_gold_p90: bool,
) -> tuple[
    PriorityReason,
    ...,
]:
    reasons: list[PriorityReason] = []

    member_count = _member_count(situation)

    if situation.situation_kind == SituationKind.TERMINAL:
        reasons.append(PriorityReason.TERMINAL_EVENT)

    if situation.max_salience_score >= 95:
        reasons.append(PriorityReason.EXCEPTIONAL_EVENT_SALIENCE)

    elif situation.max_salience_score >= 80:
        reasons.append(PriorityReason.HIGH_EVENT_SALIENCE)

    if member_count >= 5:
        reasons.append(PriorityReason.LARGE_MULTI_EVENT_SITUATION)

    elif member_count >= 2:
        reasons.append(PriorityReason.MULTI_EVENT_SITUATION)

    if situation.semantic_markers:
        reasons.append(PriorityReason.SEMANTIC_MARKER_PRESENT)

    if is_macro_anchor:
        reasons.append(PriorityReason.MACRO_INTERVAL_ANCHOR)

        if macro_lead_flip:
            reasons.append(PriorityReason.MACRO_LEAD_FLIP)

        if macro_gold_p90:
            reasons.append(PriorityReason.MACRO_GOLD_P90)

    if not reasons:
        reasons.append(PriorityReason.ROUTINE_EVENT)

    return tuple(reasons)


def build_situation_priorities(
    *,
    match_id: str,
    situations: tuple[
        TemporalSituation,
        ...,
    ],
    contexts: tuple[
        SituationStateContext,
        ...,
    ],
    macro_gold_p90_threshold: int = (MACRO_GOLD_P90_THRESHOLD),
) -> tuple[
    SituationPriority,
    ...,
]:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    if macro_gold_p90_threshold < 0:
        raise ValueError("macro_gold_p90_threshold must be non-negative")

    ordered_situations = tuple(
        sorted(
            situations,
            key=_situation_sort_key,
        )
    )

    situations_by_id = {situation.situation_id: (situation) for situation in ordered_situations}

    if len(situations_by_id) != len(ordered_situations):
        raise ValueError("Duplicate situation_id")

    contexts_by_id = {context.situation_id: context for context in contexts}

    if len(contexts_by_id) != len(contexts):
        raise ValueError("Duplicate state context situation_id")

    if set(situations_by_id) != set(contexts_by_id):
        raise ValueError("Situation/context ID coverage mismatch")

    for situation in ordered_situations:
        if situation.match_id != clean_match_id:
            raise ValueError("Situation match_id does not match")

    for context in contexts:
        if context.match_id != clean_match_id:
            raise ValueError("State context match_id does not match")

    interval_situation_ids: dict[
        tuple[str, int, int],
        list[str],
    ] = defaultdict(list)

    for context in contexts:
        interval_situation_ids[_context_key(context)].append(context.situation_id)

    interval_anchors: dict[
        tuple[str, int, int],
        TemporalSituation,
    ] = {}

    for key, situation_ids in interval_situation_ids.items():
        grouped_situations = [situations_by_id[situation_id] for situation_id in situation_ids]

        interval_anchors[key] = max(
            grouped_situations,
            key=_anchor_sort_key,
        )

    priorities: list[SituationPriority] = []

    for situation in ordered_situations:
        context = contexts_by_id[situation.situation_id]

        key = _context_key(context)

        anchor = interval_anchors[key]

        is_macro_anchor = anchor.situation_id == situation.situation_id

        macro_gold_p90 = _large_macro_change(
            context=context,
            threshold=(macro_gold_p90_threshold),
        )

        macro_lead_flip = context.leading_team_changed

        interval_change = context.interval_gold_diff_change_100_minus_200

        macro_reference = MacroContextReference(
            macro_interval_id=(
                _macro_interval_id(
                    match_id=(clean_match_id),
                    before_frame_index=(context.before_state.frame_index),
                    after_frame_index=(context.after_state.frame_index),
                )
            ),
            match_id=(clean_match_id),
            before_frame_index=(context.before_state.frame_index),
            after_frame_index=(context.after_state.frame_index),
            before_timestamp_ms=(context.before_state.timestamp_ms),
            after_timestamp_ms=(context.after_state.timestamp_ms),
            frame_interval_ms=(context.frame_interval_ms),
            before_gold_diff_100_minus_200=(context.before_gold_diff_100_minus_200),
            after_gold_diff_100_minus_200=(context.after_gold_diff_100_minus_200),
            interval_gold_diff_change_100_minus_200=(interval_change),
            leading_team_changed=(macro_lead_flip),
            large_macro_gold_change=(macro_gold_p90),
            shared_situation_count=(len(interval_situation_ids[key])),
            anchor_situation_id=(anchor.situation_id),
        )

        priorities.append(
            SituationPriority(
                situation_id=(situation.situation_id),
                match_id=(clean_match_id),
                priority_tier=(
                    _priority_tier(
                        situation=situation,
                        is_macro_anchor=(is_macro_anchor),
                        macro_lead_flip=(macro_lead_flip),
                        macro_gold_p90=(macro_gold_p90),
                    )
                ),
                event_salience_score=(situation.max_salience_score),
                situation_member_count=(_member_count(situation)),
                semantic_marker_count=(len(situation.semantic_markers)),
                macro_context=(macro_reference),
                is_macro_anchor=(is_macro_anchor),
                reasons=(
                    _priority_reasons(
                        situation=situation,
                        is_macro_anchor=(is_macro_anchor),
                        macro_lead_flip=(macro_lead_flip),
                        macro_gold_p90=(macro_gold_p90),
                    )
                ),
            )
        )

    result = tuple(
        sorted(
            priorities,
            key=_priority_sort_key,
        )
    )

    if len(result) != len(ordered_situations):
        raise RuntimeError("Priority coverage mismatch")

    return result
