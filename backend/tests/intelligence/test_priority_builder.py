from __future__ import annotations

from lol_commentary_backend.intelligence.game_state_models import (
    FrameAlignmentMode,
    GameStateSnapshot,
    SituationStateContext,
)
from lol_commentary_backend.intelligence.priority_builder import (
    build_situation_priorities,
)
from lol_commentary_backend.intelligence.priority_models import (
    PriorityReason,
    SituationPriorityTier,
)
from lol_commentary_backend.intelligence.situation_models import (
    SituationKind,
    TemporalSituation,
)


def _situation(
    *,
    index: int,
    start_timestamp_ms: int,
    salience: int,
    member_count: int = 1,
    marker_count: int = 0,
    kind: SituationKind = (SituationKind.COMBAT),
) -> TemporalSituation:
    primary = tuple(object() for _ in range(member_count - marker_count))

    markers = tuple(object() for _ in range(marker_count))

    return TemporalSituation.model_construct(
        situation_id=(f"{index + 1:064x}"),
        match_id="KR_1",
        start_timestamp_ms=(start_timestamp_ms),
        end_timestamp_ms=(start_timestamp_ms),
        duration_ms=0,
        situation_kind=kind,
        max_salience_score=(salience),
        primary_candidates=(primary),
        semantic_markers=(markers),
    )


def _snapshot(
    *,
    frame_index: int,
    timestamp_ms: int,
) -> GameStateSnapshot:
    return GameStateSnapshot.model_construct(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        teams=(),
    )


def _context(
    *,
    situation_index: int,
    before_frame: int,
    after_frame: int,
    before_gold_diff: int = 0,
    after_gold_diff: int = 0,
    lead_changed: bool = False,
) -> SituationStateContext:
    change = after_gold_diff - before_gold_diff

    return SituationStateContext.model_construct(
        situation_id=(f"{situation_index + 1:064x}"),
        match_id="KR_1",
        situation_start_timestamp_ms=(before_frame * 60_000),
        situation_end_timestamp_ms=(after_frame * 60_000),
        before_state=_snapshot(
            frame_index=before_frame,
            timestamp_ms=(before_frame * 60_000),
        ),
        after_state=_snapshot(
            frame_index=after_frame,
            timestamp_ms=(after_frame * 60_000),
        ),
        alignment_mode=(FrameAlignmentMode.DISTINCT_FRAMES),
        start_to_before_frame_ms=0,
        end_to_after_frame_ms=0,
        frame_interval_ms=((after_frame - before_frame) * 60_000),
        before_gold_diff_100_minus_200=(before_gold_diff),
        after_gold_diff_100_minus_200=(after_gold_diff),
        interval_gold_diff_change_100_minus_200=(change),
        before_leading_team_id=(
            100 if before_gold_diff > 0 else (200 if before_gold_diff < 0 else None)
        ),
        after_leading_team_id=(
            100 if after_gold_diff > 0 else (200 if after_gold_diff < 0 else None)
        ),
        leading_team_changed=(lead_changed),
    )


def test_routine_single_event_is_low() -> None:
    priorities = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=45,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )

    assert priorities[0].priority_tier == SituationPriorityTier.LOW


def test_salience_60_is_medium() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=60,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.MEDIUM


def test_salience_80_is_high() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=80,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.HIGH


def test_salience_95_is_critical() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=95,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.CRITICAL


def test_terminal_is_critical() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=50,
                kind=(SituationKind.TERMINAL),
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.CRITICAL


def test_large_situation_is_high() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=45,
                member_count=5,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.HIGH


def test_macro_signal_only_applies_to_one_anchor() -> None:
    situations = (
        _situation(
            index=0,
            start_timestamp_ms=10_000,
            salience=45,
        ),
        _situation(
            index=1,
            start_timestamp_ms=20_000,
            salience=80,
        ),
    )

    contexts = (
        _context(
            situation_index=0,
            before_frame=0,
            after_frame=1,
            before_gold_diff=1000,
            after_gold_diff=-1000,
            lead_changed=True,
        ),
        _context(
            situation_index=1,
            before_frame=0,
            after_frame=1,
            before_gold_diff=1000,
            after_gold_diff=-1000,
            lead_changed=True,
        ),
    )

    priorities = build_situation_priorities(
        match_id="KR_1",
        situations=situations,
        contexts=contexts,
    )

    anchors = [priority for priority in priorities if priority.is_macro_anchor]

    assert len(anchors) == 1

    anchor = anchors[0]

    assert anchor.situation_id == situations[1].situation_id

    macro_reason_count = sum(
        (PriorityReason.MACRO_LEAD_FLIP in priority.reasons) for priority in priorities
    )

    assert macro_reason_count == 1


def test_shared_interval_has_same_macro_id() -> None:
    situations = (
        _situation(
            index=0,
            start_timestamp_ms=10_000,
            salience=45,
        ),
        _situation(
            index=1,
            start_timestamp_ms=20_000,
            salience=60,
        ),
    )

    contexts = (
        _context(
            situation_index=0,
            before_frame=2,
            after_frame=3,
        ),
        _context(
            situation_index=1,
            before_frame=2,
            after_frame=3,
        ),
    )

    priorities = build_situation_priorities(
        match_id="KR_1",
        situations=situations,
        contexts=contexts,
    )

    assert (
        priorities[0].macro_context.macro_interval_id
        == priorities[1].macro_context.macro_interval_id
    )

    assert priorities[0].macro_context.shared_situation_count == 2


def test_lead_flip_and_p90_anchor_is_critical() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=60,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
                before_gold_diff=1000,
                after_gold_diff=-1000,
                lead_changed=True,
            ),
        ),
    )[0]

    assert priority.priority_tier == SituationPriorityTier.CRITICAL

    assert PriorityReason.MACRO_LEAD_FLIP in priority.reasons

    assert PriorityReason.MACRO_GOLD_P90 in priority.reasons


def test_macro_reference_is_not_causal() -> None:
    priority = build_situation_priorities(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                salience=45,
            ),
        ),
        contexts=(
            _context(
                situation_index=0,
                before_frame=0,
                after_frame=1,
            ),
        ),
    )[0]

    assert priority.macro_context.interpretation == "shared_macro_context_only"
