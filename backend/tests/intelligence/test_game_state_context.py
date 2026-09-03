from __future__ import annotations

import pytest

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.game_state_context import (
    build_situation_state_contexts,
)
from lol_commentary_backend.intelligence.game_state_models import (
    FrameAlignmentMode,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)

PARTICIPANT_TEAMS = {
    1: 100,
    2: 100,
    3: 200,
    4: 200,
}


def _snapshot(
    *,
    frame_index: int,
    timestamp_ms: int,
    participant_id: int,
    total_gold: int,
    xp: int = 1000,
    level: int = 5,
    minions_killed: int = 20,
    jungle_minions_killed: int = 0,
) -> ParticipantFrameSnapshot:
    return ParticipantFrameSnapshot.model_construct(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        participant_id=(participant_id),
        total_gold=total_gold,
        xp=xp,
        level=level,
        minions_killed=(minions_killed),
        jungle_minions_killed=(jungle_minions_killed),
    )


def _frame(
    *,
    frame_index: int,
    timestamp_ms: int,
    blue_gold: tuple[int, int],
    red_gold: tuple[int, int],
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    return (
        _snapshot(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            participant_id=1,
            total_gold=blue_gold[0],
        ),
        _snapshot(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            participant_id=2,
            total_gold=blue_gold[1],
        ),
        _snapshot(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            participant_id=3,
            total_gold=red_gold[0],
        ),
        _snapshot(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            participant_id=4,
            total_gold=red_gold[1],
        ),
    )


def _situation(
    *,
    index: int,
    start_timestamp_ms: int,
    end_timestamp_ms: int,
    match_id: str = "KR_1",
) -> TemporalSituation:
    return TemporalSituation.model_construct(
        situation_id=(f"{index + 1:064x}"),
        match_id=match_id,
        start_timestamp_ms=(start_timestamp_ms),
        end_timestamp_ms=(end_timestamp_ms),
    )


def _two_frames() -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    return (
        *_frame(
            frame_index=0,
            timestamp_ms=0,
            blue_gold=(
                1000,
                1000,
            ),
            red_gold=(
                900,
                900,
            ),
        ),
        *_frame(
            frame_index=1,
            timestamp_ms=60_000,
            blue_gold=(
                1600,
                1600,
            ),
            red_gold=(
                1400,
                1400,
            ),
        ),
    )


def test_distinct_frames_are_aligned() -> None:
    contexts = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                end_timestamp_ms=20_000,
            ),
        ),
        snapshots=_two_frames(),
        participant_teams=(PARTICIPANT_TEAMS),
    )

    assert len(contexts) == 1

    context = contexts[0]

    assert context.alignment_mode == FrameAlignmentMode.DISTINCT_FRAMES

    assert context.frame_interval_ms == 60_000


def test_gold_difference_is_blue_minus_red() -> None:
    context = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                end_timestamp_ms=20_000,
            ),
        ),
        snapshots=_two_frames(),
        participant_teams=(PARTICIPANT_TEAMS),
    )[0]

    assert context.before_gold_diff_100_minus_200 == 200

    assert context.after_gold_diff_100_minus_200 == 400

    assert context.interval_gold_diff_change_100_minus_200 == 200


def test_same_frame_has_no_interval_change() -> None:
    snapshots = _frame(
        frame_index=0,
        timestamp_ms=0,
        blue_gold=(
            1000,
            1000,
        ),
        red_gold=(
            900,
            900,
        ),
    )

    context = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=0,
                end_timestamp_ms=0,
            ),
        ),
        snapshots=snapshots,
        participant_teams=(PARTICIPANT_TEAMS),
    )[0]

    assert context.alignment_mode == FrameAlignmentMode.SAME_FRAME

    assert context.interval_gold_diff_change_100_minus_200 is None


def test_team_gold_is_aggregated() -> None:
    context = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=0,
                end_timestamp_ms=0,
            ),
        ),
        snapshots=_frame(
            frame_index=0,
            timestamp_ms=0,
            blue_gold=(
                1100,
                1200,
            ),
            red_gold=(
                900,
                1000,
            ),
        ),
        participant_teams=(PARTICIPANT_TEAMS),
    )[0]

    teams = {team.team_id: team for team in context.before_state.teams}

    assert teams[100].gold == 2300

    assert teams[200].gold == 1900


def test_lead_change_is_detected() -> None:
    snapshots = (
        *_frame(
            frame_index=0,
            timestamp_ms=0,
            blue_gold=(
                1000,
                1000,
            ),
            red_gold=(
                1200,
                1200,
            ),
        ),
        *_frame(
            frame_index=1,
            timestamp_ms=60_000,
            blue_gold=(
                1600,
                1600,
            ),
            red_gold=(
                1400,
                1400,
            ),
        ),
    )

    context = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                end_timestamp_ms=20_000,
            ),
        ),
        snapshots=snapshots,
        participant_teams=(PARTICIPANT_TEAMS),
    )[0]

    assert context.before_leading_team_id == 200

    assert context.after_leading_team_id == 100

    assert context.leading_team_changed is True


def test_missing_before_frame_raises() -> None:
    with pytest.raises(
        ValueError,
        match=("No frame available at or before"),
    ):
        build_situation_state_contexts(
            match_id="KR_1",
            situations=(
                _situation(
                    index=0,
                    start_timestamp_ms=10_000,
                    end_timestamp_ms=20_000,
                ),
            ),
            snapshots=_frame(
                frame_index=1,
                timestamp_ms=60_000,
                blue_gold=(
                    1000,
                    1000,
                ),
                red_gold=(
                    900,
                    900,
                ),
            ),
            participant_teams=(PARTICIPANT_TEAMS),
        )


def test_missing_after_frame_raises() -> None:
    with pytest.raises(
        ValueError,
        match=("No frame available at or after"),
    ):
        build_situation_state_contexts(
            match_id="KR_1",
            situations=(
                _situation(
                    index=0,
                    start_timestamp_ms=70_000,
                    end_timestamp_ms=80_000,
                ),
            ),
            snapshots=_two_frames(),
            participant_teams=(PARTICIPANT_TEAMS),
        )


def test_participant_coverage_mismatch_raises() -> None:
    incomplete = _frame(
        frame_index=0,
        timestamp_ms=0,
        blue_gold=(
            1000,
            1000,
        ),
        red_gold=(
            900,
            900,
        ),
    )[:-1]

    with pytest.raises(
        ValueError,
        match=("Participant coverage mismatch"),
    ):
        build_situation_state_contexts(
            match_id="KR_1",
            situations=(
                _situation(
                    index=0,
                    start_timestamp_ms=0,
                    end_timestamp_ms=0,
                ),
            ),
            snapshots=incomplete,
            participant_teams=(PARTICIPANT_TEAMS),
        )


def test_situation_match_id_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match=("Situation match_id does not match"),
    ):
        build_situation_state_contexts(
            match_id="KR_1",
            situations=(
                _situation(
                    index=0,
                    start_timestamp_ms=0,
                    end_timestamp_ms=0,
                    match_id="KR_OTHER",
                ),
            ),
            snapshots=_two_frames(),
            participant_teams=(PARTICIPANT_TEAMS),
        )


def test_interpretation_is_macro_context_only() -> None:
    context = build_situation_state_contexts(
        match_id="KR_1",
        situations=(
            _situation(
                index=0,
                start_timestamp_ms=10_000,
                end_timestamp_ms=20_000,
            ),
        ),
        snapshots=_two_frames(),
        participant_teams=(PARTICIPANT_TEAMS),
    )[0]

    assert context.interpretation == "macro_context_only"
