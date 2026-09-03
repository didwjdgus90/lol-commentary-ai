from __future__ import annotations

from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
    CommentaryCandidateType,
    SalienceBand,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)
from lol_commentary_backend.intelligence.situation_models import (
    SituationKind,
)

SOURCE_SHA = "a" * 64


def _candidate(
    *,
    sequence: int,
    timestamp_ms: int,
    candidate_type: (CommentaryCandidateType),
    actor_participant_id: (int | None) = None,
) -> CommentaryCandidate:
    return CommentaryCandidate(
        candidate_id=(f"{sequence + 1:064x}"),
        match_id="KR_1",
        source_sequence=sequence,
        source_event_sha256=(SOURCE_SHA),
        timestamp_ms=timestamp_ms,
        raw_event_type=(candidate_type.value),
        candidate_type=(candidate_type),
        salience_score=70,
        salience_band=(SalienceBand.HIGH),
        reasons=("test",),
        actor_participant_id=(actor_participant_id),
    )


def test_two_kills_within_eight_seconds_merge() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=1000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=9000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
        ),
    )

    assert len(situations) == 1

    assert len(situations[0].primary_candidates) == 2

    assert situations[0].situation_kind == SituationKind.COMBAT


def test_kills_over_eight_seconds_split() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=1000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=9001,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
        ),
    )

    assert len(situations) == 2


def test_objective_followup_at_nine_seconds_merges() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=1000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=10000,
                candidate_type=(CommentaryCandidateType.ELITE_MONSTER),
            ),
        ),
    )

    assert len(situations) == 1

    assert situations[0].situation_kind == SituationKind.MIXED


def test_objective_after_ten_seconds_splits() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=1000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=11001,
                candidate_type=(CommentaryCandidateType.ELITE_MONSTER),
            ),
        ),
    )

    assert len(situations) == 2


def test_multi_kill_becomes_semantic_marker() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
                actor_participant_id=4,
            ),
            _candidate(
                sequence=1,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.MULTI_KILL),
                actor_participant_id=4,
            ),
        ),
    )

    assert len(situations) == 1

    situation = situations[0]

    assert len(situation.primary_candidates) == 1

    assert len(situation.semantic_markers) == 1


def test_marker_within_500ms_attaches() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
                actor_participant_id=4,
            ),
            _candidate(
                sequence=1,
                timestamp_ms=5500,
                candidate_type=(CommentaryCandidateType.ACE),
                actor_participant_id=4,
            ),
        ),
    )

    assert len(situations) == 1

    assert len(situations[0].semantic_markers) == 1


def test_unmatched_marker_becomes_special_only() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
                actor_participant_id=4,
            ),
            _candidate(
                sequence=1,
                timestamp_ms=6000,
                candidate_type=(CommentaryCandidateType.ACE),
                actor_participant_id=4,
            ),
        ),
    )

    assert len(situations) == 2

    assert situations[1].situation_kind == SituationKind.SPECIAL_ONLY


def test_game_end_is_always_terminal() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.BUILDING),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=5500,
                candidate_type=(CommentaryCandidateType.GAME_END),
            ),
        ),
    )

    assert len(situations) == 2

    assert situations[1].situation_kind == SituationKind.TERMINAL


def test_nearby_buildings_form_push() -> None:
    situations = build_temporal_situations(
        match_id="KR_1",
        candidates=(
            _candidate(
                sequence=0,
                timestamp_ms=5000,
                candidate_type=(CommentaryCandidateType.BUILDING),
            ),
            _candidate(
                sequence=1,
                timestamp_ms=10000,
                candidate_type=(CommentaryCandidateType.BUILDING),
            ),
        ),
    )

    assert len(situations) == 1

    assert situations[0].situation_kind == SituationKind.PUSH


def test_situation_id_is_deterministic() -> None:
    candidates = (
        _candidate(
            sequence=0,
            timestamp_ms=5000,
            candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
        ),
        _candidate(
            sequence=1,
            timestamp_ms=7000,
            candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
        ),
    )

    first = build_temporal_situations(
        match_id="KR_1",
        candidates=candidates,
    )

    second = build_temporal_situations(
        match_id="KR_1",
        candidates=candidates,
    )

    assert first[0].situation_id == second[0].situation_id
