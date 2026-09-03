from __future__ import annotations

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidateType,
    SalienceBand,
)

SOURCE_SHA = "a" * 64


def _event(
    *,
    event_type: str,
    sequence: int = 0,
    actor_participant_id: int | None = None,
    target_participant_id: int | None = None,
    assisting_participant_ids: tuple[int, ...] = (),
    shutdown_bounty: int | None = None,
    kill_type: str | None = None,
    multi_kill_length: int | None = None,
    monster_type: str | None = None,
    monster_sub_type: str | None = None,
    building_type: str | None = None,
    tower_type: str | None = None,
) -> NormalizedGameEvent:
    return NormalizedGameEvent(
        sequence=sequence,
        frame_index=0,
        event_index=sequence,
        timestamp_ms=1000 + sequence,
        raw_event_type=event_type,
        category="system",
        known_event_type=True,
        source_event_sha256=SOURCE_SHA,
        actor_participant_id=actor_participant_id,
        target_participant_id=target_participant_id,
        assisting_participant_ids=(assisting_participant_ids),
        shutdown_bounty=shutdown_bounty,
        kill_type=kill_type,
        multi_kill_length=multi_kill_length,
        monster_type=monster_type,
        monster_sub_type=monster_sub_type,
        building_type=building_type,
        tower_type=tower_type,
    )


def test_normal_kill_is_candidate() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="CHAMPION_KILL",
                actor_participant_id=1,
                target_participant_id=6,
            ),
        ),
    )[0]

    assert candidate.candidate_type == CommentaryCandidateType.CHAMPION_KILL

    assert candidate.salience_score == 45
    assert candidate.salience_band == SalienceBand.MEDIUM


def test_large_shutdown_gets_bonus() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="CHAMPION_KILL",
                shutdown_bounty=700,
            ),
        ),
    )[0]

    assert candidate.salience_score == 80
    assert candidate.salience_band == SalienceBand.CRITICAL


def test_first_blood_is_high_salience() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="CHAMPION_SPECIAL_KILL",
                kill_type="KILL_FIRST_BLOOD",
            ),
        ),
    )[0]

    assert candidate.candidate_type == CommentaryCandidateType.FIRST_BLOOD

    assert candidate.salience_score == 70


def test_triple_kill_is_critical() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="CHAMPION_SPECIAL_KILL",
                kill_type="KILL_MULTI",
                multi_kill_length=3,
            ),
        ),
    )[0]

    assert candidate.candidate_type == CommentaryCandidateType.MULTI_KILL

    assert candidate.salience_score == 80


def test_ace_is_critical() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="CHAMPION_SPECIAL_KILL",
                kill_type="KILL_ACE",
            ),
        ),
    )[0]

    assert candidate.candidate_type == CommentaryCandidateType.ACE
    assert candidate.salience_score == 95


def test_elder_dragon_is_maximum_salience() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="ELITE_MONSTER_KILL",
                monster_type="DRAGON",
                monster_sub_type="ELDER_DRAGON",
            ),
        ),
    )[0]

    assert candidate.salience_score == 100


def test_baron_is_critical() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="ELITE_MONSTER_KILL",
                monster_type="BARON_NASHOR",
            ),
        ),
    )[0]

    assert candidate.salience_score == 95


def test_nexus_turret_is_critical() -> None:
    candidate = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="BUILDING_KILL",
                building_type="TOWER_BUILDING",
                tower_type="NEXUS_TURRET",
            ),
        ),
    )[0]

    assert candidate.salience_score == 85


def test_noise_event_is_ignored() -> None:
    candidates = build_commentary_candidates(
        match_id="KR_1",
        events=(
            _event(
                event_type="WARD_PLACED",
            ),
        ),
    )

    assert candidates == ()


def test_candidate_id_is_deterministic() -> None:
    event = _event(event_type="GAME_END")

    first = build_commentary_candidates(
        match_id="KR_1",
        events=(event,),
    )[0]

    second = build_commentary_candidates(
        match_id="KR_1",
        events=(event,),
    )[0]

    assert first.candidate_id == second.candidate_id
