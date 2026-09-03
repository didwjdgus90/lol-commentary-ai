from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
    CommentaryCandidateType,
    SalienceBand,
)

DEFAULT_MINIMUM_SALIENCE = 45


def _band(
    score: int,
) -> SalienceBand:
    if score >= 80:
        return SalienceBand.CRITICAL

    if score >= 60:
        return SalienceBand.HIGH

    if score >= 45:
        return SalienceBand.MEDIUM

    return SalienceBand.LOW


def _candidate_id(
    *,
    match_id: str,
    event: NormalizedGameEvent,
    candidate_type: CommentaryCandidateType,
) -> str:
    identity = f"{match_id}|{event.sequence}|{event.source_event_sha256}|{candidate_type.value}"

    return sha256(identity.encode("utf-8")).hexdigest()


def _shutdown_bonus(
    shutdown_bounty: int | None,
) -> tuple[int, str | None]:
    if shutdown_bounty is None or shutdown_bounty <= 0:
        return 0, None

    if shutdown_bounty >= 700:
        return 35, "shutdown>=700:+35"

    if shutdown_bounty >= 500:
        return 30, "shutdown>=500:+30"

    if shutdown_bounty >= 300:
        return 20, "shutdown>=300:+20"

    if shutdown_bounty >= 150:
        return 12, "shutdown>=150:+12"

    return 5, "shutdown>0:+5"


def _champion_kill_score(
    event: NormalizedGameEvent,
) -> tuple[int, tuple[str, ...]]:
    score = 45
    reasons = ["champion_kill:45"]

    shutdown_bonus, reason = _shutdown_bonus(event.shutdown_bounty)

    score += shutdown_bonus

    if reason is not None:
        reasons.append(reason)

    assist_count = len(event.assisting_participant_ids)

    if assist_count >= 4:
        score += 8
        reasons.append("assist_count>=4:+8")

    elif assist_count >= 3:
        score += 5
        reasons.append("assist_count>=3:+5")

    return min(score, 100), tuple(reasons)


def _special_kill_score(
    event: NormalizedGameEvent,
) -> tuple[
    CommentaryCandidateType,
    int,
    tuple[str, ...],
]:
    if event.kill_type == "KILL_FIRST_BLOOD":
        return (
            CommentaryCandidateType.FIRST_BLOOD,
            70,
            ("first_blood:70",),
        )

    if event.kill_type == "KILL_ACE":
        return (
            CommentaryCandidateType.ACE,
            95,
            ("ace:95",),
        )

    if event.kill_type == "KILL_MULTI":
        length = event.multi_kill_length or 0

        if length >= 5:
            score = 100
        elif length == 4:
            score = 90
        elif length == 3:
            score = 80
        elif length == 2:
            score = 65
        else:
            score = 60

        return (
            CommentaryCandidateType.MULTI_KILL,
            score,
            (f"multi_kill:{length}:{score}",),
        )

    return (
        CommentaryCandidateType.MULTI_KILL,
        60,
        ("special_kill_unknown:60",),
    )


def _elite_monster_score(
    event: NormalizedGameEvent,
) -> tuple[int, tuple[str, ...]]:
    if event.monster_sub_type == "ELDER_DRAGON":
        return 100, ("elder_dragon:100",)

    if event.monster_type == "BARON_NASHOR":
        return 95, ("baron_nashor:95",)

    if event.monster_type == "DRAGON":
        return 65, ("dragon:65",)

    if event.monster_type == "RIFTHERALD":
        return 60, ("rift_herald:60",)

    if event.monster_type == "HORDE":
        return 50, ("horde:50",)

    return 55, ("elite_monster_other:55",)


def _building_score(
    event: NormalizedGameEvent,
) -> tuple[int, tuple[str, ...]]:
    if event.building_type == "INHIBITOR_BUILDING":
        return 80, ("inhibitor:80",)

    if event.tower_type == "NEXUS_TURRET":
        return 85, ("nexus_turret:85",)

    if event.tower_type == "BASE_TURRET":
        return 70, ("base_turret:70",)

    if event.tower_type == "INNER_TURRET":
        return 60, ("inner_turret:60",)

    if event.tower_type == "OUTER_TURRET":
        return 50, ("outer_turret:50",)

    return 55, ("building_other:55",)


def _score_event(
    event: NormalizedGameEvent,
) -> (
    tuple[
        CommentaryCandidateType,
        int,
        tuple[str, ...],
    ]
    | None
):
    event_type = event.raw_event_type

    if event_type == "CHAMPION_KILL":
        score, reasons = _champion_kill_score(event)

        return (
            CommentaryCandidateType.CHAMPION_KILL,
            score,
            reasons,
        )

    if event_type == "CHAMPION_SPECIAL_KILL":
        return _special_kill_score(event)

    if event_type == "ELITE_MONSTER_KILL":
        score, reasons = _elite_monster_score(event)

        return (
            CommentaryCandidateType.ELITE_MONSTER,
            score,
            reasons,
        )

    if event_type == "BUILDING_KILL":
        score, reasons = _building_score(event)

        return (
            CommentaryCandidateType.BUILDING,
            score,
            reasons,
        )

    if event_type == "DRAGON_SOUL_GIVEN":
        return (
            CommentaryCandidateType.DRAGON_SOUL,
            95,
            ("dragon_soul:95",),
        )

    if event_type == "GAME_END":
        return (
            CommentaryCandidateType.GAME_END,
            100,
            ("game_end:100",),
        )

    return None


def build_commentary_candidates(
    *,
    match_id: str,
    events: Iterable[NormalizedGameEvent],
    minimum_salience: int = DEFAULT_MINIMUM_SALIENCE,
) -> tuple[CommentaryCandidate, ...]:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    if not 0 <= minimum_salience <= 100:
        raise ValueError("minimum_salience must be between 0 and 100")

    candidates: list[CommentaryCandidate] = []

    for event in events:
        scored = _score_event(event)

        if scored is None:
            continue

        (
            candidate_type,
            score,
            reasons,
        ) = scored

        if score < minimum_salience:
            continue

        candidates.append(
            CommentaryCandidate(
                candidate_id=_candidate_id(
                    match_id=clean_match_id,
                    event=event,
                    candidate_type=candidate_type,
                ),
                match_id=clean_match_id,
                source_sequence=event.sequence,
                source_event_sha256=(event.source_event_sha256),
                timestamp_ms=event.timestamp_ms,
                raw_event_type=event.raw_event_type,
                candidate_type=candidate_type,
                salience_score=score,
                salience_band=_band(score),
                reasons=reasons,
                actor_participant_id=(event.actor_participant_id),
                target_participant_id=(event.target_participant_id),
                assisting_participant_ids=(event.assisting_participant_ids),
                team_id=event.team_id,
                killer_team_id=event.killer_team_id,
                winning_team_id=(event.winning_team_id),
                position=event.position,
                shutdown_bounty=(event.shutdown_bounty),
                multi_kill_length=(event.multi_kill_length),
                kill_type=event.kill_type,
                monster_type=event.monster_type,
                monster_sub_type=(event.monster_sub_type),
                building_type=event.building_type,
                tower_type=event.tower_type,
                lane_type=event.lane_type,
                objective_name=(event.objective_name),
            )
        )

    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.timestamp_ms,
                candidate.source_sequence,
                candidate.candidate_id,
            ),
        )
    )
