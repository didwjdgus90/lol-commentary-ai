from __future__ import annotations

import pytest

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    EventCategory,
)
from lol_commentary_backend.ingestion.riot_api.normalization.normalizer import (
    normalize_match_timeline,
)


def _match() -> dict[str, object]:
    return {
        "metadata": {
            "matchId": "KR_1",
        },
        "info": {
            "gameMode": "CLASSIC",
            "queueId": 420,
            "mapId": 11,
            "gameDuration": 1800,
            "gameVersion": ("16.17.810.4348"),
            "participants": [
                {
                    "participantId": 1,
                    "teamId": 100,
                    "championId": 202,
                    "championName": "Jhin",
                    "teamPosition": ("BOTTOM"),
                    "individualPosition": ("BOTTOM"),
                    "win": True,
                },
                {
                    "participantId": 6,
                    "teamId": 200,
                    "championId": 222,
                    "championName": "Jinx",
                    "teamPosition": ("BOTTOM"),
                    "individualPosition": ("BOTTOM"),
                    "win": False,
                },
            ],
        },
    }


def _timeline(
    events: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "metadata": {
            "matchId": "KR_1",
        },
        "info": {
            "frames": [
                {
                    "timestamp": 60000,
                    "participantFrames": {
                        "1": {
                            "participantId": 1,
                            "level": 2,
                            "currentGold": 500,
                            "totalGold": 1000,
                            "xp": 400,
                            "minionsKilled": 10,
                            "jungleMinionsKilled": 0,
                            "goldPerSecond": 0,
                            "timeEnemySpentControlled": 0,
                            "position": {
                                "x": 1000,
                                "y": 2000,
                            },
                        }
                    },
                    "events": events,
                }
            ],
        },
    }


def test_normalizes_participants() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline([]),
    )

    assert result.match.match_id == "KR_1"

    assert len(result.match.participants) == 2

    assert result.match.participants[0].champion_name == "Jhin"


def test_normalizes_participant_frame() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline([]),
    )

    snapshot = result.participant_frames[0]

    assert snapshot.participant_id == 1

    assert snapshot.total_gold == 1000

    assert snapshot.minions_killed == 10

    assert snapshot.position is not None

    assert snapshot.position.x == 1000


def test_normalizes_champion_kill() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline(
            [
                {
                    "type": "CHAMPION_KILL",
                    "timestamp": 70000,
                    "killerId": 1,
                    "victimId": 6,
                    "assistingParticipantIds": [
                        2,
                        3,
                    ],
                    "bounty": 300,
                    "shutdownBounty": 0,
                    "position": {
                        "x": 4000,
                        "y": 5000,
                    },
                }
            ]
        ),
    )

    event = result.events[0]

    assert event.category == EventCategory.COMBAT

    assert event.actor_participant_id == 1

    assert event.target_participant_id == 6

    assert event.assisting_participant_ids == (
        2,
        3,
    )


def test_normalizes_item_purchase() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline(
            [
                {
                    "type": "ITEM_PURCHASED",
                    "timestamp": 80000,
                    "participantId": 1,
                    "itemId": 3508,
                }
            ]
        ),
    )

    event = result.events[0]

    assert event.category == EventCategory.ECONOMY

    assert event.item_id == 3508

    assert event.actor_participant_id == 1


def test_normalizes_elite_monster() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline(
            [
                {
                    "type": ("ELITE_MONSTER_KILL"),
                    "timestamp": 90000,
                    "killerId": 1,
                    "killerTeamId": 100,
                    "monsterType": "DRAGON",
                    "monsterSubType": ("FIRE_DRAGON"),
                }
            ]
        ),
    )

    event = result.events[0]

    assert event.category == EventCategory.OBJECTIVE

    assert event.monster_type == "DRAGON"

    assert event.monster_sub_type == "FIRE_DRAGON"


def test_unknown_event_is_preserved() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline(
            [
                {
                    "type": ("FUTURE_RIOT_EVENT"),
                    "timestamp": 100000,
                    "newField": "value",
                }
            ]
        ),
    )

    event = result.events[0]

    assert event.category == EventCategory.SYSTEM

    assert event.known_event_type is False

    assert event.raw_event_type == "FUTURE_RIOT_EVENT"

    assert event.unmapped_fields == ("newField",)


def test_unmapped_combat_fields_keep_lineage() -> None:
    result = normalize_match_timeline(
        match=_match(),
        timeline=_timeline(
            [
                {
                    "type": "CHAMPION_KILL",
                    "timestamp": 110000,
                    "killerId": 1,
                    "victimId": 6,
                    "victimDamageReceived": [{"type": ("PHYSICAL")}],
                }
            ]
        ),
    )

    event = result.events[0]

    assert "victimDamageReceived" in event.unmapped_fields

    assert len(event.source_event_sha256) == 64


def test_rejects_match_timeline_id_mismatch() -> None:
    timeline = _timeline([])

    metadata = timeline["metadata"]

    assert isinstance(
        metadata,
        dict,
    )

    metadata["matchId"] = "KR_2"

    with pytest.raises(
        ValueError,
        match="ID mismatch",
    ):
        normalize_match_timeline(
            match=_match(),
            timeline=timeline,
        )
