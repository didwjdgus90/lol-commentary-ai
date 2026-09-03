from __future__ import annotations

from hashlib import sha256
from typing import Any

from lol_commentary_backend.ingestion.riot_api.collector import (
    canonical_json_bytes,
)
from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    RIOT_NORMALIZER_VERSION,
    EventCategory,
    MapPosition,
    NormalizedGameEvent,
    NormalizedMatch,
    NormalizedMatchBundle,
    ParticipantFrameSnapshot,
    ParticipantIdentity,
)

_EVENT_CATEGORIES = {
    "CHAMPION_KILL": (EventCategory.COMBAT),
    "CHAMPION_SPECIAL_KILL": (EventCategory.COMBAT),
    "BUILDING_KILL": (EventCategory.OBJECTIVE),
    "DRAGON_SOUL_GIVEN": (EventCategory.OBJECTIVE),
    "ELITE_MONSTER_KILL": (EventCategory.OBJECTIVE),
    "OBJECTIVE_BOUNTY_PRESTART": (EventCategory.OBJECTIVE),
    "TURRET_PLATE_DESTROYED": (EventCategory.OBJECTIVE),
    "ITEM_DESTROYED": (EventCategory.ECONOMY),
    "ITEM_PURCHASED": (EventCategory.ECONOMY),
    "ITEM_SOLD": (EventCategory.ECONOMY),
    "ITEM_UNDO": (EventCategory.ECONOMY),
    "WARD_KILL": (EventCategory.VISION),
    "WARD_PLACED": (EventCategory.VISION),
    "LEVEL_UP": (EventCategory.PROGRESSION),
    "SKILL_LEVEL_UP": (EventCategory.PROGRESSION),
    "GAME_END": (EventCategory.SYSTEM),
    "PAUSE_END": (EventCategory.SYSTEM),
}


_PROMOTED_EVENT_FIELDS = {
    "type",
    "timestamp",
    "participantId",
    "killerId",
    "creatorId",
    "victimId",
    "assistingParticipantIds",
    "teamId",
    "killerTeamId",
    "winningTeam",
    "position",
    "bounty",
    "shutdownBounty",
    "killStreakLength",
    "multiKillLength",
    "killType",
    "itemId",
    "beforeId",
    "afterId",
    "goldGain",
    "level",
    "skillSlot",
    "levelUpType",
    "monsterType",
    "monsterSubType",
    "buildingType",
    "towerType",
    "laneType",
    "wardType",
    "name",
    "actualStartTime",
    "gameId",
    "realTimestamp",
}


def _object(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(f"{name} must be an object")

    return value


def _list(
    value: object,
    *,
    name: str,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        raise ValueError(f"{name} must be a list")

    return value


def _required_str(
    payload: dict[str, Any],
    key: str,
) -> str:
    value = payload.get(key)

    if (
        not isinstance(
            value,
            str,
        )
        or not value
    ):
        raise ValueError(f"{key} must be a non-empty string")

    return value


def _optional_str(
    payload: dict[str, Any],
    key: str,
) -> str | None:
    value = payload.get(key)

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(f"{key} must be a string")

    return value


def _required_int(
    payload: dict[str, Any],
    key: str,
) -> int:
    value = payload.get(key)

    if not isinstance(
        value,
        int,
    ) or isinstance(
        value,
        bool,
    ):
        raise ValueError(f"{key} must be an integer")

    return value


def _optional_int(
    payload: dict[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)

    if value is None:
        return None

    if not isinstance(
        value,
        int,
    ) or isinstance(
        value,
        bool,
    ):
        raise ValueError(f"{key} must be an integer")

    return value


def _optional_number(
    payload: dict[str, Any],
    key: str,
) -> int | float | None:
    value = payload.get(key)

    if value is None:
        return None

    if not isinstance(
        value,
        int | float,
    ) or isinstance(
        value,
        bool,
    ):
        raise ValueError(f"{key} must be numeric")

    return value


def _participant_reference(
    value: object,
) -> int | None:
    if not isinstance(
        value,
        int,
    ) or isinstance(
        value,
        bool,
    ):
        return None

    if value <= 0:
        return None

    return value


def _actor_participant_id(
    event: dict[str, Any],
) -> int | None:
    for key in (
        "participantId",
        "killerId",
        "creatorId",
    ):
        participant_id = _participant_reference(event.get(key))

        if participant_id is not None:
            return participant_id

    return None


def _assisting_participants(
    event: dict[str, Any],
) -> tuple[int, ...]:
    raw = event.get("assistingParticipantIds")

    if raw is None:
        return ()

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError("assistingParticipantIds must be a list")

    result: list[int] = []

    for value in raw:
        participant_id = _participant_reference(value)

        if participant_id is not None:
            result.append(participant_id)

    return tuple(result)


def _position(
    payload: dict[str, Any],
) -> MapPosition | None:
    raw = payload.get("position")

    if raw is None:
        return None

    position = _object(
        raw,
        name="position",
    )

    return MapPosition(
        x=_required_int(
            position,
            "x",
        ),
        y=_required_int(
            position,
            "y",
        ),
    )


def _event_sha256(
    event: dict[str, Any],
) -> str:
    return sha256(canonical_json_bytes(event)).hexdigest()


def _payload_sha256(
    payload: dict[str, Any],
) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _match_id(
    payload: dict[str, Any],
) -> str:
    metadata = _object(
        payload.get("metadata"),
        name="metadata",
    )

    return _required_str(
        metadata,
        "matchId",
    )


def _normalize_participants(
    match_info: dict[str, Any],
) -> tuple[
    ParticipantIdentity,
    ...,
]:
    raw_participants = _list(
        match_info.get("participants"),
        name="participants",
    )

    participants: list[ParticipantIdentity] = []

    seen_ids: set[int] = set()

    for raw in raw_participants:
        participant = _object(
            raw,
            name="participant",
        )

        participant_id = _required_int(
            participant,
            "participantId",
        )

        if participant_id in seen_ids:
            raise ValueError(f"Duplicate participantId: {participant_id}")

        seen_ids.add(participant_id)

        win = participant.get("win")

        if not isinstance(
            win,
            bool,
        ):
            raise ValueError("participant.win must be boolean")

        participants.append(
            ParticipantIdentity(
                participant_id=(participant_id),
                team_id=_required_int(
                    participant,
                    "teamId",
                ),
                champion_id=(
                    _required_int(
                        participant,
                        "championId",
                    )
                ),
                champion_name=(
                    _required_str(
                        participant,
                        "championName",
                    )
                ),
                team_position=(
                    _optional_str(
                        participant,
                        "teamPosition",
                    )
                ),
                individual_position=(
                    _optional_str(
                        participant,
                        "individualPosition",
                    )
                ),
                win=win,
            )
        )

    return tuple(
        sorted(
            participants,
            key=lambda item: item.participant_id,
        )
    )


def _normalize_participant_frames(
    timeline_info: dict[str, Any],
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    frames = _list(
        timeline_info.get("frames"),
        name="frames",
    )

    snapshots: list[ParticipantFrameSnapshot] = []

    for frame_index, raw_frame in enumerate(frames):
        frame = _object(
            raw_frame,
            name="frame",
        )

        timestamp = _required_int(
            frame,
            "timestamp",
        )

        raw_participant_frames = _object(
            frame.get("participantFrames"),
            name="participantFrames",
        )

        for raw_snapshot in raw_participant_frames.values():
            snapshot = _object(
                raw_snapshot,
                name="participantFrame",
            )

            snapshots.append(
                ParticipantFrameSnapshot(
                    frame_index=frame_index,
                    timestamp_ms=timestamp,
                    participant_id=(
                        _required_int(
                            snapshot,
                            "participantId",
                        )
                    ),
                    level=_optional_int(
                        snapshot,
                        "level",
                    ),
                    current_gold=(
                        _optional_int(
                            snapshot,
                            "currentGold",
                        )
                    ),
                    total_gold=(
                        _optional_int(
                            snapshot,
                            "totalGold",
                        )
                    ),
                    xp=_optional_int(
                        snapshot,
                        "xp",
                    ),
                    minions_killed=(
                        _optional_int(
                            snapshot,
                            "minionsKilled",
                        )
                    ),
                    jungle_minions_killed=(
                        _optional_int(
                            snapshot,
                            "jungleMinionsKilled",
                        )
                    ),
                    gold_per_second=(
                        _optional_int(
                            snapshot,
                            "goldPerSecond",
                        )
                    ),
                    time_enemy_spent_controlled=(
                        _optional_number(
                            snapshot,
                            "timeEnemySpentControlled",
                        )
                    ),
                    position=_position(snapshot),
                )
            )

    return tuple(snapshots)


def _normalize_events(
    timeline_info: dict[str, Any],
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    frames = _list(
        timeline_info.get("frames"),
        name="frames",
    )

    events: list[NormalizedGameEvent] = []

    sequence = 0

    for frame_index, raw_frame in enumerate(frames):
        frame = _object(
            raw_frame,
            name="frame",
        )

        raw_events = _list(
            frame.get("events"),
            name="events",
        )

        for event_index, raw_event in enumerate(raw_events):
            event = _object(
                raw_event,
                name="event",
            )

            raw_type = _required_str(
                event,
                "type",
            )

            category = _EVENT_CATEGORIES.get(
                raw_type,
                EventCategory.SYSTEM,
            )

            events.append(
                NormalizedGameEvent(
                    sequence=sequence,
                    frame_index=(frame_index),
                    event_index=(event_index),
                    timestamp_ms=(
                        _required_int(
                            event,
                            "timestamp",
                        )
                    ),
                    raw_event_type=(raw_type),
                    category=category,
                    known_event_type=(raw_type in _EVENT_CATEGORIES),
                    source_event_sha256=(_event_sha256(event)),
                    actor_participant_id=(_actor_participant_id(event)),
                    target_participant_id=(_participant_reference(event.get("victimId"))),
                    assisting_participant_ids=(_assisting_participants(event)),
                    team_id=_optional_int(
                        event,
                        "teamId",
                    ),
                    killer_team_id=(
                        _optional_int(
                            event,
                            "killerTeamId",
                        )
                    ),
                    winning_team_id=(
                        _optional_int(
                            event,
                            "winningTeam",
                        )
                    ),
                    position=_position(event),
                    bounty=_optional_int(
                        event,
                        "bounty",
                    ),
                    shutdown_bounty=(
                        _optional_int(
                            event,
                            "shutdownBounty",
                        )
                    ),
                    kill_streak_length=(
                        _optional_int(
                            event,
                            "killStreakLength",
                        )
                    ),
                    multi_kill_length=(
                        _optional_int(
                            event,
                            "multiKillLength",
                        )
                    ),
                    kill_type=_optional_str(
                        event,
                        "killType",
                    ),
                    item_id=_optional_int(
                        event,
                        "itemId",
                    ),
                    before_item_id=(
                        _optional_int(
                            event,
                            "beforeId",
                        )
                    ),
                    after_item_id=(
                        _optional_int(
                            event,
                            "afterId",
                        )
                    ),
                    gold_gain=_optional_int(
                        event,
                        "goldGain",
                    ),
                    level=_optional_int(
                        event,
                        "level",
                    ),
                    skill_slot=(
                        _optional_int(
                            event,
                            "skillSlot",
                        )
                    ),
                    level_up_type=(
                        _optional_str(
                            event,
                            "levelUpType",
                        )
                    ),
                    monster_type=(
                        _optional_str(
                            event,
                            "monsterType",
                        )
                    ),
                    monster_sub_type=(
                        _optional_str(
                            event,
                            "monsterSubType",
                        )
                    ),
                    building_type=(
                        _optional_str(
                            event,
                            "buildingType",
                        )
                    ),
                    tower_type=(
                        _optional_str(
                            event,
                            "towerType",
                        )
                    ),
                    lane_type=_optional_str(
                        event,
                        "laneType",
                    ),
                    ward_type=_optional_str(
                        event,
                        "wardType",
                    ),
                    objective_name=(
                        _optional_str(
                            event,
                            "name",
                        )
                    ),
                    actual_start_time=(
                        _optional_int(
                            event,
                            "actualStartTime",
                        )
                    ),
                    game_id=_optional_int(
                        event,
                        "gameId",
                    ),
                    real_timestamp=(
                        _optional_int(
                            event,
                            "realTimestamp",
                        )
                    ),
                    unmapped_fields=tuple(
                        sorted({str(key) for key in event} - _PROMOTED_EVENT_FIELDS)
                    ),
                )
            )

            sequence += 1

    return tuple(events)


def normalize_match_timeline(
    *,
    match: dict[str, Any],
    timeline: dict[str, Any],
) -> NormalizedMatchBundle:
    match_id = _match_id(match)

    timeline_match_id = _match_id(timeline)

    if match_id != timeline_match_id:
        raise ValueError(f"Match/timeline ID mismatch: {match_id} != {timeline_match_id}")

    match_info = _object(
        match.get("info"),
        name="match.info",
    )

    timeline_info = _object(
        timeline.get("info"),
        name="timeline.info",
    )

    participants = _normalize_participants(match_info)

    participant_frames = _normalize_participant_frames(timeline_info)

    events = _normalize_events(timeline_info)

    normalized_match = NormalizedMatch(
        normalizer_version=(RIOT_NORMALIZER_VERSION),
        match_id=match_id,
        game_mode=_required_str(
            match_info,
            "gameMode",
        ),
        queue_id=_required_int(
            match_info,
            "queueId",
        ),
        map_id=_required_int(
            match_info,
            "mapId",
        ),
        game_duration_seconds=(
            _required_int(
                match_info,
                "gameDuration",
            )
        ),
        game_version=(
            _required_str(
                match_info,
                "gameVersion",
            )
        ),
        source_match_sha256=(_payload_sha256(match)),
        source_timeline_sha256=(_payload_sha256(timeline)),
        participants=(participants),
    )

    return NormalizedMatchBundle(
        match=normalized_match,
        participant_frames=(participant_frames),
        events=events,
    )
