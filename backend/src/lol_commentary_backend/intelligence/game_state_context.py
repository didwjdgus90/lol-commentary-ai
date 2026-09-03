from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.game_state_models import (
    FrameAlignmentMode,
    GameStateSnapshot,
    SituationStateContext,
    TeamMacroState,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)

BLUE_TEAM_ID = 100
RED_TEAM_ID = 200


def _team_state_sort_key(
    state: TeamMacroState,
) -> int:
    return state.team_id


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def _context_sort_key(
    context: SituationStateContext,
) -> tuple[int, int, str]:
    return (
        context.situation_start_timestamp_ms,
        context.situation_end_timestamp_ms,
        context.situation_id,
    )


def _build_frame_states(
    *,
    snapshots: tuple[
        ParticipantFrameSnapshot,
        ...,
    ],
    participant_teams: Mapping[
        int,
        int,
    ],
) -> tuple[
    GameStateSnapshot,
    ...,
]:
    if not participant_teams:
        raise ValueError("participant_teams must not be empty")

    grouped: dict[
        int,
        list[ParticipantFrameSnapshot],
    ] = defaultdict(list)

    for snapshot in snapshots:
        grouped[snapshot.frame_index].append(snapshot)

    states: list[GameStateSnapshot] = []

    expected_participants = set(participant_teams)

    for frame_index in sorted(grouped):
        frame_snapshots = grouped[frame_index]

        participant_ids = [snapshot.participant_id for snapshot in frame_snapshots]

        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError(f"Duplicate participant snapshot in frame {frame_index}")

        observed_participants = set(participant_ids)

        if observed_participants != expected_participants:
            raise ValueError(f"Participant coverage mismatch in frame {frame_index}")

        timestamps = {snapshot.timestamp_ms for snapshot in frame_snapshots}

        if len(timestamps) != 1:
            raise ValueError("Participant snapshots in one frame have different timestamps")

        timestamp_ms = next(iter(timestamps))

        team_values: dict[
            int,
            dict[str, int],
        ] = defaultdict(
            lambda: {
                "gold": 0,
                "xp": 0,
                "levels": 0,
                "lane_cs": 0,
                "jungle_cs": 0,
            }
        )

        for snapshot in frame_snapshots:
            team_id = participant_teams.get(snapshot.participant_id)

            if team_id is None:
                raise ValueError(f"Unknown participant ID: {snapshot.participant_id}")

            values = team_values[team_id]

            values["gold"] += snapshot.total_gold or 0

            values["xp"] += snapshot.xp or 0

            values["levels"] += snapshot.level or 0

            values["lane_cs"] += snapshot.minions_killed or 0

            values["jungle_cs"] += snapshot.jungle_minions_killed or 0

        if BLUE_TEAM_ID not in team_values or RED_TEAM_ID not in team_values:
            raise ValueError("Expected Riot team IDs 100 and 200")

        teams = tuple(
            sorted(
                (
                    TeamMacroState(
                        team_id=team_id,
                        gold=values["gold"],
                        xp=values["xp"],
                        levels=(values["levels"]),
                        lane_cs=(values["lane_cs"]),
                        jungle_cs=(values["jungle_cs"]),
                    )
                    for team_id, values in team_values.items()
                ),
                key=_team_state_sort_key,
            )
        )

        states.append(
            GameStateSnapshot(
                frame_index=(frame_index),
                timestamp_ms=(timestamp_ms),
                teams=teams,
            )
        )

    return tuple(states)


def _latest_state_at_or_before(
    *,
    states: tuple[
        GameStateSnapshot,
        ...,
    ],
    timestamp_ms: int,
) -> GameStateSnapshot | None:
    result: GameStateSnapshot | None = None

    for state in states:
        if state.timestamp_ms > timestamp_ms:
            break

        result = state

    return result


def _earliest_state_at_or_after(
    *,
    states: tuple[
        GameStateSnapshot,
        ...,
    ],
    timestamp_ms: int,
) -> GameStateSnapshot | None:
    for state in states:
        if state.timestamp_ms >= timestamp_ms:
            return state

    return None


def _team_state(
    *,
    snapshot: GameStateSnapshot,
    team_id: int,
) -> TeamMacroState:
    for team in snapshot.teams:
        if team.team_id == team_id:
            return team

    raise ValueError(f"Team missing from game-state snapshot: {team_id}")


def _gold_diff(
    snapshot: GameStateSnapshot,
) -> int:
    blue = _team_state(
        snapshot=snapshot,
        team_id=BLUE_TEAM_ID,
    )

    red = _team_state(
        snapshot=snapshot,
        team_id=RED_TEAM_ID,
    )

    return blue.gold - red.gold


def _leading_team(
    gold_diff: int,
) -> int | None:
    if gold_diff > 0:
        return BLUE_TEAM_ID

    if gold_diff < 0:
        return RED_TEAM_ID

    return None


def _build_context(
    *,
    situation: TemporalSituation,
    before: GameStateSnapshot,
    after: GameStateSnapshot,
) -> SituationStateContext:
    if before.timestamp_ms > situation.start_timestamp_ms:
        raise ValueError("Before frame occurs after situation start")

    if after.timestamp_ms < situation.end_timestamp_ms:
        raise ValueError("After frame occurs before situation end")

    before_gold_diff = _gold_diff(before)

    after_gold_diff = _gold_diff(after)

    if before.frame_index == after.frame_index:
        alignment_mode = FrameAlignmentMode.SAME_FRAME

        interval_change: int | None = None

    else:
        alignment_mode = FrameAlignmentMode.DISTINCT_FRAMES

        interval_change = after_gold_diff - before_gold_diff

    before_leader = _leading_team(before_gold_diff)

    after_leader = _leading_team(after_gold_diff)

    return SituationStateContext(
        situation_id=(situation.situation_id),
        match_id=situation.match_id,
        situation_start_timestamp_ms=(situation.start_timestamp_ms),
        situation_end_timestamp_ms=(situation.end_timestamp_ms),
        before_state=before,
        after_state=after,
        alignment_mode=alignment_mode,
        start_to_before_frame_ms=(situation.start_timestamp_ms - before.timestamp_ms),
        end_to_after_frame_ms=(after.timestamp_ms - situation.end_timestamp_ms),
        frame_interval_ms=(after.timestamp_ms - before.timestamp_ms),
        before_gold_diff_100_minus_200=(before_gold_diff),
        after_gold_diff_100_minus_200=(after_gold_diff),
        interval_gold_diff_change_100_minus_200=(interval_change),
        before_leading_team_id=(before_leader),
        after_leading_team_id=(after_leader),
        leading_team_changed=(before_leader != after_leader),
    )


def build_situation_state_contexts(
    *,
    match_id: str,
    situations: tuple[
        TemporalSituation,
        ...,
    ],
    snapshots: tuple[
        ParticipantFrameSnapshot,
        ...,
    ],
    participant_teams: Mapping[
        int,
        int,
    ],
) -> tuple[
    SituationStateContext,
    ...,
]:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    states = _build_frame_states(
        snapshots=snapshots,
        participant_teams=(participant_teams),
    )

    if not states:
        raise ValueError("No participant frame states")

    ordered_situations = tuple(
        sorted(
            situations,
            key=_situation_sort_key,
        )
    )

    contexts: list[SituationStateContext] = []

    for situation in ordered_situations:
        if situation.match_id != clean_match_id:
            raise ValueError("Situation match_id does not match")

        before = _latest_state_at_or_before(
            states=states,
            timestamp_ms=(situation.start_timestamp_ms),
        )

        if before is None:
            raise ValueError(
                f"No frame available at or before situation start: {situation.situation_id}"
            )

        after = _earliest_state_at_or_after(
            states=states,
            timestamp_ms=(situation.end_timestamp_ms),
        )

        if after is None:
            raise ValueError(
                f"No frame available at or after situation end: {situation.situation_id}"
            )

        contexts.append(
            _build_context(
                situation=situation,
                before=before,
                after=after,
            )
        )

    result = tuple(
        sorted(
            contexts,
            key=_context_sort_key,
        )
    )

    if len(result) != len(ordered_situations):
        raise RuntimeError("Situation state context coverage mismatch")

    situation_ids = {situation.situation_id for situation in ordered_situations}

    context_ids = {context.situation_id for context in result}

    if situation_ids != context_ids:
        raise RuntimeError("Situation state context ID coverage mismatch")

    return result
