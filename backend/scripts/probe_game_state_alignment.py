from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)


@dataclass(frozen=True, slots=True)
class TeamTotals:
    gold: int
    xp: int
    levels: int
    lane_cs: int
    jungle_cs: int


@dataclass(frozen=True, slots=True)
class FrameState:
    frame_index: int
    timestamp_ms: int
    teams: dict[int, TeamTotals]


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _load_match(
    path: Path,
) -> NormalizedMatch:
    return NormalizedMatch.model_validate_json(path.read_text(encoding="utf-8"))


def _load_events(
    path: Path,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    events: list[NormalizedGameEvent] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            event = NormalizedGameEvent.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid event at {path}:{line_number}") from exc

        events.append(event)

    return tuple(events)


def _load_participant_frames(
    path: Path,
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    snapshots: list[ParticipantFrameSnapshot] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            snapshot = ParticipantFrameSnapshot.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid participant frame at {path}:{line_number}") from exc

        snapshots.append(snapshot)

    return tuple(snapshots)


def _participant_team_map(
    match: NormalizedMatch,
) -> dict[int, int]:
    mapping: dict[
        int,
        int,
    ] = {}

    for participant in match.participants:
        if participant.participant_id in mapping:
            raise ValueError("Duplicate participant ID")

        mapping[participant.participant_id] = participant.team_id

    return mapping


def _build_frame_states(
    *,
    snapshots: tuple[
        ParticipantFrameSnapshot,
        ...,
    ],
    participant_teams: dict[
        int,
        int,
    ],
) -> tuple[
    FrameState,
    ...,
]:
    grouped: dict[
        int,
        list[ParticipantFrameSnapshot],
    ] = defaultdict(list)

    for snapshot in snapshots:
        grouped[snapshot.frame_index].append(snapshot)

    states: list[FrameState] = []

    for frame_index in sorted(grouped):
        frame_snapshots = grouped[frame_index]

        timestamps = {snapshot.timestamp_ms for snapshot in frame_snapshots}

        if len(timestamps) != 1:
            raise ValueError("Participant snapshots in one frame have different timestamps")

        timestamp_ms = next(iter(timestamps))

        team_values: dict[
            int,
            dict[
                str,
                int,
            ],
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
                raise ValueError(
                    f"Participant frame has unknown participantId: {snapshot.participant_id}"
                )

            values = team_values[team_id]

            values["gold"] += snapshot.total_gold or 0

            values["xp"] += snapshot.xp or 0

            values["levels"] += snapshot.level or 0

            values["lane_cs"] += snapshot.minions_killed or 0

            values["jungle_cs"] += snapshot.jungle_minions_killed or 0

        teams = {
            team_id: TeamTotals(
                gold=values["gold"],
                xp=values["xp"],
                levels=values["levels"],
                lane_cs=values["lane_cs"],
                jungle_cs=values["jungle_cs"],
            )
            for team_id, values in team_values.items()
        }

        states.append(
            FrameState(
                frame_index=(frame_index),
                timestamp_ms=(timestamp_ms),
                teams=teams,
            )
        )

    return tuple(states)


def _latest_frame_at_or_before(
    *,
    states: tuple[
        FrameState,
        ...,
    ],
    timestamp_ms: int,
) -> FrameState | None:
    result: FrameState | None = None

    for state in states:
        if state.timestamp_ms > timestamp_ms:
            break

        result = state

    return result


def _earliest_frame_at_or_after(
    *,
    states: tuple[
        FrameState,
        ...,
    ],
    timestamp_ms: int,
) -> FrameState | None:
    for state in states:
        if state.timestamp_ms >= timestamp_ms:
            return state

    return None


def _percentile(
    values: list[int],
    percentile: float,
) -> int:
    if not values:
        return 0

    if not (0 <= percentile <= 1):
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)

    index = round((len(ordered) - 1) * percentile)

    return ordered[index]


def _distance_bucket(
    value_ms: int,
) -> str:
    if value_ms <= 5_000:
        return "<=5s"

    if value_ms <= 15_000:
        return "5-15s"

    if value_ms <= 30_000:
        return "15-30s"

    if value_ms <= 45_000:
        return "30-45s"

    if value_ms <= 60_000:
        return "45-60s"

    return ">60s"


def _gold_diff(
    state: FrameState,
) -> int | None:
    blue = state.teams.get(100)

    red = state.teams.get(200)

    if blue is None or red is None:
        return None

    return blue.gold - red.gold


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[
    int,
    int,
    str,
]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    aggregate = _load_json_object(root / "manifest.json")

    match_directories = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if (
                    path.is_dir()
                    and (path / "match.json").is_file()
                    and (path / "events.jsonl").is_file()
                    and (path / "participant_frames.jsonl").is_file()
                )
            ),
            key=lambda path: path.name,
        )
    )

    expected_match_count = aggregate.get("match_count")

    if not isinstance(
        expected_match_count,
        int,
    ):
        raise ValueError("Aggregate manifest has invalid match_count")

    if len(match_directories) != expected_match_count:
        raise RuntimeError(
            f"Normalized match count mismatch: {len(match_directories)} != {expected_match_count}"
        )

    if not match_directories:
        raise RuntimeError("No normalized matches found")

    total_situations = 0
    aligned_situations = 0

    same_frame_count = 0
    distinct_frame_count = 0

    missing_before = 0
    missing_after = 0

    start_lags: list[int] = []

    end_leads: list[int] = []

    alignment_spans: list[int] = []

    gold_diff_changes: list[int] = []

    start_buckets: Counter[str] = Counter()

    end_buckets: Counter[str] = Counter()

    kinds: Counter[str] = Counter()

    print("=== GAME STATE ALIGNMENT PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        snapshots = _load_participant_frames(directory / "participant_frames.jsonl")

        participant_teams = _participant_team_map(match)

        states = _build_frame_states(
            snapshots=snapshots,
            participant_teams=(participant_teams),
        )

        candidates = build_commentary_candidates(
            match_id=(directory.name),
            events=events,
        )

        situations = tuple(
            sorted(
                build_temporal_situations(
                    match_id=(directory.name),
                    candidates=(candidates),
                ),
                key=(_situation_sort_key),
            )
        )

        match_same = 0
        match_distinct = 0

        match_missing_before = 0
        match_missing_after = 0

        match_gold_swings: list[int] = []

        for situation in situations:
            total_situations += 1

            kinds[situation.situation_kind.value] += 1

            before = _latest_frame_at_or_before(
                states=states,
                timestamp_ms=(situation.start_timestamp_ms),
            )

            after = _earliest_frame_at_or_after(
                states=states,
                timestamp_ms=(situation.end_timestamp_ms),
            )

            if before is None:
                missing_before += 1
                match_missing_before += 1

            if after is None:
                missing_after += 1
                match_missing_after += 1

            if before is None or after is None:
                continue

            aligned_situations += 1

            start_lag = situation.start_timestamp_ms - before.timestamp_ms

            end_lead = after.timestamp_ms - situation.end_timestamp_ms

            alignment_span = after.timestamp_ms - before.timestamp_ms

            if start_lag < 0:
                raise RuntimeError("Previous frame occurs after situation start")

            if end_lead < 0:
                raise RuntimeError("Next frame occurs before situation end")

            if alignment_span < 0:
                raise RuntimeError("Invalid frame alignment span")

            start_lags.append(start_lag)

            end_leads.append(end_lead)

            alignment_spans.append(alignment_span)

            start_buckets[_distance_bucket(start_lag)] += 1

            end_buckets[_distance_bucket(end_lead)] += 1

            if before.frame_index == after.frame_index:
                same_frame_count += 1
                match_same += 1

            else:
                distinct_frame_count += 1
                match_distinct += 1

                before_gold_diff = _gold_diff(before)

                after_gold_diff = _gold_diff(after)

                if before_gold_diff is not None and after_gold_diff is not None:
                    change = after_gold_diff - before_gold_diff

                    gold_diff_changes.append(change)

                    match_gold_swings.append(change)

        print(f"MATCH={directory.name}")

        print(f"  frames={len(states)}")

        print(f"  situations={len(situations)}")

        print(f"  same_frame={match_same}")

        print(f"  distinct_frames={match_distinct}")

        print(f"  missing_before={match_missing_before}")

        print(f"  missing_after={match_missing_after}")

        if match_gold_swings:
            largest = max(
                match_gold_swings,
                key=abs,
            )

            print(f"  largest_gold_diff_change={largest}")

        print()

    if total_situations != 125:
        raise RuntimeError(f"Unexpected temporal situation count: {total_situations}")

    if aligned_situations + missing_before + missing_after < total_situations:
        raise RuntimeError("Alignment accounting is inconsistent")

    print("=== AGGREGATE ===")

    print(f"Situations: {total_situations}")

    print(f"Aligned: {aligned_situations}")

    print(f"Missing before: {missing_before}")

    print(f"Missing after: {missing_after}")

    print(f"Same frame: {same_frame_count}")

    print(f"Distinct frames: {distinct_frame_count}")

    print()

    if start_lags:
        print("Start -> previous frame:")

        print(f"  median_ms: {int(median(start_lags))}")

        print(f"  p90_ms: {_percentile(start_lags, 0.90)}")

        for bucket in (
            "<=5s",
            "5-15s",
            "15-30s",
            "30-45s",
            "45-60s",
            ">60s",
        ):
            print(f"  {bucket}: {start_buckets[bucket]}")

        print()

    if end_leads:
        print("End -> next frame:")

        print(f"  median_ms: {int(median(end_leads))}")

        print(f"  p90_ms: {_percentile(end_leads, 0.90)}")

        for bucket in (
            "<=5s",
            "5-15s",
            "15-30s",
            "30-45s",
            "45-60s",
            ">60s",
        ):
            print(f"  {bucket}: {end_buckets[bucket]}")

        print()

    if alignment_spans:
        print("Alignment frame span:")

        print(f"  median_ms: {int(median(alignment_spans))}")

        print(f"  p90_ms: {_percentile(alignment_spans, 0.90)}")

        print()

    if gold_diff_changes:
        absolute_changes = [abs(value) for value in gold_diff_changes]

        print("Distinct-frame gold difference changes:")

        print(f"  observations: {len(gold_diff_changes)}")

        print(f"  median_abs: {int(median(absolute_changes))}")

        print(f"  p90_abs: {_percentile(absolute_changes, 0.90)}")

        print(f"  max_abs: {max(absolute_changes)}")

        print()

    print("Situation kinds:")

    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")

    print()

    print("GAME_STATE_ALIGNMENT_PROBE=PASS")


if __name__ == "__main__":
    main()
