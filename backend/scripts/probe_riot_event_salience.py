from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _load_jsonl(
    path: Path,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        payload = json.loads(line)

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")

        records.append(payload)

    return tuple(records)


def _string_value(
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


def _integer_value(
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


def _print_string_counter(
    *,
    title: str,
    counter: Counter[str],
) -> None:
    print(f"{title}:")

    if not counter:
        print("  none")
        return

    for value, count in sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        print(f"  {value}: {count}")


def _print_integer_counter(
    *,
    title: str,
    counter: Counter[int],
) -> None:
    print(f"{title}:")

    if not counter:
        print("  none")
        return

    for value, count in sorted(
        counter.items(),
        key=lambda item: item[0],
    ):
        print(f"  {value}: {count}")


def _shutdown_bucket(
    value: int,
) -> str:
    if value <= 0:
        return "0"

    if value < 150:
        return "1-149"

    if value < 300:
        return "150-299"

    if value < 500:
        return "300-499"

    if value < 700:
        return "500-699"

    return "700+"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    processed_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    aggregate_path = processed_root / "manifest.json"

    if not aggregate_path.is_file():
        raise FileNotFoundError("Normalized aggregate manifest not found")

    aggregate = _load_json_object(aggregate_path)

    expected_event_count = aggregate.get("total_event_count")

    if not isinstance(
        expected_event_count,
        int,
    ) or isinstance(
        expected_event_count,
        bool,
    ):
        raise ValueError("Aggregate total_event_count is invalid")

    match_directories = tuple(
        sorted(
            (
                path
                for path in processed_root.iterdir()
                if (path.is_dir() and (path / "events.jsonl").is_file())
            ),
            key=lambda path: path.name,
        )
    )

    if not match_directories:
        raise RuntimeError("No normalized Riot matches found")

    event_types: Counter[str] = Counter()

    kill_bounties: Counter[int] = Counter()

    shutdown_bounties: Counter[int] = Counter()

    shutdown_buckets: Counter[str] = Counter()

    kill_streak_lengths: Counter[int] = Counter()

    assist_counts: Counter[int] = Counter()

    special_kill_types: Counter[str] = Counter()

    multi_kill_lengths: Counter[int] = Counter()

    monster_types: Counter[str] = Counter()

    monster_sub_types: Counter[str] = Counter()

    building_types: Counter[str] = Counter()

    tower_types: Counter[str] = Counter()

    building_lanes: Counter[str] = Counter()

    plate_lanes: Counter[str] = Counter()

    dragon_soul_names: Counter[str] = Counter()

    objective_bounty_teams: Counter[int] = Counter()

    winning_teams: Counter[int] = Counter()

    total_events = 0

    print("=== RIOT EVENT SALIENCE PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        events = _load_jsonl(directory / "events.jsonl")

        expected_sequence = 0

        for event in events:
            sequence = _integer_value(
                event,
                "sequence",
            )

            if sequence != expected_sequence:
                raise ValueError(
                    "Non-contiguous event "
                    "sequence in "
                    f"{directory.name}: "
                    f"expected "
                    f"{expected_sequence}, "
                    f"got {sequence}"
                )

            expected_sequence += 1

            event_type = _string_value(
                event,
                "raw_event_type",
            )

            if event_type is None:
                raise ValueError("raw_event_type missing")

            event_types[event_type] += 1

            total_events += 1

            if event_type == "CHAMPION_KILL":
                bounty = _integer_value(
                    event,
                    "bounty",
                )

                if bounty is not None:
                    kill_bounties[bounty] += 1

                shutdown = _integer_value(
                    event,
                    "shutdown_bounty",
                )

                if shutdown is not None:
                    shutdown_bounties[shutdown] += 1

                    shutdown_buckets[_shutdown_bucket(shutdown)] += 1

                streak = _integer_value(
                    event,
                    "kill_streak_length",
                )

                if streak is not None:
                    kill_streak_lengths[streak] += 1

                assists = event.get("assisting_participant_ids")

                if isinstance(
                    assists,
                    list,
                ):
                    assist_counts[len(assists)] += 1

            elif event_type == "CHAMPION_SPECIAL_KILL":
                kill_type = _string_value(
                    event,
                    "kill_type",
                )

                if kill_type:
                    special_kill_types[kill_type] += 1

                multi_length = _integer_value(
                    event,
                    "multi_kill_length",
                )

                if multi_length is not None:
                    multi_kill_lengths[multi_length] += 1

            elif event_type == "ELITE_MONSTER_KILL":
                monster_type = _string_value(
                    event,
                    "monster_type",
                )

                if monster_type:
                    monster_types[monster_type] += 1

                monster_sub_type = _string_value(
                    event,
                    "monster_sub_type",
                )

                if monster_sub_type:
                    monster_sub_types[monster_sub_type] += 1

            elif event_type == "BUILDING_KILL":
                building_type = _string_value(
                    event,
                    "building_type",
                )

                if building_type:
                    building_types[building_type] += 1

                tower_type = _string_value(
                    event,
                    "tower_type",
                )

                if tower_type:
                    tower_types[tower_type] += 1

                lane_type = _string_value(
                    event,
                    "lane_type",
                )

                if lane_type:
                    building_lanes[lane_type] += 1

            elif event_type == "TURRET_PLATE_DESTROYED":
                lane_type = _string_value(
                    event,
                    "lane_type",
                )

                if lane_type:
                    plate_lanes[lane_type] += 1

            elif event_type == "DRAGON_SOUL_GIVEN":
                soul_name = _string_value(
                    event,
                    "objective_name",
                )

                if soul_name:
                    dragon_soul_names[soul_name] += 1

            elif event_type == "OBJECTIVE_BOUNTY_PRESTART":
                team_id = _integer_value(
                    event,
                    "team_id",
                )

                if team_id is not None:
                    objective_bounty_teams[team_id] += 1

            elif event_type == "GAME_END":
                winning_team = _integer_value(
                    event,
                    "winning_team_id",
                )

                if winning_team is not None:
                    winning_teams[winning_team] += 1

        print(f"MATCH={directory.name} EVENTS={len(events)}")

    print()

    if total_events != expected_event_count:
        raise RuntimeError(
            "Processed event count "
            "does not match aggregate "
            "manifest: "
            f"{total_events} != "
            f"{expected_event_count}"
        )

    print("=== AGGREGATE ===")

    print(f"Total events: {total_events}")

    print()

    _print_string_counter(
        title="Event types",
        counter=event_types,
    )

    print()

    print("=== CHAMPION KILL ===")

    _print_integer_counter(
        title="Bounty values",
        counter=kill_bounties,
    )

    _print_integer_counter(
        title="Shutdown bounty values",
        counter=shutdown_bounties,
    )

    _print_string_counter(
        title="Shutdown bounty buckets",
        counter=shutdown_buckets,
    )

    _print_integer_counter(
        title="Kill streak lengths",
        counter=kill_streak_lengths,
    )

    _print_integer_counter(
        title="Assist counts",
        counter=assist_counts,
    )

    print()

    print("=== SPECIAL KILL ===")

    _print_string_counter(
        title="Kill types",
        counter=special_kill_types,
    )

    _print_integer_counter(
        title="Multi-kill lengths",
        counter=multi_kill_lengths,
    )

    print()

    print("=== ELITE MONSTERS ===")

    _print_string_counter(
        title="Monster types",
        counter=monster_types,
    )

    _print_string_counter(
        title="Monster sub-types",
        counter=monster_sub_types,
    )

    print()

    print("=== BUILDINGS ===")

    _print_string_counter(
        title="Building types",
        counter=building_types,
    )

    _print_string_counter(
        title="Tower types",
        counter=tower_types,
    )

    _print_string_counter(
        title="Building lanes",
        counter=building_lanes,
    )

    _print_string_counter(
        title="Plate lanes",
        counter=plate_lanes,
    )

    print()

    print("=== OTHER OBJECTIVES ===")

    _print_string_counter(
        title="Dragon soul names",
        counter=dragon_soul_names,
    )

    _print_integer_counter(
        title="Objective bounty teams",
        counter=objective_bounty_teams,
    )

    _print_integer_counter(
        title="Winning teams",
        counter=winning_teams,
    )

    print()

    print("RIOT_EVENT_SALIENCE_PROBE=PASS")


if __name__ == "__main__":
    main()
