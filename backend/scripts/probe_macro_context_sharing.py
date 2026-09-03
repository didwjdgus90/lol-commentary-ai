from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.game_state_context import (
    build_situation_state_contexts,
)
from lol_commentary_backend.intelligence.game_state_models import (
    SituationStateContext,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)


@dataclass(frozen=True, slots=True)
class SituationMacroRow:
    match_id: str

    situation_id: str

    situation_kind: str

    start_timestamp_ms: int
    end_timestamp_ms: int

    max_salience_score: int

    member_count: int

    before_frame_index: int
    after_frame_index: int

    before_timestamp_ms: int
    after_timestamp_ms: int

    before_gold_diff: int
    after_gold_diff: int

    interval_gold_diff_change: int | None

    leading_team_changed: bool

    primary_types: tuple[str, ...]
    marker_types: tuple[str, ...]

    @property
    def frame_interval_key(
        self,
    ) -> tuple[str, int, int]:
        return (
            self.match_id,
            self.before_frame_index,
            self.after_frame_index,
        )

    @property
    def absolute_gold_change(
        self,
    ) -> int | None:
        if self.interval_gold_diff_change is None:
            return None

        return abs(self.interval_gold_diff_change)


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


def _load_snapshots(
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


def _participant_teams(
    match: NormalizedMatch,
) -> dict[int, int]:
    mapping: dict[
        int,
        int,
    ] = {}

    for participant in match.participants:
        if participant.participant_id in mapping:
            raise ValueError(f"Duplicate participant ID: {participant.participant_id}")

        mapping[participant.participant_id] = participant.team_id

    return mapping


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def _context_by_id(
    contexts: tuple[
        SituationStateContext,
        ...,
    ],
) -> dict[
    str,
    SituationStateContext,
]:
    result = {context.situation_id: context for context in contexts}

    if len(result) != len(contexts):
        raise RuntimeError("Duplicate state-context situation IDs")

    return result


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


def _timestamp(
    timestamp_ms: int,
) -> str:
    seconds = timestamp_ms // 1000

    minutes = seconds // 60

    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def _group_sort_key(
    item: tuple[
        tuple[str, int, int],
        list[SituationMacroRow],
    ],
) -> tuple[int, int, int]:
    _, rows = item

    maximum_gold_change = max((row.absolute_gold_change or 0) for row in rows)

    has_lead_change = any(row.leading_team_changed for row in rows)

    return (
        len(rows),
        (1 if has_lead_change else 0),
        maximum_gold_change,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

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

    if not match_directories:
        raise RuntimeError("No normalized matches found")

    rows: list[SituationMacroRow] = []

    total_candidates = 0
    total_situations = 0
    total_contexts = 0

    print("=== MACRO CONTEXT SHARING PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        snapshots = _load_snapshots(directory / "participant_frames.jsonl")

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
                key=_situation_sort_key,
            )
        )

        contexts = build_situation_state_contexts(
            match_id=(directory.name),
            situations=(situations),
            snapshots=snapshots,
            participant_teams=(_participant_teams(match)),
        )

        context_map = _context_by_id(contexts)

        total_candidates += len(candidates)

        total_situations += len(situations)

        total_contexts += len(contexts)

        match_keys: set[tuple[str, int, int]] = set()

        for situation in situations:
            context = context_map.get(situation.situation_id)

            if context is None:
                raise RuntimeError(f"Missing state context for situation {situation.situation_id}")

            row = SituationMacroRow(
                match_id=(directory.name),
                situation_id=(situation.situation_id),
                situation_kind=(situation.situation_kind.value),
                start_timestamp_ms=(situation.start_timestamp_ms),
                end_timestamp_ms=(situation.end_timestamp_ms),
                max_salience_score=(situation.max_salience_score),
                member_count=(len(situation.primary_candidates) + len(situation.semantic_markers)),
                before_frame_index=(context.before_state.frame_index),
                after_frame_index=(context.after_state.frame_index),
                before_timestamp_ms=(context.before_state.timestamp_ms),
                after_timestamp_ms=(context.after_state.timestamp_ms),
                before_gold_diff=(context.before_gold_diff_100_minus_200),
                after_gold_diff=(context.after_gold_diff_100_minus_200),
                interval_gold_diff_change=(context.interval_gold_diff_change_100_minus_200),
                leading_team_changed=(context.leading_team_changed),
                primary_types=tuple(
                    candidate.candidate_type.value for candidate in situation.primary_candidates
                ),
                marker_types=tuple(
                    candidate.candidate_type.value for candidate in situation.semantic_markers
                ),
            )

            rows.append(row)

            match_keys.add(row.frame_interval_key)

        print(f"MATCH={directory.name}")

        print(f"  candidates={len(candidates)}")

        print(f"  situations={len(situations)}")

        print(f"  macro_intervals={len(match_keys)}")

        print()

    if total_candidates != 266:
        raise RuntimeError(f"Expected 266 candidates, got {total_candidates}")

    if total_situations != 125:
        raise RuntimeError(f"Expected 125 situations, got {total_situations}")

    if total_contexts != 125:
        raise RuntimeError(f"Expected 125 contexts, got {total_contexts}")

    if len(rows) != 125:
        raise RuntimeError(f"Expected 125 macro rows, got {len(rows)}")

    grouped: dict[
        tuple[str, int, int],
        list[SituationMacroRow],
    ] = defaultdict(list)

    for row in rows:
        grouped[row.frame_interval_key].append(row)

    unique_intervals = len(grouped)

    sharing_counts: Counter[int] = Counter(len(group) for group in grouped.values())

    shared_intervals = sum(1 for group in grouped.values() if len(group) > 1)

    singleton_intervals = sum(1 for group in grouped.values() if len(group) == 1)

    situations_in_shared_intervals = sum(len(group) for group in grouped.values() if len(group) > 1)

    max_situations_per_interval = max(len(group) for group in grouped.values())

    distinct_gold_changes = [
        abs(group[0].interval_gold_diff_change)
        for group in grouped.values()
        if (group[0].interval_gold_diff_change is not None)
    ]

    p75 = _percentile(
        distinct_gold_changes,
        0.75,
    )

    p90 = _percentile(
        distinct_gold_changes,
        0.90,
    )

    macro_p90_intervals = [
        group
        for group in grouped.values()
        if (group[0].absolute_gold_change is not None and (group[0].absolute_gold_change >= p90))
    ]

    lead_change_intervals = [group for group in grouped.values() if group[0].leading_team_changed]

    p90_situation_exposures = sum(len(group) for group in macro_p90_intervals)

    lead_change_situation_exposures = sum(len(group) for group in lead_change_intervals)

    shared_p90_intervals = sum(len(group) > 1 for group in macro_p90_intervals)

    shared_lead_intervals = sum(len(group) > 1 for group in lead_change_intervals)

    print("=== AGGREGATE ===")

    print(f"Situations: {total_situations}")

    print(f"Unique macro intervals: {unique_intervals}")

    print(f"Singleton intervals: {singleton_intervals}")

    print(f"Shared intervals: {shared_intervals}")

    print(f"Situations inside shared intervals: {situations_in_shared_intervals}")

    print(f"Max situations per interval: {max_situations_per_interval}")

    print()

    print("Situations per macro interval:")

    for count, occurrences in sorted(sharing_counts.items()):
        print(f"  {count}: {occurrences}")

    print()

    if distinct_gold_changes:
        print("Unique-interval absolute gold change:")

        print(f"  observations: {len(distinct_gold_changes)}")

        print(f"  median: {int(median(distinct_gold_changes))}")

        print(f"  p75: {p75}")

        print(f"  p90: {p90}")

        print(f"  max: {max(distinct_gold_changes)}")

        print()

    print("Macro signal sharing:")

    print(f"  p90 macro intervals: {len(macro_p90_intervals)}")

    print(f"  p90 situation exposures: {p90_situation_exposures}")

    print(f"  shared p90 intervals: {shared_p90_intervals}")

    print(f"  lead-change intervals: {len(lead_change_intervals)}")

    print(f"  lead-change situation exposures: {lead_change_situation_exposures}")

    print(f"  shared lead-change intervals: {shared_lead_intervals}")

    print()

    print("=== MOST SHARED MACRO INTERVALS ===")

    ordered_groups = tuple(
        sorted(
            grouped.items(),
            key=_group_sort_key,
            reverse=True,
        )
    )

    for (
        match_id,
        before_frame,
        after_frame,
    ), group in ordered_groups[:15]:
        first = group[0]

        gold_change = first.interval_gold_diff_change

        print(
            f"{match_id} "
            f"frames="
            f"{before_frame}"
            "->"
            f"{after_frame} "
            f"time="
            f"{_timestamp(first.before_timestamp_ms)}"
            "->"
            f"{_timestamp(first.after_timestamp_ms)} "
            f"situations="
            f"{len(group)} "
            f"lead_change="
            f"{first.leading_team_changed} "
            f"gold_change="
            f"{gold_change}"
        )

        for row in sorted(
            group,
            key=lambda value: (
                value.start_timestamp_ms,
                value.end_timestamp_ms,
                value.situation_id,
            ),
        ):
            primary = ",".join(row.primary_types) or "-"

            markers = ",".join(row.marker_types) or "-"

            print(
                "  "
                f"{_timestamp(row.start_timestamp_ms)}"
                "-"
                f"{_timestamp(row.end_timestamp_ms)} "
                f"kind="
                f"{row.situation_kind} "
                f"salience="
                f"{row.max_salience_score} "
                f"members="
                f"{row.member_count}"
            )

            print(f"    primary={primary}")

            print(f"    markers={markers}")

    print()

    print("MACRO_CONTEXT_SHARING_PROBE=PASS")


if __name__ == "__main__":
    main()
