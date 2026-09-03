from __future__ import annotations

from collections import Counter
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
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)
from lol_commentary_backend.intelligence.situation_models import (
    TemporalSituation,
)


@dataclass(frozen=True, slots=True)
class PriorityProbeRow:
    match_id: str

    start_timestamp_ms: int
    end_timestamp_ms: int

    situation_kind: str

    max_salience_score: int

    primary_count: int
    semantic_marker_count: int

    primary_types: tuple[str, ...]
    marker_types: tuple[str, ...]

    before_gold_diff: int
    after_gold_diff: int

    interval_gold_diff_change: int | None

    leading_team_changed: bool

    @property
    def member_count(self) -> int:
        return self.primary_count + self.semantic_marker_count

    @property
    def absolute_interval_gold_change(
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
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

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


def _salience_bucket(
    score: int,
) -> str:
    if score >= 80:
        return "80-100"

    if score >= 60:
        return "60-79"

    return "45-59"


def _gold_change_bucket(
    value: int,
) -> str:
    absolute = abs(value)

    if absolute <= 500:
        return "<=500"

    if absolute <= 1000:
        return "501-1000"

    if absolute <= 1500:
        return "1001-1500"

    if absolute <= 2000:
        return "1501-2000"

    if absolute <= 3000:
        return "2001-3000"

    return ">3000"


def _signal_sort_key(
    row: PriorityProbeRow,
) -> tuple[int, int, int, int]:
    return (
        (1 if row.leading_team_changed else 0),
        (row.absolute_interval_gold_change or 0),
        row.max_salience_score,
        row.member_count,
    )


def _format_timestamp(
    timestamp_ms: int,
) -> str:
    total_seconds = timestamp_ms // 1000

    minutes = total_seconds // 60

    seconds = total_seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


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

    rows: list[PriorityProbeRow] = []

    total_candidates = 0
    total_situations = 0
    total_contexts = 0

    print("=== SITUATION PRIORITY SIGNAL PROBE ===")

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

        if len(situations) != len(contexts):
            raise RuntimeError(f"Situation/context count mismatch for {directory.name}")

        contexts_by_id = {context.situation_id: (context) for context in contexts}

        if len(contexts_by_id) != len(contexts):
            raise RuntimeError("Duplicate situation context IDs")

        total_candidates += len(candidates)

        total_situations += len(situations)

        total_contexts += len(contexts)

        match_lead_changes = 0

        for situation in situations:
            context = contexts_by_id.get(situation.situation_id)

            if context is None:
                raise RuntimeError(f"Missing state context for situation {situation.situation_id}")

            if context.leading_team_changed:
                match_lead_changes += 1

            rows.append(
                PriorityProbeRow(
                    match_id=(directory.name),
                    start_timestamp_ms=(situation.start_timestamp_ms),
                    end_timestamp_ms=(situation.end_timestamp_ms),
                    situation_kind=(situation.situation_kind.value),
                    max_salience_score=(situation.max_salience_score),
                    primary_count=len(situation.primary_candidates),
                    semantic_marker_count=(len(situation.semantic_markers)),
                    primary_types=tuple(
                        candidate.candidate_type.value for candidate in situation.primary_candidates
                    ),
                    marker_types=tuple(
                        candidate.candidate_type.value for candidate in situation.semantic_markers
                    ),
                    before_gold_diff=(context.before_gold_diff_100_minus_200),
                    after_gold_diff=(context.after_gold_diff_100_minus_200),
                    interval_gold_diff_change=(context.interval_gold_diff_change_100_minus_200),
                    leading_team_changed=(context.leading_team_changed),
                )
            )

        print(f"MATCH={directory.name}")

        print(f"  candidates={len(candidates)}")

        print(f"  situations={len(situations)}")

        print(f"  contexts={len(contexts)}")

        print(f"  leading_team_changes={match_lead_changes}")

        print()

    if total_candidates != 266:
        raise RuntimeError(f"Expected 266 candidates, got {total_candidates}")

    if total_situations != 125:
        raise RuntimeError(f"Expected 125 situations, got {total_situations}")

    if total_contexts != 125:
        raise RuntimeError(f"Expected 125 contexts, got {total_contexts}")

    if len(rows) != 125:
        raise RuntimeError(f"Expected 125 priority rows, got {len(rows)}")

    situation_kinds: Counter[str] = Counter(row.situation_kind for row in rows)

    salience_buckets: Counter[str] = Counter(
        _salience_bucket(row.max_salience_score) for row in rows
    )

    member_counts: Counter[int] = Counter(row.member_count for row in rows)

    marker_counts: Counter[int] = Counter(row.semantic_marker_count for row in rows)

    gold_changes = [
        row.interval_gold_diff_change for row in rows if (row.interval_gold_diff_change is not None)
    ]

    absolute_gold_changes = [abs(value) for value in gold_changes]

    gold_buckets: Counter[str] = Counter(_gold_change_bucket(value) for value in gold_changes)

    p75_gold_change = _percentile(
        absolute_gold_changes,
        0.75,
    )

    p90_gold_change = _percentile(
        absolute_gold_changes,
        0.90,
    )

    lead_change_rows = [row for row in rows if row.leading_team_changed]

    critical_rows = [row for row in rows if (row.max_salience_score >= 80)]

    macro_p90_rows = [
        row
        for row in rows
        if (
            row.absolute_interval_gold_change is not None
            and (row.absolute_interval_gold_change >= p90_gold_change)
        )
    ]

    kind_stats: dict[
        str,
        Counter[str],
    ] = {}

    for kind in sorted(situation_kinds):
        kind_stats[kind] = Counter()

    for row in rows:
        stats = kind_stats[row.situation_kind]

        stats["total"] += 1

        if row.max_salience_score >= 80:
            stats["critical"] += 1

        if row.leading_team_changed:
            stats["lead_change"] += 1

        if row.absolute_interval_gold_change is not None and (
            row.absolute_interval_gold_change >= p90_gold_change
        ):
            stats["macro_p90"] += 1

    critical_and_lead = sum(
        (row.max_salience_score >= 80) and row.leading_team_changed for row in rows
    )

    critical_and_macro = sum(
        (row.max_salience_score >= 80)
        and (row.absolute_interval_gold_change is not None)
        and (row.absolute_interval_gold_change >= p90_gold_change)
        for row in rows
    )

    lead_and_macro = sum(
        row.leading_team_changed
        and (row.absolute_interval_gold_change is not None)
        and (row.absolute_interval_gold_change >= p90_gold_change)
        for row in rows
    )

    at_least_two_signals = 0
    all_three_signals = 0

    for row in rows:
        signals = 0

        if row.max_salience_score >= 80:
            signals += 1

        if row.leading_team_changed:
            signals += 1

        if row.absolute_interval_gold_change is not None and (
            row.absolute_interval_gold_change >= p90_gold_change
        ):
            signals += 1

        if signals >= 2:
            at_least_two_signals += 1

        if signals == 3:
            all_three_signals += 1

    print("=== AGGREGATE ===")

    print(f"Candidates: {total_candidates}")

    print(f"Situations: {total_situations}")

    print(f"State contexts: {total_contexts}")

    print()

    print("Situation kinds:")

    for kind, count in sorted(situation_kinds.items()):
        print(f"  {kind}: {count}")

    print()

    print("Situation max-salience buckets:")

    for bucket in (
        "45-59",
        "60-79",
        "80-100",
    ):
        print(f"  {bucket}: {salience_buckets[bucket]}")

    print()

    print("Situation member counts:")

    for count, occurrences in sorted(member_counts.items()):
        print(f"  {count}: {occurrences}")

    print()

    print("Semantic-marker counts:")

    for count, occurrences in sorted(marker_counts.items()):
        print(f"  {count}: {occurrences}")

    print()

    print("Absolute interval gold-diff change:")

    print(f"  observations: {len(absolute_gold_changes)}")

    print(f"  median: {int(median(absolute_gold_changes))}")

    print(f"  p75: {p75_gold_change}")

    print(f"  p90: {p90_gold_change}")

    print(f"  max: {max(absolute_gold_changes)}")

    print()

    print("Absolute interval gold change buckets:")

    for bucket in (
        "<=500",
        "501-1000",
        "1001-1500",
        "1501-2000",
        "2001-3000",
        ">3000",
    ):
        print(f"  {bucket}: {gold_buckets[bucket]}")

    print()

    print("Priority signal counts:")

    print(f"  event critical (salience>=80): {len(critical_rows)}")

    print(f"  leading-team change: {len(lead_change_rows)}")

    print(f"  macro gold p90+: {len(macro_p90_rows)}")

    print()

    print("Signal overlaps:")

    print(f"  critical + lead change: {critical_and_lead}")

    print(f"  critical + macro p90: {critical_and_macro}")

    print(f"  lead change + macro p90: {lead_and_macro}")

    print(f"  at least two signals: {at_least_two_signals}")

    print(f"  all three signals: {all_three_signals}")

    print()

    print("Signals by situation kind:")

    for kind, stats in sorted(kind_stats.items()):
        print(
            f"  {kind}: "
            f"total={stats['total']} "
            f"critical={stats['critical']} "
            f"lead_change="
            f"{stats['lead_change']} "
            f"macro_p90="
            f"{stats['macro_p90']}"
        )

    print()

    print("=== TOP SIGNAL-RICH SITUATIONS ===")

    top_rows = tuple(
        sorted(
            rows,
            key=_signal_sort_key,
            reverse=True,
        )[:20]
    )

    for row in top_rows:
        absolute_change = row.absolute_interval_gold_change

        gold_change_text = (
            "same-frame" if absolute_change is None else str(row.interval_gold_diff_change)
        )

        primary_types = ",".join(row.primary_types) or "-"

        marker_types = ",".join(row.marker_types) or "-"

        print(
            f"{row.match_id} "
            f"{_format_timestamp(row.start_timestamp_ms)}"
            "-"
            f"{_format_timestamp(row.end_timestamp_ms)} "
            f"kind={row.situation_kind} "
            f"salience="
            f"{row.max_salience_score} "
            f"members="
            f"{row.member_count} "
            f"lead_change="
            f"{row.leading_team_changed} "
            f"gold_interval_change="
            f"{gold_change_text}"
        )

        print(f"  primary={primary_types}")

        print(f"  markers={marker_types}")

        print(f"  gold_lead {row.before_gold_diff} -> {row.after_gold_diff}")

    print()

    print("SITUATION_PRIORITY_SIGNAL_PROBE=PASS")


if __name__ == "__main__":
    main()
