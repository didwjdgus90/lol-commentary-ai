from __future__ import annotations

from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import median

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
    CommentaryCandidateType,
)

_CLUSTER_WINDOWS_SECONDS = (
    3,
    5,
    8,
    10,
    12,
    15,
    20,
)

_SPECIAL_TYPES = {
    CommentaryCandidateType.FIRST_BLOOD,
    CommentaryCandidateType.MULTI_KILL,
    CommentaryCandidateType.ACE,
}

_OBJECTIVE_TYPES = {
    CommentaryCandidateType.ELITE_MONSTER,
    CommentaryCandidateType.BUILDING,
    CommentaryCandidateType.DRAGON_SOUL,
}


def _load_events(
    path: Path,
) -> tuple[NormalizedGameEvent, ...]:
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


def _percentile(
    values: list[int],
    percentile: float,
) -> int:
    if not values:
        return 0

    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    ordered = sorted(values)

    index = round((len(ordered) - 1) * percentile)

    return ordered[index]


def _gap_bucket(
    gap_ms: int,
) -> str:
    if gap_ms <= 1000:
        return "<=1s"

    if gap_ms <= 3000:
        return "1-3s"

    if gap_ms <= 5000:
        return "3-5s"

    if gap_ms <= 8000:
        return "5-8s"

    if gap_ms <= 10000:
        return "8-10s"

    if gap_ms <= 15000:
        return "10-15s"

    if gap_ms <= 20000:
        return "15-20s"

    if gap_ms <= 30000:
        return "20-30s"

    if gap_ms <= 60000:
        return "30-60s"

    return ">60s"


def _consecutive_gaps(
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ],
) -> list[int]:
    gaps: list[int] = []

    for previous, current in pairwise(candidates):
        gap = current.timestamp_ms - previous.timestamp_ms

        if gap < 0:
            raise ValueError("Candidates are not chronologically ordered")

        gaps.append(gap)

    return gaps


def _cluster_by_gap(
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ],
    *,
    max_gap_ms: int,
) -> tuple[
    tuple[
        CommentaryCandidate,
        ...,
    ],
    ...,
]:
    if not candidates:
        return ()

    clusters: list[list[CommentaryCandidate]] = [[candidates[0]]]

    for candidate in candidates[1:]:
        previous = clusters[-1][-1]

        gap = candidate.timestamp_ms - previous.timestamp_ms

        if gap <= max_gap_ms:
            clusters[-1].append(candidate)

        else:
            clusters.append([candidate])

    return tuple(tuple(cluster) for cluster in clusters)


def _cluster_signature(
    cluster: tuple[
        CommentaryCandidate,
        ...,
    ],
) -> str:
    counts: Counter[str] = Counter(candidate.candidate_type.value for candidate in cluster)

    return "+".join(
        (f"{name}:{count}" if count > 1 else name) for name, count in sorted(counts.items())
    )


def _special_kill_distances(
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ],
) -> list[int]:
    champion_kills = tuple(
        candidate
        for candidate in candidates
        if (candidate.candidate_type == CommentaryCandidateType.CHAMPION_KILL)
    )

    distances: list[int] = []

    for special in candidates:
        if special.candidate_type not in _SPECIAL_TYPES:
            continue

        possible = [
            abs(special.timestamp_ms - kill.timestamp_ms)
            for kill in champion_kills
            if (
                special.actor_participant_id is not None
                and (kill.actor_participant_id == special.actor_participant_id)
            )
        ]

        if possible:
            distances.append(min(possible))

    return distances


def _objective_followup_distances(
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ],
) -> list[int]:
    champion_kills = tuple(
        candidate
        for candidate in candidates
        if (candidate.candidate_type == CommentaryCandidateType.CHAMPION_KILL)
    )

    distances: list[int] = []

    for objective in candidates:
        if objective.candidate_type not in _OBJECTIVE_TYPES:
            continue

        previous_kills = [
            objective.timestamp_ms - kill.timestamp_ms
            for kill in champion_kills
            if (kill.timestamp_ms <= objective.timestamp_ms)
        ]

        if previous_kills:
            distances.append(min(previous_kills))

    return distances


def _print_distance_thresholds(
    *,
    title: str,
    distances: list[int],
) -> None:
    print(title)

    if not distances:
        print("  none")
        return

    thresholds = (
        500,
        1000,
        2000,
        3000,
        5000,
        8000,
        10000,
        15000,
        20000,
        30000,
    )

    print(f"  observations: {len(distances)}")

    print(f"  median_ms: {int(median(distances))}")

    print(f"  p90_ms: {_percentile(distances, 0.90)}")

    for threshold in thresholds:
        count = sum(distance <= threshold for distance in distances)

        print(f"  <= {threshold:5d} ms: {count}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    processed_root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

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
        raise RuntimeError("No normalized matches found")

    aggregate_candidate_count = 0

    aggregate_gaps: list[int] = []

    aggregate_special_distances: list[int] = []

    aggregate_objective_distances: list[int] = []

    cluster_totals: Counter[int] = Counter()

    multi_candidate_totals: Counter[int] = Counter()

    largest_clusters: Counter[int] = Counter()

    signature_counts: dict[
        int,
        Counter[str],
    ] = {seconds: Counter() for seconds in _CLUSTER_WINDOWS_SECONDS}

    print("=== COMMENTARY TEMPORAL STRUCTURE PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        events = _load_events(directory / "events.jsonl")

        candidates = build_commentary_candidates(
            match_id=directory.name,
            events=events,
        )

        gaps = _consecutive_gaps(candidates)

        special_distances = _special_kill_distances(candidates)

        objective_distances = _objective_followup_distances(candidates)

        aggregate_candidate_count += len(candidates)

        aggregate_gaps.extend(gaps)

        aggregate_special_distances.extend(special_distances)

        aggregate_objective_distances.extend(objective_distances)

        print(f"MATCH={directory.name}")

        print(f"  candidates={len(candidates)}")

        if gaps:
            print(f"  consecutive_gap_median_ms={int(median(gaps))}")

            print(f"  consecutive_gap_p90_ms={_percentile(gaps, 0.90)}")

        for seconds in _CLUSTER_WINDOWS_SECONDS:
            clusters = _cluster_by_gap(
                candidates,
                max_gap_ms=(seconds * 1000),
            )

            multi_clusters = tuple(cluster for cluster in clusters if len(cluster) > 1)

            largest = max(
                (len(cluster) for cluster in clusters),
                default=0,
            )

            cluster_totals[seconds] += len(clusters)

            multi_candidate_totals[seconds] += len(multi_clusters)

            largest_clusters[seconds] = max(
                largest_clusters[seconds],
                largest,
            )

            for cluster in multi_clusters:
                signature_counts[seconds][_cluster_signature(cluster)] += 1

            print(
                f"  window={seconds:2d}s "
                f"clusters="
                f"{len(clusters):3d} "
                f"multi="
                f"{len(multi_clusters):3d} "
                f"largest={largest}"
            )

        print()

    if aggregate_candidate_count != 266:
        raise RuntimeError(f"Unexpected commentary candidate count: {aggregate_candidate_count}")

    print("=== AGGREGATE CANDIDATE GAPS ===")

    print(f"Candidates: {aggregate_candidate_count}")

    print(f"Gaps: {len(aggregate_gaps)}")

    print(f"Median gap ms: {int(median(aggregate_gaps))}")

    print(f"P75 gap ms: {_percentile(aggregate_gaps, 0.75)}")

    print(f"P90 gap ms: {_percentile(aggregate_gaps, 0.90)}")

    print()

    gap_buckets: Counter[str] = Counter(_gap_bucket(gap) for gap in aggregate_gaps)

    print("Consecutive gap buckets:")

    for bucket in (
        "<=1s",
        "1-3s",
        "3-5s",
        "5-8s",
        "8-10s",
        "10-15s",
        "15-20s",
        "20-30s",
        "30-60s",
        ">60s",
    ):
        print(f"  {bucket}: {gap_buckets[bucket]}")

    print()

    print("=== WINDOW COMPARISON ===")

    for seconds in _CLUSTER_WINDOWS_SECONDS:
        print(
            f"window={seconds:2d}s "
            f"clusters="
            f"{cluster_totals[seconds]:3d} "
            f"multi="
            f"{multi_candidate_totals[seconds]:3d} "
            f"largest="
            f"{largest_clusters[seconds]}"
        )

    print()

    _print_distance_thresholds(
        title=("Special kill -> nearest same-actor champion kill:"),
        distances=(aggregate_special_distances),
    )

    print()

    _print_distance_thresholds(
        title=("Objective/building -> nearest previous champion kill:"),
        distances=(aggregate_objective_distances),
    )

    print()

    print("=== MULTI-CANDIDATE SIGNATURES ===")

    for seconds in (
        5,
        8,
        10,
        15,
    ):
        print(f"[window={seconds}s]")

        common = signature_counts[seconds].most_common(10)

        if not common:
            print("  none")
            continue

        for signature, count in common:
            print(f"  {signature}: {count}")

    print()

    print("COMMENTARY_TEMPORAL_PROBE=PASS")


if __name__ == "__main__":
    main()
