from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
    CommentaryCandidateType,
)
from lol_commentary_backend.intelligence.situation_models import (
    SituationKind,
    TemporalSituation,
)

PRIMARY_GAP_MS = 8_000
OBJECTIVE_FOLLOWUP_GAP_MS = 10_000
SPECIAL_MARKER_TOLERANCE_MS = 500


_SPECIAL_TYPES = {
    CommentaryCandidateType.FIRST_BLOOD,
    CommentaryCandidateType.MULTI_KILL,
    CommentaryCandidateType.ACE,
}


_OBJECTIVE_FOLLOWUP_TYPES = {
    CommentaryCandidateType.ELITE_MONSTER,
    CommentaryCandidateType.BUILDING,
    CommentaryCandidateType.DRAGON_SOUL,
}


@dataclass(slots=True)
class _SituationBucket:
    match_id: str

    primary_candidates: list[CommentaryCandidate] = field(default_factory=list)

    semantic_markers: list[CommentaryCandidate] = field(default_factory=list)


def _is_special_marker(
    candidate: CommentaryCandidate,
) -> bool:
    return candidate.candidate_type in _SPECIAL_TYPES


def _is_champion_kill(
    candidate: CommentaryCandidate,
) -> bool:
    return candidate.candidate_type == CommentaryCandidateType.CHAMPION_KILL


def _contains_combat(
    bucket: _SituationBucket,
) -> bool:
    return any(_is_champion_kill(candidate) for candidate in bucket.primary_candidates)


def _should_join_primary(
    *,
    bucket: _SituationBucket,
    candidate: CommentaryCandidate,
) -> bool:
    if not bucket.primary_candidates:
        return False

    if candidate.candidate_type == CommentaryCandidateType.GAME_END:
        return False

    previous = bucket.primary_candidates[-1]

    if previous.candidate_type == CommentaryCandidateType.GAME_END:
        return False

    gap_ms = candidate.timestamp_ms - previous.timestamp_ms

    if gap_ms < 0:
        raise ValueError("Candidates must be chronologically ordered")

    if gap_ms <= PRIMARY_GAP_MS:
        return True

    if (
        candidate.candidate_type in _OBJECTIVE_FOLLOWUP_TYPES
        and gap_ms <= OBJECTIVE_FOLLOWUP_GAP_MS
        and _contains_combat(bucket)
    ):
        return True

    return False


def _find_marker_bucket(
    *,
    buckets: list[_SituationBucket],
    marker: CommentaryCandidate,
) -> _SituationBucket | None:
    if marker.actor_participant_id is None:
        return None

    best_bucket: _SituationBucket | None = None

    best_distance: int | None = None

    for bucket in buckets:
        for candidate in bucket.primary_candidates:
            if not _is_champion_kill(candidate):
                continue

            if candidate.actor_participant_id != marker.actor_participant_id:
                continue

            distance = abs(candidate.timestamp_ms - marker.timestamp_ms)

            if distance > SPECIAL_MARKER_TOLERANCE_MS:
                continue

            if best_distance is None or distance < best_distance:
                best_bucket = bucket
                best_distance = distance

    return best_bucket


def _situation_kind(
    bucket: _SituationBucket,
) -> SituationKind:
    if not bucket.primary_candidates:
        return SituationKind.SPECIAL_ONLY

    candidate_types = {candidate.candidate_type for candidate in bucket.primary_candidates}

    if candidate_types == {CommentaryCandidateType.CHAMPION_KILL}:
        return SituationKind.COMBAT

    if candidate_types <= {
        CommentaryCandidateType.ELITE_MONSTER,
        CommentaryCandidateType.DRAGON_SOUL,
    }:
        return SituationKind.OBJECTIVE

    if candidate_types == {CommentaryCandidateType.BUILDING}:
        return SituationKind.PUSH

    if candidate_types == {CommentaryCandidateType.GAME_END}:
        return SituationKind.TERMINAL

    return SituationKind.MIXED


def _situation_id(
    bucket: _SituationBucket,
) -> str:
    primary_ids = ",".join(candidate.candidate_id for candidate in bucket.primary_candidates)

    marker_ids = ",".join(candidate.candidate_id for candidate in bucket.semantic_markers)

    identity = f"{bucket.match_id}|{primary_ids}|{marker_ids}"

    return sha256(identity.encode("utf-8")).hexdigest()


def _candidate_sort_key(
    candidate: CommentaryCandidate,
) -> tuple[int, int, str]:
    return (
        candidate.timestamp_ms,
        candidate.source_sequence,
        candidate.candidate_id,
    )


def _situation_sort_key(
    situation: TemporalSituation,
) -> tuple[int, int, str]:
    return (
        situation.start_timestamp_ms,
        situation.end_timestamp_ms,
        situation.situation_id,
    )


def _freeze_bucket(
    bucket: _SituationBucket,
) -> TemporalSituation:
    all_candidates = tuple(bucket.primary_candidates) + tuple(bucket.semantic_markers)

    if not all_candidates:
        raise ValueError("Situation must contain at least one candidate")

    start_timestamp_ms = min(candidate.timestamp_ms for candidate in all_candidates)

    end_timestamp_ms = max(candidate.timestamp_ms for candidate in all_candidates)

    max_salience_score = max(candidate.salience_score for candidate in all_candidates)

    primary_candidates = tuple(
        sorted(
            bucket.primary_candidates,
            key=_candidate_sort_key,
        )
    )

    semantic_markers = tuple(
        sorted(
            bucket.semantic_markers,
            key=_candidate_sort_key,
        )
    )

    return TemporalSituation(
        situation_id=_situation_id(bucket),
        match_id=bucket.match_id,
        start_timestamp_ms=(start_timestamp_ms),
        end_timestamp_ms=(end_timestamp_ms),
        duration_ms=(end_timestamp_ms - start_timestamp_ms),
        situation_kind=(_situation_kind(bucket)),
        max_salience_score=(max_salience_score),
        primary_candidates=(primary_candidates),
        semantic_markers=(semantic_markers),
    )


def build_temporal_situations(
    *,
    match_id: str,
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ],
) -> tuple[
    TemporalSituation,
    ...,
]:
    clean_match_id = match_id.strip()

    if not clean_match_id:
        raise ValueError("match_id must not be empty")

    ordered = tuple(
        sorted(
            candidates,
            key=_candidate_sort_key,
        )
    )

    candidate_ids = [candidate.candidate_id for candidate in ordered]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Duplicate candidate_id")

    for candidate in ordered:
        if candidate.match_id != clean_match_id:
            raise ValueError("Candidate match_id does not match")

    primary_candidates = tuple(
        candidate for candidate in ordered if not _is_special_marker(candidate)
    )

    semantic_markers = tuple(candidate for candidate in ordered if _is_special_marker(candidate))

    buckets: list[_SituationBucket] = []

    for candidate in primary_candidates:
        if buckets and _should_join_primary(
            bucket=buckets[-1],
            candidate=candidate,
        ):
            buckets[-1].primary_candidates.append(candidate)

        else:
            buckets.append(
                _SituationBucket(
                    match_id=(clean_match_id),
                    primary_candidates=[candidate],
                )
            )

    for marker in semantic_markers:
        bucket = _find_marker_bucket(
            buckets=buckets,
            marker=marker,
        )

        if bucket is not None:
            bucket.semantic_markers.append(marker)

        else:
            buckets.append(
                _SituationBucket(
                    match_id=(clean_match_id),
                    semantic_markers=[marker],
                )
            )

    situations = tuple(
        sorted(
            (_freeze_bucket(bucket) for bucket in buckets),
            key=_situation_sort_key,
        )
    )

    assigned_candidate_ids = [
        candidate.candidate_id
        for situation in situations
        for candidate in (situation.primary_candidates + situation.semantic_markers)
    ]

    if len(assigned_candidate_ids) != len(candidate_ids):
        raise RuntimeError("Candidate assignment count mismatch")

    if set(assigned_candidate_ids) != set(candidate_ids):
        raise RuntimeError("Candidate assignment coverage mismatch")

    if len(assigned_candidate_ids) != len(set(assigned_candidate_ids)):
        raise RuntimeError("Candidate assigned to multiple situations")

    return situations
