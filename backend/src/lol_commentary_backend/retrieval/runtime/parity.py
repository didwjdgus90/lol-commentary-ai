from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
)


@dataclass(frozen=True)
class RankingParity:
    query_id: str

    exact_order_match: bool
    candidate_set_match: bool

    max_score_delta: float

    left_chunk_ids: tuple[str, ...]
    right_chunk_ids: tuple[str, ...]


def compare_ranked_candidates(
    *,
    query_id: str,
    left: Sequence[RankedCandidate],
    right: Sequence[RankedCandidate],
) -> RankingParity:
    if not query_id.strip():
        raise ValueError("query_id must not be empty")

    if len(left) != len(right):
        raise ValueError("ranking lengths must match")

    left_ids = tuple(candidate.chunk_id for candidate in left)

    right_ids = tuple(candidate.chunk_id for candidate in right)

    score_deltas = [
        abs(left_candidate.score - right_candidate.score)
        for left_candidate, right_candidate in zip(
            left,
            right,
            strict=True,
        )
    ]

    return RankingParity(
        query_id=query_id,
        exact_order_match=(left_ids == right_ids),
        candidate_set_match=(set(left_ids) == set(right_ids)),
        max_score_delta=(
            max(
                score_deltas,
                default=0.0,
            )
        ),
        left_chunk_ids=left_ids,
        right_chunk_ids=right_ids,
    )
