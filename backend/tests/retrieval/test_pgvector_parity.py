import pytest

from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
)
from lol_commentary_backend.retrieval.runtime.parity import (
    compare_ranked_candidates,
)

CHUNK_A = "a" * 64
CHUNK_B = "b" * 64
CHUNK_C = "c" * 64


def _candidate(
    chunk_id: str,
    rank: int,
    score: float,
) -> RankedCandidate:
    return RankedCandidate(
        chunk_id=chunk_id,
        rank=rank,
        score=score,
    )


def test_exact_parity() -> None:
    left = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_B,
            2,
            0.8,
        ),
    ]

    right = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_B,
            2,
            0.8,
        ),
    ]

    result = compare_ranked_candidates(
        query_id="q001",
        left=left,
        right=right,
    )

    assert result.exact_order_match is True
    assert result.candidate_set_match is True
    assert result.max_score_delta == 0.0


def test_same_candidates_different_order() -> None:
    left = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_B,
            2,
            0.8,
        ),
    ]

    right = [
        _candidate(
            CHUNK_B,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_A,
            2,
            0.8,
        ),
    ]

    result = compare_ranked_candidates(
        query_id="q001",
        left=left,
        right=right,
    )

    assert result.exact_order_match is False
    assert result.candidate_set_match is True


def test_different_candidates_are_detected() -> None:
    left = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_B,
            2,
            0.8,
        ),
    ]

    right = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
        _candidate(
            CHUNK_C,
            2,
            0.8,
        ),
    ]

    result = compare_ranked_candidates(
        query_id="q001",
        left=left,
        right=right,
    )

    assert result.exact_order_match is False
    assert result.candidate_set_match is False


def test_score_delta_is_recorded() -> None:
    left = [
        _candidate(
            CHUNK_A,
            1,
            0.900000,
        ),
    ]

    right = [
        _candidate(
            CHUNK_A,
            1,
            0.899999,
        ),
    ]

    result = compare_ranked_candidates(
        query_id="q001",
        left=left,
        right=right,
    )

    assert result.exact_order_match is True

    assert result.max_score_delta == pytest.approx(0.000001)


def test_length_mismatch_is_rejected() -> None:
    left = [
        _candidate(
            CHUNK_A,
            1,
            0.9,
        ),
    ]

    right = []

    with pytest.raises(
        ValueError,
        match="lengths must match",
    ):
        compare_ranked_candidates(
            query_id="q001",
            left=left,
            right=right,
        )
