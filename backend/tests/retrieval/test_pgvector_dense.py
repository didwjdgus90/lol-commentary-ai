import pytest

from lol_commentary_backend.retrieval.runtime.pgvector_dense import (
    cosine_distance_to_score,
)


@pytest.mark.parametrize(
    (
        "distance",
        "expected_score",
    ),
    [
        (
            0.0,
            1.0,
        ),
        (
            0.1,
            0.9,
        ),
        (
            0.5,
            0.5,
        ),
        (
            1.0,
            0.0,
        ),
        (
            2.0,
            -1.0,
        ),
    ],
)
def test_cosine_distance_to_score(
    distance: float,
    expected_score: float,
) -> None:
    assert cosine_distance_to_score(distance) == pytest.approx(expected_score)
