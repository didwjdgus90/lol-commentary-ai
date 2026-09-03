import pytest

from lol_commentary_backend.retrieval.runtime.multi_pgvector_dense import (
    normalize_corpus_sha256s,
)


def test_accepts_multiple_corpus_hashes() -> None:
    first = "a" * 64
    second = "b" * 64

    assert normalize_corpus_sha256s(
        (
            first,
            second,
        )
    ) == (
        first,
        second,
    )


def test_removes_duplicate_corpus_hashes() -> None:
    first = "a" * 64
    second = "b" * 64

    assert normalize_corpus_sha256s(
        (
            first,
            second,
            first,
        )
    ) == (
        first,
        second,
    )


def test_rejects_empty_corpus_scope() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        normalize_corpus_sha256s(())


def test_rejects_invalid_corpus_hash() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid corpus",
    ):
        normalize_corpus_sha256s(("not-a-sha256",))


def test_rejects_uppercase_sha256() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid corpus",
    ):
        normalize_corpus_sha256s(("A" * 64,))
