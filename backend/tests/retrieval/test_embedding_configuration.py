import pytest

from lol_commentary_backend.retrieval.embeddings.models import (
    BGE_M3,
    EMBEDDING_CANDIDATES,
    MULTILINGUAL_E5_LARGE_INSTRUCT,
    QWEN3_EMBEDDING_06B,
)
from lol_commentary_backend.retrieval.embeddings.query import (
    LOL_PATCH_RETRIEVAL_TASK,
    format_embedding_query,
)


def test_candidate_keys_are_unique() -> None:
    keys = [candidate.key for candidate in EMBEDDING_CANDIDATES]

    assert len(keys) == len(set(keys))


def test_all_three_candidates_are_enabled() -> None:
    assert len(EMBEDDING_CANDIDATES) == 3
    assert all(candidate.experiment_enabled for candidate in EMBEDDING_CANDIDATES)


def test_candidate_limits_are_consistent() -> None:
    for candidate in EMBEDDING_CANDIDATES:
        assert candidate.experiment_max_tokens <= candidate.declared_max_tokens


def test_all_candidates_use_1024_dimensions() -> None:
    assert {candidate.embedding_dimension for candidate in EMBEDDING_CANDIDATES} == {1024}


def test_qwen_uses_instruction() -> None:
    formatted = format_embedding_query(
        "트린다미어 E 변경",
        QWEN3_EMBEDDING_06B,
    )

    assert formatted.startswith(f"Instruct: {LOL_PATCH_RETRIEVAL_TASK}\nQuery:")


def test_e5_uses_instruction() -> None:
    formatted = format_embedding_query(
        "Aphelios passive hotfix",
        MULTILINGUAL_E5_LARGE_INSTRUCT,
    )

    assert "\nQuery: Aphelios" in formatted


def test_bge_m3_uses_plain_query() -> None:
    raw = "정수 약탈자 가격 변경"

    assert format_embedding_query(raw, BGE_M3) == raw


def test_empty_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        format_embedding_query(
            "  ",
            QWEN3_EMBEDDING_06B,
        )
