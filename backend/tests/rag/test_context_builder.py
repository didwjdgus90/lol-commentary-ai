from hashlib import sha256

from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStrategy,
)


def _hash(
    value: str,
) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _hit(
    *,
    rank: int,
    character: str,
    text: str,
    score: float = 0.01,
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        chunk_id=_hash(f"chunk:{character}"),
        document_id=_hash(f"document:{character}"),
        source_record_id=_hash(f"record:{character}"),
        patch="26.1",
        locale="ko_kr",
        source_url=(f"https://example.com/{character}"),
        section_kind="hotfix",
        title=f"title-{character}",
        entity_name=None,
        heading_path=[
            "추가 패치 노트",
            f"title-{character}",
        ],
        text=text,
        score=score,
        dense_rank=rank,
        sparse_rank=None,
        dense_score=score,
        sparse_score=None,
    )


def _response(
    hits: list[RetrievalHit],
) -> RetrievalResponse:
    return RetrievalResponse(
        query=("Essence Reaver AD nerf"),
        expanded_query=("Essence Reaver AD nerf | 정수 약탈자"),
        requested_mode=(RetrievalMode.PRIMARY),
        strategy_used=(RetrievalStrategy.BGE_ALIAS_RRF),
        fallback_used=False,
        fallback_reason=None,
        top_k=max(
            len(hits),
            1,
        ),
        source_top_n=10,
        elapsed_ms=12.5,
        hits=hits,
    )


def test_preserves_retrieval_order() -> None:
    response = _response(
        [
            _hit(
                rank=1,
                character="a",
                text="first",
            ),
            _hit(
                rank=2,
                character="d",
                text="second",
            ),
        ]
    )

    bundle = build_context_bundle(response)

    assert [item.citation_id for item in bundle.evidence] == [
        "E1",
        "E2",
    ]

    assert [item.text for item in bundle.evidence] == [
        "first",
        "second",
    ]


def test_duplicate_chunk_id_is_removed() -> None:
    first = _hit(
        rank=1,
        character="a",
        text="same chunk",
    )

    duplicate = first.model_copy(
        update={
            "rank": 2,
        }
    )

    bundle = build_context_bundle(
        _response(
            [
                first,
                duplicate,
            ]
        )
    )

    assert bundle.selected_count == 1

    assert bundle.dropped_duplicate_count == 1


def test_duplicate_normalized_text_is_removed() -> None:
    response = _response(
        [
            _hit(
                rank=1,
                character="a",
                text=("공격력 55 ⇒ 50"),
            ),
            _hit(
                rank=2,
                character="d",
                text=("  공격력   55 ⇒ 50  "),
            ),
        ]
    )

    bundle = build_context_bundle(response)

    assert bundle.selected_count == 1

    assert bundle.dropped_duplicate_count == 1


def test_max_items_limits_context() -> None:
    response = _response(
        [
            _hit(
                rank=1,
                character="a",
                text="one",
            ),
            _hit(
                rank=2,
                character="d",
                text="two",
            ),
            _hit(
                rank=3,
                character="g",
                text="three",
            ),
        ]
    )

    bundle = build_context_bundle(
        response,
        max_items=2,
    )

    assert bundle.selected_count == 2
    assert bundle.dropped_limit_count == 1


def test_zero_score_is_not_filtered() -> None:
    response = _response(
        [
            _hit(
                rank=1,
                character="a",
                text="evidence",
                score=0.0,
            )
        ]
    )

    bundle = build_context_bundle(response)

    assert bundle.selected_count == 1

    assert bundle.evidence[0].retrieval_score == 0.0


def test_invalid_max_items_is_rejected() -> None:
    response = _response(
        [
            _hit(
                rank=1,
                character="a",
                text="evidence",
            )
        ]
    )

    try:
        build_context_bundle(
            response,
            max_items=0,
        )
    except ValueError as exc:
        assert "max_items must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
