from hashlib import sha256

from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)
from lol_commentary_backend.rag.context.selector import (
    SELECTION_POLICY,
    select_context_bundle,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalStrategy,
)


def _hash(
    value: str,
) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    *,
    citation_id: str,
    source_rank: int,
    title: str,
    entity_name: str | None,
    text: str,
    score: float = 0.01,
) -> ContextEvidence:
    return ContextEvidence(
        citation_id=citation_id,
        source_rank=source_rank,
        chunk_id=_hash(f"chunk:{citation_id}"),
        document_id=_hash(f"document:{citation_id}"),
        source_record_id=_hash(f"record:{citation_id}"),
        patch="26.1",
        locale="ko_kr",
        source_url=(f"https://example.com/{citation_id}"),
        section_kind="item",
        title=title,
        entity_name=entity_name,
        heading_path=(
            "패치 노트",
            title,
        ),
        text=text,
        retrieval_score=score,
        dense_rank=None,
        sparse_rank=source_rank,
    )


def _bundle(
    evidence: tuple[
        ContextEvidence,
        ...,
    ],
    *,
    query: str = ("Essence Reaver AD nerf hotfix"),
    expanded_query: str = ("Essence Reaver AD nerf hotfix | 정수 약탈자"),
) -> ContextBundle:
    return ContextBundle(
        query=query,
        expanded_query=expanded_query,
        requested_mode=(RetrievalMode.PRIMARY),
        strategy_used=(RetrievalStrategy.BGE_ALIAS_RRF),
        fallback_used=False,
        fallback_reason=None,
        retrieval_elapsed_ms=10.0,
        retrieval_top_k=max(
            len(evidence),
            1,
        ),
        source_top_n=10,
        selected_count=len(evidence),
        dropped_duplicate_count=0,
        dropped_limit_count=0,
        evidence=evidence,
    )


def test_selects_direct_entity_and_title_matches() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="정수 약탈자",
                entity_name=None,
                text="hotfix",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                title=("버그 수정 및 편의성 개선"),
                entity_name=None,
                text="rengar bugs",
            ),
            _evidence(
                citation_id="E3",
                source_rank=4,
                title="정수 약탈자",
                entity_name="정수 약탈자",
                text="item update",
            ),
        )
    )

    selected = select_context_bundle(bundle)

    assert selected.selected_count == 2

    assert [item.title for item in selected.evidence] == [
        "정수 약탈자",
        "정수 약탈자",
    ]

    assert selected.dropped_relevance_count == 1


def test_renumbers_citations_after_selection() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="정수 약탈자",
                entity_name=None,
                text="first",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                title="버그 수정",
                entity_name=None,
                text="noise",
            ),
            _evidence(
                citation_id="E3",
                source_rank=4,
                title="정수 약탈자",
                entity_name="정수 약탈자",
                text="second",
            ),
        )
    )

    selected = select_context_bundle(bundle)

    assert [item.citation_id for item in selected.evidence] == [
        "E1",
        "E2",
    ]

    assert [item.source_rank for item in selected.evidence] == [
        1,
        4,
    ]


def test_falls_back_when_no_direct_match_exists() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="챔피언 변경",
                entity_name=None,
                text="one",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                title="아이템 변경",
                entity_name=None,
                text="two",
            ),
        ),
        query="26.1 패치 요약",
        expanded_query="26.1 패치 요약",
    )

    selected = select_context_bundle(bundle)

    assert selected.selected_count == 2
    assert selected.selection_fallback_used is True

    assert selected.dropped_relevance_count == 0


def test_preserves_original_retrieval_order() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="정수 약탈자",
                entity_name=None,
                text="first",
            ),
            _evidence(
                citation_id="E2",
                source_rank=4,
                title="정수 약탈자",
                entity_name="정수 약탈자",
                text="second",
            ),
        )
    )

    selected = select_context_bundle(bundle)

    assert [item.source_rank for item in selected.evidence] == [
        1,
        4,
    ]


def test_respects_max_items() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="정수 약탈자",
                entity_name=None,
                text="one",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                title="정수 약탈자",
                entity_name="정수 약탈자",
                text="two",
            ),
        )
    )

    selected = select_context_bundle(
        bundle,
        max_items=1,
    )

    assert selected.selected_count == 1

    assert selected.dropped_limit_count == 1


def test_zero_score_direct_match_is_preserved() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                title="정수 약탈자",
                entity_name="정수 약탈자",
                text="evidence",
                score=0.0,
            ),
        )
    )

    selected = select_context_bundle(bundle)

    assert selected.selected_count == 1

    assert selected.evidence[0].retrieval_score == 0.0

    assert selected.selection_policy == SELECTION_POLICY
