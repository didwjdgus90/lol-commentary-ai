from __future__ import annotations

import pytest

from lol_commentary_backend.rag.query_planner.models import (
    RAGPriorityTier,
    RAGQueryIntent,
    RAGQueryPlan,
    RAGRetrievalQuery,
)
from lol_commentary_backend.rag.retrieval_execution.executor import (
    execute_rag_query_plans,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStrategy,
)


def _query(
    *,
    digit: int,
    text: str,
) -> RAGRetrievalQuery:
    return RAGRetrievalQuery(
        query_id=(f"{digit:064x}"),
        intent=(RAGQueryIntent.CHAMPION_PATCH),
        query_text=text,
        subject_key=(f"champion:{digit}"),
        subject_id=str(digit),
        subject_name=(f"Champion{digit}"),
    )


def _plan(
    *,
    digit: int,
    patch: str = "26.17",
    queries: tuple[
        RAGRetrievalQuery,
        ...,
    ] = (),
) -> RAGQueryPlan:
    return RAGQueryPlan(
        plan_id=(f"{digit:064x}"),
        record_id=(f"{digit + 100:064x}"),
        match_id="KR_TEST",
        situation_id=(f"{digit + 200:064x}"),
        patch=patch,
        ddragon_version="16.17.1",
        priority_tier=(RAGPriorityTier.HIGH),
        situation_kind="combat",
        queries=queries,
    )


def _hit(
    *,
    patch: str,
) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="1" * 64,
        document_id="2" * 64,
        source_record_id=("3" * 64),
        patch=patch,
        locale="ko_KR",
        source_url=("https://example.com/patch"),
        section_kind="champion",
        title="테스트",
        entity_name="테스트",
        heading_path=[
            "26.17",
            "테스트",
        ],
        text="테스트 패치 근거",
        score=0.01,
        dense_rank=1,
        sparse_rank=1,
        dense_score=0.5,
        sparse_score=3.0,
    )


class FakeRetrievalService:
    def __init__(
        self,
        *,
        fallback: bool = False,
        zero_hits: bool = False,
        response_query_override: (str | None) = None,
        hit_patch_override: (str | None) = None,
    ) -> None:
        self.fallback = fallback

        self.zero_hits = zero_hits

        self.response_query_override = response_query_override

        self.hit_patch_override = hit_patch_override

        self.calls: list[
            tuple[
                str,
                int,
                RetrievalMode,
            ]
        ] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = (RetrievalMode.AUTO),
    ) -> RetrievalResponse:
        self.calls.append(
            (
                query,
                top_k,
                mode,
            )
        )

        response_query = self.response_query_override or query

        hits = [] if self.zero_hits else [_hit(patch=(self.hit_patch_override or "26.17"))]

        strategy = (
            RetrievalStrategy.ALIAS_BM25 if self.fallback else RetrievalStrategy.BGE_ALIAS_RRF
        )

        return RetrievalResponse(
            query=response_query,
            expanded_query=(f"{response_query} | alias"),
            requested_mode=mode,
            strategy_used=strategy,
            fallback_used=(self.fallback),
            fallback_reason=("RuntimeError" if self.fallback else None),
            top_k=top_k,
            source_top_n=30,
            elapsed_ms=1.0,
            hits=hits,
        )


def test_executes_query_and_preserves_lineage() -> None:
    service = FakeRetrievalService()

    plan = _plan(
        digit=1,
        queries=(
            _query(
                digit=1,
                text=("26.17 Yone 변경 사항"),
            ),
        ),
    )

    result = execute_rag_query_plans(
        plans=(plan,),
        retrieval_service=(service),
        top_k=5,
        mode=(RetrievalMode.PRIMARY),
    )

    assert result.plan_count == 1

    assert result.query_reference_count == 1

    query_result = result.plans[0].query_results[0]

    assert query_result.plan_id == plan.plan_id

    assert query_result.query_id == plan.queries[0].query_id

    assert query_result.hits[0].patch == "26.17"


def test_reuses_exact_query_across_plans() -> None:
    text = "26.17 Yone 변경 사항"

    first = _plan(
        digit=1,
        queries=(
            _query(
                digit=1,
                text=text,
            ),
        ),
    )

    second = _plan(
        digit=2,
        queries=(
            _query(
                digit=2,
                text=text,
            ),
        ),
    )

    service = FakeRetrievalService()

    result = execute_rag_query_plans(
        plans=(
            first,
            second,
        ),
        retrieval_service=(service),
    )

    assert len(service.calls) == 1

    assert result.query_reference_count == 2

    assert result.unique_query_execution_count == 1

    assert result.cache_hit_count == 1

    assert result.plans[0].query_results[0].cache_reused is False

    assert result.plans[1].query_results[0].cache_reused is True


def test_zero_query_plan_is_preserved() -> None:
    service = FakeRetrievalService()

    plan = _plan(
        digit=1,
        queries=(),
    )

    result = execute_rag_query_plans(
        plans=(plan,),
        retrieval_service=(service),
    )

    assert service.calls == []

    assert result.query_reference_count == 0

    assert result.plans[0].query_results == ()


def test_mixed_patch_batch_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="one patch",
    ):
        execute_rag_query_plans(
            plans=(
                _plan(
                    digit=1,
                    patch="26.17",
                ),
                _plan(
                    digit=2,
                    patch="26.16",
                ),
            ),
            retrieval_service=(FakeRetrievalService()),
        )


def test_duplicate_plan_id_is_rejected() -> None:
    plan = _plan(
        digit=1,
    )

    duplicate = plan.model_copy(
        update={
            "record_id": ("f" * 64),
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate RAG plan ID",
    ):
        execute_rag_query_plans(
            plans=(
                plan,
                duplicate,
            ),
            retrieval_service=(FakeRetrievalService()),
        )


def test_response_query_mismatch_is_rejected() -> None:
    plan = _plan(
        digit=1,
        queries=(
            _query(
                digit=1,
                text="expected",
            ),
        ),
    )

    service = FakeRetrievalService(
        response_query_override=("different"),
    )

    with pytest.raises(
        ValueError,
        match=("response query"),
    ):
        execute_rag_query_plans(
            plans=(plan,),
            retrieval_service=(service),
        )


def test_hit_patch_escape_is_rejected() -> None:
    plan = _plan(
        digit=1,
        queries=(
            _query(
                digit=1,
                text="test",
            ),
        ),
    )

    service = FakeRetrievalService(
        hit_patch_override="26.16",
    )

    with pytest.raises(
        ValueError,
        match="patch scope",
    ):
        execute_rag_query_plans(
            plans=(plan,),
            retrieval_service=(service),
        )


def test_top_k_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        execute_rag_query_plans(
            plans=(
                _plan(
                    digit=1,
                ),
            ),
            retrieval_service=(FakeRetrievalService()),
            top_k=0,
        )


def test_fallback_and_zero_hit_are_counted() -> None:
    plan = _plan(
        digit=1,
        queries=(
            _query(
                digit=1,
                text="test",
            ),
        ),
    )

    result = execute_rag_query_plans(
        plans=(plan,),
        retrieval_service=(
            FakeRetrievalService(
                fallback=True,
                zero_hits=True,
            )
        ),
    )

    assert result.fallback_execution_count == 1

    assert result.zero_hit_execution_count == 1


def test_plan_and_query_order_are_preserved() -> None:
    first_query = _query(
        digit=1,
        text="first",
    )

    second_query = _query(
        digit=2,
        text="second",
    )

    first_plan = _plan(
        digit=1,
        queries=(
            first_query,
            second_query,
        ),
    )

    second_plan = _plan(
        digit=2,
        queries=(),
    )

    result = execute_rag_query_plans(
        plans=(
            first_plan,
            second_plan,
        ),
        retrieval_service=(FakeRetrievalService()),
    )

    assert result.plans[0].plan_id == first_plan.plan_id

    assert result.plans[1].plan_id == second_plan.plan_id

    assert [row.query_text for row in result.plans[0].query_results] == [
        "first",
        "second",
    ]
