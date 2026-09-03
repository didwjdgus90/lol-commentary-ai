from __future__ import annotations

from time import perf_counter
from typing import Protocol

from lol_commentary_backend.rag.query_planner.models import (
    RAGQueryPlan,
    RAGRetrievalQuery,
)
from lol_commentary_backend.rag.retrieval_execution.models import (
    RAGBatchEvidenceResult,
    RAGPlanEvidenceResult,
    RAGQueryEvidenceResult,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalResponse,
)


class RetrievalServiceProtocol(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = (RetrievalMode.AUTO),
    ) -> RetrievalResponse: ...


def _validate_response(
    *,
    plan: RAGQueryPlan,
    query: RAGRetrievalQuery,
    response: RetrievalResponse,
    top_k: int,
    mode: RetrievalMode,
) -> None:
    if response.query != query.query_text:
        raise ValueError(
            f"Retrieval response query does not match RAG query: query_id={query.query_id}"
        )

    if response.requested_mode != mode:
        raise ValueError("Retrieval response mode does not match requested mode")

    if response.top_k != top_k:
        raise ValueError("Retrieval response top_k does not match requested top_k")

    for hit in response.hits:
        if hit.patch != plan.patch:
            raise ValueError(
                "Retrieved evidence escaped "
                "the RAG plan patch scope: "
                f"plan_patch={plan.patch} "
                f"hit_patch={hit.patch} "
                f"chunk_id={hit.chunk_id}"
            )


def _query_result(
    *,
    plan: RAGQueryPlan,
    query: RAGRetrievalQuery,
    response: RetrievalResponse,
    cache_reused: bool,
) -> RAGQueryEvidenceResult:
    return RAGQueryEvidenceResult(
        query_id=query.query_id,
        plan_id=plan.plan_id,
        record_id=plan.record_id,
        match_id=plan.match_id,
        situation_id=(plan.situation_id),
        patch=plan.patch,
        intent=query.intent,
        query_text=query.query_text,
        subject_key=(query.subject_key),
        subject_id=query.subject_id,
        subject_name=(query.subject_name),
        cache_reused=cache_reused,
        expanded_query=(response.expanded_query),
        requested_mode=(response.requested_mode),
        strategy_used=(response.strategy_used),
        fallback_used=(response.fallback_used),
        fallback_reason=(response.fallback_reason),
        source_top_n=(response.source_top_n),
        source_retrieval_elapsed_ms=(response.elapsed_ms),
        hits=tuple(hit.model_copy(deep=True) for hit in response.hits),
    )


def execute_rag_query_plans(
    *,
    plans: tuple[
        RAGQueryPlan,
        ...,
    ],
    retrieval_service: (RetrievalServiceProtocol),
    top_k: int = 5,
    mode: RetrievalMode = (RetrievalMode.AUTO),
) -> RAGBatchEvidenceResult:
    if not plans:
        raise ValueError("plans must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    patches = {plan.patch for plan in plans}

    if len(patches) != 1:
        raise ValueError("One retrieval execution batch must contain exactly one patch")

    plan_ids = [plan.plan_id for plan in plans]

    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("Duplicate RAG plan ID")

    record_ids = [plan.record_id for plan in plans]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate RAG plan record ID")

    started = perf_counter()

    cache: dict[
        str,
        RetrievalResponse,
    ] = {}

    plan_results: list[RAGPlanEvidenceResult] = []

    query_reference_count = 0

    unique_query_execution_count = 0

    cache_hit_count = 0

    fallback_execution_count = 0

    zero_hit_execution_count = 0

    total_hit_references = 0

    for plan in plans:
        query_results: list[RAGQueryEvidenceResult] = []

        for query in plan.queries:
            query_reference_count += 1

            cached_response = cache.get(query.query_text)

            cache_reused = cached_response is not None

            if cached_response is None:
                response = retrieval_service.retrieve(
                    query.query_text,
                    top_k=top_k,
                    mode=mode,
                )

                _validate_response(
                    plan=plan,
                    query=query,
                    response=response,
                    top_k=top_k,
                    mode=mode,
                )

                cache[query.query_text] = response.model_copy(deep=True)

                unique_query_execution_count += 1

                if response.fallback_used:
                    fallback_execution_count += 1

                if not response.hits:
                    zero_hit_execution_count += 1

            else:
                cache_hit_count += 1

                response = cached_response.model_copy(deep=True)

                _validate_response(
                    plan=plan,
                    query=query,
                    response=response,
                    top_k=top_k,
                    mode=mode,
                )

            total_hit_references += len(response.hits)

            query_results.append(
                _query_result(
                    plan=plan,
                    query=query,
                    response=response,
                    cache_reused=(cache_reused),
                )
            )

        plan_results.append(
            RAGPlanEvidenceResult(
                plan_id=(plan.plan_id),
                record_id=(plan.record_id),
                match_id=(plan.match_id),
                situation_id=(plan.situation_id),
                patch=plan.patch,
                priority_tier=(plan.priority_tier),
                situation_kind=(plan.situation_kind),
                query_results=tuple(query_results),
            )
        )

    if unique_query_execution_count + cache_hit_count != query_reference_count:
        raise RuntimeError("Retrieval execution accounting mismatch")

    patch = next(iter(patches))

    return RAGBatchEvidenceResult(
        patch=patch,
        requested_mode=mode,
        top_k=top_k,
        plan_count=len(plans),
        query_reference_count=(query_reference_count),
        unique_query_execution_count=(unique_query_execution_count),
        cache_hit_count=(cache_hit_count),
        fallback_execution_count=(fallback_execution_count),
        zero_hit_execution_count=(zero_hit_execution_count),
        total_hit_references=(total_hit_references),
        wall_elapsed_ms=(perf_counter() - started) * 1000,
        plans=tuple(plan_results),
    )
