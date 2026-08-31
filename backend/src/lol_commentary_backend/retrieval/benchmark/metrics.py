from collections import defaultdict
from collections.abc import Callable, Sequence

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalMetricSummary,
    RetrievalQueryResult,
    RetrievedChunk,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)


def evaluate_ranking(
    case: RetrievalEvalCase,
    ranked_indices: Sequence[int],
    scores: Sequence[float],
    chunks: Sequence[PatchRagChunk],
    *,
    top_results_limit: int = 10,
) -> RetrievalQueryResult:
    if len(ranked_indices) != len(scores):
        raise ValueError("ranked_indices and scores must have the same length")

    relevant = set(case.relevant_chunk_ids)

    first_relevant_rank: int | None = None

    for rank, chunk_index in enumerate(
        ranked_indices,
        start=1,
    ):
        if chunks[chunk_index].chunk_id in relevant:
            first_relevant_rank = rank
            break

    top_1_ids = {chunks[index].chunk_id for index in ranked_indices[:1]}
    top_3_ids = {chunks[index].chunk_id for index in ranked_indices[:3]}
    top_5_ids = {chunks[index].chunk_id for index in ranked_indices[:5]}

    hit_at_1 = bool(relevant & top_1_ids)
    hit_at_3 = bool(relevant & top_3_ids)
    hit_at_5 = bool(relevant & top_5_ids)

    recall_at_5 = len(relevant & top_5_ids) / len(relevant)

    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    top_results: list[RetrievedChunk] = []

    for rank, (
        chunk_index,
        score,
    ) in enumerate(
        zip(
            ranked_indices[:top_results_limit],
            scores[:top_results_limit],
            strict=True,
        ),
        start=1,
    ):
        chunk = chunks[chunk_index]

        top_results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                entity_name=chunk.entity_name,
                score=float(score),
                relevant=(chunk.chunk_id in relevant),
            )
        )

    return RetrievalQueryResult(
        query_id=case.query_id,
        query=case.query,
        language=case.language.value,
        query_type=case.query_type.value,
        difficulty=case.difficulty.value,
        relevant_chunk_ids=list(case.relevant_chunk_ids),
        first_relevant_rank=first_relevant_rank,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        recall_at_5=recall_at_5,
        reciprocal_rank=reciprocal_rank,
        top_results=top_results,
    )


def summarize_results(
    results: Sequence[RetrievalQueryResult],
) -> RetrievalMetricSummary:
    count = len(results)

    if count == 0:
        return RetrievalMetricSummary(
            cases=0,
            hit_at_1=0.0,
            hit_at_3=0.0,
            hit_at_5=0.0,
            recall_at_5=0.0,
            mrr=0.0,
        )

    return RetrievalMetricSummary(
        cases=count,
        hit_at_1=(sum(result.hit_at_1 for result in results) / count),
        hit_at_3=(sum(result.hit_at_3 for result in results) / count),
        hit_at_5=(sum(result.hit_at_5 for result in results) / count),
        recall_at_5=(sum(result.recall_at_5 for result in results) / count),
        mrr=(sum(result.reciprocal_rank for result in results) / count),
    )


def summarize_by(
    results: Sequence[RetrievalQueryResult],
    key: Callable[
        [RetrievalQueryResult],
        str,
    ],
) -> dict[str, RetrievalMetricSummary]:
    grouped: dict[
        str,
        list[RetrievalQueryResult],
    ] = defaultdict(list)

    for result in results:
        grouped[key(result)].append(result)

    return {
        group_key: summarize_results(group_results)
        for group_key, group_results in sorted(grouped.items())
    }
