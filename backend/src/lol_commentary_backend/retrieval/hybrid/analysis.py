from collections.abc import Sequence

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalBenchmarkRun,
    RetrievalQueryResult,
)


def _by_query(
    results: Sequence[RetrievalQueryResult],
) -> dict[str, RetrievalQueryResult]:
    return {result.query_id: result for result in results}


def compare_engine_failures(
    dense_run: RetrievalBenchmarkRun,
    sparse_run: RetrievalBenchmarkRun,
) -> dict[str, object]:
    dense = _by_query(dense_run.query_results)
    sparse = _by_query(sparse_run.query_results)

    if set(dense) != set(sparse):
        raise ValueError("Runs contain different query IDs")

    query_ids = sorted(dense)

    return {
        "dense_engine": dense_run.engine_key,
        "sparse_engine": sparse_run.engine_key,
        "cases": len(query_ids),
        "dense_only_hit_at_1": [
            q for q in query_ids if dense[q].hit_at_1 and not sparse[q].hit_at_1
        ],
        "sparse_only_hit_at_1": [
            q for q in query_ids if sparse[q].hit_at_1 and not dense[q].hit_at_1
        ],
        "both_hit_at_1": [q for q in query_ids if dense[q].hit_at_1 and sparse[q].hit_at_1],
        "both_miss_at_1": [
            q for q in query_ids if not dense[q].hit_at_1 and not sparse[q].hit_at_1
        ],
        "both_miss_at_5": [
            q for q in query_ids if not dense[q].hit_at_5 and not sparse[q].hit_at_5
        ],
        "rank_rows": [
            {
                "query_id": q,
                "query": dense[q].query,
                "language": dense[q].language,
                "query_type": dense[q].query_type,
                "difficulty": dense[q].difficulty,
                "dense_first_relevant_rank": dense[q].first_relevant_rank,
                "sparse_first_relevant_rank": sparse[q].first_relevant_rank,
            }
            for q in query_ids
        ],
    }
