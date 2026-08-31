from collections.abc import Sequence
from time import perf_counter

from lol_commentary_backend.retrieval.benchmark.metrics import (
    evaluate_ranking,
    summarize_by,
    summarize_results,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalBenchmarkRun,
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.chunks.models import PatchRagChunk
from lol_commentary_backend.retrieval.evaluation.models import RetrievalEvalCase
from lol_commentary_backend.retrieval.hybrid.models import HybridRetrievalRun

HYBRID_EXPERIMENT_VERSION = "0.1.0"
DEFAULT_RRF_K = 60
DEFAULT_SOURCE_TOP_N = 10


def rrf_score(rank: int, *, rrf_k: int = DEFAULT_RRF_K) -> float:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    return 1.0 / (rrf_k + rank)


def _validate_source_runs(
    dense_run: RetrievalBenchmarkRun,
    sparse_run: RetrievalBenchmarkRun,
) -> None:
    if dense_run.engine_type != "dense":
        raise ValueError("dense_run must have engine_type=dense")
    if sparse_run.engine_type != "bm25":
        raise ValueError("sparse_run must have engine_type=bm25")
    if dense_run.corpus_sha256 != sparse_run.corpus_sha256:
        raise ValueError("Source runs used different corpora")
    if dense_run.evaluation_cases != sparse_run.evaluation_cases:
        raise ValueError("Source runs used different evaluation case counts")

    dense_ids = {result.query_id for result in dense_run.query_results}
    sparse_ids = {result.query_id for result in sparse_run.query_results}
    if dense_ids != sparse_ids:
        raise ValueError("Source runs contain different query IDs")


def _result_index(
    results: Sequence[RetrievalQueryResult],
) -> dict[str, RetrievalQueryResult]:
    return {result.query_id: result for result in results}


def _fuse_one_query(
    dense_result: RetrievalQueryResult,
    sparse_result: RetrievalQueryResult,
    case: RetrievalEvalCase,
    chunks: Sequence[PatchRagChunk],
    *,
    rrf_k: int,
    source_top_n: int,
) -> RetrievalQueryResult:
    if source_top_n <= 0:
        raise ValueError("source_top_n must be positive")

    chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}

    for source_result in (dense_result, sparse_result):
        for retrieved in source_result.top_results[:source_top_n]:
            scores[retrieved.chunk_id] = scores.get(retrieved.chunk_id, 0.0) + rrf_score(
                retrieved.rank, rrf_k=rrf_k
            )
            previous = best_rank.get(retrieved.chunk_id)
            if previous is None or retrieved.rank < previous:
                best_rank[retrieved.chunk_id] = retrieved.rank

    missing = set(scores) - set(chunk_index_by_id)
    if missing:
        raise ValueError("Benchmark result references a chunk missing from current corpus")

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_rank[chunk_id],
            chunk_id,
        ),
    )

    ranked_indices = [chunk_index_by_id[chunk_id] for chunk_id in ranked_chunk_ids]
    ranked_scores = [scores[chunk_id] for chunk_id in ranked_chunk_ids]

    return evaluate_ranking(
        case,
        ranked_indices,
        ranked_scores,
        chunks,
    )


def build_rrf_hybrid_run(
    dense_run: RetrievalBenchmarkRun,
    sparse_run: RetrievalBenchmarkRun,
    cases: Sequence[RetrievalEvalCase],
    chunks: Sequence[PatchRagChunk],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    source_top_n: int = DEFAULT_SOURCE_TOP_N,
) -> HybridRetrievalRun:
    _validate_source_runs(dense_run, sparse_run)

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if source_top_n <= 0:
        raise ValueError("source_top_n must be positive")

    dense_by_query = _result_index(dense_run.query_results)
    sparse_by_query = _result_index(sparse_run.query_results)
    case_by_query = {case.query_id: case for case in cases}

    if set(dense_by_query) != set(case_by_query):
        raise ValueError("Evaluation cases do not match benchmark query IDs")

    started = perf_counter()

    query_results = [
        _fuse_one_query(
            dense_by_query[query_id],
            sparse_by_query[query_id],
            case_by_query[query_id],
            chunks,
            rrf_k=rrf_k,
            source_top_n=source_top_n,
        )
        for query_id in sorted(dense_by_query)
    ]

    fusion_seconds = perf_counter() - started
    hybrid_key = f"rrf_{dense_run.engine_key}_{sparse_run.engine_key}"

    return HybridRetrievalRun(
        experiment_version=HYBRID_EXPERIMENT_VERSION,
        hybrid_key=hybrid_key,
        dense_engine=dense_run.engine_key,
        sparse_engine=sparse_run.engine_key,
        rrf_k=rrf_k,
        source_top_n=source_top_n,
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        corpus_sha256=dense_run.corpus_sha256,
        metrics=summarize_results(query_results),
        by_language=summarize_by(query_results, lambda result: result.language),
        by_query_type=summarize_by(query_results, lambda result: result.query_type),
        by_difficulty=summarize_by(query_results, lambda result: result.difficulty),
        fusion_total_seconds=fusion_seconds,
        fusion_query_mean_ms=(0.0 if not cases else fusion_seconds / len(cases) * 1000),
        query_results=query_results,
    )
