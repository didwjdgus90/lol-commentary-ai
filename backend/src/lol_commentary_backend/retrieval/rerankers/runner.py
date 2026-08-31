from collections.abc import Sequence
from time import perf_counter
from typing import Any

import numpy as np
import psutil

from lol_commentary_backend.retrieval.benchmark.metrics import (
    summarize_by,
    summarize_results,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)
from lol_commentary_backend.retrieval.rerankers.candidate import (
    analyze_candidate_coverage,
    require_full_candidate_coverage,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RerankerBenchmarkRun,
    RerankerCandidate,
    RerankerMemory,
    RerankerTiming,
)
from lol_commentary_backend.retrieval.rerankers.ranking import (
    rerank_query_from_scores,
)

RERANKER_BENCHMARK_VERSION = "0.1.0"


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _resolve_device(
    requested_device: str,
) -> str:
    import torch

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def run_reranker(
    candidate: RerankerCandidate,
    *,
    source_key: str,
    source_results: Sequence[RetrievalQueryResult],
    cases: Sequence[RetrievalEvalCase],
    chunks: Sequence[PatchRagChunk],
    corpus_sha256: str,
    top_n: int,
    requested_device: str,
    batch_size: int,
) -> RerankerBenchmarkRun:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    coverage = analyze_candidate_coverage(
        source_key=source_key,
        results=source_results,
        cases=cases,
        top_n=top_n,
    )
    require_full_candidate_coverage(coverage)

    result_by_query = {result.query_id: result for result in source_results}

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    pairs: list[tuple[str, str]] = []
    pair_slices: dict[
        str,
        tuple[int, int],
    ] = {}

    for case in cases:
        source_result = result_by_query[case.query_id]
        candidates = source_result.top_results[:top_n]

        start = len(pairs)

        for retrieved in candidates:
            chunk = chunk_by_id.get(retrieved.chunk_id)

            if chunk is None:
                raise ValueError("Candidate chunk is missing from current corpus")

            pairs.append(
                (
                    case.query,
                    chunk.text,
                )
            )

        pair_slices[case.query_id] = (
            start,
            len(pairs),
        )

    device = _resolve_device(requested_device)

    rss_before = _rss_mb()

    from sentence_transformers import (
        CrossEncoder,
    )

    load_started = perf_counter()

    model = CrossEncoder(
        candidate.model_id,
        device=device,
        max_length=candidate.max_length,
        trust_remote_code=(candidate.trust_remote_code),
    )

    load_seconds = perf_counter() - load_started

    rss_after_load = _rss_mb()

    inference_started = perf_counter()

    # sentence-transformers 5.4.0 exposes an invariant
    # PairInput overload that ty cannot narrow from
    # list[tuple[str, str]], although this is the documented
    # runtime input shape. Keep the runtime value unchanged
    # and isolate the third-party typing mismatch here.
    prediction_inputs: Any = pairs

    raw_scores = model.predict(
        prediction_inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    inference_seconds = perf_counter() - inference_started

    rss_after_inference = _rss_mb()

    flat_scores = np.asarray(
        raw_scores,
        dtype=np.float64,
    ).reshape(-1)

    if len(flat_scores) != len(pairs):
        raise ValueError("Reranker returned unexpected number of scores")

    query_results = []

    for case in cases:
        start, end = pair_slices[case.query_id]

        query_results.append(
            rerank_query_from_scores(
                case=case,
                source_result=result_by_query[case.query_id],
                scores=flat_scores[start:end].tolist(),
                chunks=chunks,
                top_n=top_n,
            )
        )

    metrics = summarize_results(query_results)

    return RerankerBenchmarkRun(
        benchmark_version=(RERANKER_BENCHMARK_VERSION),
        reranker_key=candidate.key,
        model_id=candidate.model_id,
        device=device,
        candidate_source=source_key,
        candidate_top_n=top_n,
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        pairs_scored=len(pairs),
        corpus_sha256=corpus_sha256,
        metrics=metrics,
        by_language=summarize_by(
            query_results,
            lambda result: result.language,
        ),
        by_query_type=summarize_by(
            query_results,
            lambda result: result.query_type,
        ),
        by_difficulty=summarize_by(
            query_results,
            lambda result: result.difficulty,
        ),
        timing=RerankerTiming(
            model_load_seconds=load_seconds,
            inference_total_seconds=(inference_seconds),
            query_mean_ms=(0.0 if not cases else inference_seconds / len(cases) * 1000),
            pair_mean_ms=(0.0 if not pairs else inference_seconds / len(pairs) * 1000),
        ),
        memory=RerankerMemory(
            rss_before_mb=rss_before,
            rss_after_load_mb=rss_after_load,
            rss_after_inference_mb=(rss_after_inference),
        ),
        query_results=query_results,
    )
