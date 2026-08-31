from hashlib import sha256

import pytest

from lol_commentary_backend.retrieval.baseline.decision import (
    build_retrieval_baseline_decision,
)
from lol_commentary_backend.retrieval.baseline.verify import (
    build_reproducibility_check,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    BenchmarkMemory,
    BenchmarkTiming,
    RetrievalBenchmarkRun,
    RetrievalMetricSummary,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RerankerBenchmarkRun,
    RerankerMemory,
    RerankerTiming,
)

HASH = sha256(b"corpus").hexdigest()


def _summary(
    hit_at_1: float,
    *,
    hit_at_3: float = 0.95,
    recall_at_5: float = 0.95,
    mrr: float = 0.88,
) -> RetrievalMetricSummary:
    return RetrievalMetricSummary(
        cases=24,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=recall_at_5,
        recall_at_5=recall_at_5,
        mrr=mrr,
    )


def _bm25(
    hit_at_1: float = 0.79,
) -> RetrievalBenchmarkRun:
    summary = _summary(hit_at_1)

    return RetrievalBenchmarkRun(
        benchmark_version="0.1.0",
        engine_key="alias_bm25",
        engine_type="bm25",
        model_id=None,
        device="cpu",
        corpus_chunks=185,
        evaluation_cases=24,
        corpus_sha256=HASH,
        embedding_dimension=None,
        metrics=summary,
        by_language={},
        by_query_type={},
        by_difficulty={},
        timing=BenchmarkTiming(
            model_load_seconds=0,
            corpus_index_seconds=0,
            query_total_seconds=0,
            query_mean_ms=1,
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=1,
            process_rss_after_load_mb=1,
            process_rss_after_index_mb=1,
            cuda_peak_allocated_mb=None,
        ),
        query_results=[],
    )


def _hybrid(
    hit_at_1: float = 0.83,
    *,
    mrr: float = 0.90,
) -> HybridRetrievalRun:
    summary = _summary(
        hit_at_1,
        mrr=mrr,
    )

    return HybridRetrievalRun(
        experiment_version="0.1.0",
        hybrid_key=("rrf_bge_m3_alias_bm25"),
        dense_engine="bge_m3",
        sparse_engine="alias_bm25",
        rrf_k=60,
        source_top_n=10,
        corpus_chunks=185,
        evaluation_cases=24,
        corpus_sha256=HASH,
        metrics=summary,
        by_language={},
        by_query_type={},
        by_difficulty={},
        fusion_total_seconds=0.001,
        fusion_query_mean_ms=0.01,
        query_results=[],
    )


def _reranker(
    key: str,
    hit_at_1: float,
) -> RerankerBenchmarkRun:
    summary = _summary(hit_at_1)

    return RerankerBenchmarkRun(
        benchmark_version="0.1.0",
        reranker_key=key,
        model_id=f"test/{key}",
        device="cpu",
        candidate_source="alias_bm25",
        candidate_top_n=10,
        corpus_chunks=185,
        evaluation_cases=24,
        pairs_scored=240,
        corpus_sha256=HASH,
        metrics=summary,
        by_language={},
        by_query_type={},
        by_difficulty={},
        timing=RerankerTiming(
            model_load_seconds=1,
            inference_total_seconds=1,
            query_mean_ms=1,
            pair_mean_ms=0.1,
        ),
        memory=RerankerMemory(
            rss_before_mb=1,
            rss_after_load_mb=2,
            rss_after_inference_mb=2,
        ),
        query_results=[],
    )


def test_primary_is_bge_alias_hybrid() -> None:
    decision = build_retrieval_baseline_decision(
        alias_bm25=_bm25(),
        bge_alias_hybrid=_hybrid(),
        reranker_runs=[
            _reranker("qwen", 0.79),
        ],
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert decision.primary.key == "rrf_bge_m3_alias_bm25"


def test_fast_fallback_is_alias_bm25() -> None:
    decision = build_retrieval_baseline_decision(
        alias_bm25=_bm25(),
        bge_alias_hybrid=_hybrid(),
        reranker_runs=[],
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert decision.fast_fallback.key == "alias_bm25"


def test_decision_rejects_stronger_reranker() -> None:
    with pytest.raises(
        ValueError,
        match="reranker exceeds",
    ):
        build_retrieval_baseline_decision(
            alias_bm25=_bm25(),
            bge_alias_hybrid=_hybrid(hit_at_1=0.83),
            reranker_runs=[
                _reranker(
                    "better",
                    0.90,
                )
            ],
            sentence_transformers_version="5.4.0",
            transformers_version="4.57.6",
        )


def test_decision_preserves_dependency_versions() -> None:
    decision = build_retrieval_baseline_decision(
        alias_bm25=_bm25(),
        bge_alias_hybrid=_hybrid(),
        reranker_runs=[],
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert decision.sentence_transformers_version == "5.4.0"
    assert decision.transformers_version == "4.57.6"


def test_reproducibility_matches_equal_metrics() -> None:
    check = build_reproducibility_check(
        reference=_hybrid(),
        reproduced=_hybrid(),
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert check.metrics_match is True


def test_reproducibility_detects_changed_metric() -> None:
    check = build_reproducibility_check(
        reference=_hybrid(hit_at_1=0.83),
        reproduced=_hybrid(hit_at_1=0.79),
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert check.metrics_match is False


def test_reproducibility_rejects_corpus_mismatch() -> None:
    reproduced = _hybrid()
    reproduced.corpus_sha256 = sha256(b"other").hexdigest()

    with pytest.raises(
        ValueError,
        match="different corpora",
    ):
        build_reproducibility_check(
            reference=_hybrid(),
            reproduced=reproduced,
            sentence_transformers_version="5.4.0",
            transformers_version="4.57.6",
        )


def test_primary_does_not_require_reranker() -> None:
    decision = build_retrieval_baseline_decision(
        alias_bm25=_bm25(),
        bge_alias_hybrid=_hybrid(),
        reranker_runs=[],
        sentence_transformers_version="5.4.0",
        transformers_version="4.57.6",
    )

    assert decision.primary.reranker_required is False
