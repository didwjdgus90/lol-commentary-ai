from copy import deepcopy
from hashlib import sha256
from typing import Literal

import pytest

from lol_commentary_backend.retrieval.benchmark.models import (
    BenchmarkMemory,
    BenchmarkTiming,
    RetrievalBenchmarkRun,
    RetrievalMetricSummary,
    RetrievalQueryResult,
    RetrievedChunk,
)
from lol_commentary_backend.retrieval.diagnostics.analyzer import (
    diagnose_fusion,
    diagnose_queries,
)
from lol_commentary_backend.retrieval.diagnostics.models import (
    FailureCategory,
    FusionEffect,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)

HASH = sha256(b"corpus").hexdigest()
CHUNK_ID = sha256(b"chunk").hexdigest()
DOC_ID = sha256(b"doc").hexdigest()


def _result(
    *,
    query_id: str = "q001",
    first_rank: int | None,
) -> RetrievalQueryResult:
    hit_1 = first_rank == 1
    hit_3 = first_rank is not None and first_rank <= 3
    hit_5 = first_rank is not None and first_rank <= 5

    return RetrievalQueryResult(
        query_id=query_id,
        query=f"query {query_id}",
        language="ko",
        query_type="item_change",
        difficulty="medium",
        relevant_chunk_ids=[CHUNK_ID],
        first_relevant_rank=first_rank,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        recall_at_5=(1.0 if hit_5 else 0.0),
        reciprocal_rank=(0.0 if first_rank is None else 1.0 / first_rank),
        top_results=[
            RetrievedChunk(
                rank=1,
                chunk_id=sha256(f"top:{query_id}".encode()).hexdigest(),
                document_id=DOC_ID,
                title="top result",
                entity_name=None,
                score=1.0,
                relevant=hit_1,
            )
        ],
    )


def _summary() -> RetrievalMetricSummary:
    return RetrievalMetricSummary(
        cases=1,
        hit_at_1=0,
        hit_at_3=0,
        hit_at_5=0,
        recall_at_5=0,
        mrr=0,
    )


def _run(
    engine: str,
    result: RetrievalQueryResult,
    *,
    engine_type: Literal["dense", "bm25"] = "dense",
) -> RetrievalBenchmarkRun:
    return RetrievalBenchmarkRun(
        benchmark_version="0.1.0",
        engine_key=engine,
        engine_type=engine_type,
        model_id=("model" if engine_type == "dense" else None),
        device="cpu",
        corpus_chunks=185,
        evaluation_cases=1,
        corpus_sha256=HASH,
        embedding_dimension=(1024 if engine_type == "dense" else None),
        metrics=_summary(),
        by_language={},
        by_query_type={},
        by_difficulty={},
        timing=BenchmarkTiming(
            model_load_seconds=0,
            corpus_index_seconds=0,
            query_total_seconds=0,
            query_mean_ms=0,
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=1,
            process_rss_after_load_mb=1,
            process_rss_after_index_mb=1,
            cuda_peak_allocated_mb=None,
        ),
        query_results=[result],
    )


def _hybrid(
    result: RetrievalQueryResult,
) -> HybridRetrievalRun:
    return HybridRetrievalRun(
        experiment_version="0.1.0",
        hybrid_key="rrf_dense_bm25",
        dense_engine="dense",
        sparse_engine="bm25",
        rrf_k=60,
        source_top_n=10,
        corpus_chunks=185,
        evaluation_cases=1,
        corpus_sha256=HASH,
        metrics=_summary(),
        by_language={},
        by_query_type={},
        by_difficulty={},
        fusion_total_seconds=0,
        fusion_query_mean_ms=0,
        query_results=[result],
    )


def test_any_top1_is_solved() -> None:
    diagnostics = diagnose_queries(
        [
            _run("a", _result(first_rank=1)),
            _run("b", _result(first_rank=8)),
        ]
    )

    assert diagnostics[0].category == FailureCategory.SOLVED_TOP1


def test_rank_two_to_ten_is_rerankable() -> None:
    diagnostics = diagnose_queries(
        [
            _run("a", _result(first_rank=4)),
            _run("b", _result(first_rank=7)),
        ]
    )

    assert diagnostics[0].category == FailureCategory.RERANKABLE_TOP10


def test_rank_above_ten_is_recall_failure() -> None:
    diagnostics = diagnose_queries(
        [
            _run("a", _result(first_rank=11)),
            _run("b", _result(first_rank=30)),
        ]
    )

    assert diagnostics[0].category == FailureCategory.CANDIDATE_RECALL_FAILURE


def test_missing_relevant_is_recall_failure() -> None:
    diagnostics = diagnose_queries(
        [
            _run("a", _result(first_rank=None)),
            _run("b", _result(first_rank=None)),
        ]
    )

    assert diagnostics[0].best_relevant_rank is None
    assert diagnostics[0].category == FailureCategory.CANDIDATE_RECALL_FAILURE


def test_best_engines_can_tie() -> None:
    diagnostics = diagnose_queries(
        [
            _run("a", _result(first_rank=2)),
            _run("b", _result(first_rank=2)),
        ]
    )

    assert diagnostics[0].best_engines == [
        "a",
        "b",
    ]


def test_new_fusion_top1_is_detected() -> None:
    diagnostics = diagnose_fusion(
        _hybrid(_result(first_rank=1)),
        _run(
            "dense",
            _result(first_rank=2),
        ),
        _run(
            "bm25",
            _result(first_rank=3),
            engine_type="bm25",
        ),
    )

    assert diagnostics[0].effect == FusionEffect.NEW_TOP1_SUCCESS


def test_lost_source_top1_is_detected() -> None:
    diagnostics = diagnose_fusion(
        _hybrid(_result(first_rank=2)),
        _run(
            "dense",
            _result(first_rank=1),
        ),
        _run(
            "bm25",
            _result(first_rank=4),
            engine_type="bm25",
        ),
    )

    assert diagnostics[0].effect == FusionEffect.LOST_SOURCE_TOP1


def test_mismatched_corpus_is_rejected() -> None:
    first = _run(
        "a",
        _result(first_rank=1),
    )
    second = deepcopy(
        _run(
            "b",
            _result(first_rank=2),
        )
    )
    second.corpus_sha256 = sha256(b"other").hexdigest()

    with pytest.raises(
        ValueError,
        match="different corpora",
    ):
        diagnose_queries([first, second])
