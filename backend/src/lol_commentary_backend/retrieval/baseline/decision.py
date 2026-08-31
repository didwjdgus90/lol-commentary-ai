from lol_commentary_backend.retrieval.baseline.models import (
    RetrievalBaselineCandidate,
    RetrievalBaselineDecision,
    RetrievalBaselineRole,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalBenchmarkRun,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RerankerBenchmarkRun,
)

BASELINE_DECISION_VERSION = "0.1.0"


def build_retrieval_baseline_decision(
    *,
    alias_bm25: RetrievalBenchmarkRun,
    bge_alias_hybrid: HybridRetrievalRun,
    reranker_runs: list[RerankerBenchmarkRun],
    sentence_transformers_version: str,
    transformers_version: str,
) -> RetrievalBaselineDecision:
    corpus_hashes = {
        alias_bm25.corpus_sha256,
        bge_alias_hybrid.corpus_sha256,
        *[run.corpus_sha256 for run in reranker_runs],
    }

    if len(corpus_hashes) != 1:
        raise ValueError("Baseline inputs used different corpora")

    evaluation_counts = {
        alias_bm25.evaluation_cases,
        bge_alias_hybrid.evaluation_cases,
        *[run.evaluation_cases for run in reranker_runs],
    }

    if len(evaluation_counts) != 1:
        raise ValueError("Baseline inputs used different evaluation case counts")

    best_reranker_hit_1 = max(
        (run.metrics.hit_at_1 for run in reranker_runs),
        default=0.0,
    )

    if best_reranker_hit_1 > bge_alias_hybrid.metrics.hit_at_1:
        raise ValueError(
            "A reranker exceeds the proposed primary baseline Hit@1. Revisit decision."
        )

    primary = RetrievalBaselineCandidate(
        key=bge_alias_hybrid.hybrid_key,
        role=RetrievalBaselineRole.PRIMARY,
        hit_at_1=(bge_alias_hybrid.metrics.hit_at_1),
        hit_at_3=(bge_alias_hybrid.metrics.hit_at_3),
        recall_at_5=(bge_alias_hybrid.metrics.recall_at_5),
        mrr=bge_alias_hybrid.metrics.mrr,
        online_dense_required=True,
        reranker_required=False,
        notes=[
            ("Highest Hit@1 among tested non-reranker retrieval pipelines."),
            ("Uses bilingual alias-expanded BM25 plus BGE-M3 dense retrieval with RRF."),
        ],
    )

    fallback = RetrievalBaselineCandidate(
        key=alias_bm25.engine_key,
        role=RetrievalBaselineRole.FAST_FALLBACK,
        hit_at_1=alias_bm25.metrics.hit_at_1,
        hit_at_3=alias_bm25.metrics.hit_at_3,
        recall_at_5=alias_bm25.metrics.recall_at_5,
        mrr=alias_bm25.metrics.mrr,
        online_dense_required=False,
        reranker_required=False,
        notes=[
            ("Alias BM25 Top-10 achieved 100% full-Gold candidate coverage."),
            ("No neural model inference is required for online candidate generation."),
        ],
    )

    rejected_rerankers = [run.reranker_key for run in reranker_runs]

    return RetrievalBaselineDecision(
        corpus_sha256=alias_bm25.corpus_sha256,
        evaluation_cases=(alias_bm25.evaluation_cases),
        decision_version=(BASELINE_DECISION_VERSION),
        primary=primary,
        fast_fallback=fallback,
        rejected_rerankers=rejected_rerankers,
        sentence_transformers_version=(sentence_transformers_version),
        transformers_version=(transformers_version),
        decision_rationale=[
            ("Rerankers did not improve Hit@1 over the strongest tested retrieval pipeline."),
            ("Qwen3 reranker matched Alias BM25 Hit@1 but added very high CPU latency."),
            ("BGE-M3 + Alias BM25 RRF is retained as the accuracy-first primary baseline."),
            ("Alias BM25 is retained as a low-cost fallback and ablation baseline."),
        ],
    )
