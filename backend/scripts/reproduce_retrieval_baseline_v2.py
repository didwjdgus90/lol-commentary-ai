import json
from pathlib import Path

import psutil
import sentence_transformers
import transformers

from lol_commentary_backend.retrieval.aliases.expander import (
    expand_bilingual_entity_query,
)
from lol_commentary_backend.retrieval.aliases.models import (
    BilingualAliasCatalog,
)
from lol_commentary_backend.retrieval.baseline.verify import (
    build_reproducibility_check,
)
from lol_commentary_backend.retrieval.benchmark.bm25 import (
    run_bm25_queries,
)
from lol_commentary_backend.retrieval.benchmark.dense import (
    run_dense_queries,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    load_chunks,
    load_eval_cases,
    validate_corpus_lineage,
)
from lol_commentary_backend.retrieval.benchmark.metrics import (
    summarize_by,
    summarize_results,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    BenchmarkMemory,
    BenchmarkTiming,
    RetrievalBenchmarkRun,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    EMBEDDING_CANDIDATES,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.hybrid.rrf import (
    DEFAULT_RRF_K,
    DEFAULT_SOURCE_TOP_N,
    build_rrf_hybrid_run,
)

BENCHMARK_VERSION = "0.1.0"


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _candidate_by_key(key: str):
    for candidate in EMBEDDING_CANDIDATES:
        if candidate.key == key:
            return candidate

    raise ValueError(f"Unknown embedding candidate: {key}")


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def _save_json(
    payload: object,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    else:
        data = payload

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    retrieval_root = project_root / "data" / "processed" / "evaluation" / "retrieval"

    chunks_path = (
        project_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "chunks.jsonl"
    )

    eval_path = retrieval_root / "patch_26_1_v1.jsonl"

    alias_catalog_path = (
        project_root
        / "data"
        / "processed"
        / "retrieval"
        / "entity_aliases"
        / "16.1.1"
        / "ko_en_aliases.json"
    )

    step37_dir = retrieval_root / "step37"
    step39_dir = retrieval_root / "step39"

    chunks = load_chunks(chunks_path)
    cases = load_eval_cases(eval_path)

    corpus_hash = validate_corpus_lineage(
        cases,
        chunks_path,
    )

    print("=== STEP 39 BASELINE REPRODUCTION ===")
    print(f"sentence-transformers: {sentence_transformers.__version__}")
    print(f"transformers: {transformers.__version__}")
    print()

    candidate = _candidate_by_key("bge_m3")

    rss_before = _memory_mb()

    (
        dense_results,
        device,
        embedding_dimension,
        load_seconds,
        corpus_seconds,
        query_seconds,
        rss_after_load,
        rss_after_index,
    ) = run_dense_queries(
        candidate,
        chunks,
        cases,
        requested_device="auto",
        batch_size=8,
    )

    dense_run = RetrievalBenchmarkRun(
        benchmark_version=BENCHMARK_VERSION,
        engine_key="bge_m3",
        engine_type="dense",
        model_id=candidate.model_id,
        device=device,
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        corpus_sha256=corpus_hash,
        embedding_dimension=embedding_dimension,
        metrics=summarize_results(dense_results),
        by_language=summarize_by(
            dense_results,
            lambda result: result.language,
        ),
        by_query_type=summarize_by(
            dense_results,
            lambda result: result.query_type,
        ),
        by_difficulty=summarize_by(
            dense_results,
            lambda result: result.difficulty,
        ),
        timing=BenchmarkTiming(
            model_load_seconds=load_seconds,
            corpus_index_seconds=corpus_seconds,
            query_total_seconds=query_seconds,
            query_mean_ms=(query_seconds / len(cases) * 1000),
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=rss_before,
            process_rss_after_load_mb=(rss_after_load),
            process_rss_after_index_mb=(rss_after_index),
            cuda_peak_allocated_mb=None,
        ),
        query_results=dense_results,
    )

    catalog = BilingualAliasCatalog.model_validate_json(
        alias_catalog_path.read_text(encoding="utf-8")
    )

    expansions = [
        expand_bilingual_entity_query(
            case.query,
            catalog,
        )
        for case in cases
    ]

    expanded_cases = [
        case.model_copy(update={"query": expansion.expanded_query})
        for case, expansion in zip(
            cases,
            expansions,
            strict=True,
        )
    ]

    sparse_rss_before = _memory_mb()

    (
        sparse_results,
        sparse_index_seconds,
        sparse_query_seconds,
    ) = run_bm25_queries(
        chunks,
        expanded_cases,
    )

    sparse_rss_after = _memory_mb()

    alias_bm25 = RetrievalBenchmarkRun(
        benchmark_version=BENCHMARK_VERSION,
        engine_key="alias_bm25",
        engine_type="bm25",
        model_id=None,
        device="cpu",
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        corpus_sha256=corpus_hash,
        embedding_dimension=None,
        metrics=summarize_results(sparse_results),
        by_language=summarize_by(
            sparse_results,
            lambda result: result.language,
        ),
        by_query_type=summarize_by(
            sparse_results,
            lambda result: result.query_type,
        ),
        by_difficulty=summarize_by(
            sparse_results,
            lambda result: result.difficulty,
        ),
        timing=BenchmarkTiming(
            model_load_seconds=0.0,
            corpus_index_seconds=(sparse_index_seconds),
            query_total_seconds=(sparse_query_seconds),
            query_mean_ms=(sparse_query_seconds / len(cases) * 1000),
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=(sparse_rss_before),
            process_rss_after_load_mb=(sparse_rss_before),
            process_rss_after_index_mb=(sparse_rss_after),
            cuda_peak_allocated_mb=None,
        ),
        query_results=sparse_results,
    )

    reproduced_hybrid = build_rrf_hybrid_run(
        dense_run,
        alias_bm25,
        cases,
        chunks,
        rrf_k=DEFAULT_RRF_K,
        source_top_n=DEFAULT_SOURCE_TOP_N,
    )

    reference_hybrid = _load_hybrid(step37_dir / "rrf_bge_m3_alias_bm25.json")

    check = build_reproducibility_check(
        reference=reference_hybrid,
        reproduced=reproduced_hybrid,
        sentence_transformers_version=(sentence_transformers.__version__),
        transformers_version=(transformers.__version__),
    )

    _save_json(
        dense_run,
        step39_dir / "bge_m3_current_env.json",
    )
    _save_json(
        alias_bm25,
        step39_dir / "alias_bm25_current_env.json",
    )
    _save_json(
        reproduced_hybrid,
        step39_dir / "rrf_bge_m3_alias_bm25_current_env.json",
    )
    _save_json(
        check,
        step39_dir / "reproducibility_check.json",
    )

    print("Reproduced BGE-M3:")
    print(f"  Hit@1: {dense_run.metrics.hit_at_1:.4f}")
    print(f"  MRR:   {dense_run.metrics.mrr:.4f}")
    print(f"  Query mean ms: {dense_run.timing.query_mean_ms:.2f}")
    print()

    print("Reproduced Alias BM25:")
    print(f"  Hit@1: {alias_bm25.metrics.hit_at_1:.4f}")
    print(f"  MRR:   {alias_bm25.metrics.mrr:.4f}")
    print()

    print("Reproduced BGE + Alias BM25 RRF:")
    print(f"  Hit@1: {reproduced_hybrid.metrics.hit_at_1:.4f}")
    print(f"  Hit@3: {reproduced_hybrid.metrics.hit_at_3:.4f}")
    print(f"  Recall@5: {reproduced_hybrid.metrics.recall_at_5:.4f}")
    print(f"  MRR: {reproduced_hybrid.metrics.mrr:.4f}")
    print()

    print("Reference:")
    print(f"  Hit@1: {reference_hybrid.metrics.hit_at_1:.4f}")
    print(f"  Hit@3: {reference_hybrid.metrics.hit_at_3:.4f}")
    print(f"  Recall@5: {reference_hybrid.metrics.recall_at_5:.4f}")
    print(f"  MRR: {reference_hybrid.metrics.mrr:.4f}")
    print()

    print(f"Metrics match: {check.metrics_match}")

    if not check.metrics_match:
        raise RuntimeError(
            "Retrieval baseline metrics changed "
            "under the pinned dependency environment. "
            "Inspect Step 39 artifacts before freezing."
        )

    print()
    print("Step 39 reproduction gate: PASS")


if __name__ == "__main__":
    main()
