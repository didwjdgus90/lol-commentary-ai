import argparse
from pathlib import Path

import psutil

from lol_commentary_backend.retrieval.benchmark.bm25 import (
    run_bm25_queries,
)
from lol_commentary_backend.retrieval.benchmark.dense import (
    run_dense_queries,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    load_chunks,
    load_eval_cases,
    save_benchmark_run,
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

BENCHMARK_VERSION = "0.1.0"


def _memory_mb() -> float:
    process = psutil.Process()

    return process.memory_info().rss / 1024 / 1024


def _cuda_peak_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _reset_cuda_peak() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _candidate_by_key(key: str):
    for candidate in EMBEDDING_CANDIDATES:
        if candidate.key == key:
            return candidate

    raise ValueError(f"Unknown dense engine: {key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        required=True,
        choices=[
            "bm25",
            *[candidate.key for candidate in EMBEDDING_CANDIDATES],
        ],
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    project_root = Path(__file__).resolve().parents[2]

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

    eval_path = (
        project_root / "data" / "processed" / "evaluation" / "retrieval" / "patch_26_1_v1.jsonl"
    )

    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"

    chunks = load_chunks(chunks_path)
    cases = load_eval_cases(eval_path)

    corpus_hash = validate_corpus_lineage(
        cases,
        chunks_path,
    )

    rss_before = _memory_mb()
    _reset_cuda_peak()

    if args.engine == "bm25":
        load_seconds = 0.0
        rss_after_load = _memory_mb()

        (
            query_results,
            index_seconds,
            query_seconds,
        ) = run_bm25_queries(
            chunks,
            cases,
        )

        rss_after_index = _memory_mb()
        device = "cpu"
        model_id = None
        embedding_dimension = None
        engine_type = "bm25"

    else:
        candidate = _candidate_by_key(args.engine)

        (
            query_results,
            device,
            embedding_dimension,
            load_seconds,
            index_seconds,
            query_seconds,
            rss_after_load,
            rss_after_index,
        ) = run_dense_queries(
            candidate,
            chunks,
            cases,
            requested_device=args.device,
            batch_size=args.batch_size,
        )

        model_id = candidate.model_id
        engine_type = "dense"

    metrics = summarize_results(query_results)

    run = RetrievalBenchmarkRun(
        benchmark_version=BENCHMARK_VERSION,
        engine_key=args.engine,
        engine_type=engine_type,
        model_id=model_id,
        device=device,
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        corpus_sha256=corpus_hash,
        embedding_dimension=embedding_dimension,
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
        timing=BenchmarkTiming(
            model_load_seconds=load_seconds,
            corpus_index_seconds=index_seconds,
            query_total_seconds=query_seconds,
            query_mean_ms=(0.0 if not cases else query_seconds / len(cases) * 1000),
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=rss_before,
            process_rss_after_load_mb=(rss_after_load),
            process_rss_after_index_mb=(rss_after_index),
            cuda_peak_allocated_mb=(_cuda_peak_mb()),
        ),
        query_results=query_results,
    )

    output_path = output_dir / f"{args.engine}.json"

    save_benchmark_run(
        run,
        output_path,
    )

    print("=== RETRIEVAL BENCHMARK RESULT ===")
    print(f"Engine: {run.engine_key}")
    print(f"Type: {run.engine_type}")
    print(f"Model: {run.model_id}")
    print(f"Device: {run.device}")
    print(f"Corpus chunks: {run.corpus_chunks}")
    print(f"Evaluation cases: {run.evaluation_cases}")
    print()

    print(f"Hit@1:    {run.metrics.hit_at_1:.4f}")
    print(f"Hit@3:    {run.metrics.hit_at_3:.4f}")
    print(f"Hit@5:    {run.metrics.hit_at_5:.4f}")
    print(f"Recall@5: {run.metrics.recall_at_5:.4f}")
    print(f"MRR:      {run.metrics.mrr:.4f}")
    print()

    print(f"Model load seconds: {run.timing.model_load_seconds:.3f}")
    print(f"Corpus index/embed seconds: {run.timing.corpus_index_seconds:.3f}")
    print(f"Query mean ms: {run.timing.query_mean_ms:.3f}")

    print()
    print("By language:")

    for key, summary in run.by_language.items():
        print(f"  {key}: Hit@1={summary.hit_at_1:.3f} MRR={summary.mrr:.3f}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
