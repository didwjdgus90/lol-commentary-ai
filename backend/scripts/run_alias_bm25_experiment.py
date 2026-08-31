import json
from pathlib import Path

import psutil

from lol_commentary_backend.retrieval.aliases.expander import (
    expand_bilingual_entity_query,
)
from lol_commentary_backend.retrieval.aliases.models import (
    BilingualAliasCatalog,
)
from lol_commentary_backend.retrieval.benchmark.bm25 import (
    run_bm25_queries,
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

BENCHMARK_VERSION = "0.1.0"


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def main() -> None:
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

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "retrieval"
        / "entity_aliases"
        / "16.1.1"
        / "ko_en_aliases.json"
    )

    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step37"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_chunks(chunks_path)
    cases = load_eval_cases(eval_path)

    corpus_hash = validate_corpus_lineage(
        cases,
        chunks_path,
    )

    catalog = BilingualAliasCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))

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

    rss_before = _memory_mb()

    (
        query_results,
        index_seconds,
        query_seconds,
    ) = run_bm25_queries(
        chunks,
        expanded_cases,
    )

    rss_after = _memory_mb()

    run = RetrievalBenchmarkRun(
        benchmark_version=BENCHMARK_VERSION,
        engine_key="alias_bm25",
        engine_type="bm25",
        model_id=None,
        device="cpu",
        corpus_chunks=len(chunks),
        evaluation_cases=len(cases),
        corpus_sha256=corpus_hash,
        embedding_dimension=None,
        metrics=summarize_results(query_results),
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
            model_load_seconds=0.0,
            corpus_index_seconds=index_seconds,
            query_total_seconds=query_seconds,
            query_mean_ms=(query_seconds / len(cases) * 1000),
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=rss_before,
            process_rss_after_load_mb=rss_before,
            process_rss_after_index_mb=rss_after,
            cuda_peak_allocated_mb=None,
        ),
        query_results=query_results,
    )

    save_benchmark_run(
        run,
        output_dir / "alias_bm25.json",
    )

    expansion_payload = [
        {
            "query_id": case.query_id,
            **expansion.model_dump(mode="json"),
        }
        for case, expansion in zip(
            cases,
            expansions,
            strict=True,
        )
    ]

    (output_dir / "query_expansions.json").write_text(
        json.dumps(
            expansion_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    changed = [row for row in expansion_payload if row["changed"]]

    print("=== STEP 37 ALIAS-EXPANDED BM25 ===")
    print(f"Expanded queries: {len(changed)}/{len(cases)}")
    print(f"Hit@1:    {run.metrics.hit_at_1:.4f}")
    print(f"Hit@3:    {run.metrics.hit_at_3:.4f}")
    print(f"Recall@5: {run.metrics.recall_at_5:.4f}")
    print(f"MRR:      {run.metrics.mrr:.4f}")

    print()
    print("Focused expansions:")

    for row in expansion_payload:
        if row["query_id"] in {
            "q002",
            "q004",
            "q016",
        }:
            print(f"  {row['query_id']}: {row['expanded_query']}")

    q016 = next(result for result in query_results if result.query_id == "q016")

    print()
    print(f"q016 first relevant rank: {q016.first_relevant_rank}")

    print()
    print(f"Saved: {output_dir / 'alias_bm25.json'}")


if __name__ == "__main__":
    main()
