import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
    load_chunks,
    load_eval_cases,
)
from lol_commentary_backend.retrieval.hybrid.rrf import (
    DEFAULT_RRF_K,
    DEFAULT_SOURCE_TOP_N,
    build_rrf_hybrid_run,
)

DENSE_ENGINES = (
    "qwen3_embedding_06b",
    "multilingual_e5_large_instruct",
    "bge_m3",
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    step34_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"
    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step35"
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

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(chunks_path)
    cases = load_eval_cases(eval_path)
    bm25 = load_benchmark_run(step34_dir / "bm25.json")
    hybrids = []

    for engine in DENSE_ENGINES:
        dense = load_benchmark_run(step34_dir / f"{engine}.json")
        hybrid = build_rrf_hybrid_run(
            dense,
            bm25,
            cases,
            chunks,
            rrf_k=DEFAULT_RRF_K,
            source_top_n=DEFAULT_SOURCE_TOP_N,
        )
        hybrids.append(hybrid)

        output_path = output_dir / f"{hybrid.hybrid_key}.json"
        output_path.write_text(
            json.dumps(
                hybrid.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    ranked = sorted(
        hybrids,
        key=lambda run: (
            run.metrics.hit_at_1,
            run.metrics.mrr,
            run.metrics.hit_at_3,
            run.metrics.recall_at_5,
        ),
        reverse=True,
    )

    comparison = [
        {
            "rank": rank,
            "hybrid": run.hybrid_key,
            "dense_engine": run.dense_engine,
            "rrf_k": run.rrf_k,
            "source_top_n": run.source_top_n,
            "hit_at_1": run.metrics.hit_at_1,
            "hit_at_3": run.metrics.hit_at_3,
            "hit_at_5": run.metrics.hit_at_5,
            "recall_at_5": run.metrics.recall_at_5,
            "mrr": run.metrics.mrr,
            "fusion_query_mean_ms": run.fusion_query_mean_ms,
        }
        for rank, run in enumerate(ranked, start=1)
    ]

    comparison_path = output_dir / "rrf_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== STEP 35 RRF HYBRID COMPARISON ===")
    print(f"Fixed RRF k: {DEFAULT_RRF_K}")
    print(f"Source depth per engine: Top-{DEFAULT_SOURCE_TOP_N}")
    print()
    print("rank | hybrid | Hit@1 | Hit@3 | Recall@5 | MRR")

    for row in comparison:
        print(
            f"{row['rank']:>4} | "
            f"{row['hybrid']:<45} | "
            f"{row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f}"
        )

    print()
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
