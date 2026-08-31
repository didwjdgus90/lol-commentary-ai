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
    step37_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step37"

    chunks = load_chunks(
        project_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "chunks.jsonl"
    )

    cases = load_eval_cases(
        project_root / "data" / "processed" / "evaluation" / "retrieval" / "patch_26_1_v1.jsonl"
    )

    alias_bm25 = load_benchmark_run(step37_dir / "alias_bm25.json")

    hybrids = []

    for engine in DENSE_ENGINES:
        dense = load_benchmark_run(step34_dir / f"{engine}.json")

        hybrid = build_rrf_hybrid_run(
            dense,
            alias_bm25,
            cases,
            chunks,
            rrf_k=DEFAULT_RRF_K,
            source_top_n=DEFAULT_SOURCE_TOP_N,
        )

        hybrids.append(hybrid)

        (step37_dir / f"{hybrid.hybrid_key}.json").write_text(
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

    print("=== STEP 37 ALIAS-BM25 RRF ===")
    print("rank | hybrid | Hit@1 | Hit@3 | Recall@5 | MRR")

    for rank, run in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank:>4} | "
            f"{run.hybrid_key:<52} | "
            f"{run.metrics.hit_at_1:.3f} | "
            f"{run.metrics.hit_at_3:.3f} | "
            f"{run.metrics.recall_at_5:.3f} | "
            f"{run.metrics.mrr:.3f}"
        )

    comparison_path = step37_dir / "alias_rrf_comparison.json"

    comparison_path.write_text(
        json.dumps(
            [
                {
                    "rank": rank,
                    "hybrid": run.hybrid_key,
                    "hit_at_1": run.metrics.hit_at_1,
                    "hit_at_3": run.metrics.hit_at_3,
                    "recall_at_5": (run.metrics.recall_at_5),
                    "mrr": run.metrics.mrr,
                }
                for rank, run in enumerate(
                    ranked,
                    start=1,
                )
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
