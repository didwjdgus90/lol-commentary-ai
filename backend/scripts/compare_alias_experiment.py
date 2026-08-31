import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)


def _load_hybrid(path: Path) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    step34 = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"
    step35 = step34.parent / "step35"
    step37 = step34.parent / "step37"

    rows = []

    baseline_bm25 = load_benchmark_run(step34 / "bm25.json")
    alias_bm25 = load_benchmark_run(step37 / "alias_bm25.json")

    rows.extend(
        [
            {
                "name": "bm25",
                "kind": "baseline_sparse",
                "hit_at_1": (baseline_bm25.metrics.hit_at_1),
                "hit_at_3": (baseline_bm25.metrics.hit_at_3),
                "recall_at_5": (baseline_bm25.metrics.recall_at_5),
                "mrr": baseline_bm25.metrics.mrr,
            },
            {
                "name": "alias_bm25",
                "kind": "alias_sparse",
                "hit_at_1": (alias_bm25.metrics.hit_at_1),
                "hit_at_3": (alias_bm25.metrics.hit_at_3),
                "recall_at_5": (alias_bm25.metrics.recall_at_5),
                "mrr": alias_bm25.metrics.mrr,
            },
        ]
    )

    baseline_hybrid = _load_hybrid(step35 / "rrf_bge_m3_bm25.json")

    alias_hybrid = _load_hybrid(step37 / "rrf_bge_m3_alias_bm25.json")

    rows.extend(
        [
            {
                "name": (baseline_hybrid.hybrid_key),
                "kind": "baseline_hybrid",
                "hit_at_1": (baseline_hybrid.metrics.hit_at_1),
                "hit_at_3": (baseline_hybrid.metrics.hit_at_3),
                "recall_at_5": (baseline_hybrid.metrics.recall_at_5),
                "mrr": (baseline_hybrid.metrics.mrr),
            },
            {
                "name": (alias_hybrid.hybrid_key),
                "kind": "alias_hybrid",
                "hit_at_1": (alias_hybrid.metrics.hit_at_1),
                "hit_at_3": (alias_hybrid.metrics.hit_at_3),
                "recall_at_5": (alias_hybrid.metrics.recall_at_5),
                "mrr": alias_hybrid.metrics.mrr,
            },
        ]
    )

    output_path = step37 / "baseline_vs_alias.json"

    output_path.write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== BASELINE VS BILINGUAL ALIAS ===")
    print("name | Hit@1 | Hit@3 | Recall@5 | MRR")

    for row in rows:
        print(
            f"{row['name']:<40} | "
            f"{row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f}"
        )

    base_q016 = next(result for result in baseline_bm25.query_results if result.query_id == "q016")
    alias_q016 = next(result for result in alias_bm25.query_results if result.query_id == "q016")

    print()
    print(f"q016 BM25 rank: {base_q016.first_relevant_rank} -> {alias_q016.first_relevant_rank}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
