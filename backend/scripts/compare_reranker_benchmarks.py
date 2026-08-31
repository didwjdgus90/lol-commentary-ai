import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RERANKER_CANDIDATES,
    CandidateSourceSelection,
    RerankerBenchmarkRun,
)


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    retrieval_root = project_root / "data" / "processed" / "evaluation" / "retrieval"
    step37 = retrieval_root / "step37"
    step38 = retrieval_root / "step38"

    selection = CandidateSourceSelection.model_validate_json(
        (step38 / "candidate_source_selection.json").read_text(encoding="utf-8")
    )

    if selection.source_key == "alias_bm25":
        source = load_benchmark_run(step37 / "alias_bm25.json")
    elif selection.source_key == "bge_alias":
        source = _load_hybrid(step37 / "rrf_bge_m3_alias_bm25.json")
    elif selection.source_key == "qwen_alias":
        source = _load_hybrid(step37 / "rrf_qwen3_embedding_06b_alias_bm25.json")
    else:
        raise ValueError("Unsupported selected source")

    rows = [
        {
            "name": (f"source:{selection.source_key}"),
            "kind": "candidate_source",
            "hit_at_1": source.metrics.hit_at_1,
            "hit_at_3": source.metrics.hit_at_3,
            "recall_at_5": (source.metrics.recall_at_5),
            "mrr": source.metrics.mrr,
            "query_mean_ms": None,
        }
    ]

    for candidate in RERANKER_CANDIDATES:
        path = step38 / f"{candidate.key}.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing reranker result: {path}")

        run = RerankerBenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))

        if run.candidate_source != selection.source_key or run.candidate_top_n != selection.top_n:
            raise ValueError("Reranker runs used different candidate source settings")

        rows.append(
            {
                "name": run.reranker_key,
                "kind": "reranker",
                "hit_at_1": (run.metrics.hit_at_1),
                "hit_at_3": (run.metrics.hit_at_3),
                "recall_at_5": (run.metrics.recall_at_5),
                "mrr": run.metrics.mrr,
                "query_mean_ms": (run.timing.query_mean_ms),
            }
        )

    reranker_rows = sorted(
        rows[1:],
        key=lambda row: (
            row["hit_at_1"],
            row["mrr"],
            row["hit_at_3"],
        ),
        reverse=True,
    )

    ranked = [
        rows[0],
        *[
            {
                **row,
                "reranker_rank": rank,
            }
            for rank, row in enumerate(
                reranker_rows,
                start=1,
            )
        ],
    ]

    output_path = step38 / "reranker_comparison.json"

    output_path.write_text(
        json.dumps(
            ranked,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP 38 RERANKER COMPARISON ===")
    print(f"Candidate source: {selection.source_key} Top-{selection.top_n}")
    print()
    print("name | Hit@1 | Hit@3 | Recall@5 | MRR | query ms")

    for row in ranked:
        latency = row["query_mean_ms"]

        latency_text = "-" if latency is None else f"{latency:.2f}"

        print(
            f"{row['name']:<38} | "
            f"{row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f} | "
            f"{latency_text}"
        )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
