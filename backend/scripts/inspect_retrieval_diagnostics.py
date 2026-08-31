import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
)
from lol_commentary_backend.retrieval.diagnostics.analyzer import (
    build_diagnostic_report,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    step34_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"

    step35_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step35"

    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step36"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qwen = load_benchmark_run(step34_dir / "qwen3_embedding_06b.json")
    e5 = load_benchmark_run(step34_dir / "multilingual_e5_large_instruct.json")
    bge = load_benchmark_run(step34_dir / "bge_m3.json")
    bm25 = load_benchmark_run(step34_dir / "bm25.json")

    best_hybrid = _load_hybrid(step35_dir / "rrf_bge_m3_bm25.json")

    report = build_diagnostic_report(
        [qwen, e5, bge, bm25],
        hybrid=best_hybrid,
        hybrid_dense_source=bge,
        hybrid_sparse_source=bm25,
    )

    output_path = output_dir / "retrieval_failure_diagnostics.json"

    output_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP 36 RETRIEVAL FAILURE DIAGNOSTICS ===")
    print(f"Cases: {report.cases}")
    print(f"Solved by at least one engine at Top-1: {len(report.solved_top1)}")
    print(f"Rerankable (best rank 2-10): {len(report.rerankable_top10)}")
    print(f"Candidate recall failures (>10 / missing): {len(report.candidate_recall_failures)}")
    print()

    print(f"Solved Top-1: {report.solved_top1}")
    print(f"Rerankable Top-10: {report.rerankable_top10}")
    print(f"Recall failures: {report.candidate_recall_failures}")

    print()
    print("Unsolved query details:")

    for item in report.query_diagnostics:
        if item.query_id in report.solved_top1:
            continue

        ranks = ", ".join(
            (f"{engine.engine}={engine.first_relevant_rank}") for engine in item.engine_diagnostics
        )

        print(
            f"  {item.query_id} | "
            f"{item.category.value} | "
            f"best={item.best_relevant_rank} | "
            f"{item.query}"
        )
        print(f"       {ranks}")

    effects: dict[str, list[str]] = {}

    for item in report.fusion_diagnostics:
        effects.setdefault(
            item.effect.value,
            [],
        ).append(item.query_id)

    print()
    print("BGE-M3 + BM25 RRF effects:")

    for effect, query_ids in sorted(effects.items()):
        print(f"  {effect}: {query_ids}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
