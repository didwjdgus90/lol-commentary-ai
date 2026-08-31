import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import load_benchmark_run


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    step34_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"
    step35_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step35"

    rows = []

    for engine in (
        "bm25",
        "qwen3_embedding_06b",
        "multilingual_e5_large_instruct",
        "bge_m3",
    ):
        run = load_benchmark_run(step34_dir / f"{engine}.json")
        rows.append(
            {
                "name": engine,
                "kind": run.engine_type,
                "hit_at_1": run.metrics.hit_at_1,
                "hit_at_3": run.metrics.hit_at_3,
                "hit_at_5": run.metrics.hit_at_5,
                "recall_at_5": run.metrics.recall_at_5,
                "mrr": run.metrics.mrr,
            }
        )

    for path in sorted(step35_dir.glob("rrf_*_bm25.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "name": payload["hybrid_key"],
                "kind": "hybrid",
                "hit_at_1": metrics["hit_at_1"],
                "hit_at_3": metrics["hit_at_3"],
                "hit_at_5": metrics["hit_at_5"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr": metrics["mrr"],
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            row["hit_at_1"],
            row["mrr"],
            row["hit_at_3"],
            row["recall_at_5"],
        ),
        reverse=True,
    )

    output = [{"rank": rank, **row} for rank, row in enumerate(ranked, start=1)]

    output_path = step35_dir / "standalone_vs_hybrid.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== STANDALONE VS HYBRID ===")
    print("rank | name | kind | Hit@1 | Hit@3 | Recall@5 | MRR")

    for row in output:
        print(
            f"{row['rank']:>4} | "
            f"{row['name']:<45} | "
            f"{row['kind']:<6} | "
            f"{row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f}"
        )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
