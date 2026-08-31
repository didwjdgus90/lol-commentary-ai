import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
)

EXPECTED_ENGINES = (
    "bm25",
    "qwen3_embedding_06b",
    "bge_m3",
    "multilingual_e5_large_instruct",
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"

    output_path = input_dir / "comparison.json"

    runs = []

    for engine in EXPECTED_ENGINES:
        path = input_dir / f"{engine}.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing benchmark result: {path}")

        runs.append(load_benchmark_run(path))

    corpus_hashes = {run.corpus_sha256 for run in runs}
    case_counts = {run.evaluation_cases for run in runs}

    if len(corpus_hashes) != 1:
        raise ValueError("Benchmark runs used different corpora")

    if len(case_counts) != 1:
        raise ValueError("Benchmark runs used different evaluation case counts")

    ranked = sorted(
        runs,
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
            "engine": run.engine_key,
            "type": run.engine_type,
            "model_id": run.model_id,
            "device": run.device,
            "hit_at_1": run.metrics.hit_at_1,
            "hit_at_3": run.metrics.hit_at_3,
            "hit_at_5": run.metrics.hit_at_5,
            "recall_at_5": (run.metrics.recall_at_5),
            "mrr": run.metrics.mrr,
            "model_load_seconds": (run.timing.model_load_seconds),
            "corpus_index_seconds": (run.timing.corpus_index_seconds),
            "query_mean_ms": (run.timing.query_mean_ms),
        }
        for rank, run in enumerate(
            ranked,
            start=1,
        )
    ]

    output_path.write_text(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP 34 RETRIEVAL COMPARISON ===")
    print("rank | engine | Hit@1 | Hit@3 | Recall@5 | MRR | query ms")

    for row in comparison:
        print(
            f"{row['rank']:>4} | "
            f"{row['engine']:<34} | "
            f"{row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | "
            f"{row['recall_at_5']:.3f} | "
            f"{row['mrr']:.3f} | "
            f"{row['query_mean_ms']:.2f}"
        )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
