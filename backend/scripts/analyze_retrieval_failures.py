import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import load_benchmark_run
from lol_commentary_backend.retrieval.hybrid.analysis import (
    compare_engine_failures,
)

DENSE_ENGINES = (
    "qwen3_embedding_06b",
    "multilingual_e5_large_instruct",
    "bge_m3",
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step34"
    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval" / "step35"
    output_dir.mkdir(parents=True, exist_ok=True)

    bm25 = load_benchmark_run(input_dir / "bm25.json")
    reports = []

    print("=== STEP 35 FAILURE / COMPLEMENTARITY ANALYSIS ===")

    for engine in DENSE_ENGINES:
        dense = load_benchmark_run(input_dir / f"{engine}.json")
        report = compare_engine_failures(dense, bm25)
        reports.append(report)

        print()
        print(f"{engine} vs BM25")
        print(f"  dense-only Hit@1: {report['dense_only_hit_at_1']}")
        print(f"  BM25-only Hit@1: {report['sparse_only_hit_at_1']}")
        print(f"  both miss Hit@1: {report['both_miss_at_1']}")
        print(f"  both miss Hit@5: {report['both_miss_at_5']}")

    output_path = output_dir / "failure_analysis.json"
    output_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
