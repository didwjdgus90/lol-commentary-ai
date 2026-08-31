import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
    load_eval_cases,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.candidate import (
    analyze_candidate_coverage,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    CandidateSourceSelection,
)

SELECTION_VERSION = "0.1.0"


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    retrieval_root = project_root / "data" / "processed" / "evaluation" / "retrieval"
    step37 = retrieval_root / "step37"

    cases = load_eval_cases(retrieval_root / "patch_26_1_v1.jsonl")

    alias_bm25 = load_benchmark_run(step37 / "alias_bm25.json")
    bge_alias = _load_hybrid(step37 / "rrf_bge_m3_alias_bm25.json")
    qwen_alias = _load_hybrid(step37 / "rrf_qwen3_embedding_06b_alias_bm25.json")

    sources = {
        "alias_bm25": (
            alias_bm25.query_results,
            alias_bm25.corpus_sha256,
        ),
        "bge_alias": (
            bge_alias.query_results,
            bge_alias.corpus_sha256,
        ),
        "qwen_alias": (
            qwen_alias.query_results,
            qwen_alias.corpus_sha256,
        ),
    }

    # Ordered by expected online retrieval cost:
    # lexical-only first, then faster dense hybrid,
    # then the slower Qwen dense hybrid.
    preferences = (
        ("alias_bm25", 5),
        ("alias_bm25", 10),
        ("bge_alias", 5),
        ("bge_alias", 10),
        ("qwen_alias", 5),
        ("qwen_alias", 10),
    )

    coverage_rows = []

    for source_key, top_n in preferences:
        results, _ = sources[source_key]

        coverage = analyze_candidate_coverage(
            source_key=source_key,
            results=results,
            cases=cases,
            top_n=top_n,
        )

        coverage_rows.append(coverage)

    selected = next(
        (row for row in coverage_rows if row.coverage_rate == 1.0),
        None,
    )

    if selected is None:
        raise ValueError("No candidate source/depth achieved 100% Gold coverage")

    _, corpus_hash = sources[selected.source_key]

    selection = CandidateSourceSelection(
        selection_version=SELECTION_VERSION,
        source_key=selected.source_key,
        top_n=selected.top_n,
        coverage_rate=(selected.coverage_rate),
        corpus_sha256=corpus_hash,
        reason=(
            "First predefined source/depth in "
            "increasing online-cost order with "
            "100% full-Gold candidate coverage."
        ),
    )

    output_dir = retrieval_root / "step38"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "candidate_source_coverage.json").write_text(
        json.dumps(
            [row.model_dump(mode="json") for row in coverage_rows],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selection_path = output_dir / "candidate_source_selection.json"

    selection_path.write_text(
        json.dumps(
            selection.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP 38 CANDIDATE SOURCE COVERAGE ===")
    print("source | top_n | full_gold | coverage | missing")

    for row in coverage_rows:
        print(
            f"{row.source_key:<12} | "
            f"{row.top_n:>5} | "
            f"{row.full_gold_cases:>2}/"
            f"{row.cases:<2} | "
            f"{row.coverage_rate:.3f} | "
            f"{row.missing_query_ids}"
        )

    print()
    print(f"Selected source: {selection.source_key}")
    print(f"Selected Top-N: {selection.top_n}")
    print(f"Gold coverage: {selection.coverage_rate:.3f}")
    print()
    print(f"Saved: {selection_path}")


if __name__ == "__main__":
    main()
