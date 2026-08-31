import argparse
import json
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
    load_chunks,
    load_eval_cases,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RERANKER_CANDIDATES,
    CandidateSourceSelection,
)
from lol_commentary_backend.retrieval.rerankers.runner import (
    run_reranker,
)


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def _candidate_by_key(key: str):
    for candidate in RERANKER_CANDIDATES:
        if candidate.key == key:
            return candidate

    raise ValueError(f"Unknown reranker: {key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reranker",
        required=True,
        choices=[candidate.key for candidate in RERANKER_CANDIDATES],
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
        raise ValueError(f"Unsupported selected source: {selection.source_key}")

    if source.corpus_sha256 != selection.corpus_sha256:
        raise ValueError("Candidate source selection is stale")

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

    cases = load_eval_cases(retrieval_root / "patch_26_1_v1.jsonl")

    candidate = _candidate_by_key(args.reranker)

    run = run_reranker(
        candidate,
        source_key=selection.source_key,
        source_results=source.query_results,
        cases=cases,
        chunks=chunks,
        corpus_sha256=(selection.corpus_sha256),
        top_n=selection.top_n,
        requested_device=args.device,
        batch_size=args.batch_size,
    )

    output_path = step38 / f"{candidate.key}.json"

    output_path.write_text(
        json.dumps(
            run.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP 38 RERANKER BENCHMARK ===")
    print(f"Reranker: {run.reranker_key}")
    print(f"Model: {run.model_id}")
    print(f"Device: {run.device}")
    print(f"Candidate source: {run.candidate_source}")
    print(f"Candidate Top-N: {run.candidate_top_n}")
    print(f"Pairs scored: {run.pairs_scored}")
    print()

    print(f"Hit@1:    {run.metrics.hit_at_1:.4f}")
    print(f"Hit@3:    {run.metrics.hit_at_3:.4f}")
    print(f"Recall@5: {run.metrics.recall_at_5:.4f}")
    print(f"MRR:      {run.metrics.mrr:.4f}")
    print()

    print(f"Model load seconds: {run.timing.model_load_seconds:.3f}")
    print(f"Rerank query mean ms: {run.timing.query_mean_ms:.3f}")
    print(f"Rerank pair mean ms: {run.timing.pair_mean_ms:.3f}")

    print()
    print("By language:")

    for key, summary in run.by_language.items():
        print(f"  {key}: Hit@1={summary.hit_at_1:.3f} MRR={summary.mrr:.3f}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
