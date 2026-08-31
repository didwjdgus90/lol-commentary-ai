from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from lol_commentary_backend.retrieval.baseline.models import (
    RetrievalBaselineDecision,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    load_chunks,
    load_eval_cases,
    validate_corpus_lineage,
)
from lol_commentary_backend.retrieval.benchmark.metrics import (
    evaluate_ranking,
    summarize_results,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    BGE_M3,
)
from lol_commentary_backend.retrieval.runtime.dense import (
    BgeDenseIndex,
)
from lol_commentary_backend.retrieval.runtime.parity import (
    compare_ranked_candidates,
)
from lol_commentary_backend.retrieval.runtime.pgvector_dense import (
    PgVectorDenseIndex,
)
from lol_commentary_backend.retrieval.storage.postgres import (
    build_vector_pool,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare frozen NumPy BGE-M3 retrieval with PostgreSQL pgvector."),
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

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


def _resolve_device(
    requested_device: str,
) -> str:
    import torch

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main() -> None:
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    if args.top_n <= 0:
        raise ValueError("top-n must be positive")

    if args.score_tolerance < 0:
        raise ValueError("score-tolerance must be nonnegative")

    project_root = Path(__file__).resolve().parents[2]

    retrieval_root = project_root / "data" / "processed" / "evaluation" / "retrieval"

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

    eval_path = retrieval_root / "patch_26_1_v1.jsonl"

    baseline_path = (
        project_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    chunks = load_chunks(chunks_path)

    cases = load_eval_cases(eval_path)

    corpus_hash = validate_corpus_lineage(
        cases,
        chunks_path,
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    if decision.corpus_sha256 != corpus_hash:
        raise RuntimeError("Frozen baseline corpus SHA does not match evaluation corpus")

    device = _resolve_device(args.device)

    from sentence_transformers import (
        SentenceTransformer,
    )

    print("=== STEP 41-6 NUMPY ↔ PGVECTOR PARITY ===")

    print(f"Corpus chunks: {len(chunks)}")

    print(f"Evaluation cases: {len(cases)}")

    print(f"Corpus SHA256: {corpus_hash}")

    print(f"Model: {BGE_M3.model_id}")

    print(f"Device: {device}")

    print(f"Top-N: {args.top_n}")

    print()

    model_started = perf_counter()

    model = SentenceTransformer(
        BGE_M3.model_id,
        device=device,
    )

    model.max_seq_length = BGE_M3.experiment_max_tokens

    model_seconds = perf_counter() - model_started

    corpus_started = perf_counter()

    corpus_embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    corpus_matrix = np.asarray(
        corpus_embeddings,
        dtype=np.float32,
    )

    corpus_seconds = perf_counter() - corpus_started

    numpy_index = BgeDenseIndex(
        chunks=chunks,
        model=model,
        corpus_embeddings=corpus_matrix,
        device=device,
    )

    pool = build_vector_pool()

    pgvector_index = PgVectorDenseIndex(
        model=model,
        pool=pool,
        corpus_sha256=corpus_hash,
        embedding_model_id=(BGE_M3.model_id),
        device=device,
    )

    pgvector_index.validate_storage()

    chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    parity_results = []

    numpy_eval_results = []
    pgvector_eval_results = []

    numpy_query_seconds = 0.0
    pgvector_query_seconds = 0.0

    try:
        for case in cases:
            numpy_started = perf_counter()

            numpy_candidates = numpy_index.search(
                case.query,
                top_n=args.top_n,
            )

            numpy_query_seconds += perf_counter() - numpy_started

            pg_started = perf_counter()

            pgvector_candidates = pgvector_index.search(
                case.query,
                top_n=args.top_n,
            )

            pgvector_query_seconds += perf_counter() - pg_started

            parity = compare_ranked_candidates(
                query_id=case.query_id,
                left=numpy_candidates,
                right=pgvector_candidates,
            )

            parity_results.append(parity)

            numpy_indices = [
                chunk_index_by_id[candidate.chunk_id] for candidate in numpy_candidates
            ]

            pgvector_indices = [
                chunk_index_by_id[candidate.chunk_id] for candidate in pgvector_candidates
            ]

            numpy_eval_results.append(
                evaluate_ranking(
                    case,
                    numpy_indices,
                    [candidate.score for candidate in numpy_candidates],
                    chunks,
                )
            )

            pgvector_eval_results.append(
                evaluate_ranking(
                    case,
                    pgvector_indices,
                    [candidate.score for candidate in pgvector_candidates],
                    chunks,
                )
            )

    finally:
        pgvector_index.close()

    numpy_metrics = summarize_results(numpy_eval_results)

    pgvector_metrics = summarize_results(pgvector_eval_results)

    exact_matches = sum(result.exact_order_match for result in parity_results)

    set_matches = sum(result.candidate_set_match for result in parity_results)

    max_score_delta = max(
        (result.max_score_delta for result in parity_results),
        default=0.0,
    )

    ranking_mismatches = [
        result.query_id for result in parity_results if not result.exact_order_match
    ]

    candidate_mismatches = [
        result.query_id for result in parity_results if not result.candidate_set_match
    ]

    score_tolerance_pass = max_score_delta <= args.score_tolerance

    metrics_match = numpy_metrics.model_dump() == pgvector_metrics.model_dump()

    print("=== PARITY RESULT ===")

    print(f"Exact Top-N order: {exact_matches}/{len(cases)}")

    print(f"Same Top-N candidate set: {set_matches}/{len(cases)}")

    print(f"Maximum score delta: {max_score_delta:.10f}")

    print(f"Score tolerance: {args.score_tolerance:.10f}")

    print(f"Score tolerance PASS: {score_tolerance_pass}")

    print()

    print(f"Ranking mismatches: {ranking_mismatches}")

    print(f"Candidate mismatches: {candidate_mismatches}")

    print()

    print("=== NUMPY METRICS ===")

    print(f"Hit@1:    {numpy_metrics.hit_at_1:.4f}")

    print(f"Hit@3:    {numpy_metrics.hit_at_3:.4f}")

    print(f"Recall@5: {numpy_metrics.recall_at_5:.4f}")

    print(f"MRR:      {numpy_metrics.mrr:.4f}")

    print()

    print("=== PGVECTOR METRICS ===")

    print(f"Hit@1:    {pgvector_metrics.hit_at_1:.4f}")

    print(f"Hit@3:    {pgvector_metrics.hit_at_3:.4f}")

    print(f"Recall@5: {pgvector_metrics.recall_at_5:.4f}")

    print(f"MRR:      {pgvector_metrics.mrr:.4f}")

    print()

    print(f"Metrics match: {metrics_match}")

    print()

    print("=== TIMING ===")

    print(f"Model load seconds: {model_seconds:.3f}")

    print(f"NumPy corpus embedding seconds: {corpus_seconds:.3f}")

    print(f"NumPy query mean ms: {numpy_query_seconds / len(cases) * 1000:.2f}")

    print(f"pgvector query mean ms: {pgvector_query_seconds / len(cases) * 1000:.2f}")

    print()

    parity_pass = (
        exact_matches == len(cases)
        and set_matches == len(cases)
        and score_tolerance_pass
        and metrics_match
    )

    print("PGVECTOR_DENSE_PARITY=" + ("PASS" if parity_pass else "FAIL"))

    if not parity_pass:
        raise RuntimeError("NumPy and pgvector dense retrieval are not equivalent")


if __name__ == "__main__":
    main()
