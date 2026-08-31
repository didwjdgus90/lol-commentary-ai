from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from lol_commentary_backend.retrieval.baseline.models import (
    RetrievalBaselineDecision,
)
from lol_commentary_backend.retrieval.runtime.pgvector_dense import (
    PgVectorDenseIndex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Smoke-test the PostgreSQL pgvector dense retrieval index."),
    )

    parser.add_argument(
        "--query",
        default=("Essence Reaver AD nerf hotfix"),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    baseline_path = (
        repository_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    index = PgVectorDenseIndex.build(
        corpus_sha256=(decision.corpus_sha256),
        requested_device=args.device,
    )

    try:
        started = perf_counter()

        results = index.search(
            args.query,
            top_n=args.top_n,
        )

        elapsed_ms = (perf_counter() - started) * 1000

        print("=== PGVECTOR DENSE SMOKE ===")

        print(f"Query: {args.query}")

        print(f"Corpus SHA256: {decision.corpus_sha256}")

        print("Embedding model: BAAI/bge-m3")

        print(f"Device: {index.device}")

        print(f"Top-N: {args.top_n}")

        print(f"Elapsed ms: {elapsed_ms:.2f}")

        print()

        for item in results:
            print(f"[{item.rank}] {item.chunk_id}")

            print(f"    score={item.score:.8f}")

        print()

        print("PGVECTOR_DENSE_SMOKE=PASS")

    finally:
        index.close()


if __name__ == "__main__":
    main()
