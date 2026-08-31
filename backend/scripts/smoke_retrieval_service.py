import argparse
from pathlib import Path
from time import perf_counter

from lol_commentary_backend.retrieval.runtime.factory import (
    build_retrieval_service,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        default=("Essence Reaver AD nerf hotfix"),
    )

    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=(RetrievalMode.FALLBACK.value),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    mode = RetrievalMode(args.mode)

    startup_started = perf_counter()

    service = build_retrieval_service(
        repository_root=repository_root,
        enable_primary=(mode != RetrievalMode.FALLBACK),
        requested_device=args.device,
    )

    startup_ms = (perf_counter() - startup_started) * 1000

    try:
        response = service.retrieve(
            args.query,
            top_k=args.top_k,
            mode=mode,
        )

        print("=== RETRIEVAL SERVICE SMOKE ===")

        print(f"Query: {response.query}")

        print(f"Expanded query: {response.expanded_query}")

        print(f"Requested mode: {response.requested_mode.value}")

        print(f"Strategy: {response.strategy_used.value}")

        print(f"Primary available: {service.primary_available}")

        print(f"Fallback used: {response.fallback_used}")

        if response.fallback_reason:
            print(f"Fallback reason: {response.fallback_reason}")

        print(f"Startup ms: {startup_ms:.2f}")

        print(f"Retrieval ms: {response.elapsed_ms:.2f}")

        print()

        for hit in response.hits:
            print(f"[{hit.rank}] {hit.title}")

            print(
                f"    entity="
                f"{hit.entity_name} "
                f"score="
                f"{hit.score:.6f} "
                f"dense_rank="
                f"{hit.dense_rank} "
                f"sparse_rank="
                f"{hit.sparse_rank}"
            )

            print(
                "    "
                + hit.text[:180].replace(
                    "\n",
                    " ",
                )
            )

        print()

        print("RETRIEVAL_SERVICE_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
