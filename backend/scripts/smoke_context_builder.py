import argparse
from pathlib import Path

from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
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
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-context-items",
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

    service = build_retrieval_service(
        repository_root=(repository_root),
        enable_primary=True,
        requested_device=args.device,
    )

    try:
        response = service.retrieve(
            args.query,
            top_k=args.top_k,
            mode=RetrievalMode.PRIMARY,
        )

        context = build_context_bundle(
            response,
            max_items=(args.max_context_items),
        )

        print("=== CONTEXT BUILDER SMOKE ===")

        print(f"Query: {context.query}")

        print(f"Strategy: {context.strategy_used.value}")

        print(f"Selected: {context.selected_count}")

        print(f"Dropped duplicates: {context.dropped_duplicate_count}")

        print(f"Dropped by limit: {context.dropped_limit_count}")

        print()

        for evidence in context.evidence:
            print(f"[{evidence.citation_id}] source_rank={evidence.source_rank}")

            print(f"    title={evidence.title}")

            print(f"    entity={evidence.entity_name}")

            print(f"    chunk_id={evidence.chunk_id}")

            print(
                "    "
                + evidence.text[:220].replace(
                    "\n",
                    " ",
                )
            )

            print()

        print("CONTEXT_BUILDER_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
