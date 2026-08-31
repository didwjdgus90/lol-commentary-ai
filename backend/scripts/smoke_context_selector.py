import argparse
from pathlib import Path

from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
from lol_commentary_backend.rag.context.formatter import (
    format_context_for_prompt,
)
from lol_commentary_backend.rag.context.selector import (
    select_context_bundle,
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
        repository_root=repository_root,
        enable_primary=True,
        requested_device=args.device,
    )

    try:
        response = service.retrieve(
            args.query,
            top_k=args.top_k,
            mode=RetrievalMode.PRIMARY,
        )

        raw_bundle = build_context_bundle(
            response,
            max_items=args.top_k,
        )

        selected_bundle = select_context_bundle(
            raw_bundle,
            max_items=(args.max_context_items),
        )

        formatted = format_context_for_prompt(selected_bundle)

        print("=== CONTEXT SELECTOR SMOKE ===")

        print(f"Query: {selected_bundle.query}")

        print(f"Selection policy: {selected_bundle.selection_policy}")

        print(f"Selection fallback: {selected_bundle.selection_fallback_used}")

        print(f"Before selection: {raw_bundle.selected_count}")

        print(f"After selection: {selected_bundle.selected_count}")

        print(f"Dropped by relevance: {selected_bundle.dropped_relevance_count}")

        print()

        for evidence in selected_bundle.evidence:
            print(f"[{evidence.citation_id}] source_rank={evidence.source_rank}")

            print(f"    title={evidence.title}")

            print(f"    entity={evidence.entity_name}")

            print()

        print(f"Formatted chars: {len(formatted)}")

        if selected_bundle.selected_count <= 0:
            raise RuntimeError("Selector returned no evidence")

        print("CONTEXT_SELECTOR_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
