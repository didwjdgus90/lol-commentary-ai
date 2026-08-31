import argparse
from pathlib import Path

from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
from lol_commentary_backend.rag.context.formatter import (
    format_context_for_prompt,
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

        bundle = build_context_bundle(
            response,
            max_items=(args.max_context_items),
        )

        formatted = format_context_for_prompt(bundle)

        print("=== CONTEXT FORMATTER SMOKE ===")

        print(f"Evidence count: {bundle.selected_count}")

        print(f"Formatted chars: {len(formatted)}")

        print()

        print(formatted)

        print()

        required_citations = {evidence.citation_id for evidence in bundle.evidence}

        missing = [
            citation_id
            for citation_id in required_citations
            if (f'id="{citation_id}"' not in formatted)
        ]

        if missing:
            raise RuntimeError(f"Missing citations in formatted context: {missing}")

        print("CONTEXT_FORMATTER_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
