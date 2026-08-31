import argparse
from pathlib import Path

from lol_commentary_backend.rag.context.budget import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    apply_context_token_budget,
)
from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
from lol_commentary_backend.rag.context.selector import (
    select_context_bundle,
)
from lol_commentary_backend.rag.context.tokens import (
    TiktokenTextCounter,
)
from lol_commentary_backend.rag.prompt.contract import (
    build_rag_prompt,
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
        "--max-context-tokens",
        type=int,
        default=(DEFAULT_CONTEXT_TOKEN_BUDGET),
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
        retrieval = service.retrieve(
            args.query,
            top_k=args.top_k,
            mode=RetrievalMode.PRIMARY,
        )

        raw_bundle = build_context_bundle(
            retrieval,
            max_items=args.top_k,
        )

        selected_bundle = select_context_bundle(
            raw_bundle,
            max_items=(args.max_context_items),
        )

        counter = TiktokenTextCounter()

        budget_result = apply_context_token_budget(
            selected_bundle,
            token_counter=counter,
            max_tokens=(args.max_context_tokens),
        )

        payload = build_rag_prompt(
            query=args.query,
            bundle=budget_result.bundle,
        )

        print("=== PROMPT CONTRACT SMOKE ===")

        print(f"Prompt version: {payload.prompt_version}")

        print(f"Evidence count: {payload.evidence_count}")

        print(f"Citation IDs: {payload.citation_ids}")

        print(f"Instructions chars: {len(payload.instructions)}")

        print(f"Input chars: {len(payload.input_text)}")

        print()

        print("=== TRUSTED INSTRUCTIONS ===")

        print(payload.instructions)

        print()

        print("=== UNTRUSTED INPUT ===")

        print(payload.input_text)

        if payload.evidence_count != len(payload.citation_ids):
            raise RuntimeError("Evidence/citation count mismatch")

        for citation_id in payload.citation_ids:
            if f'id="{citation_id}"' not in payload.input_text:
                raise RuntimeError(f"Citation missing from prompt input: {citation_id}")

        print()

        print("PROMPT_CONTRACT_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
