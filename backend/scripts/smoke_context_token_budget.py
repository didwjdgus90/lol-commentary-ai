import argparse
from pathlib import Path

from lol_commentary_backend.rag.context.budget import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    apply_context_token_budget,
)
from lol_commentary_backend.rag.context.builder import (
    build_context_bundle,
)
from lol_commentary_backend.rag.context.formatter import (
    format_context_for_prompt,
)
from lol_commentary_backend.rag.context.selector import (
    select_context_bundle,
)
from lol_commentary_backend.rag.context.tokens import (
    TiktokenTextCounter,
)
from lol_commentary_backend.retrieval.runtime.factory import (
    build_retrieval_service,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
)

TARGET_GENERATION_MODEL = "gpt-5.6-terra"


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

        counter = TiktokenTextCounter()

        budget_result = apply_context_token_budget(
            selected_bundle,
            token_counter=counter,
            max_tokens=(args.max_context_tokens),
        )

        formatted = format_context_for_prompt(budget_result.bundle)

        verified_tokens = counter.count(formatted)

        print("=== CONTEXT TOKEN BUDGET SMOKE ===")

        print(f"Target generation model: {TARGET_GENERATION_MODEL}")

        print(f"Local token encoding: {budget_result.tokenizer_name}")

        print(f"Token budget policy: {budget_result.policy}")

        print(f"Max context tokens: {budget_result.max_tokens}")

        print(f"Metadata-only tokens: {budget_result.base_tokens}")

        print(f"Used context tokens: {budget_result.used_tokens}")

        print(f"Before token budget: {budget_result.before_count}")

        print(f"After token budget: {budget_result.after_count}")

        print(f"Dropped by token budget: {budget_result.dropped_token_budget_count}")

        print(f"Formatted chars: {len(formatted)}")

        print()

        for evidence in budget_result.bundle.evidence:
            print(f"[{evidence.citation_id}] source_rank={evidence.source_rank}")

            print(f"    title={evidence.title}")

            print(f"    entity={evidence.entity_name}")

            print()

        if verified_tokens != budget_result.used_tokens:
            raise RuntimeError("Final token count mismatch")

        if budget_result.used_tokens > budget_result.max_tokens:
            raise RuntimeError("Context token budget exceeded")

        print("CONTEXT_TOKEN_BUDGET_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
