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
from lol_commentary_backend.rag.output.citation_validator import (
    validate_rag_answer,
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
            top_k=5,
            mode=RetrievalMode.PRIMARY,
        )

        raw_bundle = build_context_bundle(
            retrieval,
            max_items=5,
        )

        selected_bundle = select_context_bundle(
            raw_bundle,
            max_items=5,
        )

        budget_result = apply_context_token_budget(
            selected_bundle,
            token_counter=(TiktokenTextCounter()),
            max_tokens=(DEFAULT_CONTEXT_TOKEN_BUDGET),
        )

        prompt = build_rag_prompt(
            query=args.query,
            bundle=budget_result.bundle,
        )

        valid_answer = (
            "정수 약탈자는 26.1 추가 패치에서 "
            "총가격이 2,900골드에서 "
            "3,050골드로 증가했고 공격력이 "
            "55에서 50으로 감소했습니다. [E1] "
            "기본 26.1 변경에서는 공격력이 "
            "60에서 55로 낮아졌습니다. [E2]"
        )

        valid_result = validate_rag_answer(
            answer_text=valid_answer,
            prompt=prompt,
        )

        fabricated_answer = "정수 약탈자의 승률이 10% 증가했습니다. [E9]"

        fabricated_result = validate_rag_answer(
            answer_text=(fabricated_answer),
            prompt=prompt,
        )

        print("=== CITATION VALIDATOR SMOKE ===")

        print(f"Available citations: {prompt.citation_ids}")

        print()

        print("=== VALID ANSWER ===")

        print(valid_answer)

        print(f"Valid: {valid_result.valid}")

        print(f"Cited IDs: {valid_result.cited_ids}")

        print(f"Invalid IDs: {valid_result.invalid_citation_ids}")

        print()

        print("=== FABRICATED CITATION PROBE ===")

        print(fabricated_answer)

        print(f"Valid: {fabricated_result.valid}")

        print(f"Invalid IDs: {fabricated_result.invalid_citation_ids}")

        if not valid_result.valid:
            raise RuntimeError("Expected valid answer to pass citation validation")

        if fabricated_result.valid:
            raise RuntimeError("Fabricated citation was not rejected")

        if fabricated_result.invalid_citation_ids != ("E9",):
            raise RuntimeError("Expected E9 to be reported as invalid")

        print()

        print("VALID_CITATION_PROBE=PASS")

        print("FABRICATED_CITATION_REJECTED=PASS")

        print("CITATION_VALIDATOR_SMOKE=PASS")

    finally:
        service.close()


if __name__ == "__main__":
    main()
