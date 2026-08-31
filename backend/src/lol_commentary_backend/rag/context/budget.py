from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.rag.context.formatter import (
    format_context_for_prompt,
)
from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)
from lol_commentary_backend.rag.context.tokens import (
    TextTokenCounter,
)

TOKEN_BUDGET_POLICY = "rank_prefix_token_budget_v1"

DEFAULT_CONTEXT_TOKEN_BUDGET = 4096


class ContextBudgetResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    schema_version: Literal[1] = 1

    policy: Literal["rank_prefix_token_budget_v1"] = TOKEN_BUDGET_POLICY

    tokenizer_name: str = Field(min_length=1)

    max_tokens: int = Field(gt=0)

    base_tokens: int = Field(ge=0)

    used_tokens: int = Field(ge=0)

    before_count: int = Field(ge=0)

    after_count: int = Field(ge=0)

    dropped_token_budget_count: int = Field(ge=0)

    bundle: ContextBundle


def _renumber_evidence(
    evidence: list[ContextEvidence],
) -> tuple[ContextEvidence, ...]:
    return tuple(
        item.model_copy(
            update={
                "citation_id": f"E{index}",
            }
        )
        for index, item in enumerate(
            evidence,
            start=1,
        )
    )


def _with_evidence(
    bundle: ContextBundle,
    evidence: list[ContextEvidence],
) -> ContextBundle:
    renumbered = _renumber_evidence(evidence)

    return bundle.model_copy(
        update={
            "selected_count": len(renumbered),
            "evidence": renumbered,
        }
    )


def apply_context_token_budget(
    bundle: ContextBundle,
    *,
    token_counter: TextTokenCounter,
    max_tokens: int = (DEFAULT_CONTEXT_TOKEN_BUDGET),
) -> ContextBudgetResult:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    empty_bundle = _with_evidence(
        bundle,
        [],
    )

    base_text = format_context_for_prompt(empty_bundle)

    base_tokens = token_counter.count(base_text)

    if base_tokens > max_tokens:
        raise ValueError("Token budget is too small for retrieval context metadata")

    selected: list[ContextEvidence] = []

    for evidence in bundle.evidence:
        candidate = [
            *selected,
            evidence,
        ]

        candidate_bundle = _with_evidence(
            bundle,
            candidate,
        )

        candidate_text = format_context_for_prompt(candidate_bundle)

        candidate_tokens = token_counter.count(candidate_text)

        if candidate_tokens > max_tokens:
            break

        selected.append(evidence)

    final_bundle = _with_evidence(
        bundle,
        selected,
    )

    final_text = format_context_for_prompt(final_bundle)

    used_tokens = token_counter.count(final_text)

    return ContextBudgetResult(
        tokenizer_name=(token_counter.name),
        max_tokens=max_tokens,
        base_tokens=base_tokens,
        used_tokens=used_tokens,
        before_count=len(bundle.evidence),
        after_count=len(final_bundle.evidence),
        dropped_token_budget_count=(len(bundle.evidence) - len(final_bundle.evidence)),
        bundle=final_bundle,
    )
