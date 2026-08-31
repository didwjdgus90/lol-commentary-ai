from hashlib import sha256

import pytest

from lol_commentary_backend.rag.context.budget import (
    TOKEN_BUDGET_POLICY,
    apply_context_token_budget,
)
from lol_commentary_backend.rag.context.formatter import (
    format_context_for_prompt,
)
from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)
from lol_commentary_backend.rag.context.tokens import (
    TiktokenTextCounter,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalStrategy,
)


def _hash(
    value: str,
) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    *,
    citation_id: str,
    source_rank: int,
    text: str,
) -> ContextEvidence:
    return ContextEvidence(
        citation_id=citation_id,
        source_rank=source_rank,
        chunk_id=_hash(f"chunk:{citation_id}"),
        document_id=_hash(f"document:{citation_id}"),
        source_record_id=_hash(f"record:{citation_id}"),
        patch="26.1",
        locale="ko_kr",
        source_url=(f"https://example.com/{citation_id}"),
        section_kind="item",
        title="정수 약탈자",
        entity_name="정수 약탈자",
        heading_path=(
            "업데이트된 아이템",
            "정수 약탈자",
        ),
        text=text,
        retrieval_score=0.01,
        dense_rank=None,
        sparse_rank=source_rank,
    )


def _bundle(
    evidence: tuple[
        ContextEvidence,
        ...,
    ],
) -> ContextBundle:
    return ContextBundle(
        query=("Essence Reaver AD nerf"),
        expanded_query=("Essence Reaver AD nerf | 정수 약탈자"),
        requested_mode=(RetrievalMode.PRIMARY),
        strategy_used=(RetrievalStrategy.BGE_ALIAS_RRF),
        fallback_used=False,
        fallback_reason=None,
        retrieval_elapsed_ms=10.0,
        retrieval_top_k=max(
            len(evidence),
            1,
        ),
        source_top_n=10,
        selected_count=len(evidence),
        dropped_duplicate_count=0,
        dropped_limit_count=0,
        dropped_relevance_count=0,
        selection_policy=("entity_title_match_v1"),
        selection_fallback_used=False,
        evidence=evidence,
    )


class ConstantCounter:
    @property
    def name(self) -> str:
        return "constant"

    def count(
        self,
        text: str,
    ) -> int:
        return 100


class RankAwareCounter:
    @property
    def name(self) -> str:
        return "rank-aware"

    def count(
        self,
        text: str,
    ) -> int:
        if 'id="E3"' in text:
            return 300

        if 'id="E2"' in text:
            return 200

        if 'id="E1"' in text:
            return 100

        return 50


class MetadataHeavyCounter:
    @property
    def name(self) -> str:
        return "metadata-heavy"

    def count(
        self,
        text: str,
    ) -> int:
        return 500


def test_tiktoken_counter_counts_text() -> None:
    counter = TiktokenTextCounter()

    text = "정수 약탈자 공격력 55에서 50으로 감소"

    first = counter.count(text)
    second = counter.count(text)

    assert first > 0
    assert first == second
    assert counter.name == "o200k_base"


def test_keeps_all_evidence_when_budget_allows() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                text="first",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                text="second",
            ),
        )
    )

    result = apply_context_token_budget(
        bundle,
        token_counter=ConstantCounter(),
        max_tokens=100,
    )

    assert result.before_count == 2
    assert result.after_count == 2

    assert result.dropped_token_budget_count == 0


def test_stops_at_first_budget_overflow() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                text="first",
            ),
            _evidence(
                citation_id="E2",
                source_rank=2,
                text="second",
            ),
            _evidence(
                citation_id="E3",
                source_rank=3,
                text="third",
            ),
        )
    )

    result = apply_context_token_budget(
        bundle,
        token_counter=RankAwareCounter(),
        max_tokens=200,
    )

    assert result.after_count == 2

    assert result.dropped_token_budget_count == 1

    assert [item.source_rank for item in result.bundle.evidence] == [
        1,
        2,
    ]


def test_renumbers_citations_after_budgeting() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E3",
                source_rank=1,
                text="first",
            ),
            _evidence(
                citation_id="E8",
                source_rank=2,
                text="second",
            ),
        )
    )

    result = apply_context_token_budget(
        bundle,
        token_counter=ConstantCounter(),
        max_tokens=100,
    )

    assert [item.citation_id for item in result.bundle.evidence] == [
        "E1",
        "E2",
    ]


def test_can_return_empty_evidence() -> None:
    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                text="first",
            ),
        )
    )

    result = apply_context_token_budget(
        bundle,
        token_counter=RankAwareCounter(),
        max_tokens=50,
    )

    assert result.after_count == 0

    assert result.dropped_token_budget_count == 1


def test_rejects_non_positive_budget() -> None:
    with pytest.raises(
        ValueError,
        match="max_tokens must be positive",
    ):
        apply_context_token_budget(
            _bundle(()),
            token_counter=ConstantCounter(),
            max_tokens=0,
        )


def test_rejects_budget_smaller_than_metadata() -> None:
    with pytest.raises(
        ValueError,
        match=("Token budget is too small for retrieval context metadata"),
    ):
        apply_context_token_budget(
            _bundle(()),
            token_counter=(MetadataHeavyCounter()),
            max_tokens=100,
        )


def test_reported_used_tokens_match_formatted_bundle() -> None:
    counter = TiktokenTextCounter()

    bundle = _bundle(
        (
            _evidence(
                citation_id="E1",
                source_rank=1,
                text=("공격력 55 ⇒ 50"),
            ),
        )
    )

    result = apply_context_token_budget(
        bundle,
        token_counter=counter,
        max_tokens=4096,
    )

    formatted = format_context_for_prompt(result.bundle)

    assert result.used_tokens == counter.count(formatted)

    assert result.policy == TOKEN_BUDGET_POLICY
