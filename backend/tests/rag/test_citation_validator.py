import pytest

from lol_commentary_backend.rag.output.citation_validator import (
    validate_answer_citations,
    validate_rag_answer,
)
from lol_commentary_backend.rag.output.models import (
    CITATION_VALIDATION_POLICY,
)
from lol_commentary_backend.rag.prompt.models import (
    RagPromptPayload,
)


def _prompt(
    citation_ids: tuple[str, ...] = (
        "E1",
        "E2",
    ),
) -> RagPromptPayload:
    return RagPromptPayload(
        instructions="trusted instructions",
        input_text="untrusted input",
        evidence_count=len(citation_ids),
        citation_ids=citation_ids,
    )


def test_valid_single_citation() -> None:
    result = validate_rag_answer(
        answer_text=("공격력이 55에서 50으로 감소했습니다. [E1]"),
        prompt=_prompt(),
    )

    assert result.valid is True

    assert result.cited_ids == ("E1",)

    assert result.invalid_citation_ids == ()


def test_valid_multiple_citations() -> None:
    result = validate_rag_answer(
        answer_text=("기본 변경은 [E2]이고 추가 하향은 [E1]입니다."),
        prompt=_prompt(),
    )

    assert result.valid is True

    assert result.cited_ids == (
        "E2",
        "E1",
    )

    assert result.citation_occurrence_count == 2


def test_duplicate_citations_are_counted_but_deduplicated() -> None:
    result = validate_rag_answer(
        answer_text=("변경 A [E1] 변경 B [E1]"),
        prompt=_prompt(),
    )

    assert result.valid is True

    assert result.cited_ids == ("E1",)

    assert result.citation_occurrence_count == 2


def test_rejects_nonexistent_citation() -> None:
    result = validate_rag_answer(
        answer_text=("승률이 올랐습니다. [E9]"),
        prompt=_prompt(),
    )

    assert result.valid is False

    assert result.invalid_citation_ids == ("E9",)


def test_rejects_malformed_citation() -> None:
    result = validate_rag_answer(
        answer_text=("공격력이 감소했습니다. [E 1]"),
        prompt=_prompt(),
    )

    assert result.valid is False

    assert result.malformed_citations == ("[E 1]",)


def test_requires_citation_when_evidence_exists() -> None:
    result = validate_rag_answer(
        answer_text=("공격력이 감소했습니다."),
        prompt=_prompt(),
    )

    assert result.valid is False

    assert result.missing_required_citation is True


def test_can_disable_required_citation_policy() -> None:
    result = validate_rag_answer(
        answer_text=("공격력이 감소했습니다."),
        prompt=_prompt(),
        require_citation=False,
    )

    assert result.valid is True

    assert result.citation_required is False


def test_empty_evidence_does_not_require_citation() -> None:
    result = validate_rag_answer(
        answer_text=("공식 근거가 부족합니다."),
        prompt=_prompt(()),
    )

    assert result.valid is True

    assert result.citation_required is False


def test_rejects_fabricated_citation_when_no_evidence_exists() -> None:
    result = validate_rag_answer(
        answer_text=("공식 근거가 없습니다. [E1]"),
        prompt=_prompt(()),
    )

    assert result.valid is False

    assert result.invalid_citation_ids == ("E1",)


def test_rejects_empty_answer() -> None:
    with pytest.raises(
        ValueError,
        match=("answer_text must not be empty"),
    ):
        validate_rag_answer(
            answer_text="   ",
            prompt=_prompt(),
        )


def test_rejects_duplicate_available_ids() -> None:
    with pytest.raises(
        ValueError,
        match=("available citation IDs must be unique"),
    ):
        validate_answer_citations(
            answer_text="[E1]",
            available_citation_ids=(
                "E1",
                "E1",
            ),
        )


def test_reports_policy_version() -> None:
    result = validate_rag_answer(
        answer_text="변경됐습니다. [E1]",
        prompt=_prompt(),
    )

    assert result.policy == CITATION_VALIDATION_POLICY
