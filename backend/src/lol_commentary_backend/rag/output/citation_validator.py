import re

from lol_commentary_backend.rag.output.models import (
    CitationValidationResult,
)
from lol_commentary_backend.rag.prompt.models import (
    RagPromptPayload,
)

_VALID_CITATION_PATTERN = re.compile(r"\[(E[1-9][0-9]*)\]")

_CITATION_ID_PATTERN = re.compile(r"E[1-9][0-9]*")

_BRACKET_PATTERN = re.compile(r"\[([^\]\r\n]{1,40})\]")

_CITATION_LIKE_PATTERN = re.compile(r"(?i)^E(?:\d|\s)")


def _unique_in_order(
    values: list[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _validate_available_ids(
    citation_ids: tuple[str, ...],
) -> None:
    duplicates = len(citation_ids) != len(set(citation_ids))

    if duplicates:
        raise ValueError("available citation IDs must be unique")

    invalid = [
        citation_id
        for citation_id in citation_ids
        if (_CITATION_ID_PATTERN.fullmatch(citation_id) is None)
    ]

    if invalid:
        raise ValueError(f"Invalid available citation IDs: {invalid}")


def _find_malformed_citations(
    answer_text: str,
) -> tuple[str, ...]:
    malformed: list[str] = []

    for match in _BRACKET_PATTERN.finditer(answer_text):
        inner = match.group(1).strip()

        citation_like = _CITATION_LIKE_PATTERN.match(inner) is not None

        if not citation_like:
            continue

        if _CITATION_ID_PATTERN.fullmatch(inner) is None:
            malformed.append(match.group(0))

    return _unique_in_order(malformed)


def validate_answer_citations(
    *,
    answer_text: str,
    available_citation_ids: tuple[
        str,
        ...,
    ],
    require_citation: bool = True,
) -> CitationValidationResult:
    cleaned_answer = answer_text.strip()

    if not cleaned_answer:
        raise ValueError("answer_text must not be empty")

    _validate_available_ids(available_citation_ids)

    occurrences = [match.group(1) for match in (_VALID_CITATION_PATTERN.finditer(cleaned_answer))]

    cited_ids = _unique_in_order(occurrences)

    available = set(available_citation_ids)

    invalid_ids = _unique_in_order(
        [citation_id for citation_id in cited_ids if citation_id not in available]
    )

    malformed = _find_malformed_citations(cleaned_answer)

    citation_required = require_citation and bool(available_citation_ids)

    missing_required_citation = citation_required and not cited_ids

    valid = not invalid_ids and not malformed and not missing_required_citation

    return CitationValidationResult(
        valid=valid,
        citation_required=(citation_required),
        available_citation_ids=(available_citation_ids),
        cited_ids=cited_ids,
        citation_occurrence_count=len(occurrences),
        invalid_citation_ids=(invalid_ids),
        malformed_citations=malformed,
        missing_required_citation=(missing_required_citation),
    )


def validate_rag_answer(
    *,
    answer_text: str,
    prompt: RagPromptPayload,
    require_citation: bool = True,
) -> CitationValidationResult:
    return validate_answer_citations(
        answer_text=answer_text,
        available_citation_ids=(prompt.citation_ids),
        require_citation=(require_citation),
    )
