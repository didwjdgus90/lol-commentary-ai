from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CITATION_VALIDATION_POLICY = "citation_integrity_v1"


class CitationValidationResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    policy: Literal["citation_integrity_v1"] = CITATION_VALIDATION_POLICY

    valid: bool

    citation_required: bool

    available_citation_ids: tuple[
        str,
        ...,
    ]

    cited_ids: tuple[
        str,
        ...,
    ]

    citation_occurrence_count: int = Field(ge=0)

    invalid_citation_ids: tuple[
        str,
        ...,
    ]

    malformed_citations: tuple[
        str,
        ...,
    ]

    missing_required_citation: bool
