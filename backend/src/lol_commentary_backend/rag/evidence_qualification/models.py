from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.rag.query_planner.models import (
    RAGPriorityTier,
    RAGQueryIntent,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
)

EVIDENCE_QUALIFICATION_VERSION = "evidence_qualification_v1"

EVIDENCE_QUALIFICATION_POLICY_VERSION = "intent_scope_entity_exact_system_heading_v1"


class EvidenceQualificationStatus(StrEnum):
    QUALIFIED = "qualified"

    ABSTAIN_NO_QUALIFIED_EVIDENCE = "abstain_no_qualified_evidence"


class EvidenceQualificationReason(StrEnum):
    ENTITY_SECTION_MATCH = "entity_section_match"

    SYSTEM_HEADING_MATCH = "system_heading_match"

    NO_RETRIEVAL_HITS = "no_retrieval_hits"

    NO_ENTITY_SECTION_MATCH = "no_entity_section_match"

    NO_STRONG_SYSTEM_SCOPE = "no_strong_system_scope"


class QualifiedEvidenceHit(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    original_rank: int = Field(
        ge=1,
    )

    qualification_reason: EvidenceQualificationReason

    hit: RetrievalHit


class QualifiedQueryEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    qualification_version: Literal["evidence_qualification_v1"] = EVIDENCE_QUALIFICATION_VERSION

    policy_version: Literal["intent_scope_entity_exact_system_heading_v1"] = (
        EVIDENCE_QUALIFICATION_POLICY_VERSION
    )

    query_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    plan_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    intent: RAGQueryIntent

    query_text: str = Field(
        min_length=1,
    )

    subject_key: str = Field(
        min_length=1,
    )

    subject_id: str | None = None

    subject_name: str | None = None

    status: EvidenceQualificationStatus

    reason: EvidenceQualificationReason

    original_hit_count: int = Field(
        ge=0,
    )

    qualified_hits: tuple[
        QualifiedEvidenceHit,
        ...,
    ]

    first_qualified_rank: int | None = Field(
        default=None,
        ge=1,
    )

    @property
    def has_qualified_evidence(
        self,
    ) -> bool:
        return self.status == (EvidenceQualificationStatus.QUALIFIED)


class QualifiedPlanEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    plan_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    record_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    match_id: str = Field(
        min_length=1,
    )

    situation_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    priority_tier: RAGPriorityTier

    situation_kind: str = Field(
        min_length=1,
    )

    queries: tuple[
        QualifiedQueryEvidence,
        ...,
    ]


class QualifiedBatchEvidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    qualification_version: Literal["evidence_qualification_v1"] = EVIDENCE_QUALIFICATION_VERSION

    policy_version: Literal["intent_scope_entity_exact_system_heading_v1"] = (
        EVIDENCE_QUALIFICATION_POLICY_VERSION
    )

    patch: str = Field(
        pattern=r"^\d+\.\d+$",
    )

    plan_count: int = Field(
        ge=1,
    )

    query_count: int = Field(
        ge=0,
    )

    qualified_query_count: int = Field(
        ge=0,
    )

    abstained_query_count: int = Field(
        ge=0,
    )

    qualified_hit_count: int = Field(
        ge=0,
    )

    plans: tuple[
        QualifiedPlanEvidence,
        ...,
    ]
