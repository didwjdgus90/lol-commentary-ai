from __future__ import annotations

import json
import unicodedata

from lol_commentary_backend.rag.evidence_qualification.models import (
    EvidenceQualificationReason,
    EvidenceQualificationStatus,
    QualifiedBatchEvidence,
    QualifiedEvidenceHit,
    QualifiedPlanEvidence,
    QualifiedQueryEvidence,
)
from lol_commentary_backend.rag.query_planner.models import (
    RAGQueryIntent,
)
from lol_commentary_backend.rag.retrieval_execution.models import (
    RAGBatchEvidenceResult,
    RAGQueryEvidenceResult,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
)

ENTITY_SECTION_BY_INTENT = {
    RAGQueryIntent.CHAMPION_PATCH: ("champion"),
    RAGQueryIntent.ITEM_PATCH: ("item"),
}


SYSTEM_STRONG_TERMS = {
    RAGQueryIntent.OBJECTIVE_PATCH: (
        "드래곤",
        "바론",
        "내셔",
        "전령",
        "dragon",
        "baron",
        "herald",
    ),
    RAGQueryIntent.STRUCTURE_PATCH: (
        "포탑",
        "억제기",
        "turret",
        "inhibitor",
    ),
}


ALLOWED_SYSTEM_SECTIONS = {
    "general",
    "system",
    "objective",
    "structure",
}


def _normalize(
    value: object,
) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    return " ".join(text.casefold().split())


def _expanded_aliases(
    result: RAGQueryEvidenceResult,
) -> tuple[
    str,
    ...,
]:
    aliases: list[str] = []

    if result.subject_name:
        aliases.append(result.subject_name)

    expanded_parts = [part.strip() for part in result.expanded_query.split("|")]

    for part in expanded_parts[1:]:
        if part:
            aliases.append(part)

    normalized_seen: set[str] = set()

    unique: list[str] = []

    for alias in aliases:
        normalized = _normalize(alias)

        if not normalized:
            continue

        if normalized in (normalized_seen):
            continue

        normalized_seen.add(normalized)

        unique.append(alias)

    return tuple(unique)


def _exact_alias_match(
    *,
    value: object,
    aliases: tuple[
        str,
        ...,
    ],
) -> bool:
    normalized_value = _normalize(value)

    if not normalized_value:
        return False

    return any(normalized_value == _normalize(alias) for alias in aliases)


def _entity_hit_qualifies(
    *,
    result: RAGQueryEvidenceResult,
    hit: RetrievalHit,
) -> bool:
    expected_section = ENTITY_SECTION_BY_INTENT[result.intent]

    if _normalize(hit.section_kind) != expected_section:
        return False

    aliases = _expanded_aliases(result)

    if not aliases:
        return False

    return _exact_alias_match(
        value=hit.entity_name,
        aliases=aliases,
    )


def _system_heading_text(
    hit: RetrievalHit,
) -> str:
    return json.dumps(
        (
            hit.title,
            hit.entity_name,
            hit.heading_path,
        ),
        ensure_ascii=False,
    )


def _contains_any_term(
    *,
    text: str,
    terms: tuple[
        str,
        ...,
    ],
) -> bool:
    normalized_text = _normalize(text)

    return any(_normalize(term) in normalized_text for term in terms)


def _system_hit_qualifies(
    *,
    result: RAGQueryEvidenceResult,
    hit: RetrievalHit,
) -> bool:
    section = _normalize(hit.section_kind)

    if section not in ALLOWED_SYSTEM_SECTIONS:
        return False

    terms = SYSTEM_STRONG_TERMS[result.intent]

    return _contains_any_term(
        text=_system_heading_text(hit),
        terms=terms,
    )


def _qualification_reason(
    *,
    result: RAGQueryEvidenceResult,
) -> EvidenceQualificationReason:
    if not result.hits:
        return EvidenceQualificationReason.NO_RETRIEVAL_HITS

    if result.intent in ENTITY_SECTION_BY_INTENT:
        return EvidenceQualificationReason.NO_ENTITY_SECTION_MATCH

    return EvidenceQualificationReason.NO_STRONG_SYSTEM_SCOPE


def qualify_query_evidence(
    result: RAGQueryEvidenceResult,
) -> QualifiedQueryEvidence:
    qualified_hits: list[QualifiedEvidenceHit] = []

    if result.intent in ENTITY_SECTION_BY_INTENT:
        success_reason = EvidenceQualificationReason.ENTITY_SECTION_MATCH

        for hit in result.hits:
            if _entity_hit_qualifies(
                result=result,
                hit=hit,
            ):
                qualified_hits.append(
                    QualifiedEvidenceHit(
                        original_rank=hit.rank,
                        qualification_reason=(success_reason),
                        hit=hit.model_copy(deep=True),
                    )
                )

    elif result.intent in SYSTEM_STRONG_TERMS:
        success_reason = EvidenceQualificationReason.SYSTEM_HEADING_MATCH

        for hit in result.hits:
            if _system_hit_qualifies(
                result=result,
                hit=hit,
            ):
                qualified_hits.append(
                    QualifiedEvidenceHit(
                        original_rank=hit.rank,
                        qualification_reason=(success_reason),
                        hit=hit.model_copy(deep=True),
                    )
                )

    else:
        raise ValueError(
            f"Unsupported RAG query intent for evidence qualification: {result.intent}"
        )

    if qualified_hits:
        status = EvidenceQualificationStatus.QUALIFIED

        reason = qualified_hits[0].qualification_reason

        first_qualified_rank = min(row.original_rank for row in qualified_hits)

    else:
        status = EvidenceQualificationStatus.ABSTAIN_NO_QUALIFIED_EVIDENCE

        reason = _qualification_reason(result=result)

        first_qualified_rank = None

    return QualifiedQueryEvidence(
        query_id=result.query_id,
        plan_id=result.plan_id,
        record_id=result.record_id,
        match_id=result.match_id,
        situation_id=(result.situation_id),
        patch=result.patch,
        intent=result.intent,
        query_text=result.query_text,
        subject_key=(result.subject_key),
        subject_id=result.subject_id,
        subject_name=(result.subject_name),
        status=status,
        reason=reason,
        original_hit_count=len(result.hits),
        qualified_hits=tuple(qualified_hits),
        first_qualified_rank=(first_qualified_rank),
    )


def qualify_batch_evidence(
    batch: RAGBatchEvidenceResult,
) -> QualifiedBatchEvidence:
    plan_results: list[QualifiedPlanEvidence] = []

    query_count = 0

    qualified_query_count = 0

    abstained_query_count = 0

    qualified_hit_count = 0

    for plan in batch.plans:
        qualified_queries: list[QualifiedQueryEvidence] = []

        for query_result in plan.query_results:
            qualified = qualify_query_evidence(query_result)

            query_count += 1

            qualified_hit_count += len(qualified.qualified_hits)

            if qualified.has_qualified_evidence:
                qualified_query_count += 1

            else:
                abstained_query_count += 1

            qualified_queries.append(qualified)

        plan_results.append(
            QualifiedPlanEvidence(
                plan_id=plan.plan_id,
                record_id=(plan.record_id),
                match_id=(plan.match_id),
                situation_id=(plan.situation_id),
                patch=plan.patch,
                priority_tier=(plan.priority_tier),
                situation_kind=(plan.situation_kind),
                queries=tuple(qualified_queries),
            )
        )

    if qualified_query_count + abstained_query_count != query_count:
        raise RuntimeError("Evidence qualification accounting mismatch")

    return QualifiedBatchEvidence(
        patch=batch.patch,
        plan_count=(batch.plan_count),
        query_count=query_count,
        qualified_query_count=(qualified_query_count),
        abstained_query_count=(abstained_query_count),
        qualified_hit_count=(qualified_hit_count),
        plans=tuple(plan_results),
    )
