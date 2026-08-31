from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)

SELECTION_POLICY = "entity_title_match_v1"


def _normalize(
    value: str,
) -> str:
    return " ".join(value.casefold().split())


def _query_variants(
    bundle: ContextBundle,
) -> tuple[str, ...]:
    raw_variants = [
        bundle.query,
        *bundle.expanded_query.split("|"),
    ]

    normalized: list[str] = []

    for raw in raw_variants:
        value = _normalize(raw)

        if value and value not in normalized:
            normalized.append(value)

    return tuple(normalized)


def _evidence_targets(
    evidence: ContextEvidence,
) -> tuple[str, ...]:
    raw_targets = [
        evidence.entity_name,
        evidence.title,
    ]

    if evidence.heading_path:
        raw_targets.append(evidence.heading_path[-1])

    normalized: list[str] = []

    for raw in raw_targets:
        if raw is None:
            continue

        value = _normalize(raw)

        if value and value not in normalized:
            normalized.append(value)

    return tuple(normalized)


def _is_direct_match(
    evidence: ContextEvidence,
    *,
    query_variants: tuple[str, ...],
) -> bool:
    targets = _evidence_targets(evidence)

    for target in targets:
        for query_variant in query_variants:
            if target == query_variant or target in query_variant or query_variant in target:
                return True

    return False


def _renumber_citations(
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


def select_context_bundle(
    bundle: ContextBundle,
    *,
    max_items: int | None = None,
) -> ContextBundle:
    if not bundle.evidence:
        return bundle.model_copy(
            update={
                "selection_policy": (SELECTION_POLICY),
            }
        )

    if max_items is None:
        max_items = len(bundle.evidence)

    if max_items <= 0:
        raise ValueError("max_items must be positive")

    query_variants = _query_variants(bundle)

    direct_matches = [
        evidence
        for evidence in bundle.evidence
        if _is_direct_match(
            evidence,
            query_variants=query_variants,
        )
    ]

    if direct_matches:
        candidate_pool = direct_matches
        fallback_used = False
        dropped_relevance_count = len(bundle.evidence) - len(direct_matches)
    else:
        candidate_pool = list(bundle.evidence)
        fallback_used = True
        dropped_relevance_count = 0

    selected = candidate_pool[:max_items]

    selector_limit_drops = max(
        0,
        len(candidate_pool) - len(selected),
    )

    renumbered = _renumber_citations(selected)

    return bundle.model_copy(
        update={
            "selected_count": len(renumbered),
            "dropped_limit_count": (bundle.dropped_limit_count + selector_limit_drops),
            "dropped_relevance_count": (dropped_relevance_count),
            "selection_policy": (SELECTION_POLICY),
            "selection_fallback_used": (fallback_used),
            "evidence": renumbered,
        }
    )
