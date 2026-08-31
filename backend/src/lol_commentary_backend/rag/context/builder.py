from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
    RetrievalResponse,
)


def _normalize_text_for_dedup(
    text: str,
) -> str:
    return " ".join(text.split()).casefold()


def _build_evidence(
    hit: RetrievalHit,
    *,
    citation_number: int,
) -> ContextEvidence:
    return ContextEvidence(
        citation_id=(f"E{citation_number}"),
        source_rank=hit.rank,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        source_record_id=(hit.source_record_id),
        patch=hit.patch,
        locale=hit.locale,
        source_url=hit.source_url,
        section_kind=(hit.section_kind),
        title=hit.title,
        entity_name=hit.entity_name,
        heading_path=tuple(hit.heading_path),
        text=hit.text,
        retrieval_score=hit.score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
    )


def build_context_bundle(
    response: RetrievalResponse,
    *,
    max_items: int | None = None,
) -> ContextBundle:
    if max_items is None:
        max_items = response.top_k

    if max_items <= 0:
        raise ValueError("max_items must be positive")

    selected: list[ContextEvidence] = []

    seen_chunk_ids: set[str] = set()
    seen_texts: set[str] = set()

    dropped_duplicate_count = 0
    dropped_limit_count = 0

    for hit in response.hits:
        normalized_text = _normalize_text_for_dedup(hit.text)

        duplicate = hit.chunk_id in seen_chunk_ids or normalized_text in seen_texts

        if duplicate:
            dropped_duplicate_count += 1
            continue

        seen_chunk_ids.add(hit.chunk_id)

        seen_texts.add(normalized_text)

        if len(selected) >= max_items:
            dropped_limit_count += 1
            continue

        selected.append(
            _build_evidence(
                hit,
                citation_number=(len(selected) + 1),
            )
        )

    return ContextBundle(
        query=response.query,
        expanded_query=(response.expanded_query),
        requested_mode=(response.requested_mode),
        strategy_used=(response.strategy_used),
        fallback_used=(response.fallback_used),
        fallback_reason=(response.fallback_reason),
        retrieval_elapsed_ms=(response.elapsed_ms),
        retrieval_top_k=(response.top_k),
        source_top_n=(response.source_top_n),
        selected_count=len(selected),
        dropped_duplicate_count=(dropped_duplicate_count),
        dropped_limit_count=(dropped_limit_count),
        evidence=tuple(selected),
    )
