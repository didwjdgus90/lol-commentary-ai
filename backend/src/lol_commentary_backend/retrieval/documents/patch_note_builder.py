import json
from hashlib import sha256

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
    ResolvedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.documents.models import (
    PatchRagDocument,
    RagSourceType,
)

PATCH_RAG_DOCUMENT_BUILDER_VERSION = "0.1.0"


def _canonical_record_bytes(
    record: ResolvedPatchRecord,
) -> bytes:
    payload = {
        "builder_version": PATCH_RAG_DOCUMENT_BUILDER_VERSION,
        "record": record.model_dump(mode="json"),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _document_id(
    record: ResolvedPatchRecord,
) -> str:
    return sha256(_canonical_record_bytes(record)).hexdigest()


def _render_heading(
    record: ResolvedPatchRecord,
) -> str:
    parts = [f"{record.patch} 패치"]

    if record.entity_name is not None:
        parts.append(record.entity_name)

    parts.append(record.title)

    return " | ".join(parts)


def _render_retrieval_text(
    record: ResolvedPatchRecord,
) -> str:
    blocks: list[str] = [
        _render_heading(record),
        " > ".join(record.heading_path),
    ]

    if record.paragraphs:
        blocks.append("\n".join(record.paragraphs))

    if record.changes:
        blocks.append("\n".join(f"- {change}" for change in record.changes))

    if not record.paragraphs and not record.changes:
        blocks.append(record.content)

    return "\n\n".join(block.strip() for block in blocks if block.strip())


def build_patch_rag_document(
    record: ResolvedPatchRecord,
) -> PatchRagDocument:
    retrieval_text = _render_retrieval_text(record)

    return PatchRagDocument(
        builder_version=PATCH_RAG_DOCUMENT_BUILDER_VERSION,
        document_id=_document_id(record),
        content_sha256=sha256(retrieval_text.encode("utf-8")).hexdigest(),
        source_type=RagSourceType.PATCH_NOTE,
        source_record_id=record.record_id,
        source_order=record.order,
        patch=record.patch,
        locale=record.locale,
        source_url=record.source_url,
        source_sha256=record.source_sha256,
        ddragon_version=record.ddragon_version,
        section_kind=record.section_kind,
        entity_type=record.entity_type,
        entity_name=record.entity_name,
        entity_id=record.entity_id,
        entity_key=record.entity_key,
        resolution_method=record.resolution_method,
        entity_resolved=(
            record.entity_id is not None
            and record.entity_type
            in {
                PatchEntityType.CHAMPION,
                PatchEntityType.ITEM,
            }
        ),
        removed_from_target_map=(
            record.resolution_method == ResolutionMethod.REMOVED_FROM_TARGET_MAP
        ),
        target_map_id=record.target_map_id,
        heading_path=list(record.heading_path),
        title=record.title,
        paragraphs=list(record.paragraphs),
        changes=list(record.changes),
        retrieval_text=retrieval_text,
        context_method=record.context_method,
        context_confidence=record.context_confidence,
        context_anchor_title=record.context_anchor_title,
    )


def build_patch_rag_documents(
    records: list[ResolvedPatchRecord],
) -> list[PatchRagDocument]:
    return [build_patch_rag_document(record) for record in records]
