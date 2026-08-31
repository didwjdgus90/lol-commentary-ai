from hashlib import sha256

from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
    PatchSection,
    SectionKind,
)


def _resolve_entity(
    section: PatchSection,
) -> tuple[PatchEntityType, str | None]:
    if section.kind == SectionKind.CHAMPION:
        return PatchEntityType.CHAMPION, section.entity_name

    if section.kind == SectionKind.ITEM and section.level in {3, 4}:
        return PatchEntityType.ITEM, section.title

    return PatchEntityType.UNKNOWN, None


def _build_content(section: PatchSection) -> str:
    lines = [" > ".join(section.heading_path)]

    lines.extend(section.paragraphs)
    lines.extend(change.raw_text for change in section.changes)

    return "\n".join(line for line in lines if line).strip()


def _should_emit(section: PatchSection) -> bool:
    return bool(section.paragraphs or section.changes)


def _record_id(
    *,
    source_sha256: str,
    order: int,
    heading_path: list[str],
) -> str:
    identity = f"{source_sha256}\n{order}\n{' > '.join(heading_path)}"
    return sha256(identity.encode("utf-8")).hexdigest()


def normalize_patch_note_document(
    document: PatchNoteDocument,
) -> list[NormalizedPatchRecord]:
    source_sha256 = document.source_sha256

    if source_sha256 is None:
        raise ValueError("PatchNoteDocument.source_sha256 is required for normalization")

    records: list[NormalizedPatchRecord] = []

    def visit(section: PatchSection) -> None:
        if _should_emit(section):
            order = len(records)
            entity_type, entity_name = _resolve_entity(section)

            records.append(
                NormalizedPatchRecord(
                    record_id=_record_id(
                        source_sha256=source_sha256,
                        order=order,
                        heading_path=section.heading_path,
                    ),
                    order=order,
                    patch=document.patch,
                    locale=document.locale,
                    source_url=document.source_url,
                    source_sha256=source_sha256,
                    section_kind=section.kind.value,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    heading_path=section.heading_path,
                    title=section.title,
                    paragraphs=section.paragraphs,
                    changes=[change.raw_text for change in section.changes],
                    content=_build_content(section),
                )
            )

        for child in section.children:
            visit(child)

    for section in document.sections:
        visit(section)

    return records
