import re

from lol_commentary_backend.ingestion.patch_note_context.models import (
    ContextConfidence,
    ContextMethod,
    HotfixChampionContextHint,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
    PatchSection,
    SectionKind,
)

_ABILITY_TITLE_RE = re.compile(r"^[PQWER]\s*-\s*\S.+$")

_CONTEXT_EVIDENCE = [
    "h3_exact_champion_anchor",
    "h4_champion_detail_pattern",
]


def _has_record_content(section: PatchSection) -> bool:
    return bool(section.paragraphs or section.changes)


def _is_champion_detail_title(title: str) -> bool:
    if title == "기본 능력치":
        return True

    if title.startswith("기본 지속 효과"):
        return True

    return _ABILITY_TITLE_RE.match(title) is not None


def _unique_champion_anchor(
    section: PatchSection,
    catalog: EntityCatalog,
) -> CatalogEntity | None:
    if section.level != 3:
        return None

    candidates = catalog.exact_candidates(
        section.title,
        entity_type=PatchEntityType.CHAMPION,
    )

    if len(candidates) != 1:
        return None

    return candidates[0]


def _is_item_boundary(
    section: PatchSection,
    catalog: EntityCatalog,
) -> bool:
    if section.level != 4:
        return False

    candidates = catalog.exact_candidates(
        section.title,
        entity_type=PatchEntityType.ITEM,
    )

    return bool(candidates)


def _validate_normalized_records(
    document: PatchNoteDocument,
    records: list[NormalizedPatchRecord],
) -> dict[int, NormalizedPatchRecord]:
    source_sha256 = document.source_sha256

    if source_sha256 is None:
        raise ValueError("PatchNoteDocument.source_sha256 is required")

    records_by_order: dict[int, NormalizedPatchRecord] = {}

    for expected_order, record in enumerate(records):
        if record.order != expected_order:
            raise ValueError("Normalized records must be contiguous and ordered from zero")

        if record.source_sha256 != source_sha256:
            raise ValueError("Normalized record source SHA-256 does not match PatchNoteDocument")

        records_by_order[record.order] = record

    return records_by_order


def build_hotfix_champion_context_hints(
    document: PatchNoteDocument,
    records: list[NormalizedPatchRecord],
    catalog: EntityCatalog,
) -> list[HotfixChampionContextHint]:
    records_by_order = _validate_normalized_records(
        document,
        records,
    )

    source_sha256 = document.source_sha256

    if source_sha256 is None:
        raise ValueError("PatchNoteDocument.source_sha256 is required")

    hints: list[HotfixChampionContextHint] = []
    record_order = 0

    def visit(
        section: PatchSection,
        *,
        hotfix_mode: bool,
        active_champion: CatalogEntity | None,
        anchor_title: str | None,
    ) -> tuple[
        CatalogEntity | None,
        str | None,
    ]:
        nonlocal record_order

        if hotfix_mode:
            if section.level == 3:
                champion = _unique_champion_anchor(
                    section,
                    catalog,
                )

                if champion is not None:
                    active_champion = champion
                    anchor_title = section.title
                else:
                    active_champion = None
                    anchor_title = None

            elif section.level == 4:
                if _is_item_boundary(section, catalog):
                    active_champion = None
                    anchor_title = None
                elif active_champion is not None and _is_champion_detail_title(section.title):
                    pass
                else:
                    active_champion = None
                    anchor_title = None

        current_record: NormalizedPatchRecord | None = None

        if _has_record_content(section):
            current_record = records_by_order.get(record_order)

            if current_record is None:
                raise ValueError("Parsed document produced more records than normalized.jsonl")

            if current_record.title != section.title:
                raise ValueError(
                    "Parsed/normalized record title mismatch "
                    f"at order {record_order}: "
                    f"{section.title!r} != "
                    f"{current_record.title!r}"
                )

            if current_record.heading_path != section.heading_path:
                raise ValueError(f"Parsed/normalized heading path mismatch at order {record_order}")

            if (
                hotfix_mode
                and section.level == 4
                and active_champion is not None
                and anchor_title is not None
                and _is_champion_detail_title(section.title)
            ):
                if active_champion.entity_key is None:
                    raise ValueError("Champion catalog entity key is missing")

                hints.append(
                    HotfixChampionContextHint(
                        record_id=current_record.record_id,
                        record_order=current_record.order,
                        patch=document.patch,
                        locale=document.locale,
                        source_sha256=source_sha256,
                        champion_name=active_champion.name,
                        champion_id=active_champion.entity_id,
                        champion_key=active_champion.entity_key,
                        anchor_title=anchor_title,
                        detail_title=section.title,
                        context_method=(ContextMethod.HOTFIX_STRUCTURAL_EXACT),
                        context_confidence=(ContextConfidence.HIGH),
                        evidence=_CONTEXT_EVIDENCE,
                    )
                )

            record_order += 1

        child_active = active_champion
        child_anchor = anchor_title

        for child in section.children:
            child_active, child_anchor = visit(
                child,
                hotfix_mode=hotfix_mode,
                active_champion=child_active,
                anchor_title=child_anchor,
            )

        return active_champion, anchor_title

    for top_level in document.sections:
        hotfix_mode = top_level.kind == SectionKind.HOTFIX

        active_champion: CatalogEntity | None = None
        anchor_title: str | None = None

        if _has_record_content(top_level):
            current_record = records_by_order.get(record_order)

            if current_record is None:
                raise ValueError("Parsed document produced more records than normalized.jsonl")

            if current_record.title != top_level.title:
                raise ValueError(f"Parsed/normalized record title mismatch at order {record_order}")

            record_order += 1

        for child in top_level.children:
            active_champion, anchor_title = visit(
                child,
                hotfix_mode=hotfix_mode,
                active_champion=active_champion,
                anchor_title=anchor_title,
            )

    if record_order != len(records):
        raise ValueError(
            "Normalized.jsonl contains records that were not reproduced from PatchNoteDocument"
        )

    return hints
