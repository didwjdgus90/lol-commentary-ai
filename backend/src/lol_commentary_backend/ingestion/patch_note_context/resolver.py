from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_context.models import (
    ContextConfidence,
    HotfixChampionContextHint,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
    ResolvedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)


def _locale_key(locale: str) -> str:
    return locale.replace("-", "_").casefold()


def load_hotfix_context_hints(path: Path) -> list[HotfixChampionContextHint]:
    hints: list[HotfixChampionContextHint] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                hint = HotfixChampionContextHint.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Hotfix context hint JSONL at line {line_number}"
                ) from exc

            hints.append(hint)

    return hints


def _index_hints(
    hints: list[HotfixChampionContextHint],
) -> dict[str, HotfixChampionContextHint]:
    by_record_id: dict[str, HotfixChampionContextHint] = {}
    seen_orders: set[int] = set()

    for hint in hints:
        if hint.record_id in by_record_id:
            raise ValueError(f"Duplicate Hotfix context hint record_id: {hint.record_id}")

        if hint.record_order in seen_orders:
            raise ValueError(f"Duplicate Hotfix context hint record_order: {hint.record_order}")

        by_record_id[hint.record_id] = hint
        seen_orders.add(hint.record_order)

    return by_record_id


def _validate_hint_against_record(
    hint: HotfixChampionContextHint,
    record: ResolvedPatchRecord,
) -> None:
    if hint.record_order != record.order:
        raise ValueError(f"Hotfix context hint order does not match record {record.record_id}")

    if hint.patch != record.patch:
        raise ValueError(f"Hotfix context hint patch does not match record {record.record_id}")

    if _locale_key(hint.locale) != _locale_key(record.locale):
        raise ValueError(f"Hotfix context hint locale does not match record {record.record_id}")

    if hint.source_sha256 != record.source_sha256:
        raise ValueError(
            f"Hotfix context hint source SHA-256 does not match record {record.record_id}"
        )

    if hint.detail_title != record.title:
        raise ValueError(
            f"Hotfix context hint detail title does not match record {record.record_id}"
        )

    if record.section_kind != "hotfix":
        raise ValueError(f"Hotfix context hint points to a non-hotfix record: {record.record_id}")

    if (
        record.resolution_method != ResolutionMethod.UNRESOLVED
        or record.entity_type != PatchEntityType.UNKNOWN
    ):
        raise ValueError("Hotfix context hint may only enrich an unresolved unknown record")

    if hint.context_confidence != ContextConfidence.HIGH:
        raise ValueError("Only high-confidence Hotfix context hints may resolve an entity")


def _validate_hint_against_catalog(
    hint: HotfixChampionContextHint,
    catalog: EntityCatalog,
) -> CatalogEntity:
    candidates = catalog.exact_candidates(
        hint.champion_name,
        entity_type=PatchEntityType.CHAMPION,
    )

    if len(candidates) != 1:
        raise ValueError("Hotfix context hint champion is not unique in the Data Dragon catalog")

    champion = candidates[0]

    if champion.entity_id != hint.champion_id:
        raise ValueError("Hotfix context hint champion_id does not match the Data Dragon catalog")

    if champion.entity_key != hint.champion_key:
        raise ValueError("Hotfix context hint champion_key does not match the Data Dragon catalog")

    return champion


def apply_hotfix_champion_context_hints(
    records: list[ResolvedPatchRecord],
    hints: list[HotfixChampionContextHint],
    catalog: EntityCatalog,
) -> list[ResolvedPatchRecord]:
    hints_by_record_id = _index_hints(hints)
    consumed_hint_ids: set[str] = set()
    output: list[ResolvedPatchRecord] = []

    for record in records:
        hint = hints_by_record_id.get(record.record_id)

        if hint is None:
            output.append(record)
            continue

        _validate_hint_against_record(hint, record)
        champion = _validate_hint_against_catalog(hint, catalog)

        payload = record.model_dump(mode="python")
        payload.update(
            {
                "entity_type": PatchEntityType.CHAMPION,
                "entity_name": champion.name,
                "entity_id": champion.entity_id,
                "entity_key": champion.entity_key,
                "resolution_method": ResolutionMethod.CONTEXT_EXACT,
                "resolution_candidate_count": 1,
                "context_method": hint.context_method.value,
                "context_confidence": hint.context_confidence.value,
                "context_anchor_title": hint.anchor_title,
                "context_evidence": list(hint.evidence),
                "entity_source_sha256": champion.source_sha256,
            }
        )

        output.append(ResolvedPatchRecord.model_validate(payload))
        consumed_hint_ids.add(hint.record_id)

    missing_hint_ids = set(hints_by_record_id) - consumed_hint_ids

    if missing_hint_ids:
        sample = sorted(missing_hint_ids)[0]
        raise ValueError(f"Hotfix context hint does not match any resolved record: {sample}")

    return output
