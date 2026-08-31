import re
from decimal import Decimal, InvalidOperation

from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
    ResolvedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)

SUMMONERS_RIFT_MAP_ID = "11"
_MIN_EVIDENCE_FIELDS = 2

_MAIN_ITEM_SECTION_TITLES = frozenset(
    {
        "신규 아이템",
        "복귀 아이템",
        "업데이트된 아이템",
    }
)

_PATCH_FIELD_LABELS = (
    ("base_mana_regen_percent", "기본 마나 재생"),
    ("omnivamp_percent", "모든 피해 흡혈"),
    ("ability_haste", "스킬 가속"),
    ("total_gold", "총가격"),
    ("total_gold", "총 가격"),
    ("attack_damage", "공격력"),
    ("ability_power", "주문력"),
    ("health", "체력"),
    ("mana", "마나"),
)

_NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?%?")


def _locale_key(locale: str) -> str:
    return locale.replace("-", "_").casefold()


def _target_map_id(
    record: NormalizedPatchRecord,
) -> str | None:
    if (
        record.entity_type == PatchEntityType.ITEM
        and record.heading_path
        and record.heading_path[0] in _MAIN_ITEM_SECTION_TITLES
    ):
        return SUMMONERS_RIFT_MAP_ID

    return None


def _map_compatible_candidates(
    candidates: tuple[CatalogEntity, ...],
    target_map_id: str,
) -> tuple[CatalogEntity, ...]:
    return tuple(candidate for candidate in candidates if target_map_id in candidate.map_ids)


def _number_string(value: str) -> str | None:
    token = value.replace(",", "").replace("%", "")

    try:
        number = Decimal(token)
    except InvalidOperation:
        return None

    normalized = number.normalize()

    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))

    return format(normalized, "f")


def _effective_change_value(rest: str) -> str | None:
    if "⇒" in rest:
        value_text = rest.rsplit("⇒", maxsplit=1)[1]
    elif "→" in rest:
        value_text = rest.rsplit("→", maxsplit=1)[1]
    else:
        value_text = rest

        if not _NUMBER_RE.match(value_text.lstrip()):
            return None

    match = _NUMBER_RE.search(value_text)

    if match is None:
        return None

    return _number_string(match.group(0))


def _patch_item_evidence(
    record: NormalizedPatchRecord,
) -> dict[str, str]:
    evidence: dict[str, str] = {}

    for raw_change in record.changes:
        change = raw_change.strip()

        for field_name, label in _PATCH_FIELD_LABELS:
            prefixes = (
                label,
                f"신규 {label}",
            )

            matched_prefix = next(
                (prefix for prefix in prefixes if change.startswith(prefix)),
                None,
            )

            if matched_prefix is None:
                continue

            rest = change[len(matched_prefix) :].lstrip()

            if rest.startswith(":"):
                rest = rest[1:].lstrip()

            value = _effective_change_value(rest)

            if value is not None:
                evidence[field_name] = value

            break

    return evidence


def _strict_evidence_matches(
    record: NormalizedPatchRecord,
    candidates: tuple[CatalogEntity, ...],
) -> tuple[
    tuple[CatalogEntity, tuple[str, ...]],
    ...,
]:
    patch_evidence = _patch_item_evidence(record)

    if len(patch_evidence) < _MIN_EVIDENCE_FIELDS:
        return ()

    matches: list[tuple[CatalogEntity, tuple[str, ...]]] = []

    required_fields = set(patch_evidence)

    for candidate in candidates:
        candidate_evidence = candidate.evidence_dict()

        if not required_fields.issubset(candidate_evidence):
            continue

        if not all(
            candidate_evidence[field_name] == expected_value
            for field_name, expected_value in patch_evidence.items()
        ):
            continue

        matches.append(
            (
                candidate,
                tuple(sorted(required_fields)),
            )
        )

    return tuple(matches)


def _is_removed_from_target_map(
    record: NormalizedPatchRecord,
) -> bool:
    return any("삭제되었습니다" in change for change in record.changes)


def _resolved_record(
    record: NormalizedPatchRecord,
    *,
    catalog: EntityCatalog,
    entity: CatalogEntity | None,
    method: ResolutionMethod,
    candidate_count: int,
    original_candidate_count: int | None = None,
    target_map_id: str | None = None,
    evidence_fields: tuple[str, ...] = (),
) -> ResolvedPatchRecord:
    payload = record.model_dump(mode="python")

    if entity is not None:
        payload["entity_type"] = entity.entity_type
        payload["entity_name"] = entity.name

    payload.update(
        {
            "ddragon_version": catalog.ddragon_version,
            "entity_id": (entity.entity_id if entity is not None else None),
            "entity_key": (entity.entity_key if entity is not None else None),
            "resolution_method": method,
            "resolution_candidate_count": candidate_count,
            "resolution_original_candidate_count": (original_candidate_count),
            "target_map_id": target_map_id,
            "resolution_evidence_fields": list(evidence_fields),
            "entity_source_sha256": (entity.source_sha256 if entity is not None else None),
        }
    )

    return ResolvedPatchRecord.model_validate(payload)


def _resolve_candidates(
    record: NormalizedPatchRecord,
    *,
    catalog: EntityCatalog,
    candidates: tuple[CatalogEntity, ...],
    exact_method: ResolutionMethod,
) -> ResolvedPatchRecord:
    original_count = len(candidates)

    if not candidates:
        return _resolved_record(
            record,
            catalog=catalog,
            entity=None,
            method=ResolutionMethod.UNRESOLVED,
            candidate_count=0,
            original_candidate_count=0,
        )

    target_map_id = _target_map_id(record)

    if target_map_id is not None:
        compatible = _map_compatible_candidates(
            candidates,
            target_map_id,
        )

        if len(compatible) == 1:
            return _resolved_record(
                record,
                catalog=catalog,
                entity=compatible[0],
                method=ResolutionMethod.MAP_EXACT,
                candidate_count=1,
                original_candidate_count=original_count,
                target_map_id=target_map_id,
            )

        if not compatible:
            method = (
                ResolutionMethod.REMOVED_FROM_TARGET_MAP
                if _is_removed_from_target_map(record)
                else ResolutionMethod.MAP_INCOMPATIBLE
            )

            return _resolved_record(
                record,
                catalog=catalog,
                entity=None,
                method=method,
                candidate_count=0,
                original_candidate_count=original_count,
                target_map_id=target_map_id,
            )

        evidence_matches = _strict_evidence_matches(
            record,
            compatible,
        )

        if len(evidence_matches) == 1:
            entity, evidence_fields = evidence_matches[0]

            return _resolved_record(
                record,
                catalog=catalog,
                entity=entity,
                method=ResolutionMethod.EVIDENCE_EXACT,
                candidate_count=1,
                original_candidate_count=original_count,
                target_map_id=target_map_id,
                evidence_fields=evidence_fields,
            )

        return _resolved_record(
            record,
            catalog=catalog,
            entity=None,
            method=ResolutionMethod.AMBIGUOUS,
            candidate_count=len(compatible),
            original_candidate_count=original_count,
            target_map_id=target_map_id,
        )

    if len(candidates) == 1:
        return _resolved_record(
            record,
            catalog=catalog,
            entity=candidates[0],
            method=exact_method,
            candidate_count=1,
            original_candidate_count=1,
        )

    return _resolved_record(
        record,
        catalog=catalog,
        entity=None,
        method=ResolutionMethod.AMBIGUOUS,
        candidate_count=original_count,
        original_candidate_count=original_count,
    )


def resolve_patch_record(
    record: NormalizedPatchRecord,
    catalog: EntityCatalog,
) -> ResolvedPatchRecord:
    if _locale_key(record.locale) != _locale_key(catalog.locale):
        raise ValueError("Patch record locale does not match Data Dragon catalog locale")

    if (
        record.entity_type
        in {
            PatchEntityType.CHAMPION,
            PatchEntityType.ITEM,
        }
        and record.entity_name is not None
    ):
        candidates = catalog.exact_candidates(
            record.entity_name,
            entity_type=record.entity_type,
        )

        return _resolve_candidates(
            record,
            catalog=catalog,
            candidates=candidates,
            exact_method=ResolutionMethod.BASELINE_EXACT,
        )

    candidates = catalog.exact_candidates(record.title)

    return _resolve_candidates(
        record,
        catalog=catalog,
        candidates=candidates,
        exact_method=ResolutionMethod.TITLE_EXACT,
    )


def resolve_patch_records(
    records: list[NormalizedPatchRecord],
    catalog: EntityCatalog,
) -> list[ResolvedPatchRecord]:
    return [resolve_patch_record(record, catalog) for record in records]
