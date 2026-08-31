from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_context.models import (
    ContextConfidence,
    ContextMethod,
    HotfixChampionContextHint,
)
from lol_commentary_backend.ingestion.patch_note_context.resolver import (
    apply_hotfix_champion_context_hints,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.resolver import (
    resolve_patch_record,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)

SOURCE_SHA = sha256(b"patch-source").hexdigest()
CHAMPION_SHA = sha256(b"champion-source").hexdigest()


def _catalog() -> EntityCatalog:
    champion = CatalogEntity(
        entity_type=PatchEntityType.CHAMPION,
        entity_id="Tryndamere",
        entity_key="23",
        name="트린다미어",
        source_sha256=CHAMPION_SHA,
    )
    return EntityCatalog(
        ddragon_version="16.1.1",
        locale="ko_KR",
        champions_by_name={"트린다미어": (champion,)},
        items_by_name={},
    )


def _normalized_record() -> NormalizedPatchRecord:
    title = "E - 회전 베기"
    return NormalizedPatchRecord(
        record_id=sha256(b"hotfix-tryndamere-e").hexdigest(),
        order=10,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        section_kind="hotfix",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        heading_path=["추가 패치 노트", title],
        title=title,
        changes=["기본 피해량 : 70/105/140/175/210 ⇒ 80/120/160/200/240"],
        content=title,
    )


def _hint(
    record: NormalizedPatchRecord,
) -> HotfixChampionContextHint:
    return HotfixChampionContextHint(
        record_id=record.record_id,
        record_order=record.order,
        patch=record.patch,
        locale=record.locale,
        source_sha256=record.source_sha256,
        champion_name="트린다미어",
        champion_id="Tryndamere",
        champion_key="23",
        anchor_title="트린다미어",
        detail_title=record.title,
        context_method=ContextMethod.HOTFIX_STRUCTURAL_EXACT,
        context_confidence=ContextConfidence.HIGH,
        evidence=[
            "h3_exact_champion_anchor",
            "h4_champion_detail_pattern",
        ],
    )


def _unresolved_record():
    return resolve_patch_record(
        _normalized_record(),
        _catalog(),
    )


def test_context_hint_upgrades_unresolved_hotfix() -> None:
    enriched = apply_hotfix_champion_context_hints(
        [_unresolved_record()],
        [_hint(_normalized_record())],
        _catalog(),
    )[0]

    assert enriched.entity_type == PatchEntityType.CHAMPION
    assert enriched.entity_name == "트린다미어"
    assert enriched.entity_id == "Tryndamere"
    assert enriched.entity_key == "23"
    assert enriched.resolution_method == ResolutionMethod.CONTEXT_EXACT
    assert enriched.context_method == "hotfix_structural_exact"
    assert enriched.context_confidence == "high"
    assert enriched.context_anchor_title == "트린다미어"


def test_context_hint_does_not_overwrite_resolved_record() -> None:
    record = _unresolved_record().model_copy(
        update={
            "resolution_method": ResolutionMethod.TITLE_EXACT,
            "entity_type": PatchEntityType.CHAMPION,
            "entity_name": "트린다미어",
            "entity_id": "Tryndamere",
            "entity_key": "23",
        }
    )

    with pytest.raises(
        ValueError,
        match="only enrich an unresolved unknown",
    ):
        apply_hotfix_champion_context_hints(
            [record],
            [_hint(_normalized_record())],
            _catalog(),
        )


def test_duplicate_hint_record_id_is_rejected() -> None:
    hint = _hint(_normalized_record())

    with pytest.raises(
        ValueError,
        match="Duplicate Hotfix context hint record_id",
    ):
        apply_hotfix_champion_context_hints(
            [_unresolved_record()],
            [hint, hint],
            _catalog(),
        )


def test_hint_source_mismatch_is_rejected() -> None:
    hint = _hint(_normalized_record()).model_copy(update={"source_sha256": "0" * 64})

    with pytest.raises(ValueError, match="source SHA-256"):
        apply_hotfix_champion_context_hints(
            [_unresolved_record()],
            [hint],
            _catalog(),
        )


def test_catalog_identity_mismatch_is_rejected() -> None:
    hint = _hint(_normalized_record()).model_copy(update={"champion_id": "WrongChampion"})

    with pytest.raises(ValueError, match="champion_id"):
        apply_hotfix_champion_context_hints(
            [_unresolved_record()],
            [hint],
            _catalog(),
        )


def test_orphan_hint_is_rejected() -> None:
    hint = _hint(_normalized_record()).model_copy(
        update={"record_id": sha256(b"missing-record").hexdigest()}
    )

    with pytest.raises(
        ValueError,
        match="does not match any resolved record",
    ):
        apply_hotfix_champion_context_hints(
            [],
            [hint],
            _catalog(),
        )
