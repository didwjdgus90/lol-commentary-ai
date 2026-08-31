import json
from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    CatalogEntity,
    EntityCatalog,
    load_entity_catalog,
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
ITEM_SHA = sha256(b"item-source").hexdigest()


def _record(
    *,
    title: str,
    entity_type: PatchEntityType = PatchEntityType.UNKNOWN,
    entity_name: str | None = None,
) -> NormalizedPatchRecord:
    return NormalizedPatchRecord(
        record_id=sha256(title.encode()).hexdigest(),
        order=0,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        section_kind="hotfix",
        entity_type=entity_type,
        entity_name=entity_name,
        heading_path=["추가 패치 노트", title],
        title=title,
        content=title,
    )


def _catalog() -> EntityCatalog:
    champion = CatalogEntity(
        entity_type=PatchEntityType.CHAMPION,
        entity_id="Aphelios",
        entity_key="523",
        name="아펠리오스",
        source_sha256=CHAMPION_SHA,
    )

    item = CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id="3508",
        entity_key=None,
        name="정수 약탈자",
        source_sha256=ITEM_SHA,
    )

    return EntityCatalog(
        ddragon_version="16.1.1",
        locale="ko_KR",
        champions_by_name={
            "아펠리오스": (champion,),
        },
        items_by_name={
            "정수 약탈자": (item,),
        },
    )


def test_resolver_maps_existing_champion_to_official_id() -> None:
    resolved = resolve_patch_record(
        _record(
            title="아펠리오스",
            entity_type=PatchEntityType.CHAMPION,
            entity_name="아펠리오스",
        ),
        _catalog(),
    )

    assert resolved.entity_type == PatchEntityType.CHAMPION
    assert resolved.entity_id == "Aphelios"
    assert resolved.entity_key == "523"
    assert resolved.resolution_method == ResolutionMethod.BASELINE_EXACT


def test_resolver_promotes_unknown_exact_item_title() -> None:
    resolved = resolve_patch_record(
        _record(title="정수 약탈자"),
        _catalog(),
    )

    assert resolved.entity_type == PatchEntityType.ITEM
    assert resolved.entity_name == "정수 약탈자"
    assert resolved.entity_id == "3508"
    assert resolved.resolution_method == ResolutionMethod.TITLE_EXACT


def test_resolver_keeps_ability_title_unresolved() -> None:
    resolved = resolve_patch_record(
        _record(title="기본 지속 효과 - 암살자와 예언자"),
        _catalog(),
    )

    assert resolved.entity_type == PatchEntityType.UNKNOWN
    assert resolved.entity_id is None
    assert resolved.resolution_method == ResolutionMethod.UNRESOLVED


def test_resolver_does_not_choose_ambiguous_exact_name() -> None:
    first = CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id="1001",
        entity_key=None,
        name="중복 아이템",
        source_sha256=ITEM_SHA,
    )
    second = CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id="1002",
        entity_key=None,
        name="중복 아이템",
        source_sha256=ITEM_SHA,
    )

    catalog = EntityCatalog(
        ddragon_version="16.1.1",
        locale="ko_KR",
        champions_by_name={},
        items_by_name={
            "중복 아이템": (first, second),
        },
    )

    resolved = resolve_patch_record(
        _record(title="중복 아이템"),
        catalog,
    )

    assert resolved.entity_id is None
    assert resolved.resolution_candidate_count == 2
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_load_entity_catalog_verifies_raw_lineage(
    tmp_path,
) -> None:
    champion_payload = {
        "type": "champion",
        "version": "16.1.1",
        "data": {
            "Aphelios": {
                "id": "Aphelios",
                "key": "523",
                "name": "아펠리오스",
            }
        },
    }
    item_payload = {
        "type": "item",
        "version": "16.1.1",
        "data": {
            "3508": {
                "name": "정수 약탈자",
            }
        },
    }

    champion_content = json.dumps(
        champion_payload,
        ensure_ascii=False,
    ).encode()
    item_content = json.dumps(
        item_payload,
        ensure_ascii=False,
    ).encode()

    (tmp_path / "champion.json").write_bytes(champion_content)
    (tmp_path / "item.json").write_bytes(item_content)

    metadata = {
        "schema_version": 1,
        "collector_version": "0.1.0",
        "ddragon_version": "16.1.1",
        "locale": "ko_KR",
        "resources": {
            "champion": {
                "source_url": "https://example.com/champion.json",
                "fetched_at": "2026-08-26T00:00:00+00:00",
                "status_code": 200,
                "content_type": "application/json",
                "sha256": sha256(champion_content).hexdigest(),
                "size_bytes": len(champion_content),
            },
            "item": {
                "source_url": "https://example.com/item.json",
                "fetched_at": "2026-08-26T00:00:00+00:00",
                "status_code": 200,
                "content_type": "application/json",
                "sha256": sha256(item_content).hexdigest(),
                "size_bytes": len(item_content),
            },
        },
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = load_entity_catalog(tmp_path)

    champion_candidates = catalog.exact_candidates(
        "아펠리오스",
        entity_type=PatchEntityType.CHAMPION,
    )
    item_candidates = catalog.exact_candidates(
        "정수 약탈자",
        entity_type=PatchEntityType.ITEM,
    )

    assert champion_candidates[0].entity_id == "Aphelios"
    assert item_candidates[0].entity_id == "3508"


def test_load_entity_catalog_rejects_hash_mismatch(
    tmp_path,
) -> None:
    champion_content = json.dumps(
        {
            "version": "16.1.1",
            "data": {},
        }
    ).encode()
    item_content = json.dumps(
        {
            "version": "16.1.1",
            "data": {},
        }
    ).encode()

    (tmp_path / "champion.json").write_bytes(champion_content)
    (tmp_path / "item.json").write_bytes(item_content)

    metadata = {
        "schema_version": 1,
        "collector_version": "0.1.0",
        "ddragon_version": "16.1.1",
        "locale": "ko_KR",
        "resources": {
            "champion": {
                "source_url": "https://example.com/champion.json",
                "fetched_at": "2026-08-26T00:00:00+00:00",
                "status_code": 200,
                "content_type": "application/json",
                "sha256": "0" * 64,
                "size_bytes": len(champion_content),
            },
            "item": {
                "source_url": "https://example.com/item.json",
                "fetched_at": "2026-08-26T00:00:00+00:00",
                "status_code": 200,
                "content_type": "application/json",
                "sha256": sha256(item_content).hexdigest(),
                "size_bytes": len(item_content),
            },
        },
    }

    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        load_entity_catalog(tmp_path)
