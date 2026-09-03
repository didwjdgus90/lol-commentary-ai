import json
from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.entity_catalog.builder import (
    build_alias_index,
    build_canonical_entities,
    discover_ddragon_versions,
    normalize_alias,
)
from lol_commentary_backend.ingestion.entity_catalog.io import (
    save_entity_catalog,
)
from lol_commentary_backend.ingestion.entity_catalog.models import (
    EntityObservation,
    EntityType,
)


def _observation(
    *,
    entity_type: EntityType,
    entity_uid: str,
    riot_key: str,
    name: str,
    locale: str,
    version: str,
    riot_id: str | None = None,
    map_ids: tuple[str, ...] = (),
) -> EntityObservation:
    return EntityObservation(
        entity_type=entity_type,
        entity_uid=entity_uid,
        riot_key=riot_key,
        riot_id=riot_id,
        name=name,
        locale=locale,
        ddragon_version=version,
        source_file=("data/raw/test.json"),
        source_sha256=("a" * 64),
        map_ids=map_ids,
    )


def test_normalizes_alias_case_and_whitespace() -> None:
    assert normalize_alias("  Essence   Reaver ") == "essence reaver"


def test_merges_korean_and_english_champion() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.17.1",
        ),
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="Aatrox",
            locale="en_US",
            version="16.17.1",
        ),
    )

    entities = build_canonical_entities(rows)

    assert len(entities) == 1

    entity = entities[0]

    assert entity.entity_uid == "champion:266"

    assert "아트록스" in (entity.aliases)

    assert "Aatrox" in (entity.aliases)


def test_merges_same_entity_across_versions() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.1.1",
        ),
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.17.1",
        ),
    )

    entity = build_canonical_entities(rows)[0]

    assert entity.ddragon_versions == (
        "16.1.1",
        "16.17.1",
    )


def test_item_id_is_canonical_identity() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.17.1",
            map_ids=("11",),
        ),
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="Essence Reaver",
            locale="en_US",
            version="16.17.1",
            map_ids=("11",),
        ),
    )

    entity = build_canonical_entities(rows)[0]

    assert entity.entity_uid == "item:3508"

    assert entity.map_ids == ("11",)


def test_duplicate_item_name_remains_ambiguous() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.17.1",
        ),
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:223508",
            riot_key="223508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.17.1",
        ),
    )

    entities = build_canonical_entities(rows)

    aliases = build_alias_index(entities)

    target = next(alias for alias in aliases if (alias.normalized_alias == "정수 약탈자"))

    assert target.entity_uids == (
        "item:223508",
        "item:3508",
    )


def test_champion_technical_id_becomes_alias() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:523",
            riot_key="523",
            riot_id="Aphelios",
            name="아펠리오스",
            locale="ko_KR",
            version="16.17.1",
        ),
    )

    entity = build_canonical_entities(rows)[0]

    assert "Aphelios" in entity.aliases


def test_rejects_champion_id_conflict() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.1.1",
        ),
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="WrongAatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.17.1",
        ),
    )

    with pytest.raises(
        ValueError,
        match=("Champion Riot ID conflict"),
    ):
        build_canonical_entities(rows)


def test_unions_item_map_ids() -> None:
    rows = (
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.1.1",
            map_ids=(
                "11",
                "12",
            ),
        ),
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.17.1",
            map_ids=(
                "11",
                "21",
            ),
        ),
    )

    entity = build_canonical_entities(rows)[0]

    assert entity.map_ids == (
        "11",
        "12",
        "21",
    )


def test_discovers_versions_in_semantic_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ddragon"

    root.mkdir()

    for version in (
        "16.17.1",
        "16.2.1",
        "16.1.1",
    ):
        (root / version).mkdir()

    (root / "metadata").mkdir()

    assert discover_ddragon_versions(root) == (
        "16.1.1",
        "16.2.1",
        "16.17.1",
    )


def test_saves_catalog_with_lineage(
    tmp_path: Path,
) -> None:
    rows = (
        _observation(
            entity_type=(EntityType.CHAMPION),
            entity_uid="champion:266",
            riot_key="266",
            riot_id="Aatrox",
            name="아트록스",
            locale="ko_KR",
            version="16.17.1",
        ),
        _observation(
            entity_type=(EntityType.ITEM),
            entity_uid="item:3508",
            riot_key="3508",
            name="정수 약탈자",
            locale="ko_KR",
            version="16.17.1",
        ),
    )

    entities = build_canonical_entities(rows)

    aliases = build_alias_index(entities)

    (
        entities_path,
        aliases_path,
        metadata_path,
        metadata,
    ) = save_entity_catalog(
        output_dir=tmp_path,
        entities=entities,
        aliases=aliases,
    )

    assert entities_path.is_file()
    assert aliases_path.is_file()
    assert metadata_path.is_file()

    assert metadata.entity_count == 2
    assert metadata.champion_count == 1
    assert metadata.item_count == 1

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["catalog_version"] == "entity_catalog_v1"

    assert len(payload["entities_sha256"]) == 64
