from hashlib import sha256

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
ITEM_SHA = sha256(b"item-source").hexdigest()


def _record(
    title: str,
    changes: list[str],
) -> NormalizedPatchRecord:
    return NormalizedPatchRecord(
        record_id=sha256(title.encode()).hexdigest(),
        order=0,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        section_kind="item",
        entity_type=PatchEntityType.ITEM,
        entity_name=title,
        heading_path=["업데이트된 아이템", title],
        title=title,
        changes=changes,
        content=title,
    )


def _item(
    item_id: str,
    name: str,
    evidence: dict[str, str],
    *map_ids: str,
) -> CatalogEntity:
    return CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id=item_id,
        entity_key=None,
        name=name,
        source_sha256=ITEM_SHA,
        map_ids=frozenset(map_ids),
        item_evidence=tuple(sorted(evidence.items())),
    )


def _catalog(
    name: str,
    candidates: tuple[CatalogEntity, ...],
) -> EntityCatalog:
    return EntityCatalog(
        ddragon_version="16.1.1",
        locale="ko_KR",
        champions_by_name={},
        items_by_name={name: candidates},
    )


def test_strict_evidence_resolves_hextech_gunblade() -> None:
    name = "마법공학 총검"

    standard = _item(
        "3146",
        name,
        {
            "total_gold": "3000",
            "ability_power": "80",
            "attack_damage": "40",
            "omnivamp_percent": "10",
        },
        "11",
        "12",
        "21",
    )
    alternate = _item(
        "663146",
        name,
        {
            "total_gold": "2500",
            "ability_power": "90",
            "attack_damage": "45",
            "omnivamp_percent": "15",
        },
        "11",
    )

    record = _record(
        name,
        [
            "총가격: 3,000골드",
            "주문력: 80",
            "공격력: 40",
            "모든 피해 흡혈: 10%",
        ],
    )

    resolved = resolve_patch_record(
        record,
        _catalog(
            name,
            (standard, alternate),
        ),
    )

    assert resolved.entity_id == "3146"
    assert resolved.resolution_method == ResolutionMethod.EVIDENCE_EXACT
    assert set(resolved.resolution_evidence_fields) == {
        "ability_power",
        "attack_damage",
        "omnivamp_percent",
        "total_gold",
    }


def test_strict_evidence_resolves_echoes_of_helia() -> None:
    name = "헬리아의 메아리"

    alternate = _item(
        "326620",
        name,
        {
            "ability_power": "35",
            "health": "250",
            "base_mana_regen_percent": "150",
            "ability_haste": "20",
        },
        "11",
    )
    standard = _item(
        "6620",
        name,
        {
            "ability_power": "35",
            "health": "200",
            "base_mana_regen_percent": "125",
            "ability_haste": "20",
        },
        "11",
        "12",
        "21",
        "35",
    )

    record = _record(
        name,
        [
            "주문력 35",
            "체력 200",
            "기본 마나 재생: 125%",
            "스킬 가속 20",
        ],
    )

    resolved = resolve_patch_record(
        record,
        _catalog(
            name,
            (alternate, standard),
        ),
    )

    assert resolved.entity_id == "6620"
    assert resolved.resolution_method == ResolutionMethod.EVIDENCE_EXACT


def test_identical_evidence_keeps_variants_ambiguous() -> None:
    name = "속삭이는 머리띠"

    evidence = {
        "total_gold": "2250",
        "health": "200",
        "mana": "300",
        "base_mana_regen_percent": "75",
    }

    first = _item(
        "2526",
        name,
        evidence,
        "11",
        "12",
        "21",
    )
    second = _item(
        "322526",
        name,
        evidence,
        "11",
    )

    resolved = resolve_patch_record(
        _record(
            name,
            [
                "총가격: 2,250골드",
                "체력: 200",
                "마나: 300",
                "기본 마나 재생: 75%",
            ],
        ),
        _catalog(
            name,
            (first, second),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_single_evidence_field_is_not_enough() -> None:
    name = "무라마나"

    first = _item(
        "3042",
        name,
        {"mana": "1000"},
        "11",
        "12",
        "21",
        "35",
    )
    second = _item(
        "323042",
        name,
        {"mana": "1000"},
        "11",
    )

    resolved = resolve_patch_record(
        _record(
            name,
            ["마나: 860 ⇒ 1,000"],
        ),
        _catalog(
            name,
            (first, second),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_partial_match_does_not_resolve_redemption() -> None:
    name = "구원"

    first = _item(
        "3107",
        name,
        {
            "total_gold": "2300",
            "health": "0",
            "ability_power": "30",
        },
        "11",
        "12",
        "21",
        "35",
    )
    second = _item(
        "323107",
        name,
        {
            "total_gold": "2800",
            "health": "400",
            "ability_power": "30",
        },
        "11",
    )

    resolved = resolve_patch_record(
        _record(
            name,
            [
                "총가격: 2,300골드 ⇒ 2,250골드",
                "체력: 200 ⇒ 0",
                "신규 주문력 30",
            ],
        ),
        _catalog(
            name,
            (first, second),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_deleted_item_is_lifecycle_not_map_error() -> None:
    name = "군단의 방패"

    arena = _item(
        "223105",
        name,
        {},
        "30",
    )
    other_map = _item(
        "3105",
        name,
        {},
        "35",
    )

    resolved = resolve_patch_record(
        _record(
            name,
            ["게임에서 삭제되었습니다."],
        ),
        _catalog(
            name,
            (arena, other_map),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.target_map_id == "11"
    assert resolved.resolution_method == ResolutionMethod.REMOVED_FROM_TARGET_MAP


def test_new_prefix_is_parsed_as_structured_evidence() -> None:
    name = "테스트 아이템"

    matching = _item(
        "1001",
        name,
        {
            "ability_power": "30",
            "health": "0",
        },
        "11",
    )
    other = _item(
        "1002",
        name,
        {
            "ability_power": "20",
            "health": "200",
        },
        "11",
    )

    resolved = resolve_patch_record(
        _record(
            name,
            [
                "신규 주문력 30",
                "체력: 200 ⇒ 0",
            ],
        ),
        _catalog(
            name,
            (matching, other),
        ),
    )

    assert resolved.entity_id == "1001"
    assert resolved.resolution_method == ResolutionMethod.EVIDENCE_EXACT
