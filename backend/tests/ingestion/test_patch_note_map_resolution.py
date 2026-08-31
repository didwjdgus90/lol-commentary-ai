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


def _item_record(
    title: str,
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
        content=title,
    )


def _hotfix_record(
    title: str,
) -> NormalizedPatchRecord:
    return NormalizedPatchRecord(
        record_id=sha256(f"hotfix:{title}".encode()).hexdigest(),
        order=0,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=SOURCE_SHA,
        section_kind="hotfix",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        heading_path=["추가 패치 노트", title],
        title=title,
        content=title,
    )


def _item(
    item_id: str,
    name: str,
    *map_ids: str,
) -> CatalogEntity:
    return CatalogEntity(
        entity_type=PatchEntityType.ITEM,
        entity_id=item_id,
        entity_key=None,
        name=name,
        source_sha256=ITEM_SHA,
        map_ids=frozenset(map_ids),
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


def test_map_filter_selects_unique_summoners_rift_variant() -> None:
    name = "정수 약탈자"
    arena = _item("223508", name, "30")
    standard = _item(
        "3508",
        name,
        "11",
        "12",
        "21",
        "35",
    )

    resolved = resolve_patch_record(
        _item_record(name),
        _catalog(name, (arena, standard)),
    )

    assert resolved.entity_id == "3508"
    assert resolved.target_map_id == "11"
    assert resolved.resolution_candidate_count == 1
    assert resolved.resolution_original_candidate_count == 2
    assert resolved.resolution_method == ResolutionMethod.MAP_EXACT


def test_map_filter_keeps_multiple_map11_variants_ambiguous() -> None:
    name = "헬리아의 메아리"
    arena = _item("226620", name, "30")
    swift = _item("326620", name, "11")
    standard = _item(
        "6620",
        name,
        "11",
        "12",
        "21",
        "35",
    )

    resolved = resolve_patch_record(
        _item_record(name),
        _catalog(
            name,
            (arena, swift, standard),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.target_map_id == "11"
    assert resolved.resolution_candidate_count == 2
    assert resolved.resolution_original_candidate_count == 3
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_map_filter_marks_zero_compatible_candidates() -> None:
    name = "군단의 방패"
    unavailable = _item("223105", name)
    other_map = _item("3105", name, "35")

    resolved = resolve_patch_record(
        _item_record(name),
        _catalog(
            name,
            (unavailable, other_map),
        ),
    )

    assert resolved.entity_id is None
    assert resolved.target_map_id == "11"
    assert resolved.resolution_candidate_count == 0
    assert resolved.resolution_original_candidate_count == 2
    assert resolved.resolution_method == ResolutionMethod.MAP_INCOMPATIBLE


def test_hotfix_duplicate_title_does_not_assume_map11() -> None:
    name = "정수 약탈자"
    arena = _item("223508", name, "30")
    standard = _item(
        "3508",
        name,
        "11",
        "12",
        "21",
        "35",
    )

    resolved = resolve_patch_record(
        _hotfix_record(name),
        _catalog(name, (arena, standard)),
    )

    assert resolved.entity_id is None
    assert resolved.target_map_id is None
    assert resolved.resolution_candidate_count == 2
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS


def test_map_filter_does_not_choose_first_of_two_map11_candidates() -> None:
    name = "구원"
    first = _item("3107", name, "11", "12", "21", "35")
    second = _item("323107", name, "11")

    resolved = resolve_patch_record(
        _item_record(name),
        _catalog(name, (first, second)),
    )

    assert resolved.entity_id is None
    assert resolved.resolution_candidate_count == 2
    assert resolved.resolution_method == ResolutionMethod.AMBIGUOUS
