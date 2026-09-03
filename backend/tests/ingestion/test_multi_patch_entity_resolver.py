from lol_commentary_backend.ingestion.entity_catalog.models import (
    AliasEntry,
    CanonicalEntity,
    EntityObservation,
    EntityType,
)
from lol_commentary_backend.ingestion.entity_catalog.resolver import (
    EntityCatalogResolver,
    ResolutionStatus,
)
from lol_commentary_backend.ingestion.version_mapping.models import (
    PatchDataDragonMapping,
)


def _observation(
    *,
    entity_type: EntityType,
    entity_uid: str,
    riot_key: str,
    riot_id: str | None,
    name: str,
    map_ids: tuple[
        str,
        ...,
    ] = (),
) -> EntityObservation:
    return EntityObservation(
        entity_type=entity_type,
        entity_uid=entity_uid,
        riot_key=riot_key,
        riot_id=riot_id,
        name=name,
        locale="ko_KR",
        ddragon_version="16.17.1",
        source_file="test.json",
        source_sha256="a" * 64,
        map_ids=map_ids,
    )


def _resolver() -> EntityCatalogResolver:
    champion = CanonicalEntity(
        entity_uid="champion:266",
        entity_type=(EntityType.CHAMPION),
        riot_key="266",
        riot_id="Aatrox",
        aliases=(
            "Aatrox",
            "아트록스",
        ),
        locales=(
            "en_US",
            "ko_KR",
        ),
        ddragon_versions=("16.17.1",),
        observations=(
            _observation(
                entity_type=(EntityType.CHAMPION),
                entity_uid=("champion:266"),
                riot_key="266",
                riot_id="Aatrox",
                name="아트록스",
            ),
        ),
    )

    normal_item = CanonicalEntity(
        entity_uid="item:3508",
        entity_type=(EntityType.ITEM),
        riot_key="3508",
        aliases=(
            "Essence Reaver",
            "정수 약탈자",
        ),
        locales=(
            "en_US",
            "ko_KR",
        ),
        ddragon_versions=("16.17.1",),
        map_ids=("11",),
        observations=(
            _observation(
                entity_type=(EntityType.ITEM),
                entity_uid="item:3508",
                riot_key="3508",
                riot_id=None,
                name="정수 약탈자",
                map_ids=("11",),
            ),
        ),
    )

    arena_item = CanonicalEntity(
        entity_uid="item:223508",
        entity_type=(EntityType.ITEM),
        riot_key="223508",
        aliases=(
            "Essence Reaver",
            "정수 약탈자",
        ),
        locales=(
            "en_US",
            "ko_KR",
        ),
        ddragon_versions=("16.17.1",),
        map_ids=("30",),
        observations=(
            _observation(
                entity_type=(EntityType.ITEM),
                entity_uid=("item:223508"),
                riot_key="223508",
                riot_id=None,
                name="정수 약탈자",
                map_ids=("30",),
            ),
        ),
    )

    aliases = (
        AliasEntry(
            normalized_alias=("아트록스"),
            display_aliases=("아트록스",),
            entity_uids=("champion:266",),
        ),
        AliasEntry(
            normalized_alias=("aatrox"),
            display_aliases=("Aatrox",),
            entity_uids=("champion:266",),
        ),
        AliasEntry(
            normalized_alias=("정수 약탈자"),
            display_aliases=("정수 약탈자",),
            entity_uids=(
                "item:223508",
                "item:3508",
            ),
        ),
    )

    mapping = (
        PatchDataDragonMapping(
            patch="26.17",
            ddragon_version=("16.17.1"),
            patch_locales=(
                "en_US",
                "ko_KR",
            ),
            ddragon_locales=(
                "en_US",
                "ko_KR",
            ),
        ),
    )

    return EntityCatalogResolver(
        entities=(
            champion,
            normal_item,
            arena_item,
        ),
        aliases=aliases,
        mappings=mapping,
    )


def test_resolves_korean_champion() -> None:
    result = _resolver().resolve(
        "아트록스",
        patch="26.17",
        entity_type=(EntityType.CHAMPION),
    )

    assert result.status == ResolutionStatus.RESOLVED

    assert result.selected_entity_uid == "champion:266"


def test_resolves_english_alias_to_same_champion() -> None:
    result = _resolver().resolve(
        "Aatrox",
        patch="26.17",
        entity_type=(EntityType.CHAMPION),
    )

    assert result.selected_entity_uid == "champion:266"


def test_duplicate_item_alias_is_ambiguous_without_map() -> None:
    result = _resolver().resolve(
        "정수 약탈자",
        patch="26.17",
        entity_type=(EntityType.ITEM),
    )

    assert result.status == ResolutionStatus.AMBIGUOUS

    assert result.candidate_entity_uids == (
        "item:223508",
        "item:3508",
    )


def test_map_context_resolves_standard_item() -> None:
    result = _resolver().resolve(
        "정수 약탈자",
        patch="26.17",
        entity_type=(EntityType.ITEM),
        map_id="11",
    )

    assert result.status == ResolutionStatus.RESOLVED

    assert result.selected_entity_uid == "item:3508"


def test_map_context_resolves_arena_item() -> None:
    result = _resolver().resolve(
        "정수 약탈자",
        patch="26.17",
        entity_type=(EntityType.ITEM),
        map_id="30",
    )

    assert result.selected_entity_uid == "item:223508"


def test_unknown_alias_returns_not_found() -> None:
    result = _resolver().resolve(
        "존재하지않는아이템",
        patch="26.17",
    )

    assert result.status == ResolutionStatus.NOT_FOUND


def test_unknown_patch_is_rejected() -> None:
    try:
        _resolver().resolve(
            "아트록스",
            patch="99.99",
        )
    except ValueError as exc:
        assert "No Data Dragon mapping" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
