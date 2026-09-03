import pytest

from lol_commentary_backend.ingestion.entity_catalog.models import (
    CanonicalEntity,
    EntityObservation,
    EntityType,
)
from lol_commentary_backend.retrieval.aliases.multi_catalog import (
    build_multi_patch_alias_catalog,
)


def _observation(
    *,
    name: str,
    locale: str,
    version: str,
    source_sha256: str = "a" * 64,
) -> EntityObservation:
    return EntityObservation(
        entity_type=(EntityType.CHAMPION),
        entity_uid="champion:266",
        riot_key="266",
        riot_id="Aatrox",
        name=name,
        locale=locale,
        ddragon_version=version,
        source_file="test.json",
        source_sha256=(source_sha256),
    )


def _entity(
    observations: tuple[
        EntityObservation,
        ...,
    ],
) -> CanonicalEntity:
    return CanonicalEntity(
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
        ddragon_versions=tuple(
            sorted({observation.ddragon_version for observation in observations})
        ),
        observations=observations,
    )


def test_builds_korean_english_pair() -> None:
    entity = _entity(
        (
            _observation(
                name="아트록스",
                locale="ko_KR",
                version="16.17.1",
            ),
            _observation(
                name="Aatrox",
                locale="en_US",
                version="16.17.1",
            ),
        )
    )

    catalog = build_multi_patch_alias_catalog((entity,))

    assert len(catalog.aliases) == 1

    alias = catalog.aliases[0]

    assert alias.ko_name == ("아트록스")

    assert alias.en_name == ("Aatrox")

    assert alias.entity_key == "266"


def test_deduplicates_same_pair_across_versions() -> None:
    entity = _entity(
        (
            _observation(
                name="아트록스",
                locale="ko_KR",
                version="16.1.1",
            ),
            _observation(
                name="Aatrox",
                locale="en_US",
                version="16.1.1",
            ),
            _observation(
                name="아트록스",
                locale="ko_KR",
                version="16.17.1",
            ),
            _observation(
                name="Aatrox",
                locale="en_US",
                version="16.17.1",
            ),
        )
    )

    catalog = build_multi_patch_alias_catalog((entity,))

    assert len(catalog.aliases) == 1


def test_preserves_historical_name_pair() -> None:
    entity = _entity(
        (
            _observation(
                name="옛 이름",
                locale="ko_KR",
                version="16.1.1",
            ),
            _observation(
                name="Old Name",
                locale="en_US",
                version="16.1.1",
            ),
            _observation(
                name="새 이름",
                locale="ko_KR",
                version="16.17.1",
            ),
            _observation(
                name="New Name",
                locale="en_US",
                version="16.17.1",
            ),
        )
    )

    catalog = build_multi_patch_alias_catalog((entity,))

    pairs = {
        (
            alias.ko_name,
            alias.en_name,
        )
        for alias in catalog.aliases
    }

    assert pairs == {
        (
            "옛 이름",
            "Old Name",
        ),
        (
            "새 이름",
            "New Name",
        ),
    }


def test_composite_source_hash_is_sha256() -> None:
    entity = _entity(
        (
            _observation(
                name="아트록스",
                locale="ko_KR",
                version="16.17.1",
                source_sha256="a" * 64,
            ),
            _observation(
                name="Aatrox",
                locale="en_US",
                version="16.17.1",
                source_sha256="b" * 64,
            ),
        )
    )

    catalog = build_multi_patch_alias_catalog((entity,))

    assert len(catalog.source_hashes) == 2

    assert all(len(record.sha256) == 64 for record in catalog.source_hashes)


def test_rejects_catalog_without_bilingual_pair() -> None:
    entity = _entity(
        (
            _observation(
                name="아트록스",
                locale="ko_KR",
                version="16.17.1",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="No bilingual aliases",
    ):
        build_multi_patch_alias_catalog((entity,))
