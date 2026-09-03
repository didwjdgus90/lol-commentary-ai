import pytest

from lol_commentary_backend.ingestion.sources.models import (
    SourceAuth,
    SourceFamily,
)
from lol_commentary_backend.ingestion.sources.registry import (
    DATA_SOURCE_REGISTRY,
    get_data_source,
    list_data_sources,
)


def test_registry_ids_are_unique() -> None:
    source_ids = tuple(DATA_SOURCE_REGISTRY)

    assert len(source_ids) == len(set(source_ids))


def test_all_sources_use_https() -> None:
    for source in DATA_SOURCE_REGISTRY.values():
        assert source.url_template.startswith("https://")


def test_public_probe_sources_are_concrete() -> None:
    probe_sources = list_data_sources(probe_enabled=True)

    assert probe_sources

    for source in probe_sources:
        assert source.auth == SourceAuth.PUBLIC

        assert "{" not in source.url_template


def test_patch_note_locales_are_registered() -> None:
    ko = get_data_source("patch_notes_ko_index")

    en = get_data_source("patch_notes_en_index")

    assert ko.locales == ("ko_KR",)

    assert en.locales == ("en_US",)


def test_ddragon_core_sources_are_registered() -> None:
    required = {
        "ddragon_versions",
        "ddragon_languages",
        "ddragon_champions",
        "ddragon_champion_detail",
        "ddragon_items",
        "ddragon_runes",
        "ddragon_summoner_spells",
    }

    assert required.issubset(DATA_SOURCE_REGISTRY)


def test_riot_api_sources_require_api_key() -> None:
    sources = list_data_sources(family=SourceFamily.RIOT_API)

    assert sources

    for source in sources:
        assert source.auth == SourceAuth.RIOT_API_KEY


def test_live_sources_are_local_runtime_only() -> None:
    sources = list_data_sources(family=SourceFamily.LIVE_CLIENT)

    assert sources

    for source in sources:
        assert source.auth == SourceAuth.LOCAL_GAME_CLIENT

        assert "127.0.0.1:2999" in source.url_template

        assert source.runtime_only is True


def test_match_timeline_is_registered() -> None:
    source = get_data_source("riot_match_timeline")

    assert source.family == SourceFamily.RIOT_API

    assert "/timeline" in source.url_template


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown data source",
    ):
        get_data_source("does_not_exist")


def test_registry_contains_all_required_families() -> None:
    families = {source.family for source in (DATA_SOURCE_REGISTRY.values())}

    assert families == {
        SourceFamily.PATCH_NOTES,
        SourceFamily.DDRAGON,
        SourceFamily.GAME_CONSTANTS,
        SourceFamily.RIOT_API,
        SourceFamily.LIVE_CLIENT,
        SourceFamily.REPLAY,
    }
