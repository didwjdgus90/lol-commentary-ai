from types import MappingProxyType

from lol_commentary_backend.ingestion.sources.models import (
    DataSourceSpec,
    SourceAuth,
    SourceFamily,
    SourceFormat,
    SourcePurpose,
)

_SOURCE_SPECS = (
    DataSourceSpec(
        source_id="patch_notes_ko_index",
        name="League of Legends Korean Patch Notes",
        family=SourceFamily.PATCH_NOTES,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.HTML,
        url_template=("https://www.leagueoflegends.com/ko-kr/news/tags/patch-notes/"),
        purposes=(SourcePurpose.RAG,),
        storage_subdir="patch_notes",
        locales=("ko_KR",),
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="patch_notes_en_index",
        name="League of Legends English Patch Notes",
        family=SourceFamily.PATCH_NOTES,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.HTML,
        url_template=("https://www.leagueoflegends.com/en-us/news/tags/patch-notes/"),
        purposes=(SourcePurpose.RAG,),
        storage_subdir="patch_notes",
        locales=("en_US",),
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="ddragon_versions",
        name="Data Dragon Versions",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://ddragon.leagueoflegends.com/api/versions.json"),
        purposes=(SourcePurpose.ENTITY,),
        storage_subdir="ddragon/metadata",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="ddragon_languages",
        name="Data Dragon Languages",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://ddragon.leagueoflegends.com/cdn/languages.json"),
        purposes=(SourcePurpose.ENTITY,),
        storage_subdir="ddragon/metadata",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="ddragon_champions",
        name="Data Dragon Champion Summary",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/champion.json"
        ),
        purposes=(
            SourcePurpose.RAG,
            SourcePurpose.ENTITY,
        ),
        storage_subdir="ddragon/champions",
        locales=(
            "ko_KR",
            "en_US",
        ),
        versioned=True,
    ),
    DataSourceSpec(
        source_id="ddragon_champion_detail",
        name="Data Dragon Champion Detail",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://ddragon.leagueoflegends.com/"
            "cdn/{version}/data/{locale}/"
            "champion/{champion_id}.json"
        ),
        purposes=(
            SourcePurpose.RAG,
            SourcePurpose.ENTITY,
        ),
        storage_subdir=("ddragon/champion_detail"),
        locales=(
            "ko_KR",
            "en_US",
        ),
        versioned=True,
    ),
    DataSourceSpec(
        source_id="ddragon_items",
        name="Data Dragon Items",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/item.json"),
        purposes=(
            SourcePurpose.RAG,
            SourcePurpose.ENTITY,
        ),
        storage_subdir="ddragon/items",
        locales=(
            "ko_KR",
            "en_US",
        ),
        versioned=True,
    ),
    DataSourceSpec(
        source_id="ddragon_runes",
        name="Data Dragon Runes Reforged",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/runesReforged.json"
        ),
        purposes=(
            SourcePurpose.RAG,
            SourcePurpose.ENTITY,
        ),
        storage_subdir="ddragon/runes",
        locales=(
            "ko_KR",
            "en_US",
        ),
        versioned=True,
    ),
    DataSourceSpec(
        source_id="ddragon_summoner_spells",
        name="Data Dragon Summoner Spells",
        family=SourceFamily.DDRAGON,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://ddragon.leagueoflegends.com/cdn/{version}/data/{locale}/summoner.json"
        ),
        purposes=(
            SourcePurpose.RAG,
            SourcePurpose.ENTITY,
        ),
        storage_subdir=("ddragon/summoner_spells"),
        locales=(
            "ko_KR",
            "en_US",
        ),
        versioned=True,
    ),
    DataSourceSpec(
        source_id="game_constants_seasons",
        name="Riot Season Constants",
        family=SourceFamily.GAME_CONSTANTS,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://static.developer.riotgames.com/docs/lol/seasons.json"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="game_constants",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="game_constants_queues",
        name="Riot Queue Constants",
        family=SourceFamily.GAME_CONSTANTS,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://static.developer.riotgames.com/docs/lol/queues.json"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="game_constants",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="game_constants_maps",
        name="Riot Map Constants",
        family=SourceFamily.GAME_CONSTANTS,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://static.developer.riotgames.com/docs/lol/maps.json"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="game_constants",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="game_constants_modes",
        name="Riot Game Mode Constants",
        family=SourceFamily.GAME_CONSTANTS,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://static.developer.riotgames.com/docs/lol/gameModes.json"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="game_constants",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="game_constants_types",
        name="Riot Game Type Constants",
        family=SourceFamily.GAME_CONSTANTS,
        auth=SourceAuth.PUBLIC,
        response_format=SourceFormat.JSON,
        url_template=("https://static.developer.riotgames.com/docs/lol/gameTypes.json"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="game_constants",
        probe_enabled=True,
    ),
    DataSourceSpec(
        source_id="riot_account_by_riot_id",
        name="Riot Account by Riot ID",
        family=SourceFamily.RIOT_API,
        auth=SourceAuth.RIOT_API_KEY,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://{region}.api.riotgames.com/"
            "riot/account/v1/accounts/by-riot-id/"
            "{game_name}/{tag_line}"
        ),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="riot_api/accounts",
    ),
    DataSourceSpec(
        source_id="riot_match_ids_by_puuid",
        name="Riot Match IDs by PUUID",
        family=SourceFamily.RIOT_API,
        auth=SourceAuth.RIOT_API_KEY,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        ),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="riot_api/match_ids",
    ),
    DataSourceSpec(
        source_id="riot_match_detail",
        name="Riot Match Detail",
        family=SourceFamily.RIOT_API,
        auth=SourceAuth.RIOT_API_KEY,
        response_format=SourceFormat.JSON,
        url_template=("https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="riot_api/matches",
    ),
    DataSourceSpec(
        source_id="riot_match_timeline",
        name="Riot Match Timeline",
        family=SourceFamily.RIOT_API,
        auth=SourceAuth.RIOT_API_KEY,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        ),
        purposes=(SourcePurpose.MATCH,),
        storage_subdir="riot_api/timelines",
    ),
    DataSourceSpec(
        source_id="riot_spectator_active_game",
        name="Riot Spectator Active Game",
        family=SourceFamily.RIOT_API,
        auth=SourceAuth.RIOT_API_KEY,
        response_format=SourceFormat.JSON,
        url_template=(
            "https://{platform}.api.riotgames.com/lol/spectator/v5/active-games/by-summoner/{puuid}"
        ),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="riot_api/spectator",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="live_all_game_data",
        name="Live Client All Game Data",
        family=SourceFamily.LIVE_CLIENT,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/liveclientdata/allgamedata"),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="live",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="live_events",
        name="Live Client Events",
        family=SourceFamily.LIVE_CLIENT,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/liveclientdata/eventdata"),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="live/events",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="live_game_stats",
        name="Live Client Game Stats",
        family=SourceFamily.LIVE_CLIENT,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/liveclientdata/gamestats"),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="live/game_stats",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="live_player_list",
        name="Live Client Player List",
        family=SourceFamily.LIVE_CLIENT,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/liveclientdata/playerlist"),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="live/players",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="live_active_player",
        name="Live Client Active Player",
        family=SourceFamily.LIVE_CLIENT,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/liveclientdata/activeplayer"),
        purposes=(SourcePurpose.LIVE,),
        storage_subdir="live/active_player",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="game_client_openapi",
        name="Game Client OpenAPI",
        family=SourceFamily.REPLAY,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/swagger/v3/openapi.json"),
        purposes=(SourcePurpose.REPLAY,),
        storage_subdir="replay/openapi",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="replay_game",
        name="Replay Game State",
        family=SourceFamily.REPLAY,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/replay/game"),
        purposes=(SourcePurpose.REPLAY,),
        storage_subdir="replay/game",
        runtime_only=True,
    ),
    DataSourceSpec(
        source_id="replay_playback",
        name="Replay Playback",
        family=SourceFamily.REPLAY,
        auth=SourceAuth.LOCAL_GAME_CLIENT,
        response_format=SourceFormat.JSON,
        url_template=("https://127.0.0.1:2999/replay/playback"),
        purposes=(SourcePurpose.REPLAY,),
        storage_subdir="replay/playback",
        runtime_only=True,
    ),
)


DATA_SOURCE_REGISTRY = MappingProxyType({source.source_id: source for source in _SOURCE_SPECS})


def get_data_source(
    source_id: str,
) -> DataSourceSpec:
    try:
        return DATA_SOURCE_REGISTRY[source_id]
    except KeyError as exc:
        raise KeyError(f"Unknown data source: {source_id}") from exc


def list_data_sources(
    *,
    family: SourceFamily | None = None,
    auth: SourceAuth | None = None,
    probe_enabled: bool | None = None,
) -> tuple[DataSourceSpec, ...]:
    sources: list[DataSourceSpec] = []

    for source in _SOURCE_SPECS:
        if family is not None and source.family != family:
            continue

        if auth is not None and source.auth != auth:
            continue

        if probe_enabled is not None and source.probe_enabled != probe_enabled:
            continue

        sources.append(source)

    return tuple(sources)
