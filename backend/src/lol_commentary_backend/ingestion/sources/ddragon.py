from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from lol_commentary_backend.ingestion.sources.registry import (
    get_data_source,
)

DataDragonLocale = Literal[
    "ko_KR",
    "en_US",
]

SnapshotAction = Literal[
    "downloaded",
    "reused",
]

ChampionDetailScope = Literal[
    "none",
    "latest",
    "all",
]


_VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\."
    r"(?P<minor>\d+)\."
    r"(?P<build>\d+)$"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

_USER_AGENT = "lol-commentary-ai/ddragon-collector-v1"


class StaticSnapshotRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source_id: str = Field(
        min_length=1,
    )

    resource_kind: str = Field(
        min_length=1,
    )

    version: str | None = None

    locale: str | None = None

    entity_id: str | None = None

    source_url: str = Field(
        min_length=1,
    )

    file_path: str = Field(
        min_length=1,
    )

    content_sha256: str = Field(
        pattern=_SHA256_PATTERN,
    )

    byte_count: int = Field(
        gt=0,
    )

    snapshot_at: datetime


class StaticCollectionItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    action: SnapshotAction

    manifest: StaticSnapshotRecord


class DataDragonCollectionSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    selected_versions: tuple[
        str,
        ...,
    ]

    locales: tuple[
        str,
        ...,
    ]

    champion_detail_scope: ChampionDetailScope

    downloaded_count: int = Field(
        ge=0,
    )

    reused_count: int = Field(
        ge=0,
    )

    ddragon_manifest_path: str = Field(
        min_length=1,
    )

    game_constants_manifest_path: str = Field(
        min_length=1,
    )

    champion_detail_count: int = Field(
        ge=0,
    )

    items: tuple[
        StaticCollectionItem,
        ...,
    ]


def sha256_hex(
    content: bytes,
) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_ddragon_version(
    version: str,
) -> tuple[
    int,
    int,
    int,
]:
    match = _VERSION_PATTERN.fullmatch(version)

    if match is None:
        raise ValueError(f"Unsupported Data Dragon version: {version}")

    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("build")),
    )


def select_latest_builds(
    *,
    available_versions: list[str],
    major: int,
    start_minor: int,
    end_minor: int,
) -> tuple[str, ...]:
    if start_minor > end_minor:
        raise ValueError("start_minor must not be after end_minor")

    candidates: dict[
        int,
        tuple[int, str],
    ] = {}

    for version in available_versions:
        try:
            (
                version_major,
                minor,
                build,
            ) = parse_ddragon_version(version)
        except ValueError:
            continue

        if version_major != major:
            continue

        if not (start_minor <= minor <= end_minor):
            continue

        current = candidates.get(minor)

        if current is None or build > current[0]:
            candidates[minor] = (
                build,
                version,
            )

    requested_minors = set(
        range(
            start_minor,
            end_minor + 1,
        )
    )

    missing = sorted(requested_minors - set(candidates))

    if missing:
        raise ValueError(f"Missing Data Dragon minor versions: {missing}")

    return tuple(
        candidates[minor][1]
        for minor in range(
            start_minor,
            end_minor + 1,
        )
    )


def validate_explicit_versions(
    *,
    requested_versions: tuple[
        str,
        ...,
    ],
    available_versions: list[str],
) -> tuple[str, ...]:
    if not requested_versions:
        raise ValueError("At least one Data Dragon version must be requested")

    available = set(available_versions)

    missing = [version for version in requested_versions if version not in available]

    if missing:
        raise ValueError(f"Unknown Data Dragon versions: {missing}")

    unique = tuple(dict.fromkeys(requested_versions))

    return tuple(
        sorted(
            unique,
            key=parse_ddragon_version,
        )
    )


def extract_champion_ids(
    payload: object,
) -> tuple[str, ...]:
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Champion summary must be a JSON object")

    data = payload.get("data")

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError("Champion summary has no data object")

    champion_ids = [key for key in data if isinstance(key, str) and key]

    if not champion_ids:
        raise ValueError("Champion summary contains no champions")

    return tuple(sorted(champion_ids))


def _relative_posix(
    *,
    path: Path,
    repository_root: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def _snapshot_time(
    path: Path,
) -> datetime:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=UTC,
    )


def _write_bytes_atomic(
    *,
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_bytes(content)

    temporary.replace(path)


def _decode_json(
    *,
    content: bytes,
    source_url: str,
) -> object:
    try:
        return json.loads(content)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(f"Invalid JSON received from {source_url}") from exc


def _fetch_or_reuse_json(
    *,
    client: httpx.Client,
    repository_root: Path,
    path: Path,
    source_id: str,
    resource_kind: str,
    source_url: str,
    version: str | None = None,
    locale: str | None = None,
    entity_id: str | None = None,
    overwrite: bool,
) -> tuple[
    StaticCollectionItem,
    object,
]:
    if path.exists() and not overwrite:
        content = path.read_bytes()

        payload = _decode_json(
            content=content,
            source_url=source_url,
        )

        action: SnapshotAction = "reused"

    else:
        response = client.get(source_url)

        response.raise_for_status()

        if not response.content:
            raise RuntimeError(f"Empty response from {source_url}")

        payload = _decode_json(
            content=response.content,
            source_url=source_url,
        )

        _write_bytes_atomic(
            path=path,
            content=response.content,
        )

        content = response.content

        action = "downloaded"

    record = StaticSnapshotRecord(
        source_id=source_id,
        resource_kind=resource_kind,
        version=version,
        locale=locale,
        entity_id=entity_id,
        source_url=source_url,
        file_path=_relative_posix(
            path=path,
            repository_root=(repository_root),
        ),
        content_sha256=sha256_hex(content),
        byte_count=len(content),
        snapshot_at=_snapshot_time(path),
    )

    return (
        StaticCollectionItem(
            action=action,
            manifest=record,
        ),
        payload,
    )


def _manifest_key(
    record: StaticSnapshotRecord,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
]:
    return (
        record.source_id,
        record.version or "",
        record.locale or "",
        record.entity_id or "",
        record.resource_kind,
    )


def _manifest_sort_key(
    record: StaticSnapshotRecord,
) -> tuple[
    tuple[int, int, int],
    str,
    str,
    str,
    str,
]:
    version_key = (
        parse_ddragon_version(record.version)
        if record.version and _VERSION_PATTERN.fullmatch(record.version)
        else (-1, -1, -1)
    )

    return (
        version_key,
        record.locale or "",
        record.resource_kind,
        record.entity_id or "",
        record.source_id,
    )


def _read_existing_manifest(
    path: Path,
) -> list[StaticSnapshotRecord]:
    if not path.exists():
        return []

    records: list[StaticSnapshotRecord] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        records.append(StaticSnapshotRecord.model_validate_json(line))

    return records


def write_merged_manifest(
    *,
    path: Path,
    records: tuple[
        StaticSnapshotRecord,
        ...,
    ],
) -> None:
    merged = {_manifest_key(record): record for record in (_read_existing_manifest(path))}

    for record in records:
        merged[_manifest_key(record)] = record

    ordered = sorted(
        merged.values(),
        key=_manifest_sort_key,
    )

    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in ordered
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        ("\n".join(lines) + "\n" if lines else ""),
        encoding="utf-8",
    )

    temporary.replace(path)


def _metadata_target(
    *,
    repository_root: Path,
    filename: str,
) -> Path:
    return repository_root / "data" / "raw" / "ddragon" / "metadata" / filename


def _version_target(
    *,
    repository_root: Path,
    version: str,
    locale: str,
    relative_path: str,
) -> Path:
    return repository_root / "data" / "raw" / "ddragon" / version / locale / relative_path


def _constant_target(
    *,
    repository_root: Path,
    filename: str,
) -> Path:
    return repository_root / "data" / "raw" / "game_constants" / filename


def _format_source_url(
    *,
    source_id: str,
    **values: str,
) -> str:
    source = get_data_source(source_id)

    return source.url_template.format(**values)


def _collect_metadata(
    *,
    client: httpx.Client,
    repository_root: Path,
    overwrite: bool,
) -> tuple[
    list[StaticCollectionItem],
    list[str],
    list[str],
]:
    items: list[StaticCollectionItem] = []

    versions_source = get_data_source("ddragon_versions")

    versions_item, versions_payload = _fetch_or_reuse_json(
        client=client,
        repository_root=(repository_root),
        path=_metadata_target(
            repository_root=(repository_root),
            filename="versions.json",
        ),
        source_id=(versions_source.source_id),
        resource_kind=("metadata_versions"),
        source_url=(versions_source.url_template),
        overwrite=overwrite,
    )

    items.append(versions_item)

    if not isinstance(
        versions_payload,
        list,
    ) or not all(isinstance(value, str) for value in versions_payload):
        raise RuntimeError("Data Dragon versions payload is invalid")

    languages_source = get_data_source("ddragon_languages")

    (
        languages_item,
        languages_payload,
    ) = _fetch_or_reuse_json(
        client=client,
        repository_root=(repository_root),
        path=_metadata_target(
            repository_root=(repository_root),
            filename="languages.json",
        ),
        source_id=(languages_source.source_id),
        resource_kind=("metadata_languages"),
        source_url=(languages_source.url_template),
        overwrite=overwrite,
    )

    items.append(languages_item)

    if not isinstance(
        languages_payload,
        list,
    ) or not all(isinstance(value, str) for value in languages_payload):
        raise RuntimeError("Data Dragon languages payload is invalid")

    return (
        items,
        list(versions_payload),
        list(languages_payload),
    )


def _collect_core_version_data(
    *,
    client: httpx.Client,
    repository_root: Path,
    version: str,
    locale: DataDragonLocale,
    collect_details: bool,
    overwrite: bool,
) -> tuple[
    list[StaticCollectionItem],
    int,
]:
    items: list[StaticCollectionItem] = []

    champion_url = _format_source_url(
        source_id=("ddragon_champions"),
        version=version,
        locale=locale,
    )

    champion_item, champion_payload = _fetch_or_reuse_json(
        client=client,
        repository_root=(repository_root),
        path=_version_target(
            repository_root=(repository_root),
            version=version,
            locale=locale,
            relative_path=("champion/champion.json"),
        ),
        source_id=("ddragon_champions"),
        resource_kind=("champion_summary"),
        source_url=champion_url,
        version=version,
        locale=locale,
        overwrite=overwrite,
    )

    items.append(champion_item)

    champion_ids = extract_champion_ids(champion_payload)

    resource_specs = (
        (
            "ddragon_items",
            "items",
            "item/item.json",
        ),
        (
            "ddragon_runes",
            "runes",
            "runes/runesReforged.json",
        ),
        (
            "ddragon_summoner_spells",
            "summoner_spells",
            "summoner/summoner.json",
        ),
    )

    for (
        source_id,
        resource_kind,
        relative_path,
    ) in resource_specs:
        source_url = _format_source_url(
            source_id=source_id,
            version=version,
            locale=locale,
        )

        item, _ = _fetch_or_reuse_json(
            client=client,
            repository_root=(repository_root),
            path=_version_target(
                repository_root=(repository_root),
                version=version,
                locale=locale,
                relative_path=(relative_path),
            ),
            source_id=source_id,
            resource_kind=(resource_kind),
            source_url=(source_url),
            version=version,
            locale=locale,
            overwrite=overwrite,
        )

        items.append(item)

    detail_count = 0

    if collect_details:
        for champion_id in champion_ids:
            source_url = _format_source_url(
                source_id=("ddragon_champion_detail"),
                version=version,
                locale=locale,
                champion_id=(champion_id),
            )

            detail_item, _ = _fetch_or_reuse_json(
                client=client,
                repository_root=(repository_root),
                path=_version_target(
                    repository_root=(repository_root),
                    version=version,
                    locale=locale,
                    relative_path=(f"champion/detail/{champion_id}.json"),
                ),
                source_id=("ddragon_champion_detail"),
                resource_kind=("champion_detail"),
                source_url=source_url,
                version=version,
                locale=locale,
                entity_id=champion_id,
                overwrite=overwrite,
            )

            items.append(detail_item)

            detail_count += 1

    return (
        items,
        detail_count,
    )


def _collect_game_constants(
    *,
    client: httpx.Client,
    repository_root: Path,
    overwrite: bool,
) -> list[StaticCollectionItem]:
    specs = (
        (
            "game_constants_seasons",
            "seasons.json",
            "seasons",
        ),
        (
            "game_constants_queues",
            "queues.json",
            "queues",
        ),
        (
            "game_constants_maps",
            "maps.json",
            "maps",
        ),
        (
            "game_constants_modes",
            "gameModes.json",
            "game_modes",
        ),
        (
            "game_constants_types",
            "gameTypes.json",
            "game_types",
        ),
    )

    items: list[StaticCollectionItem] = []

    for (
        source_id,
        filename,
        resource_kind,
    ) in specs:
        source = get_data_source(source_id)

        item, _ = _fetch_or_reuse_json(
            client=client,
            repository_root=(repository_root),
            path=_constant_target(
                repository_root=(repository_root),
                filename=filename,
            ),
            source_id=source_id,
            resource_kind=(resource_kind),
            source_url=(source.url_template),
            overwrite=overwrite,
        )

        items.append(item)

    return items


def collect_ddragon_data(
    *,
    repository_root: Path,
    locales: tuple[
        DataDragonLocale,
        ...,
    ] = (
        "ko_KR",
        "en_US",
    ),
    explicit_versions: tuple[
        str,
        ...,
    ]
    | None = None,
    major: int = 16,
    start_minor: int = 1,
    end_minor: int = 17,
    champion_detail_scope: (ChampionDetailScope) = "latest",
    timeout_seconds: float = 30.0,
    overwrite: bool = False,
) -> DataDragonCollectionSummary:
    if not locales:
        raise ValueError("At least one locale must be requested")

    all_items: list[StaticCollectionItem] = []

    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
    ) as client:
        (
            metadata_items,
            available_versions,
            available_languages,
        ) = _collect_metadata(
            client=client,
            repository_root=(repository_root),
            overwrite=overwrite,
        )

        all_items.extend(metadata_items)

        missing_locales = [locale for locale in locales if (locale not in available_languages)]

        if missing_locales:
            raise ValueError(f"Unsupported Data Dragon locales: {missing_locales}")

        if explicit_versions:
            selected_versions = validate_explicit_versions(
                requested_versions=(explicit_versions),
                available_versions=(available_versions),
            )

        else:
            selected_versions = select_latest_builds(
                available_versions=(available_versions),
                major=major,
                start_minor=(start_minor),
                end_minor=(end_minor),
            )

        latest_selected = max(
            selected_versions,
            key=parse_ddragon_version,
        )

        champion_detail_count = 0

        for version in selected_versions:
            for locale in locales:
                collect_details = champion_detail_scope == "all" or (
                    champion_detail_scope == "latest" and version == latest_selected
                )

                (
                    version_items,
                    version_detail_count,
                ) = _collect_core_version_data(
                    client=client,
                    repository_root=(repository_root),
                    version=version,
                    locale=locale,
                    collect_details=(collect_details),
                    overwrite=overwrite,
                )

                all_items.extend(version_items)

                champion_detail_count += version_detail_count

        constant_items = _collect_game_constants(
            client=client,
            repository_root=(repository_root),
            overwrite=overwrite,
        )

        all_items.extend(constant_items)

    ddragon_records = tuple(
        item.manifest for item in all_items if (item.manifest.source_id.startswith("ddragon_"))
    )

    constant_records = tuple(
        item.manifest
        for item in all_items
        if (item.manifest.source_id.startswith("game_constants_"))
    )

    ddragon_manifest = repository_root / "data" / "manifests" / "ddragon" / "raw_sources.jsonl"

    constants_manifest = (
        repository_root / "data" / "manifests" / "game_constants" / "raw_sources.jsonl"
    )

    write_merged_manifest(
        path=ddragon_manifest,
        records=ddragon_records,
    )

    write_merged_manifest(
        path=constants_manifest,
        records=constant_records,
    )

    downloaded_count = sum(item.action == "downloaded" for item in all_items)

    reused_count = sum(item.action == "reused" for item in all_items)

    return DataDragonCollectionSummary(
        selected_versions=(selected_versions),
        locales=tuple(locales),
        champion_detail_scope=(champion_detail_scope),
        downloaded_count=(downloaded_count),
        reused_count=(reused_count),
        ddragon_manifest_path=(
            _relative_posix(
                path=ddragon_manifest,
                repository_root=(repository_root),
            )
        ),
        game_constants_manifest_path=(
            _relative_posix(
                path=constants_manifest,
                repository_root=(repository_root),
            )
        ),
        champion_detail_count=(champion_detail_count),
        items=tuple(all_items),
    )
