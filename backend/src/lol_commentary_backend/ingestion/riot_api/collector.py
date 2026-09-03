from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import (
    Mapping,
)
from datetime import (
    UTC,
    datetime,
)
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from lol_commentary_backend.ingestion.riot_api.models import (
    RiotAccount,
    RiotMatchIds,
)
from lol_commentary_backend.ingestion.riot_api.raw_models import (
    RIOT_RAW_COLLECTOR_VERSION,
    RiotRawCollectionResult,
    RiotRawManifestRecord,
    RiotRawResourceType,
)

_MATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class RiotRawCollectorClient(Protocol):
    def get_account_by_riot_id(
        self,
        *,
        game_name: str,
        tag_line: str,
    ) -> RiotAccount: ...

    def get_match_ids_by_puuid(
        self,
        *,
        puuid: str,
        start: int = 0,
        count: int = 20,
    ) -> RiotMatchIds: ...

    def get_match(
        self,
        match_id: str,
    ) -> dict[str, object]: ...

    def get_timeline(
        self,
        match_id: str,
    ) -> dict[str, object]: ...


def parse_riot_id(
    riot_id: str,
) -> tuple[
    str,
    str,
]:
    normalized = unicodedata.normalize(
        "NFKC",
        riot_id,
    ).strip()

    if normalized.count("#") != 1:
        raise ValueError("Riot ID must use exactly one '#' separator, for example GameName#KR1")

    game_name, tag_line = normalized.split(
        "#",
        1,
    )

    game_name = game_name.strip()
    tag_line = tag_line.strip()

    if not game_name:
        raise ValueError("Riot ID game name must not be empty")

    if not tag_line:
        raise ValueError("Riot ID tag line must not be empty")

    return (
        game_name,
        tag_line,
    )


def account_identity_sha256(
    *,
    game_name: str,
    tag_line: str,
) -> str:
    normalized = (
        unicodedata.normalize(
            "NFKC",
            f"{game_name}#{tag_line}",
        )
        .strip()
        .casefold()
    )

    return sha256(normalized.encode("utf-8")).hexdigest()


def canonical_json_bytes(
    payload: object,
) -> bytes:
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n"
    )

    return text.encode("utf-8")


def _bytes_sha256(
    payload: bytes,
) -> str:
    return sha256(payload).hexdigest()


def _file_sha256(
    path: Path,
) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_match_id(
    match_id: str,
) -> str:
    normalized = match_id.strip()

    if not normalized:
        raise ValueError("match_id must not be empty")

    if _MATCH_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("match_id contains unsupported characters")

    return normalized


def _atomic_write_bytes(
    *,
    path: Path,
    payload: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_bytes(payload)

    temporary.replace(path)


def _relative_path(
    *,
    repository_root: Path,
    path: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def _load_manifest(
    path: Path,
) -> dict[
    str,
    RiotRawManifestRecord,
]:
    if not path.is_file():
        return {}

    records: dict[
        str,
        RiotRawManifestRecord,
    ] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = RiotRawManifestRecord.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid Riot raw manifest record at line {line_number}") from exc

        if record.resource_key in records:
            raise ValueError(f"Duplicate Riot raw manifest resource_key: {record.resource_key}")

        records[record.resource_key] = record

    return records


def _write_manifest(
    *,
    path: Path,
    records: Mapping[
        str,
        RiotRawManifestRecord,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        for _, record in sorted(
            records.items(),
            key=lambda item: item[0],
        )
    ]

    payload = "\n".join(lines) + ("\n" if lines else "")

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(path)


def _record_is_reusable(
    *,
    repository_root: Path,
    record: (RiotRawManifestRecord | None),
) -> bool:
    if record is None:
        return False

    file_path = repository_root / record.file_path

    if not file_path.is_file():
        return False

    if file_path.stat().st_size != record.size_bytes:
        return False

    return _file_sha256(file_path) == record.sha256


def _snapshot_record(
    *,
    repository_root: Path,
    path: Path,
    payload: object,
    resource_key: str,
    resource_type: (RiotRawResourceType),
    source_endpoint: str,
    account_identity: (str | None) = None,
    match_id: str | None = None,
) -> RiotRawManifestRecord:
    body = canonical_json_bytes(payload)

    _atomic_write_bytes(
        path=path,
        payload=body,
    )

    return RiotRawManifestRecord(
        collector_version=(RIOT_RAW_COLLECTOR_VERSION),
        resource_key=(resource_key),
        resource_type=(resource_type),
        source_endpoint=(source_endpoint),
        fetched_at=datetime.now(UTC),
        file_path=_relative_path(
            repository_root=(repository_root),
            path=path,
        ),
        sha256=_bytes_sha256(body),
        size_bytes=len(body),
        account_identity_sha256=(account_identity),
        match_id=match_id,
    )


def collect_riot_match_snapshots(
    *,
    client: RiotRawCollectorClient,
    repository_root: Path,
    game_name: str,
    tag_line: str,
    start: int = 0,
    count: int = 3,
    force_refresh: bool = False,
) -> RiotRawCollectionResult:
    clean_game_name = game_name.strip()

    clean_tag_line = tag_line.strip().removeprefix("#").strip()

    if not clean_game_name:
        raise ValueError("game_name must not be empty")

    if not clean_tag_line:
        raise ValueError("tag_line must not be empty")

    if start < 0:
        raise ValueError("start must not be negative")

    if not 1 <= count <= 100:
        raise ValueError("count must be between 1 and 100")

    identity_sha = account_identity_sha256(
        game_name=(clean_game_name),
        tag_line=(clean_tag_line),
    )

    raw_root = repository_root / "data" / "raw" / "riot_api"

    manifest_path = repository_root / "data" / "manifests" / "riot_api" / "raw_sources.jsonl"

    records = _load_manifest(manifest_path)

    downloaded = 0
    reused = 0

    account = client.get_account_by_riot_id(
        game_name=(clean_game_name),
        tag_line=(clean_tag_line),
    )

    account_key = f"account:{identity_sha}"

    account_path = raw_root / "accounts" / identity_sha / "account.json"

    account_payload = account.model_dump(
        mode="json",
        by_alias=True,
    )

    account_body = canonical_json_bytes(account_payload)

    existing_account = records.get(account_key)

    account_reusable = (
        not force_refresh
        and _record_is_reusable(
            repository_root=(repository_root),
            record=existing_account,
        )
        and existing_account is not None
        and existing_account.sha256 == _bytes_sha256(account_body)
    )

    if account_reusable:
        reused += 1

    else:
        account_record = _snapshot_record(
            repository_root=(repository_root),
            path=account_path,
            payload=account_payload,
            resource_key=(account_key),
            resource_type=(RiotRawResourceType.ACCOUNT),
            source_endpoint=("/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}"),
            account_identity=(identity_sha),
        )

        records[account_key] = account_record

        downloaded += 1

    match_ids_result = client.get_match_ids_by_puuid(
        puuid=account.puuid,
        start=start,
        count=count,
    )

    for raw_match_id in match_ids_result.match_ids:
        match_id = _validate_match_id(raw_match_id)

        match_directory = raw_root / "matches" / match_id

        match_key = f"match:{match_id}"

        match_path = match_directory / "match.json"

        existing_match = records.get(match_key)

        if not force_refresh and _record_is_reusable(
            repository_root=(repository_root),
            record=(existing_match),
        ):
            reused += 1

        else:
            match_payload = client.get_match(match_id)

            match_record = _snapshot_record(
                repository_root=(repository_root),
                path=match_path,
                payload=(match_payload),
                resource_key=(match_key),
                resource_type=(RiotRawResourceType.MATCH),
                source_endpoint=("/lol/match/v5/matches/{matchId}"),
                account_identity=(identity_sha),
                match_id=(match_id),
            )

            records[match_key] = match_record

            downloaded += 1

        timeline_key = f"timeline:{match_id}"

        timeline_path = match_directory / "timeline.json"

        existing_timeline = records.get(timeline_key)

        if not force_refresh and _record_is_reusable(
            repository_root=(repository_root),
            record=(existing_timeline),
        ):
            reused += 1

        else:
            timeline_payload = client.get_timeline(match_id)

            timeline_record = _snapshot_record(
                repository_root=(repository_root),
                path=timeline_path,
                payload=(timeline_payload),
                resource_key=(timeline_key),
                resource_type=(RiotRawResourceType.TIMELINE),
                source_endpoint=("/lol/match/v5/matches/{matchId}/timeline"),
                account_identity=(identity_sha),
                match_id=(match_id),
            )

            records[timeline_key] = timeline_record

            downloaded += 1

    _write_manifest(
        path=manifest_path,
        records=records,
    )

    return RiotRawCollectionResult(
        requested_match_count=count,
        returned_match_count=len(match_ids_result.match_ids),
        downloaded_resources=(downloaded),
        reused_resources=reused,
        manifest_record_count=len(records),
        match_ids=(match_ids_result.match_ids),
    )
