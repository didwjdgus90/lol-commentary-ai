from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.riot_api.collector import (
    canonical_json_bytes,
)
from lol_commentary_backend.ingestion.riot_api.normalization.artifacts import (
    build_normalized_riot_dataset,
)
from lol_commentary_backend.ingestion.riot_api.raw_models import (
    RIOT_RAW_COLLECTOR_VERSION,
    RiotRawManifestRecord,
    RiotRawResourceType,
)


def _match() -> dict[str, object]:
    return {
        "metadata": {
            "matchId": "KR_1",
        },
        "info": {
            "gameMode": "CLASSIC",
            "queueId": 420,
            "mapId": 11,
            "gameDuration": 1800,
            "gameVersion": ("16.17.810.4348"),
            "participants": [
                {
                    "participantId": 1,
                    "teamId": 100,
                    "championId": 202,
                    "championName": "Jhin",
                    "teamPosition": ("BOTTOM"),
                    "individualPosition": ("BOTTOM"),
                    "win": True,
                    "puuid": ("secret-puuid"),
                    "riotIdGameName": ("Secret Player"),
                    "riotIdTagline": ("KR1"),
                }
            ],
        },
    }


def _timeline() -> dict[str, object]:
    return {
        "metadata": {
            "matchId": "KR_1",
        },
        "info": {
            "frames": [
                {
                    "timestamp": 60000,
                    "participantFrames": {
                        "1": {
                            "participantId": 1,
                            "level": 2,
                            "currentGold": 500,
                            "totalGold": 1000,
                            "xp": 400,
                            "minionsKilled": 10,
                            "jungleMinionsKilled": 0,
                            "goldPerSecond": 0,
                            "timeEnemySpentControlled": 0,
                            "position": {
                                "x": 1000,
                                "y": 2000,
                            },
                        }
                    },
                    "events": [
                        {
                            "type": ("ITEM_PURCHASED"),
                            "timestamp": 61000,
                            "participantId": 1,
                            "itemId": 3508,
                        }
                    ],
                }
            ],
        },
    }


def _write_raw(
    repository_root: Path,
) -> None:
    raw_root = repository_root / "data" / "raw" / "riot_api" / "matches" / "KR_1"

    raw_root.mkdir(parents=True)

    match_bytes = canonical_json_bytes(_match())

    timeline_bytes = canonical_json_bytes(_timeline())

    match_path = raw_root / "match.json"

    timeline_path = raw_root / "timeline.json"

    match_path.write_bytes(match_bytes)

    timeline_path.write_bytes(timeline_bytes)

    records = (
        RiotRawManifestRecord(
            collector_version=(RIOT_RAW_COLLECTOR_VERSION),
            resource_key="match:KR_1",
            resource_type=(RiotRawResourceType.MATCH),
            source_endpoint=("/lol/match/v5/matches/{matchId}"),
            fetched_at=("2026-08-31T00:00:00Z"),
            file_path=("data/raw/riot_api/matches/KR_1/match.json"),
            sha256=sha256(match_bytes).hexdigest(),
            size_bytes=len(match_bytes),
            match_id="KR_1",
        ),
        RiotRawManifestRecord(
            collector_version=(RIOT_RAW_COLLECTOR_VERSION),
            resource_key=("timeline:KR_1"),
            resource_type=(RiotRawResourceType.TIMELINE),
            source_endpoint=("/lol/match/v5/matches/{matchId}/timeline"),
            fetched_at=("2026-08-31T00:00:00Z"),
            file_path=("data/raw/riot_api/matches/KR_1/timeline.json"),
            sha256=sha256(timeline_bytes).hexdigest(),
            size_bytes=len(timeline_bytes),
            match_id="KR_1",
        ),
    )

    manifest_path = repository_root / "data" / "manifests" / "riot_api" / "raw_sources.jsonl"

    manifest_path.parent.mkdir(parents=True)

    text = "\n".join(
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )

    manifest_path.write_text(
        text + "\n",
        encoding="utf-8",
    )


def test_builds_processed_files(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    result = build_normalized_riot_dataset(repository_root=tmp_path)

    assert result.built_match_count == 1

    root = tmp_path / "data" / "processed" / "riot_api" / "normalized_v1" / "KR_1"

    assert (root / "match.json").is_file()

    assert (root / "participant_frames.jsonl").is_file()

    assert (root / "events.jsonl").is_file()

    assert (root / "manifest.json").is_file()


def test_aggregate_counts(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    result = build_normalized_riot_dataset(repository_root=tmp_path)

    manifest = result.manifest

    assert manifest.match_count == 1

    assert manifest.total_participant_count == 1

    assert manifest.total_participant_frame_count == 1

    assert manifest.total_event_count == 1


def test_processed_output_drops_player_identity(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    build_normalized_riot_dataset(repository_root=tmp_path)

    processed_root = tmp_path / "data" / "processed" / "riot_api" / "normalized_v1"

    text = "\n".join(path.read_text(encoding="utf-8") for path in (processed_root.rglob("*.json*")))

    assert "secret-puuid" not in text

    assert "Secret Player" not in text


def test_second_build_is_reused(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    first = build_normalized_riot_dataset(repository_root=tmp_path)

    second = build_normalized_riot_dataset(repository_root=tmp_path)

    assert first.built_match_count == 1

    assert second.built_match_count == 0

    assert second.reused_match_count == 1

    assert second.aggregate_manifest_changed is False


def test_processed_manifest_captures_raw_sha(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    build_normalized_riot_dataset(repository_root=tmp_path)

    path = tmp_path / "data" / "processed" / "riot_api" / "normalized_v1" / "KR_1" / "manifest.json"

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["source_match_sha256"]) == 64

    assert len(payload["source_timeline_sha256"]) == 64


def test_corrupted_raw_source_is_rejected(
    tmp_path: Path,
) -> None:
    _write_raw(tmp_path)

    match_path = tmp_path / "data" / "raw" / "riot_api" / "matches" / "KR_1" / "match.json"

    match_path.write_text(
        "corrupted",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"size mismatch|SHA256",
    ):
        build_normalized_riot_dataset(repository_root=tmp_path)
