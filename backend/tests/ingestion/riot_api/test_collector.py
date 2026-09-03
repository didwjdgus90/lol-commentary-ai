from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.riot_api.collector import (
    canonical_json_bytes,
    collect_riot_match_snapshots,
    parse_riot_id,
)
from lol_commentary_backend.ingestion.riot_api.models import (
    RiotAccount,
    RiotMatchIds,
)


class FakeRiotClient:
    def __init__(
        self,
    ) -> None:
        self.account_calls = 0
        self.match_id_calls = 0
        self.match_calls = 0
        self.timeline_calls = 0

    def get_account_by_riot_id(
        self,
        *,
        game_name: str,
        tag_line: str,
    ) -> RiotAccount:
        self.account_calls += 1

        return RiotAccount(
            puuid="test-puuid",
            gameName=game_name,
            tagLine=tag_line,
        )

    def get_match_ids_by_puuid(
        self,
        *,
        puuid: str,
        start: int = 0,
        count: int = 20,
    ) -> RiotMatchIds:
        self.match_id_calls += 1

        assert puuid == "test-puuid"

        return RiotMatchIds(
            puuid=puuid,
            start=start,
            count=count,
            match_ids=("KR_100",),
        )

    def get_match(
        self,
        match_id: str,
    ) -> dict[str, object]:
        self.match_calls += 1

        return {
            "metadata": {
                "matchId": match_id,
            },
            "info": {
                "gameDuration": 1800,
            },
        }

    def get_timeline(
        self,
        match_id: str,
    ) -> dict[str, object]:
        self.timeline_calls += 1

        return {
            "metadata": {
                "matchId": match_id,
            },
            "info": {
                "frames": [
                    {
                        "timestamp": 0,
                    }
                ],
            },
        }


def test_parse_riot_id() -> None:
    assert parse_riot_id("Player#KR1") == (
        "Player",
        "KR1",
    )


def test_parse_riot_id_rejects_double_hash() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        parse_riot_id("Player##KR1")


def test_canonical_json_is_deterministic() -> None:
    first = canonical_json_bytes(
        {
            "b": 2,
            "a": 1,
        }
    )

    second = canonical_json_bytes(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert first == second


def test_first_collection_writes_raw_snapshots(
    tmp_path: Path,
) -> None:
    client = FakeRiotClient()

    result = collect_riot_match_snapshots(
        client=client,
        repository_root=(tmp_path),
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    assert result.downloaded_resources == 3

    assert result.reused_resources == 0

    assert client.account_calls == 1

    assert client.match_id_calls == 1

    assert client.match_calls == 1

    assert client.timeline_calls == 1

    match_path = tmp_path / "data" / "raw" / "riot_api" / "matches" / "KR_100" / "match.json"

    timeline_path = match_path.parent / "timeline.json"

    manifest_path = tmp_path / "data" / "manifests" / "riot_api" / "raw_sources.jsonl"

    assert match_path.is_file()

    assert timeline_path.is_file()

    assert manifest_path.is_file()

    records = [
        json.loads(line)
        for line in (manifest_path.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]

    assert len(records) == 3


def test_second_collection_reuses_match_and_timeline(
    tmp_path: Path,
) -> None:
    first_client = FakeRiotClient()

    collect_riot_match_snapshots(
        client=first_client,
        repository_root=tmp_path,
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    second_client = FakeRiotClient()

    result = collect_riot_match_snapshots(
        client=second_client,
        repository_root=(tmp_path),
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    assert result.downloaded_resources == 0

    assert result.reused_resources == 3

    assert second_client.account_calls == 1

    assert second_client.match_id_calls == 1

    assert second_client.match_calls == 0

    assert second_client.timeline_calls == 0


def test_corrupted_match_is_refetched(
    tmp_path: Path,
) -> None:
    first_client = FakeRiotClient()

    collect_riot_match_snapshots(
        client=first_client,
        repository_root=tmp_path,
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    match_path = tmp_path / "data" / "raw" / "riot_api" / "matches" / "KR_100" / "match.json"

    match_path.write_text(
        "corrupted",
        encoding="utf-8",
    )

    second_client = FakeRiotClient()

    result = collect_riot_match_snapshots(
        client=second_client,
        repository_root=(tmp_path),
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    assert second_client.match_calls == 1

    assert second_client.timeline_calls == 0

    assert result.downloaded_resources == 1

    assert result.reused_resources == 2


def test_manifest_does_not_contain_api_key(
    tmp_path: Path,
) -> None:
    client = FakeRiotClient()

    collect_riot_match_snapshots(
        client=client,
        repository_root=tmp_path,
        game_name="Player",
        tag_line="KR1",
        count=1,
    )

    manifest_path = tmp_path / "data" / "manifests" / "riot_api" / "raw_sources.jsonl"

    text = manifest_path.read_text(encoding="utf-8")

    assert "RGAPI-" not in text
