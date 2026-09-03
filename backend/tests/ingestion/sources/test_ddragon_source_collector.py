from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.sources.ddragon import (
    StaticSnapshotRecord,
    extract_champion_ids,
    parse_ddragon_version,
    select_latest_builds,
    sha256_hex,
    validate_explicit_versions,
    write_merged_manifest,
)


def test_parses_semantic_ddragon_version() -> None:
    assert parse_ddragon_version("16.17.1") == (
        16,
        17,
        1,
    )


def test_rejects_non_semantic_version() -> None:
    with pytest.raises(
        ValueError,
        match=("Unsupported Data Dragon"),
    ):
        parse_ddragon_version("lolpatch_7.20")


def test_selects_latest_build_for_each_minor() -> None:
    versions = [
        "16.3.1",
        "16.2.2",
        "16.2.1",
        "16.1.1",
        "15.24.1",
    ]

    selected = select_latest_builds(
        available_versions=(versions),
        major=16,
        start_minor=1,
        end_minor=3,
    )

    assert selected == (
        "16.1.1",
        "16.2.2",
        "16.3.1",
    )


def test_rejects_missing_minor_version() -> None:
    with pytest.raises(
        ValueError,
        match=("Missing Data Dragon"),
    ):
        select_latest_builds(
            available_versions=[
                "16.1.1",
                "16.3.1",
            ],
            major=16,
            start_minor=1,
            end_minor=3,
        )


def test_validates_explicit_versions() -> None:
    selected = validate_explicit_versions(
        requested_versions=(
            "16.17.1",
            "16.1.1",
        ),
        available_versions=[
            "16.1.1",
            "16.17.1",
        ],
    )

    assert selected == (
        "16.1.1",
        "16.17.1",
    )


def test_rejects_unknown_explicit_version() -> None:
    with pytest.raises(
        ValueError,
        match=("Unknown Data Dragon"),
    ):
        validate_explicit_versions(
            requested_versions=("99.1.1",),
            available_versions=[
                "16.17.1",
            ],
        )


def test_extracts_sorted_champion_ids() -> None:
    payload = {
        "data": {
            "Zed": {},
            "Ahri": {},
            "Aatrox": {},
        }
    }

    assert extract_champion_ids(payload) == (
        "Aatrox",
        "Ahri",
        "Zed",
    )


def test_rejects_invalid_champion_summary() -> None:
    with pytest.raises(
        ValueError,
        match="no data object",
    ):
        extract_champion_ids({})


def test_sha256_is_deterministic() -> None:
    first = sha256_hex(b"ddragon")

    second = sha256_hex(b"ddragon")

    assert first == second
    assert len(first) == 64


def test_manifest_merge_preserves_old_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"

    old = StaticSnapshotRecord(
        source_id="ddragon_items",
        resource_kind="items",
        version="16.1.1",
        locale="ko_KR",
        source_url=("https://example.com/old"),
        file_path="old.json",
        content_sha256="a" * 64,
        byte_count=10,
        snapshot_at=("2026-01-01T00:00:00Z"),
    )

    new = StaticSnapshotRecord(
        source_id="ddragon_items",
        resource_kind="items",
        version="16.17.1",
        locale="ko_KR",
        source_url=("https://example.com/new"),
        file_path="new.json",
        content_sha256="b" * 64,
        byte_count=20,
        snapshot_at=("2026-08-31T00:00:00Z"),
    )

    write_merged_manifest(
        path=path,
        records=(old,),
    )

    write_merged_manifest(
        path=path,
        records=(new,),
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    assert '"version": "16.1.1"' in lines[0]

    assert '"version": "16.17.1"' in lines[1]


def test_manifest_merge_updates_same_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"

    first = StaticSnapshotRecord(
        source_id="ddragon_items",
        resource_kind="items",
        version="16.17.1",
        locale="ko_KR",
        source_url=("https://example.com/items"),
        file_path="items.json",
        content_sha256="a" * 64,
        byte_count=10,
        snapshot_at=("2026-08-30T00:00:00Z"),
    )

    second = first.model_copy(
        update={
            "content_sha256": ("b" * 64),
            "byte_count": 20,
        }
    )

    write_merged_manifest(
        path=path,
        records=(first,),
    )

    write_merged_manifest(
        path=path,
        records=(second,),
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    assert f'"content_sha256": "{"b" * 64}"' in lines[0]
