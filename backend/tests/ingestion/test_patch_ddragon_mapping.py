import json
from pathlib import Path

import pytest

from lol_commentary_backend.ingestion.version_mapping.builder import (
    build_patch_ddragon_mappings,
    save_patch_ddragon_mapping,
)


def _write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _ddragon_rows(
    version: str,
) -> list[dict]:
    rows: list[dict] = []

    for locale in (
        "ko_KR",
        "en_US",
    ):
        for resource in (
            "champion_summary",
            "items",
            "runes",
            "summoner_spells",
        ):
            rows.append(
                {
                    "version": version,
                    "locale": locale,
                    "resource_kind": resource,
                }
            )

    return rows


def test_maps_patch_minor_to_collected_ddragon_minor(
    tmp_path: Path,
) -> None:
    patch_manifest = tmp_path / "patch.jsonl"

    ddragon_manifest = tmp_path / "ddragon.jsonl"

    _write_jsonl(
        patch_manifest,
        [
            {
                "patch": "26.1",
                "locale": "ko_KR",
            },
            {
                "patch": "26.1",
                "locale": "en_US",
            },
        ],
    )

    _write_jsonl(
        ddragon_manifest,
        _ddragon_rows("16.1.1"),
    )

    mappings = build_patch_ddragon_mappings(
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
        start_patch="26.1",
        end_patch="26.1",
    )

    assert len(mappings) == 1

    assert mappings[0].patch == "26.1"

    assert mappings[0].ddragon_version == "16.1.1"


def test_selects_latest_build_for_minor(
    tmp_path: Path,
) -> None:
    patch_manifest = tmp_path / "patch.jsonl"

    ddragon_manifest = tmp_path / "ddragon.jsonl"

    _write_jsonl(
        patch_manifest,
        [
            {
                "patch": "26.1",
                "locale": "ko_KR",
            },
            {
                "patch": "26.1",
                "locale": "en_US",
            },
        ],
    )

    rows = _ddragon_rows("16.1.1") + _ddragon_rows("16.1.2")

    _write_jsonl(
        ddragon_manifest,
        rows,
    )

    mappings = build_patch_ddragon_mappings(
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
        start_patch="26.1",
        end_patch="26.1",
    )

    assert mappings[0].ddragon_version == "16.1.2"


def test_rejects_missing_patch_locale(
    tmp_path: Path,
) -> None:
    patch_manifest = tmp_path / "patch.jsonl"

    ddragon_manifest = tmp_path / "ddragon.jsonl"

    _write_jsonl(
        patch_manifest,
        [
            {
                "patch": "26.1",
                "locale": "ko_KR",
            },
        ],
    )

    _write_jsonl(
        ddragon_manifest,
        _ddragon_rows("16.1.1"),
    )

    with pytest.raises(
        ValueError,
        match="missing patch-note",
    ):
        build_patch_ddragon_mappings(
            patch_manifest_path=(patch_manifest),
            ddragon_manifest_path=(ddragon_manifest),
            start_patch="26.1",
            end_patch="26.1",
        )


def test_rejects_incomplete_ddragon_snapshot(
    tmp_path: Path,
) -> None:
    patch_manifest = tmp_path / "patch.jsonl"

    ddragon_manifest = tmp_path / "ddragon.jsonl"

    _write_jsonl(
        patch_manifest,
        [
            {
                "patch": "26.1",
                "locale": "ko_KR",
            },
            {
                "patch": "26.1",
                "locale": "en_US",
            },
        ],
    )

    _write_jsonl(
        ddragon_manifest,
        [
            {
                "version": "16.1.1",
                "locale": "ko_KR",
                "resource_kind": ("champion_summary"),
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match=("no complete Data Dragon"),
    ):
        build_patch_ddragon_mappings(
            patch_manifest_path=(patch_manifest),
            ddragon_manifest_path=(ddragon_manifest),
            start_patch="26.1",
            end_patch="26.1",
        )


def test_saves_mapping_lineage(
    tmp_path: Path,
) -> None:
    patch_manifest = tmp_path / "patch.jsonl"

    ddragon_manifest = tmp_path / "ddragon.jsonl"

    _write_jsonl(
        patch_manifest,
        [
            {
                "patch": "26.1",
                "locale": "ko_KR",
            },
            {
                "patch": "26.1",
                "locale": "en_US",
            },
        ],
    )

    _write_jsonl(
        ddragon_manifest,
        _ddragon_rows("16.1.1"),
    )

    mappings = build_patch_ddragon_mappings(
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
        start_patch="26.1",
        end_patch="26.1",
    )

    (
        mappings_path,
        metadata_path,
        metadata,
    ) = save_patch_ddragon_mapping(
        output_dir=(tmp_path / "output"),
        mappings=mappings,
        patch_manifest_path=(patch_manifest),
        ddragon_manifest_path=(ddragon_manifest),
    )

    assert mappings_path.is_file()
    assert metadata_path.is_file()

    assert metadata.mapping_count == 1

    assert len(metadata.patch_manifest_sha256) == 64

    assert len(metadata.ddragon_manifest_sha256) == 64
