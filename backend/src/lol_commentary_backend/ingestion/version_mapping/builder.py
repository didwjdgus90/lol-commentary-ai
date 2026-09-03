from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.version_mapping.models import (
    PatchDataDragonMapping,
    VersionMappingMetadata,
)

_REQUIRED_LOCALES = (
    "ko_KR",
    "en_US",
)

_REQUIRED_DDRAGON_RESOURCES = {
    "champion_summary",
    "items",
    "runes",
    "summoner_spells",
}


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")

        rows.append(payload)

    return rows


def _patch_minor(
    patch: str,
) -> int:
    major, minor = patch.split(".")

    if major != "26":
        raise ValueError(f"Step 43-5 only supports 2026 patches: {patch}")

    return int(minor)


def _ddragon_key(
    version: str,
) -> tuple[
    int,
    int,
    int,
]:
    parts = version.split(".")

    if len(parts) != 3:
        raise ValueError(f"Invalid Data Dragon version: {version}")

    major_text, minor_text, build_text = parts

    try:
        major = int(major_text)
        minor = int(minor_text)
        build = int(build_text)
    except ValueError as exc:
        raise ValueError(f"Invalid Data Dragon version: {version}") from exc

    return (
        major,
        minor,
        build,
    )


def _build_patch_locale_index(
    rows: list[dict[str, Any]],
) -> dict[
    str,
    set[str],
]:
    result: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in rows:
        patch = row.get("patch")
        locale = row.get("locale")

        if isinstance(patch, str) and isinstance(
            locale,
            str,
        ):
            result[patch].add(locale)

    return result


def _build_ddragon_matrix(
    rows: list[dict[str, Any]],
) -> dict[
    str,
    dict[
        str,
        set[str],
    ],
]:
    matrix: dict[
        str,
        dict[
            str,
            set[str],
        ],
    ] = defaultdict(lambda: defaultdict(set))

    for row in rows:
        version = row.get("version")

        locale = row.get("locale")

        resource_kind = row.get("resource_kind")

        if not isinstance(
            version,
            str,
        ):
            continue

        if not isinstance(
            locale,
            str,
        ):
            continue

        if not isinstance(
            resource_kind,
            str,
        ):
            continue

        if not version.startswith("16."):
            continue

        matrix[version][locale].add(resource_kind)

    return matrix


def _complete_ddragon_versions(
    matrix: dict[
        str,
        dict[
            str,
            set[str],
        ],
    ],
) -> tuple[str, ...]:
    complete: list[str] = []

    for version, locale_data in matrix.items():
        valid = True

        for locale in _REQUIRED_LOCALES:
            resources = locale_data.get(
                locale,
                set(),
            )

            if not (_REQUIRED_DDRAGON_RESOURCES <= resources):
                valid = False
                break

        if valid:
            complete.append(version)

    return tuple(
        sorted(
            complete,
            key=_ddragon_key,
        )
    )


def _select_latest_build_by_minor(
    versions: tuple[
        str,
        ...,
    ],
) -> dict[
    int,
    str,
]:
    selected: dict[
        int,
        str,
    ] = {}

    for version in versions:
        major, minor, build = _ddragon_key(version)

        if major != 16:
            continue

        current = selected.get(minor)

        if current is None:
            selected[minor] = version
            continue

        current_build = _ddragon_key(current)[2]

        if build > current_build:
            selected[minor] = version

    return selected


def build_patch_ddragon_mappings(
    *,
    patch_manifest_path: Path,
    ddragon_manifest_path: Path,
    start_patch: str = "26.1",
    end_patch: str = "26.17",
) -> tuple[
    PatchDataDragonMapping,
    ...,
]:
    patch_rows = _read_jsonl(patch_manifest_path)

    ddragon_rows = _read_jsonl(ddragon_manifest_path)

    patch_index = _build_patch_locale_index(patch_rows)

    ddragon_matrix = _build_ddragon_matrix(ddragon_rows)

    complete_versions = _complete_ddragon_versions(ddragon_matrix)

    selected_by_minor = _select_latest_build_by_minor(complete_versions)

    start_minor = _patch_minor(start_patch)

    end_minor = _patch_minor(end_patch)

    if start_minor > end_minor:
        raise ValueError("start_patch must not be after end_patch")

    mappings: list[PatchDataDragonMapping] = []

    for minor in range(
        start_minor,
        end_minor + 1,
    ):
        patch = f"26.{minor}"

        patch_locales = patch_index.get(
            patch,
            set(),
        )

        missing_patch_locales = set(_REQUIRED_LOCALES) - patch_locales

        if missing_patch_locales:
            raise ValueError(
                f"{patch} is missing patch-note locales: {sorted(missing_patch_locales)}"
            )

        ddragon_version = selected_by_minor.get(minor)

        if ddragon_version is None:
            raise ValueError(f"{patch} has no complete Data Dragon snapshot")

        ddragon_locales = tuple(sorted(ddragon_matrix[ddragon_version]))

        mappings.append(
            PatchDataDragonMapping(
                patch=patch,
                ddragon_version=(ddragon_version),
                patch_locales=(tuple(sorted(patch_locales))),
                ddragon_locales=(ddragon_locales),
            )
        )

    return tuple(mappings)


def save_patch_ddragon_mapping(
    *,
    output_dir: Path,
    mappings: tuple[
        PatchDataDragonMapping,
        ...,
    ],
    patch_manifest_path: Path,
    ddragon_manifest_path: Path,
) -> tuple[
    Path,
    Path,
    VersionMappingMetadata,
]:
    if not mappings:
        raise ValueError("mappings must not be empty")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mappings_path = output_dir / "patch_ddragon.jsonl"

    metadata_path = output_dir / "metadata.json"

    lines = [
        json.dumps(
            mapping.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        for mapping in mappings
    ]

    temporary = mappings_path.with_suffix(".jsonl.tmp")

    temporary.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    temporary.replace(mappings_path)

    metadata = VersionMappingMetadata(
        mapping_count=len(mappings),
        first_patch=(mappings[0].patch),
        last_patch=(mappings[-1].patch),
        patch_manifest_sha256=(_sha256_file(patch_manifest_path)),
        ddragon_manifest_sha256=(_sha256_file(ddragon_manifest_path)),
        mappings_sha256=(_sha256_file(mappings_path)),
    )

    temporary_metadata = metadata_path.with_suffix(".json.tmp")

    temporary_metadata.write_text(
        json.dumps(
            metadata.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_metadata.replace(metadata_path)

    return (
        mappings_path,
        metadata_path,
        metadata,
    )
