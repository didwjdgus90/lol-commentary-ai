from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field
from selectolax.parser import HTMLParser

from lol_commentary_backend.ingestion.sources.registry import (
    get_data_source,
)

PatchLocale = Literal[
    "ko_KR",
    "en_US",
]

PatchCollectionAction = Literal[
    "downloaded",
    "reused",
]


_LOCALE_SITE_PATH: dict[
    PatchLocale,
    str,
] = {
    "ko_KR": "ko-kr",
    "en_US": "en-us",
}

_LOCALE_INDEX_SOURCE: dict[
    PatchLocale,
    str,
] = {
    "ko_KR": "patch_notes_ko_index",
    "en_US": "patch_notes_en_index",
}

_PATCH_PATTERN = re.compile(
    r"^(?P<major>\d{2})\."
    r"(?P<minor>[1-9]\d?)$"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"

_OFFICIAL_HOST = "www.leagueoflegends.com"

_USER_AGENT = "lol-commentary-ai/patch-note-collector-v1"


class PatchNoteManifestRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    source_id: str = Field(
        min_length=1,
    )

    patch: str = Field(
        pattern=(r"^\d{2}\.[1-9]\d?$"),
    )

    locale: PatchLocale

    title: str = Field(
        min_length=1,
    )

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


class PatchNoteCollectionItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    action: PatchCollectionAction

    manifest: PatchNoteManifestRecord


class PatchNoteCollectionSummary(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1

    requested_count: int = Field(
        ge=0,
    )

    downloaded_count: int = Field(
        ge=0,
    )

    reused_count: int = Field(
        ge=0,
    )

    manifest_path: str = Field(
        min_length=1,
    )

    items: tuple[
        PatchNoteCollectionItem,
        ...,
    ]


def _parse_patch(
    patch: str,
) -> tuple[int, int]:
    match = _PATCH_PATTERN.fullmatch(patch)

    if match is None:
        raise ValueError(f"Invalid patch version: {patch}")

    return (
        int(match.group("major")),
        int(match.group("minor")),
    )


def build_patch_sequence(
    *,
    start_patch: str,
    end_patch: str,
) -> tuple[str, ...]:
    start_major, start_minor = _parse_patch(start_patch)

    end_major, end_minor = _parse_patch(end_patch)

    if start_major != end_major:
        raise ValueError("Patch range must stay within one major version")

    if start_minor > end_minor:
        raise ValueError("start_patch must not be after end_patch")

    return tuple(
        f"{start_major}.{minor}"
        for minor in range(
            start_minor,
            end_minor + 1,
        )
    )


def sha256_hex(
    content: bytes,
) -> str:
    return hashlib.sha256(content).hexdigest()


def build_candidate_urls(
    *,
    patch: str,
    locale: PatchLocale,
) -> tuple[str, ...]:
    major, minor = _parse_patch(patch)

    site_locale = _LOCALE_SITE_PATH[locale]

    base = f"https://{_OFFICIAL_HOST}/{site_locale}/news/game-updates/"

    return (
        (f"{base}patch-{major}-{minor}-notes/"),
        (f"{base}league-of-legends-patch-{major}-{minor}-notes/"),
    )


def _contains_patch(
    value: str,
    patch: str,
) -> bool:
    major, minor = _parse_patch(patch)

    dot_pattern = re.compile(
        rf"(?<!\d)"
        rf"{major}\.{minor}"
        rf"(?!\d)"
    )

    dash_pattern = re.compile(
        rf"(?<!\d)"
        rf"{major}-{minor}"
        rf"(?!\d)"
    )

    return dot_pattern.search(value) is not None or dash_pattern.search(value) is not None


def discover_patch_note_links(
    *,
    index_html: str,
    index_url: str,
    target_patches: tuple[
        str,
        ...,
    ],
) -> dict[str, str]:
    tree = HTMLParser(index_html)

    discovered: dict[
        str,
        str,
    ] = {}

    for anchor in tree.css("a"):
        href = anchor.attributes.get("href")

        if not href:
            continue

        absolute_url = urljoin(
            index_url,
            href,
        )

        parsed = urlparse(absolute_url)

        if parsed.netloc != _OFFICIAL_HOST:
            continue

        if "/news/game-updates/" not in parsed.path:
            continue

        anchor_text = " ".join(
            anchor.text(
                separator=" ",
                strip=True,
            ).split()
        )

        searchable = f"{anchor_text} {parsed.path}"

        for patch in target_patches:
            if patch in discovered:
                continue

            if _contains_patch(
                searchable,
                patch,
            ):
                discovered[patch] = absolute_url

    return discovered


def validate_patch_note_html(
    *,
    html: str,
    patch: str,
) -> str:
    tree = HTMLParser(html)

    heading = tree.css_first("h1")

    if heading is None:
        raise ValueError("Patch note page has no h1")

    title = " ".join(
        heading.text(
            separator=" ",
            strip=True,
        ).split()
    )

    if not title:
        raise ValueError("Patch note title is empty")

    if not _contains_patch(
        title,
        patch,
    ):
        raise ValueError(f"Patch note title does not match {patch}: {title}")

    return title


def _validate_official_url(
    url: str,
) -> None:
    parsed = urlparse(url)

    if (
        parsed.scheme != "https"
        or parsed.netloc != _OFFICIAL_HOST
        or "/news/game-updates/" not in parsed.path
    ):
        raise ValueError(
            f"Resolved patch note URL is not an official League game update URL: {url}"
        )


def _unique_urls(
    urls: list[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(urls))


def resolve_patch_note_page(
    *,
    client: httpx.Client,
    patch: str,
    locale: PatchLocale,
    discovered_url: str | None,
) -> tuple[
    str,
    str,
    bytes,
]:
    candidates: list[str] = []

    if discovered_url:
        candidates.append(discovered_url)

    candidates.extend(
        build_candidate_urls(
            patch=patch,
            locale=locale,
        )
    )

    errors: list[str] = []

    for url in _unique_urls(candidates):
        _validate_official_url(url)

        response = client.get(url)

        if response.status_code == httpx.codes.NOT_FOUND:
            errors.append(f"{url} -> 404")
            continue

        response.raise_for_status()

        if not response.content:
            errors.append(f"{url} -> empty body")
            continue

        content_type = response.headers.get(
            "content-type",
            "",
        )

        if "text/html" not in content_type:
            errors.append(f"{url} -> unexpected content-type {content_type}")
            continue

        try:
            title = validate_patch_note_html(
                html=response.text,
                patch=patch,
            )
        except ValueError as exc:
            errors.append(f"{url} -> {exc}")
            continue

        final_url = str(response.url)

        _validate_official_url(final_url)

        return (
            final_url,
            title,
            response.content,
        )

    raise RuntimeError(
        f"Could not resolve official patch note for {patch} {locale}. Attempts: {errors}"
    )


def _canonical_target_path(
    *,
    repository_root: Path,
    patch: str,
    locale: PatchLocale,
) -> Path:
    locale_dir = locale.lower()

    return repository_root / "data" / "raw" / "patch_notes" / patch / locale_dir / "official.html"


def _find_existing_html(
    target_path: Path,
) -> Path | None:
    if target_path.exists():
        return target_path

    target_dir = target_path.parent

    if not target_dir.exists():
        return None

    html_files = sorted(path for path in target_dir.glob("*.html") if path.is_file())

    if len(html_files) > 1:
        raise RuntimeError(f"Multiple existing HTML snapshots found in {target_dir}")

    if html_files:
        return html_files[0]

    return None


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


def _snapshot_time(
    path: Path,
) -> datetime:
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=UTC,
    )


def _relative_posix(
    *,
    path: Path,
    repository_root: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def _manifest_sort_key(
    record: PatchNoteManifestRecord,
) -> tuple[
    int,
    int,
    str,
]:
    major, minor = _parse_patch(record.patch)

    return (
        major,
        minor,
        record.locale,
    )


def write_patch_note_manifest(
    *,
    repository_root: Path,
    records: tuple[
        PatchNoteManifestRecord,
        ...,
    ],
) -> Path:
    manifest_path = repository_root / "data" / "manifests" / "patch_notes" / "raw_sources.jsonl"

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = sorted(
        records,
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

    payload = "\n".join(lines) + "\n" if lines else ""

    temporary = manifest_path.with_suffix(".jsonl.tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(manifest_path)

    return manifest_path


def collect_patch_notes(
    *,
    repository_root: Path,
    start_patch: str,
    end_patch: str,
    locales: tuple[
        PatchLocale,
        ...,
    ] = (
        "ko_KR",
        "en_US",
    ),
    timeout_seconds: float = 30.0,
    overwrite: bool = False,
) -> PatchNoteCollectionSummary:
    patches = build_patch_sequence(
        start_patch=start_patch,
        end_patch=end_patch,
    )

    if not locales:
        raise ValueError("At least one locale must be requested")

    discovered_by_locale: dict[
        PatchLocale,
        dict[str, str],
    ] = {}

    items: list[PatchNoteCollectionItem] = []

    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html",
        },
    ) as client:
        for locale in locales:
            source = get_data_source(_LOCALE_INDEX_SOURCE[locale])

            response = client.get(source.url_template)

            response.raise_for_status()

            discovered_by_locale[locale] = discover_patch_note_links(
                index_html=(response.text),
                index_url=(source.url_template),
                target_patches=patches,
            )

        for patch in patches:
            for locale in locales:
                discovered_url = discovered_by_locale[locale].get(patch)

                (
                    source_url,
                    remote_title,
                    remote_content,
                ) = resolve_patch_note_page(
                    client=client,
                    patch=patch,
                    locale=locale,
                    discovered_url=(discovered_url),
                )

                target_path = _canonical_target_path(
                    repository_root=(repository_root),
                    patch=patch,
                    locale=locale,
                )

                existing_path = None if overwrite else _find_existing_html(target_path)

                if existing_path is not None:
                    content = existing_path.read_bytes()

                    try:
                        existing_title = validate_patch_note_html(
                            html=(content.decode("utf-8")),
                            patch=patch,
                        )
                    except (
                        UnicodeDecodeError,
                        ValueError,
                    ) as exc:
                        raise RuntimeError(
                            f"Existing patch snapshot is invalid: {existing_path}"
                        ) from exc

                    snapshot_path = existing_path

                    title = existing_title

                    action: PatchCollectionAction = "reused"

                else:
                    _write_bytes_atomic(
                        path=target_path,
                        content=(remote_content),
                    )

                    snapshot_path = target_path

                    content = remote_content

                    title = remote_title

                    action = "downloaded"

                source_id = _LOCALE_INDEX_SOURCE[locale]

                manifest = PatchNoteManifestRecord(
                    source_id=source_id,
                    patch=patch,
                    locale=locale,
                    title=title,
                    source_url=(source_url),
                    file_path=(
                        _relative_posix(
                            path=(snapshot_path),
                            repository_root=(repository_root),
                        )
                    ),
                    content_sha256=(sha256_hex(content)),
                    byte_count=len(content),
                    snapshot_at=(_snapshot_time(snapshot_path)),
                )

                items.append(
                    PatchNoteCollectionItem(
                        action=action,
                        manifest=manifest,
                    )
                )

    records = tuple(item.manifest for item in items)

    manifest_path = write_patch_note_manifest(
        repository_root=(repository_root),
        records=records,
    )

    downloaded_count = sum(item.action == "downloaded" for item in items)

    reused_count = sum(item.action == "reused" for item in items)

    return PatchNoteCollectionSummary(
        requested_count=(len(patches) * len(locales)),
        downloaded_count=(downloaded_count),
        reused_count=(reused_count),
        manifest_path=(
            _relative_posix(
                path=manifest_path,
                repository_root=(repository_root),
            )
        ),
        items=tuple(items),
    )
