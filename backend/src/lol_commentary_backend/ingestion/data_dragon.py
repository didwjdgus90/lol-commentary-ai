import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import httpx

BASE_URL = "https://ddragon.leagueoflegends.com"
VERSIONS_URL = f"{BASE_URL}/api/versions.json"

DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
DEFAULT_HEADERS = {
    "User-Agent": "LoLCommentaryAI/0.1 (educational data collector)",
}
SCHEMA_VERSION = 1
COLLECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class DataDragonTarget:
    version: str
    locale: str


@dataclass(frozen=True)
class RawDataDragonResource:
    name: str
    source_url: str
    fetched_at: datetime
    status_code: int
    content_type: str | None
    content: bytes
    sha256: str


@dataclass(frozen=True)
class RawDataDragonBundle:
    target: DataDragonTarget
    versions: RawDataDragonResource
    champion: RawDataDragonResource
    item: RawDataDragonResource


def _resource_url(
    target: DataDragonTarget,
    resource_name: str,
) -> str:
    return f"{BASE_URL}/cdn/{target.version}/data/{target.locale}/{resource_name}.json"


def _fetch_resource(
    client: httpx.Client,
    *,
    name: str,
    source_url: str,
) -> RawDataDragonResource:
    response = client.get(source_url)
    response.raise_for_status()

    content = response.content

    return RawDataDragonResource(
        name=name,
        source_url=source_url,
        fetched_at=datetime.now(UTC),
        status_code=response.status_code,
        content_type=response.headers.get("content-type"),
        content=content,
        sha256=sha256(content).hexdigest(),
    )


def _validate_version(
    versions_resource: RawDataDragonResource,
    target: DataDragonTarget,
) -> None:
    try:
        payload = json.loads(versions_resource.content)
    except json.JSONDecodeError as exc:
        raise ValueError("Data Dragon versions.json is not valid JSON") from exc

    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        raise ValueError("Data Dragon versions.json must be a list of strings")

    if target.version not in payload:
        raise ValueError(f"Data Dragon version not found: {target.version}")


def fetch_data_dragon_bundle(
    target: DataDragonTarget,
    *,
    transport: httpx.BaseTransport | None = None,
) -> RawDataDragonBundle:
    with httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        transport=transport,
    ) as client:
        versions = _fetch_resource(
            client,
            name="versions",
            source_url=VERSIONS_URL,
        )
        _validate_version(versions, target)

        champion = _fetch_resource(
            client,
            name="champion",
            source_url=_resource_url(target, "champion"),
        )
        item = _fetch_resource(
            client,
            name="item",
            source_url=_resource_url(target, "item"),
        )

    return RawDataDragonBundle(
        target=target,
        versions=versions,
        champion=champion,
        item=item,
    )


def save_raw_data_dragon_bundle(
    bundle: RawDataDragonBundle,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    resources = (
        bundle.versions,
        bundle.champion,
        bundle.item,
    )

    saved_paths: dict[str, Path] = {}

    for resource in resources:
        path = output_dir / f"{resource.name}.json"
        path.write_bytes(resource.content)
        saved_paths[resource.name] = path

    metadata_path = output_dir / "metadata.json"

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "ddragon_version": bundle.target.version,
        "locale": bundle.target.locale,
        "resources": {
            resource.name: {
                "source_url": resource.source_url,
                "fetched_at": resource.fetched_at.isoformat(),
                "status_code": resource.status_code,
                "content_type": resource.content_type,
                "sha256": resource.sha256,
                "size_bytes": len(resource.content),
            }
            for resource in resources
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        saved_paths["versions"],
        saved_paths["champion"],
        saved_paths["item"],
        metadata_path,
    )
