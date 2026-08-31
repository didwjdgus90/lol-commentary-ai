import json
from hashlib import sha256

import httpx
import pytest

from lol_commentary_backend.ingestion.data_dragon import (
    DataDragonTarget,
    fetch_data_dragon_bundle,
    save_raw_data_dragon_bundle,
)

TARGET = DataDragonTarget(
    version="16.1.1",
    locale="ko_KR",
)


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path

    if path == "/api/versions.json":
        content = json.dumps(["16.2.1", "16.1.1", "15.24.1"]).encode()
    elif path.endswith("/champion.json"):
        content = json.dumps(
            {
                "type": "champion",
                "version": "16.1.1",
                "data": {
                    "Aphelios": {
                        "id": "Aphelios",
                        "key": "523",
                        "name": "아펠리오스",
                    }
                },
            },
            ensure_ascii=False,
        ).encode()
    elif path.endswith("/item.json"):
        content = json.dumps(
            {
                "type": "item",
                "version": "16.1.1",
                "data": {
                    "3508": {
                        "name": "정수 약탈자",
                    }
                },
            },
            ensure_ascii=False,
        ).encode()
    else:
        return httpx.Response(404, request=request)

    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=content,
        request=request,
    )


def test_fetch_data_dragon_bundle_uses_version_and_locale() -> None:
    bundle = fetch_data_dragon_bundle(
        TARGET,
        transport=httpx.MockTransport(_handler),
    )

    assert bundle.target.version == "16.1.1"
    assert bundle.target.locale == "ko_KR"

    assert (
        bundle.champion.source_url == "https://ddragon.leagueoflegends.com/"
        "cdn/16.1.1/data/ko_KR/champion.json"
    )
    assert (
        bundle.item.source_url == "https://ddragon.leagueoflegends.com/"
        "cdn/16.1.1/data/ko_KR/item.json"
    )


def test_fetch_data_dragon_bundle_rejects_unknown_version() -> None:
    target = DataDragonTarget(
        version="99.99.99",
        locale="ko_KR",
    )

    with pytest.raises(
        ValueError,
        match="Data Dragon version not found",
    ):
        fetch_data_dragon_bundle(
            target,
            transport=httpx.MockTransport(_handler),
        )


def test_save_raw_data_dragon_bundle_writes_lineage(
    tmp_path,
) -> None:
    bundle = fetch_data_dragon_bundle(
        TARGET,
        transport=httpx.MockTransport(_handler),
    )

    (
        versions_path,
        champion_path,
        item_path,
        metadata_path,
    ) = save_raw_data_dragon_bundle(bundle, tmp_path)

    assert versions_path.is_file()
    assert champion_path.is_file()
    assert item_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["ddragon_version"] == "16.1.1"
    assert metadata["locale"] == "ko_KR"

    assert metadata["resources"]["champion"]["sha256"] == (
        sha256(champion_path.read_bytes()).hexdigest()
    )
    assert metadata["resources"]["item"]["sha256"] == (sha256(item_path.read_bytes()).hexdigest())
