from __future__ import annotations

import httpx
import pytest

from lol_commentary_backend.ingestion.riot_api.client import (
    RiotApiClient,
)
from lol_commentary_backend.ingestion.riot_api.errors import (
    RiotApiAuthenticationError,
    RiotApiNotFoundError,
)

TEST_KEY = "RGAPI-test-key-not-real"


def test_account_by_riot_id() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.headers["X-Riot-Token"] == TEST_KEY

        assert "/riot/account/v1/accounts/by-riot-id/" in str(request.url)

        return httpx.Response(
            200,
            json={
                "puuid": "test-puuid",
                "gameName": "Test Player",
                "tagLine": "KR1",
            },
        )

    transport = httpx.MockTransport(handler)

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=transport,
    ) as client:
        account = client.get_account_by_riot_id(
            game_name="Test Player",
            tag_line="KR1",
        )

    assert account.puuid == "test-puuid"

    assert account.game_name == "Test Player"


def test_match_ids_by_puuid() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.params["start"] == "5"

        assert request.url.params["count"] == "10"

        return httpx.Response(
            200,
            json=[
                "KR_1",
                "KR_2",
            ],
        )

    transport = httpx.MockTransport(handler)

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=transport,
    ) as client:
        result = client.get_match_ids_by_puuid(
            puuid="test-puuid",
            start=5,
            count=10,
        )

    assert result.match_ids == (
        "KR_1",
        "KR_2",
    )


def test_retries_429_using_retry_after() -> None:
    attempts = 0

    slept: list[float] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            return httpx.Response(
                429,
                headers={
                    "Retry-After": "2",
                },
                request=request,
            )

        return httpx.Response(
            200,
            json=[],
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=transport,
        sleep_fn=slept.append,
    ) as client:
        result = client.get_match_ids_by_puuid(
            puuid="test-puuid",
            count=1,
        )

    assert attempts == 2

    assert slept == [2.0]

    assert result.match_ids == ()


def test_retries_server_error() -> None:
    attempts = 0

    slept: list[float] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            return httpx.Response(
                503,
                request=request,
            )

        return httpx.Response(
            200,
            json=[],
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=transport,
        sleep_fn=slept.append,
    ) as client:
        client.get_match_ids_by_puuid(
            puuid="test-puuid",
            count=1,
        )

    assert attempts == 2

    assert slept == [0.5]


@pytest.mark.parametrize(
    "status_code",
    [
        401,
        403,
    ],
)
def test_classifies_authentication_errors(
    status_code: int,
) -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code,
            request=request,
        )

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RiotApiAuthenticationError):
            client.get_match("KR_1")


def test_classifies_not_found() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            404,
            request=request,
        )

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RiotApiNotFoundError):
            client.get_account_by_riot_id(
                game_name="missing",
                tag_line="KR1",
            )


def test_reads_match_detail() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "metadata": {
                    "matchId": "KR_1",
                },
                "info": {},
            },
            request=request,
        )

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=httpx.MockTransport(handler),
    ) as client:
        payload = client.get_match("KR_1")

    assert payload["metadata"]["matchId"] == "KR_1"


def test_reads_timeline() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert str(request.url).endswith("/timeline")

        return httpx.Response(
            200,
            json={
                "metadata": {
                    "matchId": "KR_1",
                },
                "info": {
                    "frames": [],
                },
            },
            request=request,
        )

    with RiotApiClient(
        api_key=TEST_KEY,
        transport=httpx.MockTransport(handler),
    ) as client:
        payload = client.get_timeline("KR_1")

    assert payload["info"]["frames"] == []


def test_rejects_invalid_match_count() -> None:
    with RiotApiClient(
        api_key=TEST_KEY,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=[],
                request=request,
            )
        ),
    ) as client:
        with pytest.raises(
            ValueError,
            match="between 1 and 100",
        ):
            client.get_match_ids_by_puuid(
                puuid="test",
                count=101,
            )
