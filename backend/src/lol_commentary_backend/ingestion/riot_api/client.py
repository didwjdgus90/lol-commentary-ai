from __future__ import annotations

import os
import time
from collections.abc import (
    Callable,
)
from typing import Any
from urllib.parse import quote

import httpx

from lol_commentary_backend.ingestion.riot_api.errors import (
    RiotApiAuthenticationError,
    RiotApiError,
    RiotApiNotFoundError,
    RiotApiRateLimitError,
    RiotApiServerError,
    RiotApiTransportError,
)
from lol_commentary_backend.ingestion.riot_api.models import (
    RiotAccount,
    RiotMatchIds,
)

RIOT_API_KEY_ENV = "RIOT_API_KEY"

ASIA_REGIONAL_BASE_URL = "https://asia.api.riotgames.com"

DEFAULT_TIMEOUT_SECONDS = 15.0

DEFAULT_MAX_ATTEMPTS = 4

DEFAULT_BASE_BACKOFF_SECONDS = 0.5

_USER_AGENT = "lol-commentary-ai/riot-api-client-v1"

type RiotQueryParam = str | int | float | bool | None


def resolve_riot_api_key(
    explicit_key: str | None = None,
) -> str:
    if explicit_key is not None:
        candidate = explicit_key.strip()

        if candidate:
            return candidate

    candidate = os.getenv(
        RIOT_API_KEY_ENV,
        "",
    ).strip()

    if not candidate:
        raise RuntimeError(
            "RIOT_API_KEY is not set. "
            "Store the Riot API key "
            "locally and load it into "
            "the process environment."
        )

    if candidate == "change-me":
        raise RuntimeError("RIOT_API_KEY still uses the change-me placeholder.")

    return candidate


def _retry_after_seconds(
    response: httpx.Response,
) -> float:
    value = response.headers.get("Retry-After")

    if value is None:
        return 1.0

    try:
        parsed = float(value)
    except ValueError:
        return 1.0

    return max(
        parsed,
        0.0,
    )


def _validate_json_object(
    payload: object,
    *,
    resource_name: str,
) -> dict[str, Any]:
    if not isinstance(
        payload,
        dict,
    ):
        raise RiotApiError(f"Expected Riot API JSON object for {resource_name}")

    return payload


class RiotApiClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        regional_base_url: str = (ASIA_REGIONAL_BASE_URL),
        timeout_seconds: float = (DEFAULT_TIMEOUT_SECONDS),
        max_attempts: int = (DEFAULT_MAX_ATTEMPTS),
        base_backoff_seconds: float = (DEFAULT_BASE_BACKOFF_SECONDS),
        transport: (httpx.BaseTransport | None) = None,
        sleep_fn: Callable[
            [float],
            None,
        ] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        if base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must not be negative")

        self._api_key = resolve_riot_api_key(api_key)

        self._max_attempts = max_attempts

        self._base_backoff_seconds = base_backoff_seconds

        self._sleep = sleep_fn

        self._client = httpx.Client(
            base_url=(regional_base_url),
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
            headers={
                "Accept": ("application/json"),
                "User-Agent": _USER_AGENT,
                "X-Riot-Token": (self._api_key),
            },
        )

    def __enter__(
        self,
    ) -> RiotApiClient:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def close(
        self,
    ) -> None:
        self._client.close()

    def _backoff_seconds(
        self,
        attempt: int,
    ) -> float:
        return self._base_backoff_seconds * (2 ** (attempt - 1))

    def _request_json(
        self,
        path: str,
        *,
        params: dict[
            str,
            RiotQueryParam,
        ]
        | None = None,
    ) -> object:
        for attempt in range(
            1,
            self._max_attempts + 1,
        ):
            try:
                response = self._client.get(
                    path,
                    params=params,
                )

            except httpx.TransportError as exc:
                if attempt < self._max_attempts:
                    self._sleep(self._backoff_seconds(attempt))

                    continue

                raise (RiotApiTransportError("Riot API network request failed")) from exc

            if response.status_code in {
                401,
                403,
            }:
                raise (
                    RiotApiAuthenticationError(
                        "Riot API "
                        "authentication failed. "
                        "The development API key "
                        "may be missing, invalid, "
                        "or expired.",
                        status_code=(response.status_code),
                    )
                )

            if response.status_code == 404:
                raise RiotApiNotFoundError(
                    "Riot API resource was not found",
                    status_code=404,
                )

            if response.status_code == 429:
                retry_after = _retry_after_seconds(response)

                if attempt < self._max_attempts:
                    self._sleep(retry_after)

                    continue

                raise RiotApiRateLimitError(
                    "Riot API rate limit was exceeded",
                    retry_after_seconds=(retry_after),
                )

            if 500 <= response.status_code <= 599:
                if attempt < self._max_attempts:
                    self._sleep(self._backoff_seconds(attempt))

                    continue

                raise RiotApiServerError(
                    "Riot API server request failed after retries",
                    status_code=(response.status_code),
                )

            if not response.is_success:
                raise RiotApiError(
                    f"Unexpected Riot API response status: {response.status_code}",
                    status_code=(response.status_code),
                )

            try:
                return response.json()

            except ValueError as exc:
                raise RiotApiError("Riot API returned invalid JSON") from exc

        raise RuntimeError("Unreachable Riot API retry state")

    def get_account_by_riot_id(
        self,
        *,
        game_name: str,
        tag_line: str,
    ) -> RiotAccount:
        clean_game_name = game_name.strip()

        clean_tag_line = tag_line.strip()

        if not clean_game_name:
            raise ValueError("game_name must not be empty")

        if not clean_tag_line:
            raise ValueError("tag_line must not be empty")

        encoded_game_name = quote(
            clean_game_name,
            safe="",
        )

        encoded_tag_line = quote(
            clean_tag_line,
            safe="",
        )

        payload = self._request_json(
            f"/riot/account/v1/accounts/by-riot-id/{encoded_game_name}/{encoded_tag_line}"
        )

        return RiotAccount.model_validate(payload)

    def get_match_ids_by_puuid(
        self,
        *,
        puuid: str,
        start: int = 0,
        count: int = 20,
    ) -> RiotMatchIds:
        clean_puuid = puuid.strip()

        if not clean_puuid:
            raise ValueError("puuid must not be empty")

        if start < 0:
            raise ValueError("start must not be negative")

        if not 1 <= count <= 100:
            raise ValueError("count must be between 1 and 100")

        encoded_puuid = quote(
            clean_puuid,
            safe="",
        )

        payload = self._request_json(
            (f"/lol/match/v5/matches/by-puuid/{encoded_puuid}/ids"),
            params={
                "start": start,
                "count": count,
            },
        )

        if not isinstance(
            payload,
            list,
        ) or not all(
            isinstance(
                match_id,
                str,
            )
            and match_id
            for match_id in payload
        ):
            raise RiotApiError("Riot Match-V5 match ID response is invalid")

        return RiotMatchIds(
            puuid=clean_puuid,
            start=start,
            count=count,
            match_ids=tuple(payload),
        )

    def get_match(
        self,
        match_id: str,
    ) -> dict[str, Any]:
        clean_match_id = match_id.strip()

        if not clean_match_id:
            raise ValueError("match_id must not be empty")

        encoded_match_id = quote(
            clean_match_id,
            safe="",
        )

        payload = self._request_json(f"/lol/match/v5/matches/{encoded_match_id}")

        return _validate_json_object(
            payload,
            resource_name="match",
        )

    def get_timeline(
        self,
        match_id: str,
    ) -> dict[str, Any]:
        clean_match_id = match_id.strip()

        if not clean_match_id:
            raise ValueError("match_id must not be empty")

        encoded_match_id = quote(
            clean_match_id,
            safe="",
        )

        payload = self._request_json(f"/lol/match/v5/matches/{encoded_match_id}/timeline")

        return _validate_json_object(
            payload,
            resource_name="timeline",
        )
