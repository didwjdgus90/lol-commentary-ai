from __future__ import annotations


class RiotApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)

        self.status_code = status_code


class RiotApiAuthenticationError(RiotApiError):
    pass


class RiotApiNotFoundError(RiotApiError):
    pass


class RiotApiRateLimitError(RiotApiError):
    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
    ) -> None:
        super().__init__(
            message,
            status_code=429,
        )

        self.retry_after_seconds = retry_after_seconds


class RiotApiServerError(RiotApiError):
    pass


class RiotApiTransportError(RiotApiError):
    pass
