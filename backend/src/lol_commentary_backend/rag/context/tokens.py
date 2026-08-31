from typing import Protocol

import tiktoken

DEFAULT_TOKEN_ENCODING = "o200k_base"


class TextTokenCounter(Protocol):
    @property
    def name(self) -> str: ...

    def count(
        self,
        text: str,
    ) -> int: ...


class TiktokenTextCounter:
    def __init__(
        self,
        *,
        encoding_name: str = DEFAULT_TOKEN_ENCODING,
    ) -> None:
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return self._encoding_name

    def count(
        self,
        text: str,
    ) -> int:
        return len(self._encoding.encode(text))
