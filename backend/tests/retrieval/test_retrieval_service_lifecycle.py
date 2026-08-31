from hashlib import sha256
from types import SimpleNamespace
from typing import cast

from lol_commentary_backend.retrieval.aliases.models import (
    QueryExpansion,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
)
from lol_commentary_backend.retrieval.runtime.service import (
    RetrievalService,
)


class FakeSparse:
    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> tuple[
        QueryExpansion,
        list[RankedCandidate],
    ]:
        raise AssertionError("search should not be called")


class FakeClosableDense:
    def __init__(
        self,
    ) -> None:
        self.close_calls = 0

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> list[RankedCandidate]:
        raise AssertionError("search should not be called")

    def close(
        self,
    ) -> None:
        self.close_calls += 1


def _chunk() -> PatchRagChunk:
    chunk_id = sha256(b"chunk").hexdigest()

    chunk = SimpleNamespace(
        chunk_id=chunk_id,
    )

    return cast(
        PatchRagChunk,
        chunk,
    )


def test_service_closes_dense_resource() -> None:
    dense = FakeClosableDense()

    service = RetrievalService(
        chunks=[_chunk()],
        sparse_retriever=FakeSparse(),
        dense_retriever=dense,
        source_top_n=10,
        rrf_k=60,
    )

    service.close()

    assert dense.close_calls == 1


def test_service_close_is_idempotent() -> None:
    dense = FakeClosableDense()

    service = RetrievalService(
        chunks=[_chunk()],
        sparse_retriever=FakeSparse(),
        dense_retriever=dense,
        source_top_n=10,
        rrf_k=60,
    )

    service.close()
    service.close()

    assert dense.close_calls == 1
