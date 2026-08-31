from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.aliases.models import (
    QueryExpansion,
)
from lol_commentary_backend.retrieval.chunks.models import (
    ChunkStrategy,
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.fusion import (
    fuse_rrf_candidates,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
    RetrievalMode,
    RetrievalStrategy,
)
from lol_commentary_backend.retrieval.runtime.service import (
    RetrievalService,
)

HASH = sha256(b"source").hexdigest()


def _chunk(index: int) -> PatchRagChunk:
    text = f"document {index}"

    return PatchRagChunk(
        chunker_version="0.1.0",
        chunk_id=sha256(f"chunk:{index}".encode()).hexdigest(),
        content_sha256=sha256(text.encode()).hexdigest(),
        document_id=sha256(f"doc:{index}".encode()).hexdigest(),
        document_content_sha256=HASH,
        source_record_id=sha256(f"record:{index}".encode()).hexdigest(),
        chunk_index=0,
        chunk_count=1,
        char_count=len(text),
        strategy=ChunkStrategy.SINGLE_DOCUMENT,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=HASH,
        ddragon_version="16.1.1",
        section_kind="hotfix",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        entity_id=None,
        entity_key=None,
        entity_resolved=False,
        resolution_method=ResolutionMethod.UNRESOLVED,
        removed_from_target_map=False,
        target_map_id=None,
        heading_path=[text],
        title=text,
        text=text,
    )


def _candidate(
    chunk: PatchRagChunk,
    *,
    rank: int,
    score: float,
) -> RankedCandidate:
    return RankedCandidate(
        chunk_id=chunk.chunk_id,
        rank=rank,
        score=score,
    )


class FakeSparse:
    def __init__(
        self,
        candidates: list[RankedCandidate],
    ) -> None:
        self.candidates = candidates
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> tuple[
        QueryExpansion,
        list[RankedCandidate],
    ]:
        self.calls.append(query)

        return (
            QueryExpansion(
                original_query=query,
                expanded_query=(f"{query} | 정수 약탈자"),
                matched_entity_keys=["item:3508"],
                added_aliases=["정수 약탈자"],
                changed=True,
            ),
            self.candidates[:top_n],
        )


class FakeDense:
    def __init__(
        self,
        candidates: list[RankedCandidate],
        *,
        fail: bool = False,
    ) -> None:
        self.candidates = candidates
        self.fail = fail
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> list[RankedCandidate]:
        self.calls.append(query)

        if self.fail:
            raise RuntimeError("dense failure")

        return self.candidates[:top_n]


def _service(
    *,
    dense: FakeDense | None,
) -> tuple[
    RetrievalService,
    FakeSparse,
]:
    chunks = [
        _chunk(0),
        _chunk(1),
        _chunk(2),
    ]

    sparse = FakeSparse(
        [
            _candidate(
                chunks[1],
                rank=1,
                score=3.0,
            ),
            _candidate(
                chunks[0],
                rank=2,
                score=2.0,
            ),
            _candidate(
                chunks[2],
                rank=3,
                score=1.0,
            ),
        ]
    )

    return (
        RetrievalService(
            chunks=chunks,
            sparse_retriever=sparse,
            dense_retriever=dense,
            source_top_n=3,
            rrf_k=60,
        ),
        sparse,
    )


def test_rrf_rewards_cross_engine_overlap() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]

    fused = fuse_rrf_candidates(
        [
            _candidate(
                chunks[0],
                rank=1,
                score=0.9,
            ),
            _candidate(
                chunks[1],
                rank=2,
                score=0.8,
            ),
        ],
        [
            _candidate(
                chunks[1],
                rank=1,
                score=10.0,
            ),
            _candidate(
                chunks[0],
                rank=2,
                score=9.0,
            ),
        ],
        top_k=2,
        rrf_k=60,
    )

    assert len(fused) == 2
    assert {item.chunk_id for item in fused} == {chunk.chunk_id for chunk in chunks}


def test_rrf_ties_are_deterministic() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]

    first = fuse_rrf_candidates(
        [
            _candidate(
                chunks[0],
                rank=1,
                score=1.0,
            )
        ],
        [
            _candidate(
                chunks[1],
                rank=1,
                score=1.0,
            )
        ],
        top_k=2,
        rrf_k=60,
    )

    second = fuse_rrf_candidates(
        [
            _candidate(
                chunks[0],
                rank=1,
                score=1.0,
            )
        ],
        [
            _candidate(
                chunks[1],
                rank=1,
                score=1.0,
            )
        ],
        top_k=2,
        rrf_k=60,
    )

    assert first == second


def test_fallback_mode_skips_dense() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
        _chunk(2),
    ]
    dense = FakeDense(
        [
            _candidate(
                chunks[0],
                rank=1,
                score=1.0,
            )
        ]
    )
    service, _ = _service(dense=dense)

    response = service.retrieve(
        "Essence Reaver",
        top_k=2,
        mode=RetrievalMode.FALLBACK,
    )

    assert dense.calls == []
    assert response.strategy_used == RetrievalStrategy.ALIAS_BM25
    assert response.fallback_used is False


def test_primary_uses_original_query_for_dense() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
        _chunk(2),
    ]
    dense = FakeDense(
        [
            _candidate(
                chunks[0],
                rank=1,
                score=1.0,
            )
        ]
    )
    service, sparse = _service(dense=dense)

    response = service.retrieve(
        "Essence Reaver",
        top_k=2,
        mode=RetrievalMode.PRIMARY,
    )

    assert sparse.calls == ["Essence Reaver"]
    assert dense.calls == ["Essence Reaver"]
    assert response.expanded_query == "Essence Reaver | 정수 약탈자"
    assert response.strategy_used == RetrievalStrategy.BGE_ALIAS_RRF


def test_auto_falls_back_on_dense_failure() -> None:
    service, _ = _service(
        dense=FakeDense(
            [],
            fail=True,
        )
    )

    response = service.retrieve(
        "test",
        top_k=3,
        mode=RetrievalMode.AUTO,
    )

    assert response.fallback_used is True
    assert response.strategy_used == RetrievalStrategy.ALIAS_BM25
    assert response.fallback_reason == "RuntimeError"


def test_primary_propagates_dense_failure() -> None:
    service, _ = _service(
        dense=FakeDense(
            [],
            fail=True,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Primary retrieval failed",
    ):
        service.retrieve(
            "test",
            top_k=3,
            mode=RetrievalMode.PRIMARY,
        )


def test_auto_without_dense_uses_fallback() -> None:
    service, _ = _service(dense=None)

    response = service.retrieve(
        "test",
        top_k=3,
        mode=RetrievalMode.AUTO,
    )

    assert response.fallback_used is True
    assert response.fallback_reason == "dense_unavailable"


def test_top_k_cannot_exceed_frozen_source_depth() -> None:
    service, _ = _service(dense=None)

    with pytest.raises(
        ValueError,
        match="source_top_n",
    ):
        service.retrieve(
            "test",
            top_k=4,
        )
