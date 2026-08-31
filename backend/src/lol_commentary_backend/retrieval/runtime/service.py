from collections.abc import Sequence
from time import perf_counter
from typing import Protocol, runtime_checkable

from lol_commentary_backend.retrieval.aliases.models import (
    QueryExpansion,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.fusion import (
    fuse_rrf_candidates,
)
from lol_commentary_backend.retrieval.runtime.models import (
    FusedCandidate,
    RankedCandidate,
    RetrievalHit,
    RetrievalMode,
    RetrievalResponse,
    RetrievalStrategy,
)


class SparseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> tuple[
        QueryExpansion,
        list[RankedCandidate],
    ]: ...


class DenseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> list[RankedCandidate]: ...


@runtime_checkable
class ClosableResource(Protocol):
    def close(
        self,
    ) -> None: ...


class RetrievalService:
    def __init__(
        self,
        *,
        chunks: Sequence[PatchRagChunk],
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever | None,
        source_top_n: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")

        if source_top_n <= 0:
            raise ValueError("source_top_n must be positive")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        if len(self._chunks_by_id) != len(chunks):
            raise ValueError("chunk IDs must be unique")

        self._sparse = sparse_retriever
        self._dense = dense_retriever
        self._source_top_n = source_top_n
        self._rrf_k = rrf_k
        self._closed = False

    @property
    def primary_available(self) -> bool:
        return self._dense is not None

    def close(
        self,
    ) -> None:
        if self._closed:
            return

        dense = self._dense

        if dense is not None and isinstance(
            dense,
            ClosableResource,
        ):
            dense.close()

        self._closed = True

    def _hit_from_fused(
        self,
        candidate: FusedCandidate,
        *,
        rank: int,
    ) -> RetrievalHit:
        chunk = self._chunks_by_id.get(candidate.chunk_id)

        if chunk is None:
            raise ValueError("Retriever returned unknown chunk ID")

        return RetrievalHit(
            rank=rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_record_id=(chunk.source_record_id),
            patch=chunk.patch,
            locale=chunk.locale,
            source_url=str(chunk.source_url),
            section_kind=chunk.section_kind,
            title=chunk.title,
            entity_name=chunk.entity_name,
            heading_path=chunk.heading_path,
            text=chunk.text,
            score=candidate.score,
            dense_rank=candidate.dense_rank,
            sparse_rank=candidate.sparse_rank,
            dense_score=candidate.dense_score,
            sparse_score=candidate.sparse_score,
        )

    def _fallback_hits(
        self,
        sparse: Sequence[RankedCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        fused = [
            FusedCandidate(
                chunk_id=item.chunk_id,
                score=item.score,
                sparse_rank=item.rank,
                sparse_score=item.score,
            )
            for item in sparse[:top_k]
        ]

        return [
            self._hit_from_fused(
                item,
                rank=rank,
            )
            for rank, item in enumerate(
                fused,
                start=1,
            )
        ]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.AUTO,
    ) -> RetrievalResponse:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self._source_top_n:
            raise ValueError("top_k cannot exceed source_top_n for the frozen retrieval baseline")

        started = perf_counter()

        expansion, sparse = self._sparse.search(
            normalized_query,
            top_n=self._source_top_n,
        )

        if mode == RetrievalMode.FALLBACK:
            hits = self._fallback_hits(
                sparse,
                top_k=top_k,
            )

            return RetrievalResponse(
                query=normalized_query,
                expanded_query=(expansion.expanded_query),
                requested_mode=mode,
                strategy_used=(RetrievalStrategy.ALIAS_BM25),
                fallback_used=False,
                fallback_reason=None,
                top_k=top_k,
                source_top_n=self._source_top_n,
                elapsed_ms=(perf_counter() - started) * 1000,
                hits=hits,
            )

        if self._dense is None:
            if mode == RetrievalMode.PRIMARY:
                raise RuntimeError("Primary retrieval is unavailable")

            hits = self._fallback_hits(
                sparse,
                top_k=top_k,
            )

            return RetrievalResponse(
                query=normalized_query,
                expanded_query=(expansion.expanded_query),
                requested_mode=mode,
                strategy_used=(RetrievalStrategy.ALIAS_BM25),
                fallback_used=True,
                fallback_reason="dense_unavailable",
                top_k=top_k,
                source_top_n=self._source_top_n,
                elapsed_ms=(perf_counter() - started) * 1000,
                hits=hits,
            )

        try:
            # Preserve the frozen experiment contract:
            # dense receives the original query, while
            # only BM25 receives bilingual alias expansion.
            dense = self._dense.search(
                normalized_query,
                top_n=self._source_top_n,
            )
        except Exception as exc:
            if mode == RetrievalMode.PRIMARY:
                raise RuntimeError("Primary retrieval failed") from exc

            hits = self._fallback_hits(
                sparse,
                top_k=top_k,
            )

            return RetrievalResponse(
                query=normalized_query,
                expanded_query=(expansion.expanded_query),
                requested_mode=mode,
                strategy_used=(RetrievalStrategy.ALIAS_BM25),
                fallback_used=True,
                fallback_reason=type(exc).__name__,
                top_k=top_k,
                source_top_n=self._source_top_n,
                elapsed_ms=(perf_counter() - started) * 1000,
                hits=hits,
            )

        fused = fuse_rrf_candidates(
            dense,
            sparse,
            top_k=top_k,
            rrf_k=self._rrf_k,
        )

        hits = [
            self._hit_from_fused(
                item,
                rank=rank,
            )
            for rank, item in enumerate(
                fused,
                start=1,
            )
        ]

        return RetrievalResponse(
            query=normalized_query,
            expanded_query=(expansion.expanded_query),
            requested_mode=mode,
            strategy_used=(RetrievalStrategy.BGE_ALIAS_RRF),
            fallback_used=False,
            fallback_reason=None,
            top_k=top_k,
            source_top_n=self._source_top_n,
            elapsed_ms=(perf_counter() - started) * 1000,
            hits=hits,
        )
