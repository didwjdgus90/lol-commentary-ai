import numpy as np
from rank_bm25 import BM25Okapi

from lol_commentary_backend.retrieval.aliases.expander import (
    expand_bilingual_entity_query,
)
from lol_commentary_backend.retrieval.aliases.models import (
    BilingualAliasCatalog,
    QueryExpansion,
)
from lol_commentary_backend.retrieval.benchmark.bm25 import (
    bm25_tokens,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
)


class AliasBm25Index:
    def __init__(
        self,
        chunks: list[PatchRagChunk],
        alias_catalog: BilingualAliasCatalog,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")

        self._chunks = chunks
        self._alias_catalog = alias_catalog

        tokenized_corpus = [bm25_tokens(chunk.text) for chunk in chunks]

        self._index = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> tuple[
        QueryExpansion,
        list[RankedCandidate],
    ]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        expansion = expand_bilingual_entity_query(
            query,
            self._alias_catalog,
        )

        scores = np.asarray(
            self._index.get_scores(bm25_tokens(expansion.expanded_query)),
            dtype=np.float64,
        )

        ranked = np.argsort(
            -scores,
            kind="stable",
        )[:top_n]

        candidates = [
            RankedCandidate(
                chunk_id=self._chunks[int(index)].chunk_id,
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(
                ranked,
                start=1,
            )
        ]

        return expansion, candidates
