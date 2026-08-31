from collections.abc import Sequence

from lol_commentary_backend.retrieval.hybrid.rrf import (
    rrf_score,
)
from lol_commentary_backend.retrieval.runtime.models import (
    FusedCandidate,
    RankedCandidate,
)


def fuse_rrf_candidates(
    dense: Sequence[RankedCandidate],
    sparse: Sequence[RankedCandidate],
    *,
    top_k: int,
    rrf_k: int,
) -> list[FusedCandidate]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    scores: dict[str, float] = {}
    dense_by_id = {item.chunk_id: item for item in dense}
    sparse_by_id = {item.chunk_id: item for item in sparse}

    for item in dense:
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + rrf_score(
            item.rank,
            rrf_k=rrf_k,
        )

    for item in sparse:
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + rrf_score(
            item.rank,
            rrf_k=rrf_k,
        )

    def _best_rank(
        chunk_id: str,
    ) -> int:
        ranks = [
            item.rank
            for item in (
                dense_by_id.get(chunk_id),
                sparse_by_id.get(chunk_id),
            )
            if item is not None
        ]

        return min(ranks)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            _best_rank(chunk_id),
            chunk_id,
        ),
    )[:top_k]

    return [
        FusedCandidate(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            dense_rank=(dense_by_id[chunk_id].rank if chunk_id in dense_by_id else None),
            sparse_rank=(sparse_by_id[chunk_id].rank if chunk_id in sparse_by_id else None),
            dense_score=(dense_by_id[chunk_id].score if chunk_id in dense_by_id else None),
            sparse_score=(sparse_by_id[chunk_id].score if chunk_id in sparse_by_id else None),
        )
        for chunk_id in ranked_ids
    ]
