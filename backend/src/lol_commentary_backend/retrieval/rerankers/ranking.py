from collections.abc import Sequence

import numpy as np

from lol_commentary_backend.retrieval.benchmark.metrics import (
    evaluate_ranking,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)


def rerank_query_from_scores(
    *,
    case: RetrievalEvalCase,
    source_result: RetrievalQueryResult,
    scores: Sequence[float],
    chunks: Sequence[PatchRagChunk],
    top_n: int,
) -> RetrievalQueryResult:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    candidates = source_result.top_results[:top_n]

    if len(scores) != len(candidates):
        raise ValueError("scores length must match candidate count")

    relevant = set(case.relevant_chunk_ids)
    candidate_ids = {item.chunk_id for item in candidates}

    if not relevant.issubset(candidate_ids):
        raise ValueError(f"{case.query_id}: Gold chunk is missing from reranker candidates")

    chunk_index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    missing = candidate_ids - set(chunk_index_by_id)

    if missing:
        raise ValueError("Candidate references chunk missing from current corpus")

    score_array = np.asarray(
        scores,
        dtype=np.float64,
    )

    order = np.argsort(
        -score_array,
        kind="stable",
    )

    ranked_indices = [chunk_index_by_id[candidates[index].chunk_id] for index in order]

    ranked_scores = [float(score_array[index]) for index in order]

    return evaluate_ranking(
        case,
        ranked_indices,
        ranked_scores,
        chunks,
    )
