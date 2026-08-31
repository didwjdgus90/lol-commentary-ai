from time import perf_counter

import numpy as np
import psutil

from lol_commentary_backend.retrieval.benchmark.metrics import (
    evaluate_ranking,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    EmbeddingCandidate,
)
from lol_commentary_backend.retrieval.embeddings.query import (
    format_embedding_query,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)


def _resolve_device(
    requested_device: str,
) -> str:
    import torch

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def run_dense_queries(
    candidate: EmbeddingCandidate,
    chunks: list[PatchRagChunk],
    cases: list[RetrievalEvalCase],
    *,
    requested_device: str,
    batch_size: int,
) -> tuple[
    list[RetrievalQueryResult],
    str,
    int,
    float,
    float,
    float,
    float,
    float,
]:
    from sentence_transformers import (
        SentenceTransformer,
    )

    device = _resolve_device(requested_device)

    load_started = perf_counter()

    model = SentenceTransformer(
        candidate.model_id,
        device=device,
    )
    model.max_seq_length = candidate.experiment_max_tokens

    load_seconds = perf_counter() - load_started
    rss_after_load_mb = psutil.Process().memory_info().rss / 1024 / 1024

    corpus_texts = [chunk.text for chunk in chunks]

    corpus_started = perf_counter()

    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    corpus_seconds = perf_counter() - corpus_started
    rss_after_index_mb = psutil.Process().memory_info().rss / 1024 / 1024

    if corpus_embeddings.ndim != 2:
        raise ValueError("Expected 2D corpus embeddings")

    embedding_dimension = int(corpus_embeddings.shape[1])

    formatted_queries = [
        format_embedding_query(
            case.query,
            candidate,
        )
        for case in cases
    ]

    query_started = perf_counter()

    query_embeddings = model.encode(
        formatted_queries,
        batch_size=1,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_seconds = perf_counter() - query_started

    scores = query_embeddings @ corpus_embeddings.T

    results: list[RetrievalQueryResult] = []

    for case_index, case in enumerate(cases):
        row = scores[case_index]

        ranked = np.argsort(
            -row,
            kind="stable",
        )

        ranked_indices = ranked.tolist()
        ranked_scores = row[ranked].tolist()

        results.append(
            evaluate_ranking(
                case,
                ranked_indices,
                ranked_scores,
                chunks,
            )
        )

    return (
        results,
        device,
        embedding_dimension,
        load_seconds,
        corpus_seconds,
        query_seconds,
        rss_after_load_mb,
        rss_after_index_mb,
    )
