import re
from time import perf_counter

import numpy as np
from rank_bm25 import BM25Okapi

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

_TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-zA-Z]+|[0-9]+(?:[./][0-9]+)*")


def bm25_tokens(text: str) -> list[str]:
    tokens: list[str] = []

    for raw_token in _TOKEN_PATTERN.findall(text.casefold()):
        tokens.append(raw_token)

        if raw_token and all("가" <= char <= "힣" for char in raw_token) and len(raw_token) >= 2:
            tokens.extend(
                f"ko2:{raw_token[index : index + 2]}" for index in range(len(raw_token) - 1)
            )

    return tokens


def run_bm25_queries(
    chunks: list[PatchRagChunk],
    cases: list[RetrievalEvalCase],
) -> tuple[
    list[RetrievalQueryResult],
    float,
    float,
]:
    start_index = perf_counter()

    tokenized_corpus = [bm25_tokens(chunk.text) for chunk in chunks]

    index = BM25Okapi(tokenized_corpus)

    index_seconds = perf_counter() - start_index

    results: list[RetrievalQueryResult] = []

    start_queries = perf_counter()

    for case in cases:
        scores_array = np.asarray(
            index.get_scores(bm25_tokens(case.query)),
            dtype=np.float64,
        )

        ranked = np.argsort(
            -scores_array,
            kind="stable",
        )

        ranked_indices = ranked.tolist()
        ranked_scores = scores_array[ranked].tolist()

        results.append(
            evaluate_ranking(
                case,
                ranked_indices,
                ranked_scores,
                chunks,
            )
        )

    query_seconds = perf_counter() - start_queries

    return (
        results,
        index_seconds,
        query_seconds,
    )
