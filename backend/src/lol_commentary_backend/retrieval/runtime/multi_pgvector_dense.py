from __future__ import annotations

import re
from typing import Any

import numpy as np
from psycopg_pool import ConnectionPool

from lol_commentary_backend.retrieval.embeddings.models import (
    BGE_M3,
)
from lol_commentary_backend.retrieval.embeddings.query import (
    format_embedding_query,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
)
from lol_commentary_backend.retrieval.runtime.pgvector_dense import (
    cosine_distance_to_score,
)
from lol_commentary_backend.retrieval.storage.postgres import (
    build_vector_pool,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def normalize_corpus_sha256s(
    corpus_sha256s: tuple[str, ...],
) -> tuple[str, ...]:
    if not corpus_sha256s:
        raise ValueError("corpus_sha256s must not be empty")

    normalized: list[str] = []

    seen: set[str] = set()

    for value in corpus_sha256s:
        corpus_sha256 = value.strip()

        if _SHA256_PATTERN.fullmatch(corpus_sha256) is None:
            raise ValueError(f"Invalid corpus SHA-256: {value}")

        if corpus_sha256 in seen:
            continue

        seen.add(corpus_sha256)

        normalized.append(corpus_sha256)

    return tuple(normalized)


def _resolve_device(
    requested_device: str,
) -> str:
    import torch

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


class MultiCorpusPgVectorDenseIndex:
    def __init__(
        self,
        *,
        model: Any,
        pool: ConnectionPool,
        corpus_sha256s: tuple[
            str,
            ...,
        ],
        embedding_model_id: str,
        device: str,
    ) -> None:
        if not embedding_model_id.strip():
            raise ValueError("embedding_model_id must not be empty")

        self._model = model

        self._pool = pool

        self._corpus_sha256s = normalize_corpus_sha256s(corpus_sha256s)

        self._embedding_model_id = embedding_model_id

        self.device = device

    @property
    def corpus_sha256s(
        self,
    ) -> tuple[str, ...]:
        return self._corpus_sha256s

    @classmethod
    def build(
        cls,
        *,
        corpus_sha256s: tuple[
            str,
            ...,
        ],
        database_url: str | None = None,
        requested_device: str = "auto",
        pool_min_size: int = 1,
        pool_max_size: int = 4,
    ) -> MultiCorpusPgVectorDenseIndex:
        from sentence_transformers import (
            SentenceTransformer,
        )

        normalized_corpora = normalize_corpus_sha256s(corpus_sha256s)

        device = _resolve_device(requested_device)

        model = SentenceTransformer(
            BGE_M3.model_id,
            device=device,
        )

        model.max_seq_length = BGE_M3.experiment_max_tokens

        pool = build_vector_pool(
            database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
        )

        instance = cls(
            model=model,
            pool=pool,
            corpus_sha256s=(normalized_corpora),
            embedding_model_id=(BGE_M3.model_id),
            device=device,
        )

        try:
            instance.validate_storage()
        except Exception:
            pool.close()
            raise

        return instance

    def validate_storage(
        self,
    ) -> None:
        corpus_values = list(self._corpus_sha256s)

        with self._pool.connection() as connection:
            corpus_rows = connection.execute(
                """
                    SELECT
                        corpus_sha256,
                        chunk_count
                    FROM retrieval_corpora
                    WHERE corpus_sha256
                        = ANY(%s::text[])
                    """,
                (corpus_values,),
            ).fetchall()

            expected_by_corpus = {str(row[0]): int(row[1]) for row in corpus_rows}

            missing_corpora = [
                corpus_sha256
                for corpus_sha256 in self._corpus_sha256s
                if corpus_sha256 not in expected_by_corpus
            ]

            if missing_corpora:
                raise RuntimeError(
                    "Multi-corpus retrieval "
                    "storage is incomplete. "
                    "Missing corpora: "
                    f"{missing_corpora}"
                )

            embedding_rows = connection.execute(
                """
                    SELECT
                        corpus_sha256,
                        COUNT(*),
                        MIN(
                            embedding_dimension
                        ),
                        MAX(
                            embedding_dimension
                        ),
                        BOOL_AND(normalized)
                    FROM rag_chunk_embeddings
                    WHERE corpus_sha256
                        = ANY(%s::text[])
                      AND embedding_model_id
                        = %s
                    GROUP BY corpus_sha256
                    """,
                (
                    corpus_values,
                    self._embedding_model_id,
                ),
            ).fetchall()

        embedding_by_corpus = {str(row[0]): row for row in embedding_rows}

        for corpus_sha256 in self._corpus_sha256s:
            expected_count = expected_by_corpus[corpus_sha256]

            row = embedding_by_corpus.get(corpus_sha256)

            if row is None:
                raise RuntimeError(f"No embeddings stored for corpus {corpus_sha256}")

            embedding_count = int(row[1])

            min_dimension = row[2]

            max_dimension = row[3]

            all_normalized = row[4]

            if embedding_count != expected_count:
                raise RuntimeError(
                    "Stored embedding count "
                    "mismatch for corpus "
                    f"{corpus_sha256}: "
                    f"expected "
                    f"{expected_count}, "
                    f"got "
                    f"{embedding_count}"
                )

            if (
                min_dimension != BGE_M3.embedding_dimension
                or max_dimension != BGE_M3.embedding_dimension
            ):
                raise RuntimeError("Stored embedding dimension does not match BGE-M3")

            if all_normalized is not True:
                raise RuntimeError("Stored BGE-M3 embeddings must be normalized")

    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        formatted_query = format_embedding_query(
            query,
            BGE_M3,
        )

        embedding = self._model.encode(
            [formatted_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        matrix = np.asarray(
            embedding,
            dtype=np.float32,
        )

        expected_shape = (
            1,
            BGE_M3.embedding_dimension,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Unexpected query embedding shape: expected {expected_shape}, got {matrix.shape}"
            )

        if not np.isfinite(matrix).all():
            raise ValueError("Query embedding contains non-finite values")

        norm = float(np.linalg.norm(matrix[0]))

        if not np.isclose(
            norm,
            1.0,
            atol=1e-4,
        ):
            raise ValueError("Query embedding must have unit L2 norm")

        return matrix[0]

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> list[RankedCandidate]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if top_n <= 0:
            raise ValueError("top_n must be positive")

        query_embedding = self._encode_query(normalized_query)

        corpus_values = list(self._corpus_sha256s)

        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    MIN(
                        embedding <=> %s
                    ) AS cosine_distance
                FROM rag_chunk_embeddings
                WHERE corpus_sha256
                    = ANY(%s::text[])
                  AND embedding_model_id
                    = %s
                GROUP BY chunk_id
                ORDER BY
                    cosine_distance ASC,
                    chunk_id ASC
                LIMIT %s
                """,
                (
                    query_embedding,
                    corpus_values,
                    self._embedding_model_id,
                    top_n,
                ),
            ).fetchall()

        return [
            RankedCandidate(
                chunk_id=str(row[0]),
                rank=rank,
                score=(cosine_distance_to_score(float(row[1]))),
            )
            for rank, row in enumerate(
                rows,
                start=1,
            )
        ]

    def close(
        self,
    ) -> None:
        self._pool.close()
