from __future__ import annotations

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
from lol_commentary_backend.retrieval.storage.postgres import (
    build_vector_pool,
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


def cosine_distance_to_score(
    distance: float,
) -> float:
    return 1.0 - distance


class PgVectorDenseIndex:
    def __init__(
        self,
        *,
        model: Any,
        pool: ConnectionPool,
        corpus_sha256: str,
        embedding_model_id: str,
        device: str,
    ) -> None:
        if not corpus_sha256:
            raise ValueError("corpus_sha256 must not be empty")

        if not embedding_model_id:
            raise ValueError("embedding_model_id must not be empty")

        self._model = model
        self._pool = pool

        self._corpus_sha256 = corpus_sha256

        self._embedding_model_id = embedding_model_id

        self.device = device

    @classmethod
    def build(
        cls,
        *,
        corpus_sha256: str,
        database_url: str | None = None,
        requested_device: str = "auto",
        pool_min_size: int = 1,
        pool_max_size: int = 4,
    ) -> PgVectorDenseIndex:
        from sentence_transformers import (
            SentenceTransformer,
        )

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
            corpus_sha256=corpus_sha256,
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
        with self._pool.connection() as connection:
            corpus_row = connection.execute(
                """
                SELECT chunk_count
                FROM retrieval_corpora
                WHERE corpus_sha256 = %s
                """,
                (self._corpus_sha256,),
            ).fetchone()

            if corpus_row is None:
                raise RuntimeError("Frozen retrieval corpus is not stored in PostgreSQL")

            expected_count = int(corpus_row[0])

            embedding_row = connection.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(embedding_dimension),
                    MAX(embedding_dimension),
                    BOOL_AND(normalized)
                FROM rag_chunk_embeddings
                WHERE corpus_sha256 = %s
                  AND embedding_model_id = %s
                """,
                (
                    self._corpus_sha256,
                    self._embedding_model_id,
                ),
            ).fetchone()

            if embedding_row is None:
                raise RuntimeError("Failed to validate stored embeddings")

            embedding_count = int(embedding_row[0])

            min_dimension = embedding_row[1]

            max_dimension = embedding_row[2]

            all_normalized = embedding_row[3]

        if embedding_count != expected_count:
            raise RuntimeError(
                f"Stored embedding count mismatch: expected {expected_count}, got {embedding_count}"
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

        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    embedding <=> %s
                        AS cosine_distance
                FROM rag_chunk_embeddings
                WHERE corpus_sha256 = %s
                  AND embedding_model_id = %s
                ORDER BY
                    cosine_distance ASC,
                    chunk_id ASC
                LIMIT %s
                """,
                (
                    query_embedding,
                    self._corpus_sha256,
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
