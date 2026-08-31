from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg

from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)


@dataclass(frozen=True)
class CorpusManifest:
    corpus_sha256: str

    patch: str
    locale: str

    chunker_version: str
    chunk_count: int

    embedding_model_id: str
    embedding_dimension: int

    normalized: bool


@dataclass(frozen=True)
class StorageCounts:
    corpora: int
    chunks: int
    embeddings: int


def build_corpus_manifest(
    chunks: Sequence[PatchRagChunk],
    *,
    corpus_sha256: str,
    embedding_model_id: str,
    embedding_dimension: int,
    normalized: bool,
) -> CorpusManifest:
    if not chunks:
        raise ValueError("chunks must not be empty")

    if len(corpus_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in corpus_sha256
    ):
        raise ValueError("corpus_sha256 must be a lowercase SHA-256 hex string")

    if not embedding_model_id.strip():
        raise ValueError("embedding_model_id must not be empty")

    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be positive")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk IDs must be unique")

    patches = {chunk.patch for chunk in chunks}

    locales = {chunk.locale for chunk in chunks}

    chunker_versions = {chunk.chunker_version for chunk in chunks}

    if len(patches) != 1:
        raise ValueError("all chunks must belong to one patch")

    if len(locales) != 1:
        raise ValueError("all chunks must belong to one locale")

    if len(chunker_versions) != 1:
        raise ValueError("all chunks must use one chunker version")

    return CorpusManifest(
        corpus_sha256=corpus_sha256,
        patch=next(iter(patches)),
        locale=next(iter(locales)),
        chunker_version=next(iter(chunker_versions)),
        chunk_count=len(chunks),
        embedding_model_id=embedding_model_id,
        embedding_dimension=embedding_dimension,
        normalized=normalized,
    )


def validate_embedding_matrix(
    embeddings: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
    normalized: bool,
) -> np.ndarray:
    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")

    expected_shape = (
        expected_rows,
        expected_dimension,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"embedding matrix shape mismatch: expected {expected_shape}, got {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")

    if normalized:
        norms = np.linalg.norm(
            matrix,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4,
        ):
            raise ValueError("normalized embeddings must have unit L2 norm")

    return matrix


def _upsert_corpus(
    connection: psycopg.Connection[Any],
    manifest: CorpusManifest,
) -> None:
    connection.execute(
        """
        INSERT INTO retrieval_corpora (
            corpus_sha256,
            patch,
            locale,
            chunker_version,
            chunk_count
        )
        VALUES (
            %(corpus_sha256)s,
            %(patch)s,
            %(locale)s,
            %(chunker_version)s,
            %(chunk_count)s
        )
        ON CONFLICT (corpus_sha256)
        DO NOTHING
        """,
        {
            "corpus_sha256": manifest.corpus_sha256,
            "patch": manifest.patch,
            "locale": manifest.locale,
            "chunker_version": manifest.chunker_version,
            "chunk_count": manifest.chunk_count,
        },
    )


def _chunk_row(
    manifest: CorpusManifest,
    chunk: PatchRagChunk,
) -> dict[str, object]:
    return {
        "corpus_sha256": manifest.corpus_sha256,
        "schema_version": chunk.schema_version,
        "chunker_version": chunk.chunker_version,
        "chunk_id": chunk.chunk_id,
        "content_sha256": chunk.content_sha256,
        "document_id": chunk.document_id,
        "document_content_sha256": chunk.document_content_sha256,
        "source_record_id": chunk.source_record_id,
        "chunk_index": chunk.chunk_index,
        "chunk_count": chunk.chunk_count,
        "char_count": chunk.char_count,
        "strategy": chunk.strategy.value,
        "patch": chunk.patch,
        "locale": chunk.locale,
        "source_url": str(chunk.source_url),
        "source_sha256": chunk.source_sha256,
        "ddragon_version": chunk.ddragon_version,
        "section_kind": chunk.section_kind,
        "entity_type": chunk.entity_type.value,
        "entity_name": chunk.entity_name,
        "entity_id": chunk.entity_id,
        "entity_key": chunk.entity_key,
        "entity_resolved": chunk.entity_resolved,
        "resolution_method": chunk.resolution_method.value,
        "removed_from_target_map": chunk.removed_from_target_map,
        "target_map_id": chunk.target_map_id,
        "heading_path": chunk.heading_path,
        "title": chunk.title,
        "chunk_text": chunk.text,
    }


def _upsert_chunks(
    connection: psycopg.Connection[Any],
    manifest: CorpusManifest,
    chunks: Sequence[PatchRagChunk],
) -> None:
    rows = [
        _chunk_row(
            manifest,
            chunk,
        )
        for chunk in chunks
    ]

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO rag_chunks (
                corpus_sha256,
                schema_version,
                chunker_version,
                chunk_id,
                content_sha256,
                document_id,
                document_content_sha256,
                source_record_id,
                chunk_index,
                chunk_count,
                char_count,
                strategy,
                patch,
                locale,
                source_url,
                source_sha256,
                ddragon_version,
                section_kind,
                entity_type,
                entity_name,
                entity_id,
                entity_key,
                entity_resolved,
                resolution_method,
                removed_from_target_map,
                target_map_id,
                heading_path,
                title,
                chunk_text
            )
            VALUES (
                %(corpus_sha256)s,
                %(schema_version)s,
                %(chunker_version)s,
                %(chunk_id)s,
                %(content_sha256)s,
                %(document_id)s,
                %(document_content_sha256)s,
                %(source_record_id)s,
                %(chunk_index)s,
                %(chunk_count)s,
                %(char_count)s,
                %(strategy)s,
                %(patch)s,
                %(locale)s,
                %(source_url)s,
                %(source_sha256)s,
                %(ddragon_version)s,
                %(section_kind)s,
                %(entity_type)s,
                %(entity_name)s,
                %(entity_id)s,
                %(entity_key)s,
                %(entity_resolved)s,
                %(resolution_method)s,
                %(removed_from_target_map)s,
                %(target_map_id)s,
                %(heading_path)s,
                %(title)s,
                %(chunk_text)s
            )
            ON CONFLICT (
                corpus_sha256,
                chunk_id
            )
            DO UPDATE SET
                schema_version =
                    EXCLUDED.schema_version,

                chunker_version =
                    EXCLUDED.chunker_version,

                content_sha256 =
                    EXCLUDED.content_sha256,

                document_id =
                    EXCLUDED.document_id,

                document_content_sha256 =
                    EXCLUDED.document_content_sha256,

                source_record_id =
                    EXCLUDED.source_record_id,

                chunk_index =
                    EXCLUDED.chunk_index,

                chunk_count =
                    EXCLUDED.chunk_count,

                char_count =
                    EXCLUDED.char_count,

                strategy =
                    EXCLUDED.strategy,

                patch =
                    EXCLUDED.patch,

                locale =
                    EXCLUDED.locale,

                source_url =
                    EXCLUDED.source_url,

                source_sha256 =
                    EXCLUDED.source_sha256,

                ddragon_version =
                    EXCLUDED.ddragon_version,

                section_kind =
                    EXCLUDED.section_kind,

                entity_type =
                    EXCLUDED.entity_type,

                entity_name =
                    EXCLUDED.entity_name,

                entity_id =
                    EXCLUDED.entity_id,

                entity_key =
                    EXCLUDED.entity_key,

                entity_resolved =
                    EXCLUDED.entity_resolved,

                resolution_method =
                    EXCLUDED.resolution_method,

                removed_from_target_map =
                    EXCLUDED.removed_from_target_map,

                target_map_id =
                    EXCLUDED.target_map_id,

                heading_path =
                    EXCLUDED.heading_path,

                title =
                    EXCLUDED.title,

                chunk_text =
                    EXCLUDED.chunk_text
            """,
            rows,
        )


def _upsert_embeddings(
    connection: psycopg.Connection[Any],
    manifest: CorpusManifest,
    chunks: Sequence[PatchRagChunk],
    embeddings: np.ndarray,
) -> None:
    rows = [
        {
            "corpus_sha256": manifest.corpus_sha256,
            "chunk_id": chunk.chunk_id,
            "embedding_model_id": manifest.embedding_model_id,
            "embedding_dimension": manifest.embedding_dimension,
            "normalized": manifest.normalized,
            "embedding": embedding,
        }
        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        )
    ]

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO rag_chunk_embeddings (
                corpus_sha256,
                chunk_id,
                embedding_model_id,
                embedding_dimension,
                normalized,
                embedding
            )
            VALUES (
                %(corpus_sha256)s,
                %(chunk_id)s,
                %(embedding_model_id)s,
                %(embedding_dimension)s,
                %(normalized)s,
                %(embedding)s
            )
            ON CONFLICT (
                corpus_sha256,
                chunk_id,
                embedding_model_id
            )
            DO UPDATE SET
                embedding_dimension =
                    EXCLUDED.embedding_dimension,

                normalized =
                    EXCLUDED.normalized,

                embedding =
                    EXCLUDED.embedding,

                created_at =
                    NOW()
            """,
            rows,
        )


def fetch_storage_counts(
    connection: psycopg.Connection[Any],
    manifest: CorpusManifest,
) -> StorageCounts:
    corpus_result = connection.execute(
        """
        SELECT COUNT(*)
        FROM retrieval_corpora
        WHERE corpus_sha256 = %s
        """,
        (manifest.corpus_sha256,),
    ).fetchone()

    chunk_result = connection.execute(
        """
        SELECT COUNT(*)
        FROM rag_chunks
        WHERE corpus_sha256 = %s
        """,
        (manifest.corpus_sha256,),
    ).fetchone()

    embedding_result = connection.execute(
        """
        SELECT COUNT(*)
        FROM rag_chunk_embeddings
        WHERE corpus_sha256 = %s
          AND embedding_model_id = %s
        """,
        (
            manifest.corpus_sha256,
            manifest.embedding_model_id,
        ),
    ).fetchone()

    if corpus_result is None or chunk_result is None or embedding_result is None:
        raise RuntimeError("failed to read storage counts")

    return StorageCounts(
        corpora=int(corpus_result[0]),
        chunks=int(chunk_result[0]),
        embeddings=int(embedding_result[0]),
    )


def upsert_retrieval_corpus(
    connection: psycopg.Connection[Any],
    *,
    manifest: CorpusManifest,
    chunks: Sequence[PatchRagChunk],
    embeddings: np.ndarray,
) -> StorageCounts:
    matrix = validate_embedding_matrix(
        embeddings,
        expected_rows=len(chunks),
        expected_dimension=(manifest.embedding_dimension),
        normalized=manifest.normalized,
    )

    if len(chunks) != manifest.chunk_count:
        raise ValueError("manifest chunk_count does not match chunks")

    _upsert_corpus(
        connection,
        manifest,
    )

    _upsert_chunks(
        connection,
        manifest,
        chunks,
    )

    _upsert_embeddings(
        connection,
        manifest,
        chunks,
        matrix,
    )

    counts = fetch_storage_counts(
        connection,
        manifest,
    )

    if counts.corpora != 1:
        raise RuntimeError("expected exactly one corpus row")

    if counts.chunks != manifest.chunk_count:
        raise RuntimeError(
            f"stored chunk count mismatch: expected {manifest.chunk_count}, got {counts.chunks}"
        )

    if counts.embeddings != manifest.chunk_count:
        raise RuntimeError(
            "stored embedding count mismatch: "
            f"expected {manifest.chunk_count}, "
            f"got {counts.embeddings}"
        )

    return counts
