from hashlib import sha256
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.storage.pgvector_ingest import (
    build_corpus_manifest,
    validate_embedding_matrix,
)

HASH = sha256(b"source").hexdigest()


def _chunk(
    index: int,
    *,
    patch: str = "26.1",
) -> PatchRagChunk:
    chunk = SimpleNamespace(
        chunk_id=sha256(f"chunk:{index}".encode()).hexdigest(),
        patch=patch,
        locale="ko_kr",
        chunker_version="0.1.0",
    )

    return cast(
        PatchRagChunk,
        chunk,
    )


def test_build_corpus_manifest() -> None:
    chunks = [
        _chunk(1),
        _chunk(2),
    ]

    manifest = build_corpus_manifest(
        chunks,
        corpus_sha256=HASH,
        embedding_model_id="BAAI/bge-m3",
        embedding_dimension=1024,
        normalized=True,
    )

    assert manifest.patch == "26.1"
    assert manifest.locale == "ko_kr"

    assert manifest.chunker_version == "0.1.0"

    assert manifest.chunk_count == 2

    assert manifest.embedding_dimension == 1024


def test_manifest_rejects_mixed_patch() -> None:
    chunks = [
        _chunk(
            1,
            patch="26.1",
        ),
        _chunk(
            2,
            patch="26.2",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="one patch",
    ):
        build_corpus_manifest(
            chunks,
            corpus_sha256=HASH,
            embedding_model_id=("BAAI/bge-m3"),
            embedding_dimension=1024,
            normalized=True,
        )


def test_embedding_matrix_accepts_normalized_vectors() -> None:
    matrix = np.zeros(
        (
            2,
            1024,
        ),
        dtype=np.float32,
    )

    matrix[
        0,
        0,
    ] = 1.0

    matrix[
        1,
        1,
    ] = 1.0

    result = validate_embedding_matrix(
        matrix,
        expected_rows=2,
        expected_dimension=1024,
        normalized=True,
    )

    assert result.shape == (
        2,
        1024,
    )

    assert result.dtype == np.float32


def test_embedding_matrix_rejects_wrong_dimension() -> None:
    matrix = np.zeros(
        (
            2,
            10,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="shape mismatch",
    ):
        validate_embedding_matrix(
            matrix,
            expected_rows=2,
            expected_dimension=1024,
            normalized=False,
        )


def test_embedding_matrix_rejects_non_unit_vectors() -> None:
    matrix = np.ones(
        (
            2,
            1024,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="unit L2 norm",
    ):
        validate_embedding_matrix(
            matrix,
            expected_rows=2,
            expected_dimension=1024,
            normalized=True,
        )
