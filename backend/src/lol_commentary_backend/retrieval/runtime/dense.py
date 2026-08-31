from typing import Any

import numpy as np

from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    BGE_M3,
)
from lol_commentary_backend.retrieval.embeddings.query import (
    format_embedding_query,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RankedCandidate,
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


class BgeDenseIndex:
    def __init__(
        self,
        *,
        chunks: list[PatchRagChunk],
        model: Any,
        corpus_embeddings: np.ndarray,
        device: str,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")

        if corpus_embeddings.ndim != 2:
            raise ValueError("corpus_embeddings must be 2D")

        if corpus_embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding row count must match chunks")

        self._chunks = chunks
        self._model = model
        self._corpus_embeddings = corpus_embeddings
        self.device = device

    @classmethod
    def build(
        cls,
        chunks: list[PatchRagChunk],
        *,
        requested_device: str = "auto",
        batch_size: int = 8,
    ) -> "BgeDenseIndex":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        from sentence_transformers import (
            SentenceTransformer,
        )

        device = _resolve_device(requested_device)

        model = SentenceTransformer(
            BGE_M3.model_id,
            device=device,
        )

        model.max_seq_length = BGE_M3.experiment_max_tokens

        corpus_embeddings = model.encode(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return cls(
            chunks=chunks,
            model=model,
            corpus_embeddings=np.asarray(
                corpus_embeddings,
                dtype=np.float32,
            ),
            device=device,
        )

    def search(
        self,
        query: str,
        *,
        top_n: int,
    ) -> list[RankedCandidate]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        formatted_query = format_embedding_query(
            query,
            BGE_M3,
        )

        query_embedding = self._model.encode(
            [formatted_query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_array = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_array.ndim != 2 or query_array.shape[0] != 1:
            raise ValueError("Expected one 2D query embedding")

        scores = (query_array @ self._corpus_embeddings.T)[0]

        ranked = np.argsort(
            -scores,
            kind="stable",
        )[:top_n]

        return [
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
