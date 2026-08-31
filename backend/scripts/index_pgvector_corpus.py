from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lol_commentary_backend.retrieval.baseline.models import (
    RetrievalBaselineDecision,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    file_sha256,
    load_chunks,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    BGE_M3,
)
from lol_commentary_backend.retrieval.storage.pgvector_ingest import (
    build_corpus_manifest,
    upsert_retrieval_corpus,
)
from lol_commentary_backend.retrieval.storage.postgres import (
    connect_database,
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


def _validate_dependency_versions(
    decision: RetrievalBaselineDecision,
) -> None:
    import sentence_transformers
    import transformers

    if sentence_transformers.__version__ != decision.sentence_transformers_version:
        raise ValueError("sentence-transformers version differs from frozen retrieval baseline")

    if transformers.__version__ != decision.transformers_version:
        raise ValueError("transformers version differs from frozen retrieval baseline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Embed the frozen LoL chunk corpus and upsert it into pgvector."),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=("Embedding device: auto, cpu, cuda, etc."),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("SentenceTransformer corpus encoding batch size."),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    repository_root = Path(__file__).resolve().parents[2]

    chunks_path = (
        repository_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "chunks.jsonl"
    )

    baseline_path = (
        repository_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    current_corpus_hash = file_sha256(chunks_path)

    if current_corpus_hash != decision.corpus_sha256:
        raise ValueError(
            "Chunk corpus differs from frozen "
            "retrieval baseline. "
            "Do not index an unverified corpus."
        )

    _validate_dependency_versions(decision)

    chunks = load_chunks(chunks_path)

    device = _resolve_device(args.device)

    from sentence_transformers import (
        SentenceTransformer,
    )

    model = SentenceTransformer(
        BGE_M3.model_id,
        device=device,
    )

    model.max_seq_length = BGE_M3.experiment_max_tokens

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    manifest = build_corpus_manifest(
        chunks,
        corpus_sha256=(current_corpus_hash),
        embedding_model_id=(BGE_M3.model_id),
        embedding_dimension=(BGE_M3.embedding_dimension),
        normalized=True,
    )

    print("=== PGVECTOR CORPUS INGESTION ===")
    print(f"Corpus SHA256: {manifest.corpus_sha256}")
    print(f"Patch: {manifest.patch}")
    print(f"Locale: {manifest.locale}")
    print(f"Chunker version: {manifest.chunker_version}")
    print(f"Chunks: {manifest.chunk_count}")
    print(f"Embedding model: {manifest.embedding_model_id}")
    print(f"Embedding dimension: {manifest.embedding_dimension}")
    print(f"Normalized: {manifest.normalized}")
    print(f"Device: {device}")

    with connect_database() as connection:
        counts = upsert_retrieval_corpus(
            connection,
            manifest=manifest,
            chunks=chunks,
            embeddings=embedding_matrix,
        )

    print()
    print("=== STORAGE VERIFICATION ===")
    print(f"Corpora: {counts.corpora}")
    print(f"Chunks: {counts.chunks}")
    print(f"Embeddings: {counts.embeddings}")
    print("PGVECTOR_CORPUS_INGESTION=PASS")


if __name__ == "__main__":
    main()
