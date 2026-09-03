from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from lol_commentary_backend.ingestion.multi_patch.models import (
    MultiPatchCorpusManifest,
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--patch",
        action="append",
        dest="patches",
    )

    parser.add_argument(
        "--locale",
        action="append",
        choices=(
            "ko_KR",
            "en_US",
        ),
        dest="locales",
    )

    parser.add_argument(
        "--force-reembed",
        action="store_true",
    )

    return parser.parse_args()


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


def _stored_counts(
    connection: psycopg.Connection[Any],
    *,
    corpus_sha256: str,
    embedding_model_id: str,
) -> tuple[
    int,
    int,
    int,
]:
    corpus_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM retrieval_corpora
        WHERE corpus_sha256 = %s
        """,
        (corpus_sha256,),
    ).fetchone()

    chunk_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM rag_chunks
        WHERE corpus_sha256 = %s
        """,
        (corpus_sha256,),
    ).fetchone()

    embedding_row = connection.execute(
        """
            SELECT COUNT(*)
            FROM rag_chunk_embeddings
            WHERE corpus_sha256 = %s
              AND embedding_model_id = %s
            """,
        (
            corpus_sha256,
            embedding_model_id,
        ),
    ).fetchone()

    if corpus_row is None or chunk_row is None or embedding_row is None:
        raise RuntimeError("Could not read pgvector storage counts")

    return (
        int(corpus_row[0]),
        int(chunk_row[0]),
        int(embedding_row[0]),
    )


def _aggregate_counts(
    connection: psycopg.Connection[Any],
    *,
    corpus_sha256s: tuple[
        str,
        ...,
    ],
    embedding_model_id: str,
) -> tuple[
    int,
    int,
    int,
]:
    corpus_values = list(corpus_sha256s)

    corpus_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM retrieval_corpora
        WHERE corpus_sha256
            = ANY(%s::text[])
        """,
        (corpus_values,),
    ).fetchone()

    chunk_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM rag_chunks
        WHERE corpus_sha256
            = ANY(%s::text[])
        """,
        (corpus_values,),
    ).fetchone()

    embedding_row = connection.execute(
        """
            SELECT COUNT(*)
            FROM rag_chunk_embeddings
            WHERE corpus_sha256
                = ANY(%s::text[])
              AND embedding_model_id = %s
            """,
        (
            corpus_values,
            embedding_model_id,
        ),
    ).fetchone()

    if corpus_row is None or chunk_row is None or embedding_row is None:
        raise RuntimeError("Could not read aggregate storage counts")

    return (
        int(corpus_row[0]),
        int(chunk_row[0]),
        int(embedding_row[0]),
    )


def main() -> None:
    args = _parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    repository_root = Path(__file__).resolve().parents[2]

    aggregate_path = (
        repository_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "multi_patch_v1"
        / "manifest.json"
    )

    baseline_path = (
        repository_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    aggregate = MultiPatchCorpusManifest.model_validate_json(
        aggregate_path.read_text(encoding="utf-8")
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    _validate_dependency_versions(decision)

    patch_filter = set(args.patches) if args.patches else None

    locale_filter = set(args.locales) if args.locales else None

    selected_shards = [
        shard
        for shard in aggregate.shards
        if (patch_filter is None or shard.patch in patch_filter)
        and (locale_filter is None or shard.locale in locale_filter)
    ]

    if not selected_shards:
        raise ValueError("No corpus shards selected")

    corpus_sha256s = tuple(shard.corpus_sha256 for shard in selected_shards)

    if len(set(corpus_sha256s)) != len(corpus_sha256s):
        raise RuntimeError("Selected corpus hashes must be unique")

    expected_chunks = sum(shard.chunk_count for shard in selected_shards)

    print("=== MULTI-PATCH PGVECTOR INDEXING ===")

    print(f"Selected shards: {len(selected_shards)}")

    print(f"Expected chunks: {expected_chunks}")

    print(f"Embedding model: {BGE_M3.model_id}")

    print()

    device = _resolve_device(args.device)

    model = None

    downloaded = 0
    reused = 0

    with connect_database() as connection:
        for index, shard in enumerate(
            selected_shards,
            start=1,
        ):
            manifest_path = repository_root / shard.manifest_path

            chunks_path = manifest_path.parent / "chunks.jsonl"

            current_sha = file_sha256(chunks_path)

            if current_sha != shard.corpus_sha256:
                raise ValueError(f"Corpus lineage mismatch for {shard.patch} {shard.locale}")

            chunks = load_chunks(chunks_path)

            if len(chunks) != shard.chunk_count:
                raise ValueError(f"Chunk count mismatch for {shard.patch} {shard.locale}")

            patches = {chunk.patch for chunk in chunks}

            locales = {chunk.locale for chunk in chunks}

            if patches != {shard.patch}:
                raise ValueError(f"Patch mismatch inside shard {shard.patch} {shard.locale}")

            if locales != {shard.locale}:
                raise ValueError(f"Locale mismatch inside shard {shard.patch} {shard.locale}")

            corpus_manifest = build_corpus_manifest(
                chunks,
                corpus_sha256=(shard.corpus_sha256),
                embedding_model_id=(BGE_M3.model_id),
                embedding_dimension=(BGE_M3.embedding_dimension),
                normalized=True,
            )

            (
                corpus_count,
                chunk_count,
                embedding_count,
            ) = _stored_counts(
                connection,
                corpus_sha256=(shard.corpus_sha256),
                embedding_model_id=(BGE_M3.model_id),
            )

            complete = (
                corpus_count == 1
                and chunk_count == shard.chunk_count
                and embedding_count == shard.chunk_count
            )

            print(f"[{index}/{len(selected_shards)}] {shard.patch} {shard.locale}")

            if complete and not args.force_reembed:
                reused += 1

                print("  ACTION=reused")

                print(f"  CHUNKS={chunk_count}")

                continue

            if model is None:
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
                batch_size=(args.batch_size),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            matrix = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            counts = upsert_retrieval_corpus(
                connection,
                manifest=(corpus_manifest),
                chunks=chunks,
                embeddings=matrix,
            )

            connection.commit()

            downloaded += 1

            print("  ACTION=indexed")

            print(f"  CHUNKS={counts.chunks}")

            print(f"  EMBEDDINGS={counts.embeddings}")

    with connect_database() as connection:
        (
            final_corpora,
            final_chunks,
            final_embeddings,
        ) = _aggregate_counts(
            connection,
            corpus_sha256s=(corpus_sha256s),
            embedding_model_id=(BGE_M3.model_id),
        )

    print()
    print("=== STORAGE VERIFICATION ===")

    print(f"Selected corpora: {final_corpora}")

    print(f"Stored chunks: {final_chunks}")

    print(f"Stored embeddings: {final_embeddings}")

    print(f"Indexed shards: {downloaded}")

    print(f"Reused shards: {reused}")

    if final_corpora != len(selected_shards):
        raise RuntimeError("Stored corpus count mismatch")

    if final_chunks != expected_chunks:
        raise RuntimeError("Stored chunk count mismatch")

    if final_embeddings != expected_chunks:
        raise RuntimeError("Stored embedding count mismatch")

    print()

    print("MULTI_PATCH_PGVECTOR_INDEX=PASS")


if __name__ == "__main__":
    main()
