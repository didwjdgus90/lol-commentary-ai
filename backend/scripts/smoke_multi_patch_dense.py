from __future__ import annotations

import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.multi_patch.models import (
    MultiPatchCorpusManifest,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    load_chunks,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.multi_pgvector_dense import (
    MultiCorpusPgVectorDenseIndex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        default=("정수 약탈자 변경"),
    )

    parser.add_argument(
        "--locale",
        choices=(
            "ko_KR",
            "en_US",
        ),
        default="ko_KR",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.top_n <= 0:
        raise ValueError("top-n must be positive")

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

    aggregate = MultiPatchCorpusManifest.model_validate_json(
        aggregate_path.read_text(encoding="utf-8")
    )

    selected = [shard for shard in aggregate.shards if shard.locale == args.locale]

    if len(selected) <= 1:
        raise RuntimeError("Smoke requires more than one corpus")

    corpus_sha256s = tuple(shard.corpus_sha256 for shard in selected)

    chunks_by_id: dict[
        str,
        PatchRagChunk,
    ] = {}

    for shard in selected:
        manifest_path = repository_root / shard.manifest_path

        chunks_path = manifest_path.parent / "chunks.jsonl"

        for chunk in load_chunks(chunks_path):
            existing = chunks_by_id.get(chunk.chunk_id)

            if existing is not None and existing != chunk:
                raise RuntimeError("Conflicting duplicate chunk ID across corpora")

            chunks_by_id[chunk.chunk_id] = chunk

    print("=== MULTI-CORPUS DENSE SMOKE ===")

    print(f"Locale: {args.locale}")

    print(f"Corpus count: {len(corpus_sha256s)}")

    print(f"Loaded unique chunks: {len(chunks_by_id)}")

    print(f"Query: {args.query}")

    print()

    index = MultiCorpusPgVectorDenseIndex.build(
        corpus_sha256s=(corpus_sha256s),
        requested_device=(args.device),
    )

    try:
        results = index.search(
            args.query,
            top_n=args.top_n,
        )

    finally:
        index.close()

    if not results:
        raise RuntimeError("Dense search returned no results")

    for candidate in results:
        chunk = chunks_by_id.get(candidate.chunk_id)

        if chunk is None:
            raise RuntimeError("Dense search returned unknown chunk ID")

        print(f"#{candidate.rank} score={candidate.score:.6f}")

        print(f"  PATCH={chunk.patch}")

        print(f"  LOCALE={chunk.locale}")

        print(f"  TITLE={chunk.title}")

        print(f"  ENTITY={chunk.entity_name}")

        print(f"  CHUNK={chunk.chunk_id}")

        print()

    print("MULTI_CORPUS_DENSE_SMOKE=PASS")


if __name__ == "__main__":
    main()
