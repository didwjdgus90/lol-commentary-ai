from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lol_commentary_backend.ingestion.multi_patch.models import (
    CorpusShardReference,
    MultiPatchCorpusManifest,
)
from lol_commentary_backend.retrieval.aliases.models import (
    BilingualAliasCatalog,
)
from lol_commentary_backend.retrieval.baseline.models import (
    RetrievalBaselineDecision,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    file_sha256,
    load_chunks,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.runtime.multi_pgvector_dense import (
    MultiCorpusPgVectorDenseIndex,
)
from lol_commentary_backend.retrieval.runtime.service import (
    RetrievalService,
)
from lol_commentary_backend.retrieval.runtime.sparse import (
    AliasBm25Index,
)

DEFAULT_MULTI_SOURCE_TOP_N = 30
DEFAULT_MULTI_RRF_K = 60


@dataclass(
    frozen=True,
)
class MultiPatchRetrievalScope:
    chunks: tuple[
        PatchRagChunk,
        ...,
    ]

    corpus_sha256s: tuple[
        str,
        ...,
    ]

    patches: tuple[
        str,
        ...,
    ]

    locales: tuple[
        str,
        ...,
    ]


def select_multi_patch_shards(
    manifest: MultiPatchCorpusManifest,
    *,
    locales: tuple[
        str,
        ...,
    ],
    patches: tuple[
        str,
        ...,
    ]
    | None = None,
) -> tuple[
    CorpusShardReference,
    ...,
]:
    if not locales:
        raise ValueError("locales must not be empty")

    locale_scope = set(locales)

    patch_scope = set(patches) if patches is not None else None

    selected = tuple(
        shard
        for shard in manifest.shards
        if shard.locale in locale_scope and (patch_scope is None or shard.patch in patch_scope)
    )

    if not selected:
        raise ValueError("No multi-patch corpus shards selected")

    return selected


def load_multi_patch_scope(
    *,
    repository_root: Path,
    locales: tuple[
        str,
        ...,
    ] = ("ko_KR",),
    patches: tuple[
        str,
        ...,
    ]
    | None = None,
) -> MultiPatchRetrievalScope:
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

    shards = select_multi_patch_shards(
        aggregate,
        locales=locales,
        patches=patches,
    )

    chunks_by_id: dict[
        str,
        PatchRagChunk,
    ] = {}

    corpus_sha256s: list[str] = []

    for shard in shards:
        manifest_path = repository_root / shard.manifest_path

        chunks_path = manifest_path.parent / "chunks.jsonl"

        current_sha256 = file_sha256(chunks_path)

        if current_sha256 != shard.corpus_sha256:
            raise ValueError(f"Corpus lineage mismatch for {shard.patch} {shard.locale}")

        shard_chunks = load_chunks(chunks_path)

        if len(shard_chunks) != shard.chunk_count:
            raise ValueError(f"Chunk count mismatch for {shard.patch} {shard.locale}")

        for chunk in shard_chunks:
            if chunk.patch != shard.patch:
                raise ValueError("Chunk patch does not match shard manifest")

            if chunk.locale != shard.locale:
                raise ValueError("Chunk locale does not match shard manifest")

            existing = chunks_by_id.get(chunk.chunk_id)

            if existing is not None and existing != chunk:
                raise RuntimeError("Conflicting duplicate chunk ID across multi-patch shards")

            chunks_by_id.setdefault(
                chunk.chunk_id,
                chunk,
            )

        corpus_sha256s.append(shard.corpus_sha256)

    if len(set(corpus_sha256s)) != len(corpus_sha256s):
        raise RuntimeError("Corpus SHA-256 values must be unique")

    ordered_patches = tuple(dict.fromkeys(shard.patch for shard in shards))

    ordered_locales = tuple(dict.fromkeys(shard.locale for shard in shards))

    return MultiPatchRetrievalScope(
        chunks=tuple(chunks_by_id.values()),
        corpus_sha256s=tuple(corpus_sha256s),
        patches=ordered_patches,
        locales=ordered_locales,
    )


def _validate_embedding_versions(
    repository_root: Path,
) -> None:
    baseline_path = (
        repository_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    import sentence_transformers
    import transformers

    if sentence_transformers.__version__ != decision.sentence_transformers_version:
        raise ValueError("sentence-transformers version differs from frozen embedding contract")

    if transformers.__version__ != decision.transformers_version:
        raise ValueError("transformers version differs from frozen embedding contract")


def build_multi_patch_retrieval_service(
    *,
    repository_root: Path,
    locales: tuple[
        str,
        ...,
    ] = ("ko_KR",),
    patches: tuple[
        str,
        ...,
    ]
    | None = None,
    enable_primary: bool = True,
    requested_device: str = "auto",
    database_url: str | None = None,
    source_top_n: int = (DEFAULT_MULTI_SOURCE_TOP_N),
    rrf_k: int = (DEFAULT_MULTI_RRF_K),
) -> RetrievalService:
    scope = load_multi_patch_scope(
        repository_root=(repository_root),
        locales=locales,
        patches=patches,
    )

    alias_catalog_path = (
        repository_root
        / "data"
        / "processed"
        / "retrieval"
        / "entity_aliases"
        / "multi_patch_v1"
        / "ko_en_aliases.json"
    )

    alias_catalog = BilingualAliasCatalog.model_validate_json(
        alias_catalog_path.read_text(encoding="utf-8")
    )

    sparse = AliasBm25Index(
        list(scope.chunks),
        alias_catalog,
    )

    dense = None

    if enable_primary:
        _validate_embedding_versions(repository_root)

        dense = MultiCorpusPgVectorDenseIndex.build(
            corpus_sha256s=(scope.corpus_sha256s),
            database_url=(database_url),
            requested_device=(requested_device),
        )

    return RetrievalService(
        chunks=scope.chunks,
        sparse_retriever=sparse,
        dense_retriever=dense,
        source_top_n=source_top_n,
        rrf_k=rrf_k,
    )
