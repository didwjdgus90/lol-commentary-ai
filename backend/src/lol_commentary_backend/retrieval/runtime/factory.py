from pathlib import Path

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
from lol_commentary_backend.retrieval.hybrid.rrf import (
    DEFAULT_RRF_K,
    DEFAULT_SOURCE_TOP_N,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalStrategy,
)
from lol_commentary_backend.retrieval.runtime.pgvector_dense import (
    PgVectorDenseIndex,
)
from lol_commentary_backend.retrieval.runtime.service import (
    RetrievalService,
)
from lol_commentary_backend.retrieval.runtime.sparse import (
    AliasBm25Index,
)


def _validate_baseline_contract(
    decision: RetrievalBaselineDecision,
) -> None:
    if decision.primary.key != RetrievalStrategy.BGE_ALIAS_RRF.value:
        raise ValueError("Runtime primary strategy does not match frozen baseline")

    if decision.fast_fallback.key != RetrievalStrategy.ALIAS_BM25.value:
        raise ValueError("Runtime fallback strategy does not match frozen baseline")


def _validate_dependency_versions(
    decision: RetrievalBaselineDecision,
) -> None:
    import sentence_transformers
    import transformers

    if sentence_transformers.__version__ != decision.sentence_transformers_version:
        raise ValueError("sentence-transformers version differs from frozen retrieval baseline")

    if transformers.__version__ != decision.transformers_version:
        raise ValueError("transformers version differs from frozen retrieval baseline")


def build_retrieval_service(
    *,
    repository_root: Path,
    enable_primary: bool = True,
    requested_device: str = "auto",
    strict_versions: bool = True,
    database_url: str | None = None,
    pool_min_size: int = 1,
    pool_max_size: int = 4,
) -> RetrievalService:
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

    alias_catalog_path = (
        repository_root
        / "data"
        / "processed"
        / "retrieval"
        / "entity_aliases"
        / "16.1.1"
        / "ko_en_aliases.json"
    )

    baseline_path = (
        repository_root / "backend" / "evaluation" / "retrieval" / "retrieval_baseline_v2.json"
    )

    decision = RetrievalBaselineDecision.model_validate_json(
        baseline_path.read_text(encoding="utf-8")
    )

    _validate_baseline_contract(decision)

    current_corpus_hash = file_sha256(chunks_path)

    if current_corpus_hash != decision.corpus_sha256:
        raise ValueError("Chunk corpus differs from frozen retrieval baseline")

    if enable_primary and strict_versions:
        _validate_dependency_versions(decision)

    chunks = load_chunks(chunks_path)

    alias_catalog = BilingualAliasCatalog.model_validate_json(
        alias_catalog_path.read_text(encoding="utf-8")
    )

    sparse = AliasBm25Index(
        chunks,
        alias_catalog,
    )

    dense = (
        PgVectorDenseIndex.build(
            corpus_sha256=(decision.corpus_sha256),
            database_url=database_url,
            requested_device=(requested_device),
            pool_min_size=pool_min_size,
            pool_max_size=pool_max_size,
        )
        if enable_primary
        else None
    )

    return RetrievalService(
        chunks=chunks,
        sparse_retriever=sparse,
        dense_retriever=dense,
        source_top_n=(DEFAULT_SOURCE_TOP_N),
        rrf_k=DEFAULT_RRF_K,
    )
