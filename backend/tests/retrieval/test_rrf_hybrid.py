from hashlib import sha256
from typing import Literal

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.benchmark.models import (
    BenchmarkMemory,
    BenchmarkTiming,
    RetrievalBenchmarkRun,
    RetrievalMetricSummary,
    RetrievalQueryResult,
    RetrievedChunk,
)
from lol_commentary_backend.retrieval.chunks.models import (
    ChunkStrategy,
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    QueryDifficulty,
    QueryLanguage,
    QueryType,
    RetrievalEvalCase,
    RetrievalTargetSelector,
    TargetGranularity,
)
from lol_commentary_backend.retrieval.hybrid.rrf import (
    build_rrf_hybrid_run,
    rrf_score,
)

HASH = sha256(b"source").hexdigest()


def _chunk(index: int) -> PatchRagChunk:
    text = f"chunk {index}"
    return PatchRagChunk(
        chunker_version="0.1.0",
        chunk_id=sha256(f"chunk:{index}".encode()).hexdigest(),
        content_sha256=sha256(text.encode()).hexdigest(),
        document_id=sha256(f"doc:{index}".encode()).hexdigest(),
        document_content_sha256=HASH,
        source_record_id=sha256(f"record:{index}".encode()).hexdigest(),
        chunk_index=0,
        chunk_count=1,
        char_count=len(text),
        strategy=ChunkStrategy.SINGLE_DOCUMENT,
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=HASH,
        ddragon_version="16.1.1",
        section_kind="hotfix",
        entity_type=PatchEntityType.UNKNOWN,
        entity_name=None,
        entity_id=None,
        entity_key=None,
        entity_resolved=False,
        resolution_method=ResolutionMethod.UNRESOLVED,
        removed_from_target_map=False,
        target_map_id=None,
        heading_path=[text],
        title=text,
        text=text,
    )


def _query_result(
    *,
    ranked_chunks: list[PatchRagChunk],
    relevant_chunk_id: str,
) -> RetrievalQueryResult:
    first_rank = None
    top_results = []

    for rank, chunk in enumerate(ranked_chunks, start=1):
        relevant = chunk.chunk_id == relevant_chunk_id
        if relevant and first_rank is None:
            first_rank = rank
        top_results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                entity_name=None,
                score=float(len(ranked_chunks) - rank),
                relevant=relevant,
            )
        )

    return RetrievalQueryResult(
        query_id="q001",
        query="test query",
        language="ko",
        query_type="system",
        difficulty="medium",
        relevant_chunk_ids=[relevant_chunk_id],
        first_relevant_rank=first_rank,
        hit_at_1=first_rank == 1,
        hit_at_3=first_rank is not None and first_rank <= 3,
        hit_at_5=first_rank is not None and first_rank <= 5,
        recall_at_5=(1.0 if first_rank is not None and first_rank <= 5 else 0.0),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        top_results=top_results,
    )


def _run(
    *,
    engine_key: str,
    engine_type: Literal["dense", "bm25"],
    result: RetrievalQueryResult,
) -> RetrievalBenchmarkRun:
    summary = RetrievalMetricSummary(
        cases=1,
        hit_at_1=float(result.hit_at_1),
        hit_at_3=float(result.hit_at_3),
        hit_at_5=float(result.hit_at_5),
        recall_at_5=result.recall_at_5,
        mrr=result.reciprocal_rank,
    )
    return RetrievalBenchmarkRun(
        benchmark_version="0.1.0",
        engine_key=engine_key,
        engine_type=engine_type,
        model_id="test/model" if engine_type == "dense" else None,
        device="cpu",
        corpus_chunks=3,
        evaluation_cases=1,
        corpus_sha256=HASH,
        embedding_dimension=1024 if engine_type == "dense" else None,
        metrics=summary,
        by_language={"ko": summary},
        by_query_type={"system": summary},
        by_difficulty={"medium": summary},
        timing=BenchmarkTiming(
            model_load_seconds=0,
            corpus_index_seconds=0,
            query_total_seconds=0,
            query_mean_ms=0,
        ),
        memory=BenchmarkMemory(
            process_rss_before_mb=1,
            process_rss_after_load_mb=1,
            process_rss_after_index_mb=1,
            cuda_peak_allocated_mb=None,
        ),
        query_results=[result],
    )


def _case(relevant_chunk_id: str) -> RetrievalEvalCase:
    return RetrievalEvalCase(
        dataset_version="0.1.0",
        query_id="q001",
        query="test query",
        language=QueryLanguage.KO,
        query_type=QueryType.SYSTEM,
        difficulty=QueryDifficulty.MEDIUM,
        target_granularity=TargetGranularity.ANSWER_CHUNK,
        relevant_document_ids=[HASH],
        relevant_chunk_ids=[relevant_chunk_id],
        relevant_source_record_ids=[HASH],
        selector=RetrievalTargetSelector(
            patch="26.1",
            title="test",
        ),
        rationale="test",
        corpus_sha256=HASH,
    )


def test_rrf_score_matches_formula() -> None:
    assert rrf_score(1, rrf_k=60) == pytest.approx(1 / 61)


def test_rrf_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        rrf_score(0)


def test_hybrid_can_rescue_sparse_only_relevant() -> None:
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    relevant_id = chunks[2].chunk_id

    hybrid = build_rrf_hybrid_run(
        _run(
            engine_key="dense",
            engine_type="dense",
            result=_query_result(
                ranked_chunks=[chunks[0], chunks[1], chunks[2]],
                relevant_chunk_id=relevant_id,
            ),
        ),
        _run(
            engine_key="bm25",
            engine_type="bm25",
            result=_query_result(
                ranked_chunks=[chunks[2], chunks[0], chunks[1]],
                relevant_chunk_id=relevant_id,
            ),
        ),
        [_case(relevant_id)],
        chunks,
    )

    assert hybrid.metrics.hit_at_3 == 1.0


def test_hybrid_uses_union_of_candidates() -> None:
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    relevant_id = chunks[2].chunk_id

    hybrid = build_rrf_hybrid_run(
        _run(
            engine_key="dense",
            engine_type="dense",
            result=_query_result(
                ranked_chunks=[chunks[0], chunks[1]],
                relevant_chunk_id=relevant_id,
            ),
        ),
        _run(
            engine_key="bm25",
            engine_type="bm25",
            result=_query_result(
                ranked_chunks=[chunks[2]],
                relevant_chunk_id=relevant_id,
            ),
        ),
        [_case(relevant_id)],
        chunks,
    )

    returned = {item.chunk_id for item in hybrid.query_results[0].top_results}
    assert returned == {chunk.chunk_id for chunk in chunks}


def test_hybrid_is_deterministic() -> None:
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    relevant_id = chunks[0].chunk_id
    dense = _run(
        engine_key="dense",
        engine_type="dense",
        result=_query_result(
            ranked_chunks=chunks,
            relevant_chunk_id=relevant_id,
        ),
    )
    sparse = _run(
        engine_key="bm25",
        engine_type="bm25",
        result=_query_result(
            ranked_chunks=list(reversed(chunks)),
            relevant_chunk_id=relevant_id,
        ),
    )

    first = build_rrf_hybrid_run(dense, sparse, [_case(relevant_id)], chunks)
    second = build_rrf_hybrid_run(dense, sparse, [_case(relevant_id)], chunks)

    assert [item.chunk_id for item in first.query_results[0].top_results] == [
        item.chunk_id for item in second.query_results[0].top_results
    ]


def test_hybrid_rejects_wrong_source_types() -> None:
    chunks = [_chunk(0)]
    result = _query_result(
        ranked_chunks=chunks,
        relevant_chunk_id=chunks[0].chunk_id,
    )
    dense = _run(
        engine_key="dense",
        engine_type="dense",
        result=result,
    )

    with pytest.raises(ValueError, match="sparse_run"):
        build_rrf_hybrid_run(
            dense,
            dense,
            [_case(chunks[0].chunk_id)],
            chunks,
        )


def test_hybrid_rejects_invalid_source_top_n() -> None:
    chunks = [_chunk(0)]
    result = _query_result(
        ranked_chunks=chunks,
        relevant_chunk_id=chunks[0].chunk_id,
    )

    with pytest.raises(ValueError, match="source_top_n"):
        build_rrf_hybrid_run(
            _run(
                engine_key="dense",
                engine_type="dense",
                result=result,
            ),
            _run(
                engine_key="bm25",
                engine_type="bm25",
                result=result,
            ),
            [_case(chunks[0].chunk_id)],
            chunks,
            source_top_n=0,
        )


def test_hybrid_preserves_corpus_hash() -> None:
    chunks = [_chunk(0)]
    result = _query_result(
        ranked_chunks=chunks,
        relevant_chunk_id=chunks[0].chunk_id,
    )

    hybrid = build_rrf_hybrid_run(
        _run(
            engine_key="dense",
            engine_type="dense",
            result=result,
        ),
        _run(
            engine_key="bm25",
            engine_type="bm25",
            result=result,
        ),
        [_case(chunks[0].chunk_id)],
        chunks,
    )

    assert hybrid.corpus_sha256 == HASH
