from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.benchmark.models import (
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
from lol_commentary_backend.retrieval.rerankers.candidate import (
    analyze_candidate_coverage,
    require_full_candidate_coverage,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RERANKER_CANDIDATES,
)
from lol_commentary_backend.retrieval.rerankers.ranking import (
    rerank_query_from_scores,
)

HASH = sha256(b"source").hexdigest()


def _chunk(index: int) -> PatchRagChunk:
    text = f"document {index}"

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


def _case(
    relevant_chunk_ids: list[str],
) -> RetrievalEvalCase:
    return RetrievalEvalCase(
        dataset_version="0.1.0",
        query_id="q001",
        query="테스트 질의",
        language=QueryLanguage.KO,
        query_type=QueryType.SYSTEM,
        difficulty=QueryDifficulty.MEDIUM,
        target_granularity=(TargetGranularity.ANSWER_CHUNK),
        relevant_document_ids=[HASH],
        relevant_chunk_ids=relevant_chunk_ids,
        relevant_source_record_ids=[HASH],
        selector=RetrievalTargetSelector(
            patch="26.1",
            title="test",
        ),
        rationale="reranker test",
        corpus_sha256=HASH,
    )


def _source_result(
    chunks: list[PatchRagChunk],
    relevant_ids: set[str],
) -> RetrievalQueryResult:
    top_results = [
        RetrievedChunk(
            rank=rank,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            entity_name=None,
            score=float(len(chunks) - rank),
            relevant=(chunk.chunk_id in relevant_ids),
        )
        for rank, chunk in enumerate(
            chunks,
            start=1,
        )
    ]

    first_rank = next(
        (result.rank for result in top_results if result.relevant),
        None,
    )

    return RetrievalQueryResult(
        query_id="q001",
        query="테스트 질의",
        language="ko",
        query_type="system",
        difficulty="medium",
        relevant_chunk_ids=sorted(relevant_ids),
        first_relevant_rank=first_rank,
        hit_at_1=first_rank == 1,
        hit_at_3=(first_rank is not None and first_rank <= 3),
        hit_at_5=(first_rank is not None and first_rank <= 5),
        recall_at_5=(
            len(relevant_ids & {item.chunk_id for item in top_results[:5]}) / len(relevant_ids)
        ),
        reciprocal_rank=(0.0 if first_rank is None else 1.0 / first_rank),
        top_results=top_results,
    )


def test_reranker_candidate_keys_are_unique() -> None:
    keys = [candidate.key for candidate in RERANKER_CANDIDATES]

    assert len(keys) == len(set(keys))
    assert len(keys) == 3


def test_candidate_coverage_is_full() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]
    case = _case([chunks[1].chunk_id])
    source = _source_result(
        chunks,
        {chunks[1].chunk_id},
    )

    coverage = analyze_candidate_coverage(
        source_key="source",
        results=[source],
        cases=[case],
        top_n=2,
    )

    assert coverage.coverage_rate == 1.0
    assert coverage.missing_query_ids == []


def test_candidate_coverage_detects_missing_gold() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]
    case = _case([chunks[1].chunk_id])
    source = _source_result(
        [chunks[0]],
        {chunks[1].chunk_id},
    )

    coverage = analyze_candidate_coverage(
        source_key="source",
        results=[source],
        cases=[case],
        top_n=1,
    )

    assert coverage.coverage_rate == 0.0
    assert coverage.missing_query_ids == ["q001"]


def test_full_coverage_guard_rejects_missing() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]
    coverage = analyze_candidate_coverage(
        source_key="source",
        results=[
            _source_result(
                [chunks[0]],
                {chunks[1].chunk_id},
            )
        ],
        cases=[_case([chunks[1].chunk_id])],
        top_n=1,
    )

    with pytest.raises(
        ValueError,
        match="100% Gold coverage",
    ):
        require_full_candidate_coverage(coverage)


def test_reranker_scores_can_promote_gold_to_top1() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
        _chunk(2),
    ]
    relevant = chunks[1].chunk_id

    result = rerank_query_from_scores(
        case=_case([relevant]),
        source_result=_source_result(
            chunks,
            {relevant},
        ),
        scores=[0.1, 0.9, 0.2],
        chunks=chunks,
        top_n=3,
    )

    assert result.hit_at_1 is True
    assert result.first_relevant_rank == 1


def test_reranking_is_stable_for_tied_scores() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]
    relevant = chunks[1].chunk_id

    result = rerank_query_from_scores(
        case=_case([relevant]),
        source_result=_source_result(
            chunks,
            {relevant},
        ),
        scores=[0.5, 0.5],
        chunks=chunks,
        top_n=2,
    )

    assert result.first_relevant_rank == 2


def test_score_count_mismatch_is_rejected() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]
    relevant = chunks[1].chunk_id

    with pytest.raises(
        ValueError,
        match="scores length",
    ):
        rerank_query_from_scores(
            case=_case([relevant]),
            source_result=_source_result(
                chunks,
                {relevant},
            ),
            scores=[0.5],
            chunks=chunks,
            top_n=2,
        )


def test_reranking_rejects_missing_gold_candidate() -> None:
    chunks = [
        _chunk(0),
        _chunk(1),
    ]

    with pytest.raises(
        ValueError,
        match="Gold chunk",
    ):
        rerank_query_from_scores(
            case=_case([chunks[1].chunk_id]),
            source_result=_source_result(
                [chunks[0]],
                {chunks[1].chunk_id},
            ),
            scores=[0.9],
            chunks=chunks,
            top_n=1,
        )
