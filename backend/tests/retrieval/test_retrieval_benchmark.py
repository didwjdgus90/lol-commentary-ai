from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.benchmark.bm25 import (
    bm25_tokens,
)
from lol_commentary_backend.retrieval.benchmark.metrics import (
    evaluate_ranking,
    summarize_by,
    summarize_results,
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

HASH = sha256(b"source").hexdigest()


def _chunk(
    index: int,
    title: str,
) -> PatchRagChunk:
    text = f"26.1 패치 {title}"

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
        heading_path=[title],
        title=title,
        text=text,
    )


def _case(
    relevant_chunk_ids: list[str],
    *,
    query_id: str = "q001",
    language: QueryLanguage = QueryLanguage.KO,
) -> RetrievalEvalCase:
    return RetrievalEvalCase(
        dataset_version="0.1.0",
        query_id=query_id,
        query="테스트 질의",
        language=language,
        query_type=QueryType.SYSTEM,
        difficulty=QueryDifficulty.MEDIUM,
        target_granularity=(TargetGranularity.ANSWER_CHUNK),
        relevant_document_ids=[HASH],
        relevant_chunk_ids=relevant_chunk_ids,
        relevant_source_record_ids=[HASH],
        selector=RetrievalTargetSelector(
            patch="26.1",
            title="테스트",
        ),
        rationale="metric test",
        corpus_sha256=HASH,
    )


def test_hit_at_one_and_mrr_are_correct() -> None:
    chunks = [
        _chunk(0, "정답"),
        _chunk(1, "오답"),
    ]
    case = _case([chunks[0].chunk_id])

    result = evaluate_ranking(
        case,
        [0, 1],
        [0.9, 0.1],
        chunks,
    )

    assert result.hit_at_1 is True
    assert result.reciprocal_rank == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    chunks = [
        _chunk(0, "오답1"),
        _chunk(1, "오답2"),
        _chunk(2, "정답"),
    ]
    case = _case([chunks[2].chunk_id])

    result = evaluate_ranking(
        case,
        [0, 1, 2],
        [0.9, 0.8, 0.7],
        chunks,
    )

    assert result.hit_at_1 is False
    assert result.hit_at_3 is True
    assert result.reciprocal_rank == pytest.approx(1 / 3)


def test_recall_at_five_supports_multiple_gold_chunks() -> None:
    chunks = [_chunk(index, f"문서{index}") for index in range(6)]
    case = _case(
        [
            chunks[1].chunk_id,
            chunks[5].chunk_id,
        ]
    )

    result = evaluate_ranking(
        case,
        [0, 1, 2, 3, 4, 5],
        [6, 5, 4, 3, 2, 1],
        chunks,
    )

    assert result.recall_at_5 == 0.5


def test_summary_averages_query_metrics() -> None:
    chunks = [
        _chunk(0, "정답"),
        _chunk(1, "오답"),
    ]

    first = evaluate_ranking(
        _case(
            [chunks[0].chunk_id],
            query_id="q001",
        ),
        [0, 1],
        [0.9, 0.1],
        chunks,
    )
    second = evaluate_ranking(
        _case(
            [chunks[0].chunk_id],
            query_id="q002",
        ),
        [1, 0],
        [0.9, 0.1],
        chunks,
    )

    summary = summarize_results([first, second])

    assert summary.hit_at_1 == 0.5
    assert summary.mrr == 0.75


def test_group_summary_uses_requested_key() -> None:
    chunks = [
        _chunk(0, "정답"),
        _chunk(1, "오답"),
    ]

    ko = evaluate_ranking(
        _case(
            [chunks[0].chunk_id],
            query_id="q001",
            language=QueryLanguage.KO,
        ),
        [0, 1],
        [1.0, 0.0],
        chunks,
    )
    en = evaluate_ranking(
        _case(
            [chunks[0].chunk_id],
            query_id="q002",
            language=QueryLanguage.EN,
        ),
        [1, 0],
        [1.0, 0.0],
        chunks,
    )

    grouped = summarize_by(
        [ko, en],
        lambda result: result.language,
    )

    assert grouped["ko"].hit_at_1 == 1.0
    assert grouped["en"].hit_at_1 == 0.0


def test_bm25_tokenizer_keeps_exact_numbers() -> None:
    tokens = bm25_tokens("26.1 공격력 55에서 50")

    assert "26.1" in tokens
    assert "55" in tokens
    assert "50" in tokens


def test_bm25_tokenizer_adds_korean_bigrams() -> None:
    tokens = bm25_tokens("삭제되었습니다")

    assert "삭제되었습니다" in tokens
    assert "ko2:삭제" in tokens


def test_ranking_length_mismatch_is_rejected() -> None:
    chunks = [_chunk(0, "정답")]

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_ranking(
            _case([chunks[0].chunk_id]),
            [0],
            [],
            chunks,
        )
