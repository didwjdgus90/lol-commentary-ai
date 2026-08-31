from hashlib import sha256

import pytest

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.retrieval.chunks.models import (
    ChunkStrategy,
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.builder import (
    build_retrieval_eval_cases,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    QueryDifficulty,
    QueryLanguage,
    QueryType,
    RetrievalEvalSeed,
    RetrievalTargetSelector,
    TargetGranularity,
)

HASH = sha256(b"source").hexdigest()


def _chunk(
    *,
    document_seed: str,
    chunk_index: int,
    chunk_count: int,
    title: str,
    entity_name: str | None,
    text: str,
) -> PatchRagChunk:
    document_id = sha256(document_seed.encode()).hexdigest()

    return PatchRagChunk(
        chunker_version="0.1.0",
        chunk_id=sha256((f"{document_seed}:{chunk_index}:{text}").encode()).hexdigest(),
        content_sha256=sha256(text.encode()).hexdigest(),
        document_id=document_id,
        document_content_sha256=HASH,
        source_record_id=sha256(f"record:{document_seed}".encode()).hexdigest(),
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        char_count=len(text),
        strategy=(
            ChunkStrategy.SINGLE_DOCUMENT if chunk_count == 1 else ChunkStrategy.SEMANTIC_SPLIT
        ),
        patch="26.1",
        locale="ko_kr",
        source_url=("https://www.leagueoflegends.com/ko-kr/news/game-updates/patch-26-1-notes/"),
        source_sha256=HASH,
        ddragon_version="16.1.1",
        section_kind="hotfix",
        entity_type=(
            PatchEntityType.CHAMPION if entity_name is not None else PatchEntityType.UNKNOWN
        ),
        entity_name=entity_name,
        entity_id=("Tryndamere" if entity_name == "트린다미어" else None),
        entity_key=("23" if entity_name == "트린다미어" else None),
        entity_resolved=entity_name is not None,
        resolution_method=(
            ResolutionMethod.CONTEXT_EXACT
            if entity_name is not None
            else ResolutionMethod.UNRESOLVED
        ),
        removed_from_target_map=False,
        target_map_id=None,
        heading_path=[
            "추가 패치 노트",
            title,
        ],
        title=title,
        text=text,
    )


def _seed(
    *,
    query_id: str = "q001",
    title: str = "E - 회전 베기",
    entity_name: str | None = "트린다미어",
    required_text: str | None = None,
) -> RetrievalEvalSeed:
    return RetrievalEvalSeed(
        query_id=query_id,
        query=f"테스트 질문 {query_id}",
        language=QueryLanguage.KO,
        query_type=QueryType.CHAMPION_ABILITY,
        difficulty=QueryDifficulty.EASY,
        selector=RetrievalTargetSelector(
            patch="26.1",
            title=title,
            entity_name=entity_name,
            section_kind="hotfix",
            required_text_contains=required_text,
        ),
        rationale="테스트용 gold target",
    )


def test_document_scope_keeps_all_parent_chunks() -> None:
    chunks = [
        _chunk(
            document_seed="a",
            chunk_index=0,
            chunk_count=2,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="첫 번째 변경",
        ),
        _chunk(
            document_seed="a",
            chunk_index=1,
            chunk_count=2,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="두 번째 변경",
        ),
    ]

    case = build_retrieval_eval_cases(
        [_seed()],
        chunks,
        corpus_hash=HASH,
    )[0]

    assert case.target_granularity == TargetGranularity.DOCUMENT
    assert len(case.relevant_chunk_ids) == 2


def test_required_text_selects_answer_chunk() -> None:
    chunks = [
        _chunk(
            document_seed="a",
            chunk_index=0,
            chunk_count=2,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="기본 피해량 : 70 ⇒ 80",
        ),
        _chunk(
            document_seed="a",
            chunk_index=1,
            chunk_count=2,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="다른 설명",
        ),
    ]

    case = build_retrieval_eval_cases(
        [_seed(required_text="70 ⇒ 80")],
        chunks,
        corpus_hash=HASH,
    )[0]

    assert case.target_granularity == TargetGranularity.ANSWER_CHUNK
    assert len(case.relevant_chunk_ids) == 1


def test_entity_name_disambiguates_same_title() -> None:
    chunks = [
        _chunk(
            document_seed="nilah",
            chunk_index=0,
            chunk_count=1,
            title="기본 능력치",
            entity_name="닐라",
            text="닐라 기본 능력치",
        ),
        _chunk(
            document_seed="quinn",
            chunk_index=0,
            chunk_count=1,
            title="기본 능력치",
            entity_name="퀸",
            text="퀸 기본 능력치",
        ),
    ]

    seed = _seed(
        title="기본 능력치",
        entity_name="퀸",
    )

    case = build_retrieval_eval_cases(
        [seed],
        chunks,
        corpus_hash=HASH,
    )[0]

    assert len(case.relevant_document_ids) == 1


def test_missing_target_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="matched no chunks",
    ):
        build_retrieval_eval_cases(
            [_seed()],
            [],
            corpus_hash=HASH,
        )


def test_multiple_documents_are_rejected() -> None:
    chunks = [
        _chunk(
            document_seed="a",
            chunk_index=0,
            chunk_count=1,
            title="버그 수정",
            entity_name=None,
            text="첫 번째",
        ),
        _chunk(
            document_seed="b",
            chunk_index=0,
            chunk_count=1,
            title="버그 수정",
            entity_name=None,
            text="두 번째",
        ),
    ]

    seed = _seed(
        title="버그 수정",
        entity_name=None,
    )

    with pytest.raises(
        ValueError,
        match="multiple documents",
    ):
        build_retrieval_eval_cases(
            [seed],
            chunks,
            corpus_hash=HASH,
        )


def test_missing_required_text_is_rejected() -> None:
    chunks = [
        _chunk(
            document_seed="a",
            chunk_index=0,
            chunk_count=1,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="기본 피해량 : 70 ⇒ 80",
        )
    ]

    with pytest.raises(
        ValueError,
        match="required answer text",
    ):
        build_retrieval_eval_cases(
            [_seed(required_text="존재하지 않는 값")],
            chunks,
            corpus_hash=HASH,
        )


def test_duplicate_query_id_is_rejected() -> None:
    seed = _seed()

    with pytest.raises(
        ValueError,
        match="Duplicate retrieval query_id",
    ):
        build_retrieval_eval_cases(
            [seed, seed],
            [],
            corpus_hash=HASH,
        )


def test_corpus_hash_is_preserved() -> None:
    chunks = [
        _chunk(
            document_seed="a",
            chunk_index=0,
            chunk_count=1,
            title="E - 회전 베기",
            entity_name="트린다미어",
            text="기본 피해량",
        )
    ]

    case = build_retrieval_eval_cases(
        [_seed()],
        chunks,
        corpus_hash=HASH,
    )[0]

    assert case.corpus_sha256 == HASH
