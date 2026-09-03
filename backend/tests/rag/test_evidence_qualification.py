from __future__ import annotations

from lol_commentary_backend.rag.evidence_qualification.models import (
    EvidenceQualificationReason,
    EvidenceQualificationStatus,
)
from lol_commentary_backend.rag.evidence_qualification.qualifier import (
    qualify_query_evidence,
)
from lol_commentary_backend.rag.query_planner.models import (
    RAGQueryIntent,
)
from lol_commentary_backend.rag.retrieval_execution.models import (
    RAGQueryEvidenceResult,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalHit,
    RetrievalMode,
    RetrievalStrategy,
)


def _hit(
    *,
    rank: int = 1,
    section: str = "champion",
    title: str = "요네",
    entity: str | None = "요네",
    heading_path: list[str] | None = None,
    text: str = "근거",
) -> RetrievalHit:
    return RetrievalHit(
        rank=rank,
        chunk_id=(f"{rank:064x}"),
        document_id=(f"{rank + 100:064x}"),
        source_record_id=(f"{rank + 200:064x}"),
        patch="26.17",
        locale="ko_KR",
        source_url=(f"https://example.com/{rank}"),
        section_kind=section,
        title=title,
        entity_name=entity,
        heading_path=(
            heading_path
            if heading_path is not None
            else [
                "챔피언",
                title,
            ]
        ),
        text=text,
        score=0.03,
        dense_rank=rank,
        sparse_rank=rank,
        dense_score=0.5,
        sparse_score=3.0,
    )


def _result(
    *,
    intent: RAGQueryIntent,
    subject_name: str | None,
    expanded_query: str,
    hits: tuple[
        RetrievalHit,
        ...,
    ],
) -> RAGQueryEvidenceResult:
    return RAGQueryEvidenceResult(
        query_id="1" * 64,
        plan_id="2" * 64,
        record_id="3" * 64,
        match_id="KR_TEST",
        situation_id="4" * 64,
        patch="26.17",
        intent=intent,
        query_text="test query",
        subject_key="test:1",
        subject_id="1",
        subject_name=subject_name,
        cache_reused=False,
        expanded_query=expanded_query,
        requested_mode=(RetrievalMode.PRIMARY),
        strategy_used=(RetrievalStrategy.BGE_ALIAS_RRF),
        fallback_used=False,
        source_top_n=30,
        source_retrieval_elapsed_ms=1.0,
        hits=hits,
    )


def test_champion_exact_entity_is_qualified() -> None:
    result = _result(
        intent=(RAGQueryIntent.CHAMPION_PATCH),
        subject_name="Yone",
        expanded_query=("26.17 Yone 변경 사항 | 요네"),
        hits=(_hit(),),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.status == (EvidenceQualificationStatus.QUALIFIED)

    assert qualified.reason == (EvidenceQualificationReason.ENTITY_SECTION_MATCH)

    assert qualified.first_qualified_rank == 1


def test_cross_language_alias_is_supported() -> None:
    result = _result(
        intent=(RAGQueryIntent.CHAMPION_PATCH),
        subject_name="Yone",
        expanded_query=("26.17 Yone 변경 사항 | 요네"),
        hits=(_hit(entity="요네"),),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.has_qualified_evidence is True


def test_wrong_entity_is_rejected() -> None:
    result = _result(
        intent=(RAGQueryIntent.CHAMPION_PATCH),
        subject_name="Yone",
        expanded_query=("26.17 Yone 변경 사항 | 요네"),
        hits=(
            _hit(
                title="야스오",
                entity="야스오",
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.status == (EvidenceQualificationStatus.ABSTAIN_NO_QUALIFIED_EVIDENCE)


def test_right_entity_wrong_section_is_rejected() -> None:
    result = _result(
        intent=(RAGQueryIntent.ITEM_PATCH),
        subject_name="무한의 대검",
        expanded_query=("26.17 무한의 대검 변경 사항 | Infinity Edge"),
        hits=(
            _hit(
                section="game_mode",
                title="증강",
                entity=None,
                text=("무한의 대검 업그레이드"),
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.reason == (EvidenceQualificationReason.NO_ENTITY_SECTION_MATCH)


def test_item_exact_entity_and_section_is_qualified() -> None:
    result = _result(
        intent=(RAGQueryIntent.ITEM_PATCH),
        subject_name="갈라진 하늘",
        expanded_query=("26.17 갈라진 하늘 변경 사항 | Sundered Sky"),
        hits=(
            _hit(
                section="item",
                title="갈라진 하늘",
                entity="갈라진 하늘",
                heading_path=[
                    "아이템",
                    "갈라진 하늘",
                ],
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.has_qualified_evidence is True


def test_objective_body_only_match_is_rejected() -> None:
    result = _result(
        intent=(RAGQueryIntent.OBJECTIVE_PATCH),
        subject_name=None,
        expanded_query=("26.17 오브젝트 변경 사항"),
        hits=(
            _hit(
                section="general",
                title="챔피언",
                entity=None,
                heading_path=[
                    "클래식",
                    "챔피언",
                ],
                text=("드래곤과 바론이라는 단어가 본문에 있음"),
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.reason == (EvidenceQualificationReason.NO_STRONG_SYSTEM_SCOPE)


def test_objective_heading_match_is_qualified() -> None:
    result = _result(
        intent=(RAGQueryIntent.OBJECTIVE_PATCH),
        subject_name=None,
        expanded_query=("26.17 오브젝트 변경 사항"),
        hits=(
            _hit(
                section="general",
                title="드래곤 변경",
                entity=None,
                heading_path=[
                    "게임플레이",
                    "드래곤",
                ],
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.has_qualified_evidence is True


def test_generic_structure_word_is_not_enough() -> None:
    result = _result(
        intent=(RAGQueryIntent.STRUCTURE_PATCH),
        subject_name=None,
        expanded_query=("26.17 구조물 변경 사항"),
        hits=(
            _hit(
                section="general",
                title="아트",
                entity=None,
                heading_path=[
                    "클래식",
                    "아트",
                ],
                text=("의도치 않게 배치된 구조물이 제거됩니다."),
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.has_qualified_evidence is False


def test_turret_heading_is_qualified() -> None:
    result = _result(
        intent=(RAGQueryIntent.STRUCTURE_PATCH),
        subject_name=None,
        expanded_query=("26.17 구조물 변경 사항"),
        hits=(
            _hit(
                section="general",
                title="포탑",
                entity=None,
                heading_path=[
                    "게임플레이",
                    "포탑",
                ],
            ),
        ),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.has_qualified_evidence is True


def test_zero_hits_abstains_explicitly() -> None:
    result = _result(
        intent=(RAGQueryIntent.ITEM_PATCH),
        subject_name="크라켄 학살자",
        expanded_query=("26.17 크라켄 학살자 변경 사항 | Kraken Slayer"),
        hits=(),
    )

    qualified = qualify_query_evidence(result)

    assert qualified.status == (EvidenceQualificationStatus.ABSTAIN_NO_QUALIFIED_EVIDENCE)

    assert qualified.reason == (EvidenceQualificationReason.NO_RETRIEVAL_HITS)
