from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from lol_commentary_backend.rag.evidence_qualification.models import (
    EvidenceQualificationStatus,
)
from lol_commentary_backend.rag.evidence_qualification.qualifier import (
    qualify_batch_evidence,
)
from lol_commentary_backend.rag.query_planner.models import (
    RAGPriorityTier,
    RAGQueryIntent,
    RAGQueryPlan,
    RAGRetrievalQuery,
)
from lol_commentary_backend.rag.retrieval_execution.executor import (
    execute_rag_query_plans,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
)
from lol_commentary_backend.retrieval.runtime.multi_factory import (
    build_multi_patch_retrieval_service,
)

PATCH = "26.17"

TOP_K = 10


def _hash(
    value: str,
) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _query(
    *,
    label: str,
    intent: RAGQueryIntent,
    query_text: str,
    subject_key: str,
    subject_id: str | None = None,
    subject_name: str | None = None,
) -> RAGRetrievalQuery:
    return RAGRetrievalQuery(
        query_id=_hash(f"query:{label}"),
        intent=intent,
        query_text=query_text,
        subject_key=subject_key,
        subject_id=subject_id,
        subject_name=subject_name,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    database_url = os.environ.get("LOL_DATABASE_URL")

    if not database_url:
        raise RuntimeError("LOL_DATABASE_URL is not set")

    queries = (
        _query(
            label="yone",
            intent=(RAGQueryIntent.CHAMPION_PATCH),
            query_text=("26.17 Yone 변경 사항"),
            subject_key="champion:777",
            subject_id="777",
            subject_name="Yone",
        ),
        _query(
            label="infinity",
            intent=(RAGQueryIntent.ITEM_PATCH),
            query_text=("26.17 무한의 대검 변경 사항"),
            subject_key="item:3031",
            subject_id="3031",
            subject_name="무한의 대검",
        ),
        _query(
            label="kraken",
            intent=(RAGQueryIntent.ITEM_PATCH),
            query_text=("26.17 크라켄 학살자 변경 사항"),
            subject_key="item:6672",
            subject_id="6672",
            subject_name="크라켄 학살자",
        ),
        _query(
            label="objective",
            intent=(RAGQueryIntent.OBJECTIVE_PATCH),
            query_text=("26.17 오브젝트 드래곤 바론 전령 변경 사항"),
            subject_key=("system:objective"),
            subject_name=("드래곤 바론 전령"),
        ),
        _query(
            label="structure",
            intent=(RAGQueryIntent.STRUCTURE_PATCH),
            query_text=("26.17 포탑 억제기 구조물 변경 사항"),
            subject_key=("system:structure"),
            subject_name=("포탑 억제기 구조물"),
        ),
    )

    plan = RAGQueryPlan(
        plan_id=_hash("qualification:plan"),
        record_id=_hash("qualification:record"),
        match_id=("KR_STEP_45_E_C_SMOKE"),
        situation_id=_hash("qualification:situation"),
        patch=PATCH,
        ddragon_version="16.17.1",
        priority_tier=(RAGPriorityTier.HIGH),
        situation_kind="mixed",
        queries=queries,
    )

    print("=== STEP 45-E-C EVIDENCE QUALIFICATION SMOKE ===")

    print()

    service = build_multi_patch_retrieval_service(
        repository_root=(repository_root),
        locales=("ko_KR",),
        patches=(PATCH,),
        enable_primary=True,
        requested_device="auto",
        database_url=(database_url),
        source_top_n=30,
        rrf_k=60,
    )

    if not service.primary_available:
        raise RuntimeError("Primary retrieval must be available")

    retrieval_batch = execute_rag_query_plans(
        plans=(plan,),
        retrieval_service=service,
        top_k=TOP_K,
        mode=(RetrievalMode.PRIMARY),
    )

    qualified_batch = qualify_batch_evidence(retrieval_batch)

    rows = qualified_batch.plans[0].queries

    print("=== QUALIFICATION RESULTS ===")

    print()

    for row in rows:
        print(f"QUERY={row.query_text}")

        print(f"  intent={row.intent.value}")

        print(f"  status={row.status.value}")

        print(f"  reason={row.reason.value}")

        print(f"  original_hits={row.original_hit_count}")

        print(f"  qualified_hits={len(row.qualified_hits)}")

        print(
            "  first_qualified_rank="
            + (str(row.first_qualified_rank) if (row.first_qualified_rank is not None) else "NONE")
        )

        for qualified_hit in row.qualified_hits[:3]:
            hit = qualified_hit.hit

            print(
                "    QUALIFIED "
                f"rank="
                f"{qualified_hit.original_rank} "
                f"section={hit.section_kind} "
                f"title={hit.title} "
                f"entity={hit.entity_name}"
            )

        print()

    result_by_intent_subject = {row.subject_key: row for row in rows}

    yone = result_by_intent_subject["champion:777"]

    infinity = result_by_intent_subject["item:3031"]

    kraken = result_by_intent_subject["item:6672"]

    objective = result_by_intent_subject["system:objective"]

    structure = result_by_intent_subject["system:structure"]

    if yone.status != (EvidenceQualificationStatus.QUALIFIED):
        raise RuntimeError("Yone evidence should qualify")

    for row in (
        infinity,
        kraken,
        objective,
        structure,
    ):
        if row.status != (EvidenceQualificationStatus.ABSTAIN_NO_QUALIFIED_EVIDENCE):
            raise RuntimeError(f"Expected strict abstention for {row.subject_key}")

    if yone.first_qualified_rank != 1:
        raise RuntimeError("Expected Yone qualified evidence at rank 1")

    if qualified_batch.qualified_query_count != 1:
        raise RuntimeError("Expected exactly 1 qualified query")

    if qualified_batch.abstained_query_count != 4:
        raise RuntimeError("Expected exactly 4 abstained queries")

    print("=== AGGREGATE ===")

    print(f"query_count={qualified_batch.query_count}")

    print(f"qualified_query_count={qualified_batch.qualified_query_count}")

    print(f"abstained_query_count={qualified_batch.abstained_query_count}")

    print(f"qualified_hit_count={qualified_batch.qualified_hit_count}")

    print()

    print("GLOBAL_SCORE_THRESHOLD=DISALLOWED")

    print("ENTITY_SCOPE_GATE=PASS")

    print("SYSTEM_HEADING_GATE=PASS")

    print("ABSTENTION_GATE=PASS")

    print()

    print("EVIDENCE_QUALIFICATION_V1_SMOKE=PASS")


if __name__ == "__main__":
    main()
