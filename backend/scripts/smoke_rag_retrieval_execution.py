from __future__ import annotations

import os
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from lol_commentary_backend.rag.query_planner.models import (
    RAGFocusChampion,
    RAGFocusRole,
    RAGPlannerInput,
    RAGPriorityTier,
    RAGSelectedItem,
)
from lol_commentary_backend.rag.query_planner.planner import (
    build_rag_query_plan,
)
from lol_commentary_backend.rag.retrieval_execution.executor import (
    execute_rag_query_plans,
)
from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalStrategy,
)
from lol_commentary_backend.retrieval.runtime.multi_factory import (
    build_multi_patch_retrieval_service,
)

PATCH = "26.17"

DDRAGON_VERSION = "16.17.1"

TOP_K = 5


def _hash(
    value: str,
) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _planner_input(
    *,
    label: str,
    champion: bool = False,
    item_id: int | None = None,
    item_name: str | None = None,
    objective: bool = False,
    structure: bool = False,
    priority: RAGPriorityTier = (RAGPriorityTier.HIGH),
) -> RAGPlannerInput:
    champions = (
        (
            RAGFocusChampion(
                participant_id=1,
                champion_id=777,
                champion_name="Yone",
                role=RAGFocusRole.ACTOR,
            ),
        )
        if champion
        else ()
    )

    items: tuple[
        RAGSelectedItem,
        ...,
    ] = ()

    if item_id is not None or item_name is not None:
        if item_id is None or item_name is None:
            raise ValueError("item_id and item_name must be supplied together")

        items = (
            RAGSelectedItem(
                participant_id=1,
                item_id=item_id,
                item_name=item_name,
                score=90,
                tier="high",
                age_ms_at_situation_start=30_000,
                source_event_sha256=_hash(f"{label}:item"),
            ),
        )

    return RAGPlannerInput(
        record_id=_hash(f"{label}:record"),
        match_id="KR_STEP_45_D_SMOKE",
        situation_id=_hash(f"{label}:situation"),
        patch=PATCH,
        ddragon_version=DDRAGON_VERSION,
        priority_tier=priority,
        situation_kind="combat",
        focus_champions=champions,
        selected_items=items,
        objective_context=objective,
        structure_context=structure,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    database_url = os.environ.get("LOL_DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "LOL_DATABASE_URL is not set. Run scripts/load_dev_env.ps1 before this smoke test."
        )

    first_plan = build_rag_query_plan(
        _planner_input(
            label="first",
            champion=True,
            item_id=3031,
            item_name="무한의 대검",
            objective=True,
            structure=True,
            priority=(RAGPriorityTier.CRITICAL),
        )
    )

    second_plan = build_rag_query_plan(
        _planner_input(
            label="second",
            champion=True,
            item_id=6672,
            item_name="크라켄 학살자",
            structure=True,
        )
    )

    third_plan = build_rag_query_plan(
        _planner_input(
            label="third",
            priority=(RAGPriorityTier.LOW),
        )
    )

    plans = (
        first_plan,
        second_plan,
        third_plan,
    )

    print("=== STEP 45-D RAG RETRIEVAL EXECUTION SMOKE ===")

    print()

    print(f"PATCH_SCOPE={PATCH}")

    print(f"PLAN_COUNT={len(plans)}")

    print(f"QUERY_REFERENCES={sum(plan.query_count for plan in plans)}")

    print()

    startup_started = perf_counter()

    service = build_multi_patch_retrieval_service(
        repository_root=repository_root,
        locales=("ko_KR",),
        patches=(PATCH,),
        enable_primary=True,
        requested_device="auto",
        database_url=database_url,
        source_top_n=30,
        rrf_k=60,
    )

    startup_ms = (perf_counter() - startup_started) * 1000

    if not service.primary_available:
        raise RuntimeError("Primary retrieval must be available")

    batch = execute_rag_query_plans(
        plans=plans,
        retrieval_service=service,
        top_k=TOP_K,
        mode=RetrievalMode.PRIMARY,
    )

    executed_results = [
        result for plan in batch.plans for result in plan.query_results if not result.cache_reused
    ]

    strategy_counts = Counter(result.strategy_used.value for result in executed_results)

    intent_counts = Counter(
        result.intent.value for plan in batch.plans for result in plan.query_results
    )

    print("=== EXECUTION SUMMARY ===")

    print(f"startup_ms={startup_ms:.2f}")

    print(f"wall_elapsed_ms={batch.wall_elapsed_ms:.2f}")

    print(f"query_references={batch.query_reference_count}")

    print(f"unique_query_executions={batch.unique_query_execution_count}")

    print(f"cache_hits={batch.cache_hit_count}")

    print(f"fallback_executions={batch.fallback_execution_count}")

    print(f"zero_hit_executions={batch.zero_hit_execution_count}")

    print(f"total_hit_references={batch.total_hit_references}")

    print()

    print("=== INTENT REFERENCES ===")

    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")

    print()

    print("=== EXECUTED STRATEGIES ===")

    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count}")

    print()

    print("=== UNIQUE QUERY RESULTS ===")

    for result in executed_results:
        print(f"QUERY={result.query_text}")

        print(f"  intent={result.intent.value}")

        print(f"  strategy={result.strategy_used.value}")

        print(f"  expanded={result.expanded_query}")

        print(f"  hits={len(result.hits)}")

        if result.hits:
            first_hit = result.hits[0]

            print(f"  top1_patch={first_hit.patch}")

            print(f"  top1_title={first_hit.title}")

            print(f"  top1_entity={first_hit.entity_name}")

            print(f"  top1_score={first_hit.score:.6f}")

            print(f"  top1_dense_rank={first_hit.dense_rank}")

            print(f"  top1_sparse_rank={first_hit.sparse_rank}")

        print()

    payload = batch.model_dump_json().casefold()

    pii_tokens = (
        "puuid",
        "summonerid",
        "summoner_id",
        "gamename",
        "game_name",
        "tagline",
        "tag_line",
    )

    pii_hits = [token for token in pii_tokens if token in payload]

    if batch.plan_count != 3:
        raise RuntimeError("Expected 3 smoke plans")

    if batch.query_reference_count != 7:
        raise RuntimeError("Expected 7 query references")

    if batch.unique_query_execution_count != 5:
        raise RuntimeError("Expected 5 unique retrieval executions")

    if batch.cache_hit_count != 2:
        raise RuntimeError("Expected 2 retrieval cache hits")

    if batch.fallback_execution_count != 0:
        raise RuntimeError("Primary smoke must not use fallback")

    if batch.zero_hit_execution_count != 0:
        raise RuntimeError("Expected evidence for every smoke query")

    if pii_hits:
        raise RuntimeError("PII-like token leaked into evidence artifact: " + ",".join(pii_hits))

    for result in executed_results:
        if result.strategy_used != RetrievalStrategy.BGE_ALIAS_RRF:
            raise RuntimeError("Primary smoke did not use hybrid retrieval")

        for hit in result.hits:
            if hit.patch != PATCH:
                raise RuntimeError("Patch scope escaped")

            if hit.locale != "ko_KR":
                raise RuntimeError("Locale scope escaped")

    print("PATCH_SCOPE=PASS")

    print("CACHE_DEDUP=PASS")

    print("PRIMARY_HYBRID=PASS")

    print("PII_BOUNDARY=PASS")

    print()

    print("RAG_RETRIEVAL_EXECUTION_V1_SMOKE=PASS")


if __name__ == "__main__":
    main()
