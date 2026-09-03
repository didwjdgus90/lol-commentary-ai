from __future__ import annotations

import json
from hashlib import sha256

from lol_commentary_backend.rag.query_planner.models import (
    RAG_QUERY_PLAN_VERSION,
    RAG_QUERY_PLANNER_POLICY_VERSION,
    RAGFocusChampion,
    RAGFocusRole,
    RAGPlannerInput,
    RAGQueryIntent,
    RAGQueryPlan,
    RAGRetrievalQuery,
    RAGSelectedItem,
)

MAX_CHAMPION_QUERIES = 2

MAX_ITEM_QUERIES = 2

MAX_PLAN_QUERIES = 6


_ROLE_ORDER = {
    RAGFocusRole.ACTOR: 0,
    RAGFocusRole.TARGET: 1,
}


def _query_id(
    *,
    record_id: str,
    intent: RAGQueryIntent,
    subject_key: str,
    query_text: str,
) -> str:
    payload = {
        "record_id": record_id,
        "intent": intent.value,
        "subject_key": subject_key,
        "query_text": query_text,
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return sha256(canonical.encode("utf-8")).hexdigest()


def _champion_sort_key(
    champion: RAGFocusChampion,
) -> tuple[
    int,
    int,
    int,
    str,
]:
    return (
        _ROLE_ORDER[champion.role],
        champion.participant_id,
        champion.champion_id,
        champion.champion_name,
    )


def _select_champions(
    champions: tuple[
        RAGFocusChampion,
        ...,
    ],
) -> tuple[
    RAGFocusChampion,
    ...,
]:
    ordered = sorted(
        champions,
        key=_champion_sort_key,
    )

    result: list[RAGFocusChampion] = []

    seen_champion_ids: set[int] = set()

    for champion in ordered:
        if champion.champion_id in seen_champion_ids:
            continue

        seen_champion_ids.add(champion.champion_id)

        result.append(champion)

        if len(result) >= MAX_CHAMPION_QUERIES:
            break

    return tuple(result)


def _item_sort_key(
    item: RAGSelectedItem,
) -> tuple[
    int,
    int,
    int,
    int,
    str,
]:
    return (
        -item.score,
        item.age_ms_at_situation_start,
        item.item_id,
        item.participant_id,
        item.source_event_sha256,
    )


def _select_items(
    items: tuple[
        RAGSelectedItem,
        ...,
    ],
) -> tuple[
    RAGSelectedItem,
    ...,
]:
    ordered = sorted(
        items,
        key=_item_sort_key,
    )

    result: list[RAGSelectedItem] = []

    seen_item_ids: set[int] = set()

    for item in ordered:
        if item.item_id in seen_item_ids:
            continue

        seen_item_ids.add(item.item_id)

        result.append(item)

        if len(result) >= MAX_ITEM_QUERIES:
            break

    return tuple(result)


def _champion_query(
    *,
    planner_input: RAGPlannerInput,
    champion: RAGFocusChampion,
) -> RAGRetrievalQuery:
    query_text = f"{planner_input.patch} {champion.champion_name} 변경 사항"

    subject_key = f"champion:{champion.champion_id}"

    return RAGRetrievalQuery(
        query_id=_query_id(
            record_id=(planner_input.record_id),
            intent=(RAGQueryIntent.CHAMPION_PATCH),
            subject_key=subject_key,
            query_text=query_text,
        ),
        intent=(RAGQueryIntent.CHAMPION_PATCH),
        query_text=query_text,
        subject_key=subject_key,
        subject_id=str(champion.champion_id),
        subject_name=(champion.champion_name),
        participant_id=(champion.participant_id),
        focus_role=champion.role,
    )


def _item_query(
    *,
    planner_input: RAGPlannerInput,
    item: RAGSelectedItem,
) -> RAGRetrievalQuery:
    query_text = f"{planner_input.patch} {item.item_name} 변경 사항"

    subject_key = f"item:{item.item_id}"

    return RAGRetrievalQuery(
        query_id=_query_id(
            record_id=(planner_input.record_id),
            intent=(RAGQueryIntent.ITEM_PATCH),
            subject_key=subject_key,
            query_text=query_text,
        ),
        intent=(RAGQueryIntent.ITEM_PATCH),
        query_text=query_text,
        subject_key=subject_key,
        subject_id=str(item.item_id),
        subject_name=(item.item_name),
        participant_id=(item.participant_id),
        source_event_sha256=(item.source_event_sha256),
    )


def _system_query(
    *,
    planner_input: RAGPlannerInput,
    intent: RAGQueryIntent,
) -> RAGRetrievalQuery:
    if intent == RAGQueryIntent.OBJECTIVE_PATCH:
        query_text = f"{planner_input.patch} 오브젝트 드래곤 바론 전령 변경 사항"

        subject_key = "system:objective"

        subject_name = "드래곤 바론 전령"

    elif intent == RAGQueryIntent.STRUCTURE_PATCH:
        query_text = f"{planner_input.patch} 포탑 억제기 구조물 변경 사항"

        subject_key = "system:structure"

        subject_name = "포탑 억제기 구조물"

    else:
        raise ValueError(f"Unsupported system query intent: {intent}")

    return RAGRetrievalQuery(
        query_id=_query_id(
            record_id=(planner_input.record_id),
            intent=intent,
            subject_key=subject_key,
            query_text=query_text,
        ),
        intent=intent,
        query_text=query_text,
        subject_key=subject_key,
        subject_name=subject_name,
    )


def _plan_id(
    *,
    planner_input: RAGPlannerInput,
    queries: tuple[
        RAGRetrievalQuery,
        ...,
    ],
) -> str:
    payload = {
        "plan_version": (RAG_QUERY_PLAN_VERSION),
        "planner_policy_version": (RAG_QUERY_PLANNER_POLICY_VERSION),
        "record_id": (planner_input.record_id),
        "match_id": (planner_input.match_id),
        "situation_id": (planner_input.situation_id),
        "patch": (planner_input.patch),
        "ddragon_version": (planner_input.ddragon_version),
        "priority_tier": (planner_input.priority_tier.value),
        "situation_kind": (planner_input.situation_kind),
        "queries": [query.model_dump(mode="json") for query in queries],
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return sha256(canonical.encode("utf-8")).hexdigest()


def build_rag_query_plan(
    planner_input: RAGPlannerInput,
) -> RAGQueryPlan:
    queries: list[RAGRetrievalQuery] = []

    for champion in _select_champions(planner_input.focus_champions):
        queries.append(
            _champion_query(
                planner_input=(planner_input),
                champion=champion,
            )
        )

    for item in _select_items(planner_input.selected_items):
        queries.append(
            _item_query(
                planner_input=(planner_input),
                item=item,
            )
        )

    if planner_input.objective_context:
        queries.append(
            _system_query(
                planner_input=(planner_input),
                intent=(RAGQueryIntent.OBJECTIVE_PATCH),
            )
        )

    if planner_input.structure_context:
        queries.append(
            _system_query(
                planner_input=(planner_input),
                intent=(RAGQueryIntent.STRUCTURE_PATCH),
            )
        )

    ordered_queries = tuple(queries)

    if len(ordered_queries) > MAX_PLAN_QUERIES:
        raise RuntimeError("RAG query plan exceeded maximum query count")

    query_ids = [query.query_id for query in ordered_queries]

    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("Duplicate query ID inside RAG plan")

    return RAGQueryPlan(
        plan_id=_plan_id(
            planner_input=(planner_input),
            queries=ordered_queries,
        ),
        record_id=(planner_input.record_id),
        match_id=(planner_input.match_id),
        situation_id=(planner_input.situation_id),
        patch=planner_input.patch,
        ddragon_version=(planner_input.ddragon_version),
        priority_tier=(planner_input.priority_tier),
        situation_kind=(planner_input.situation_kind),
        queries=ordered_queries,
    )


def build_rag_query_plans(
    inputs: tuple[
        RAGPlannerInput,
        ...,
    ],
) -> tuple[
    RAGQueryPlan,
    ...,
]:
    record_ids = [planner_input.record_id for planner_input in inputs]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate planner input record ID")

    plans = tuple(build_rag_query_plan(planner_input) for planner_input in inputs)

    plan_ids = [plan.plan_id for plan in plans]

    if len(plan_ids) != len(set(plan_ids)):
        raise RuntimeError("Duplicate RAG query plan ID")

    return plans
