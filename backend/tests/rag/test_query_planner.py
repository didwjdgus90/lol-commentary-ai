from __future__ import annotations

import pytest

from lol_commentary_backend.rag.query_planner.models import (
    RAGFocusChampion,
    RAGFocusRole,
    RAGPlannerInput,
    RAGPriorityTier,
    RAGQueryIntent,
    RAGSelectedItem,
)
from lol_commentary_backend.rag.query_planner.planner import (
    build_rag_query_plan,
    build_rag_query_plans,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _champion(
    *,
    participant_id: int,
    champion_id: int,
    name: str,
    role: RAGFocusRole,
) -> RAGFocusChampion:
    return RAGFocusChampion(
        participant_id=(participant_id),
        champion_id=champion_id,
        champion_name=name,
        role=role,
    )


def _item(
    *,
    participant_id: int,
    item_id: int,
    name: str,
    score: int,
    source_sha256: str,
    age_ms: int = 30_000,
) -> RAGSelectedItem:
    return RAGSelectedItem(
        participant_id=(participant_id),
        item_id=item_id,
        item_name=name,
        score=score,
        tier=("high" if score >= 85 else "medium"),
        age_ms_at_situation_start=(age_ms),
        source_event_sha256=(source_sha256),
    )


def _input(
    *,
    record_id: str = HASH_A,
    focus_champions: tuple[
        RAGFocusChampion,
        ...,
    ] = (),
    selected_items: tuple[
        RAGSelectedItem,
        ...,
    ] = (),
    objective: bool = False,
    structure: bool = False,
) -> RAGPlannerInput:
    return RAGPlannerInput(
        record_id=record_id,
        match_id="KR_TEST",
        situation_id=HASH_B,
        patch="26.17",
        ddragon_version="16.17.1",
        priority_tier=(RAGPriorityTier.HIGH),
        situation_kind="combat",
        focus_champions=(focus_champions),
        selected_items=(selected_items),
        objective_context=objective,
        structure_context=structure,
    )


def test_builds_champion_patch_query() -> None:
    planner_input = _input(
        focus_champions=(
            _champion(
                participant_id=1,
                champion_id=266,
                name="Aatrox",
                role=(RAGFocusRole.ACTOR),
            ),
        )
    )

    plan = build_rag_query_plan(planner_input)

    assert plan.query_count == 1

    query = plan.queries[0]

    assert query.intent == RAGQueryIntent.CHAMPION_PATCH

    assert query.query_text == "26.17 Aatrox 변경 사항"


def test_actor_is_ordered_before_target() -> None:
    planner_input = _input(
        focus_champions=(
            _champion(
                participant_id=2,
                champion_id=22,
                name="Ashe",
                role=(RAGFocusRole.TARGET),
            ),
            _champion(
                participant_id=1,
                champion_id=266,
                name="Aatrox",
                role=(RAGFocusRole.ACTOR),
            ),
        )
    )

    plan = build_rag_query_plan(planner_input)

    assert plan.queries[0].subject_name == "Aatrox"

    assert plan.queries[1].subject_name == "Ashe"


def test_champion_queries_are_capped_at_two() -> None:
    champions = tuple(
        _champion(
            participant_id=index,
            champion_id=(100 + index),
            name=f"Champion{index}",
            role=(RAGFocusRole.ACTOR),
        )
        for index in range(
            1,
            5,
        )
    )

    plan = build_rag_query_plan(_input(focus_champions=(champions)))

    champion_queries = [
        query for query in plan.queries if (query.intent == RAGQueryIntent.CHAMPION_PATCH)
    ]

    assert len(champion_queries) == 2


def test_duplicate_champion_is_deduplicated() -> None:
    plan = build_rag_query_plan(
        _input(
            focus_champions=(
                _champion(
                    participant_id=1,
                    champion_id=266,
                    name="Aatrox",
                    role=(RAGFocusRole.ACTOR),
                ),
                _champion(
                    participant_id=2,
                    champion_id=266,
                    name="Aatrox",
                    role=(RAGFocusRole.TARGET),
                ),
            )
        )
    )

    assert plan.query_count == 1


def test_builds_item_patch_query() -> None:
    plan = build_rag_query_plan(
        _input(
            selected_items=(
                _item(
                    participant_id=1,
                    item_id=3031,
                    name="무한의 대검",
                    score=90,
                    source_sha256=(HASH_C),
                ),
            )
        )
    )

    query = plan.queries[0]

    assert query.intent == RAGQueryIntent.ITEM_PATCH

    assert query.query_text == ("26.17 무한의 대검 변경 사항")

    assert query.source_event_sha256 == HASH_C


def test_duplicate_item_keeps_best_signal() -> None:
    plan = build_rag_query_plan(
        _input(
            selected_items=(
                _item(
                    participant_id=1,
                    item_id=3031,
                    name="무한의 대검",
                    score=70,
                    source_sha256=("1" * 64),
                ),
                _item(
                    participant_id=2,
                    item_id=3031,
                    name="무한의 대검",
                    score=90,
                    source_sha256=("2" * 64),
                ),
            )
        )
    )

    assert plan.query_count == 1

    assert plan.queries[0].source_event_sha256 == "2" * 64


def test_builds_objective_query() -> None:
    plan = build_rag_query_plan(_input(objective=True))

    assert plan.query_count == 1

    assert plan.queries[0].intent == (RAGQueryIntent.OBJECTIVE_PATCH)


def test_builds_structure_query() -> None:
    plan = build_rag_query_plan(_input(structure=True))

    assert plan.query_count == 1

    assert plan.queries[0].intent == (RAGQueryIntent.STRUCTURE_PATCH)


def test_plan_never_exceeds_six_queries() -> None:
    champions = tuple(
        _champion(
            participant_id=index,
            champion_id=(100 + index),
            name=f"C{index}",
            role=(RAGFocusRole.ACTOR),
        )
        for index in range(
            1,
            4,
        )
    )

    items = tuple(
        _item(
            participant_id=index,
            item_id=(3000 + index),
            name=f"Item{index}",
            score=90,
            source_sha256=(f"{index:064x}"),
        )
        for index in range(
            1,
            4,
        )
    )

    plan = build_rag_query_plan(
        _input(
            focus_champions=(champions),
            selected_items=items,
            objective=True,
            structure=True,
        )
    )

    assert plan.query_count == 6


def test_empty_and_deterministic_plan() -> None:
    planner_input = _input()

    first = build_rag_query_plan(planner_input)

    second = build_rag_query_plan(planner_input)

    assert first.queries == ()

    assert first.plan_id == second.plan_id


def test_duplicate_batch_record_id_is_rejected() -> None:
    planner_input = _input()

    with pytest.raises(
        ValueError,
        match="Duplicate planner",
    ):
        build_rag_query_plans(
            (
                planner_input,
                planner_input,
            )
        )
