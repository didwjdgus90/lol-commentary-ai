from __future__ import annotations

import pytest

from lol_commentary_backend.intelligence.item_evidence_models import (
    ItemEvidenceAction,
)
from lol_commentary_backend.intelligence.item_power_spike_models import (
    ItemPowerSpikeContext,
    ItemPowerSpikeEvaluation,
    ItemPowerSpikeTier,
)
from lol_commentary_backend.intelligence.item_power_spike_selection_models import (
    SelectedItemPowerSpikeTier,
)
from lol_commentary_backend.intelligence.item_power_spike_selector import (
    select_item_power_spike_contexts,
)


def _evaluation(
    *,
    item_id: int = 3071,
    score: int = 90,
    age_ms: int = 30_000,
    participant_id: int = 1,
    source_digit: int = 1,
    gold_total: int = 3000,
    depth: int | None = 3,
    from_ids: tuple[int, ...] = (
        1036,
        3067,
    ),
    into_ids: tuple[int, ...] = (),
    commentary_candidate: bool = True,
) -> ItemPowerSpikeEvaluation:
    return ItemPowerSpikeEvaluation(
        participant_id=participant_id,
        item_id=item_id,
        name_ko=f"아이템 {item_id}",
        name_en=f"Item {item_id}",
        action=ItemEvidenceAction.PURCHASED,
        timestamp_ms=100_000,
        age_ms_at_situation_start=age_ms,
        gold_total=gold_total,
        depth=depth,
        from_item_ids=from_ids,
        into_item_ids=into_ids,
        tags=("Damage",),
        score=score,
        tier=(ItemPowerSpikeTier.HIGH if score >= 75 else ItemPowerSpikeTier.MEDIUM),
        commentary_candidate=commentary_candidate,
        reasons=(),
        source_event_sha256=f"{source_digit:064x}",
    )


def _context(
    *,
    context_digit: int,
    record_digit: int,
    situation_digit: int,
    evaluations: tuple[
        ItemPowerSpikeEvaluation,
        ...,
    ],
) -> ItemPowerSpikeContext:
    return ItemPowerSpikeContext(
        context_id=f"{context_digit:064x}",
        match_id="KR_TEST",
        record_id=f"{record_digit:064x}",
        situation_id=f"{situation_digit:064x}",
        ddragon_version="16.17.1",
        ddragon_catalog_sha256="a" * 64,
        evaluations=evaluations,
        top_signals=(),
    )


def test_score_85_is_high() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    score=85,
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert result[0].selected_signals[0].tier == SelectedItemPowerSpikeTier.HIGH


def test_score_60_is_medium() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    score=60,
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert result[0].selected_signals[0].tier == SelectedItemPowerSpikeTier.MEDIUM


def test_score_below_60_is_filtered() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    score=59,
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert result[0].selected_signals == ()


def test_signal_older_than_180_seconds_is_filtered() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    age_ms=180_001,
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert result[0].selected_signals == ()


def test_low_value_intermediate_is_filtered() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    item_id=3047,
                    score=80,
                    gold_total=1200,
                    depth=2,
                    from_ids=(
                        1001,
                        1029,
                    ),
                    into_ids=(3174,),
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert result[0].selected_signals == ()


def test_high_value_intermediate_is_allowed() -> None:
    contexts = (
        _context(
            context_digit=1,
            record_digit=1,
            situation_digit=1,
            evaluations=(
                _evaluation(
                    score=85,
                    gold_total=2200,
                    depth=3,
                    from_ids=(
                        1001,
                        1036,
                    ),
                    into_ids=(9999,),
                ),
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=contexts,
    )

    assert (
        len(
            result[0].selected_signals,
        )
        == 1
    )


def test_same_purchase_source_selects_nearest_context() -> None:
    source_digit = 33

    first = _context(
        context_digit=1,
        record_digit=1,
        situation_digit=1,
        evaluations=(
            _evaluation(
                age_ms=120_000,
                source_digit=source_digit,
            ),
        ),
    )

    second = _context(
        context_digit=2,
        record_digit=2,
        situation_digit=2,
        evaluations=(
            _evaluation(
                age_ms=30_000,
                source_digit=source_digit,
            ),
        ),
    )

    result = select_item_power_spike_contexts(
        power_contexts=(
            first,
            second,
        ),
    )

    assert result[0].selected_signals == ()

    assert (
        len(
            result[1].selected_signals,
        )
        == 1
    )


def test_per_context_limit_is_enforced() -> None:
    evaluations = tuple(
        _evaluation(
            item_id=3000 + index,
            score=90 - index,
            source_digit=index + 1,
        )
        for index in range(4)
    )

    context = _context(
        context_digit=1,
        record_digit=1,
        situation_digit=1,
        evaluations=evaluations,
    )

    result = select_item_power_spike_contexts(
        power_contexts=(context,),
        max_signals_per_context=2,
    )

    assert (
        len(
            result[0].selected_signals,
        )
        == 2
    )


def test_selection_is_deterministic() -> None:
    context = _context(
        context_digit=1,
        record_digit=1,
        situation_digit=1,
        evaluations=(_evaluation(),),
    )

    first = select_item_power_spike_contexts(
        power_contexts=(context,),
    )

    second = select_item_power_spike_contexts(
        power_contexts=(context,),
    )

    assert first[0].selection_id == second[0].selection_id


def test_cross_match_batch_is_rejected() -> None:
    first = _context(
        context_digit=1,
        record_digit=1,
        situation_digit=1,
        evaluations=(_evaluation(),),
    )

    second = _context(
        context_digit=2,
        record_digit=2,
        situation_digit=2,
        evaluations=(
            _evaluation(
                source_digit=2,
            ),
        ),
    ).model_copy(
        update={
            "match_id": "KR_OTHER",
        }
    )

    with pytest.raises(
        ValueError,
        match="one match only",
    ):
        select_item_power_spike_contexts(
            power_contexts=(
                first,
                second,
            ),
        )
