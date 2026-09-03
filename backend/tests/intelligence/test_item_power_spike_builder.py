from __future__ import annotations

from lol_commentary_backend.intelligence.item_evidence_models import (
    ConfirmedItemEvidence,
    ItemEvidenceAction,
    ParticipantItemEvidence,
    SituationItemEvidenceContext,
)
from lol_commentary_backend.intelligence.item_metadata_models import (
    DDragonItemCatalogInfo,
    DDragonItemMetadata,
)
from lol_commentary_backend.intelligence.item_metadata_resolver import (
    DDragonItemMetadataResolver,
)
from lol_commentary_backend.intelligence.item_power_spike_builder import (
    build_item_power_spike_context,
)
from lol_commentary_backend.intelligence.item_power_spike_models import (
    ItemPowerSpikeReason,
    ItemPowerSpikeTier,
)


def _metadata(
    *,
    item_id: int,
    gold_total: int,
    tags: tuple[
        str,
        ...,
    ] = (),
    from_item_ids: tuple[
        int,
        ...,
    ] = (),
    into_item_ids: tuple[
        int,
        ...,
    ] = (),
    depth: int | None = None,
    purchasable: bool = True,
) -> DDragonItemMetadata:
    return DDragonItemMetadata(
        ddragon_version="16.17.1",
        item_id=item_id,
        name_ko=f"아이템 {item_id}",
        name_en=f"Item {item_id}",
        gold_base=gold_total,
        gold_total=gold_total,
        gold_sell=0,
        purchasable=purchasable,
        tags=tags,
        from_item_ids=(from_item_ids),
        into_item_ids=(into_item_ids),
        map_ids=(11,),
        depth=depth,
    )


def _resolver(
    *items: DDragonItemMetadata,
) -> DDragonItemMetadataResolver:
    info = DDragonItemCatalogInfo(
        ddragon_version="16.17.1",
        raw_item_count=len(items),
        item_count=len(items),
        skipped_item_count=0,
        skipped_item_ids=(),
        ko_source_path=("data/raw/ddragon/16.17.1/ko_KR/item/item.json"),
        en_source_path=("data/raw/ddragon/16.17.1/en_US/item/item.json"),
        ko_source_sha256=("a" * 64),
        en_source_sha256=("b" * 64),
        catalog_sha256=("c" * 64),
    )

    return DDragonItemMetadataResolver(
        catalog_info=info,
        items=tuple(items),
    )


def _evidence(
    *,
    item_id: int,
    age_ms: int,
    action: ItemEvidenceAction = (ItemEvidenceAction.PURCHASED),
    sequence: int = 1,
) -> ConfirmedItemEvidence:
    situation_start = 500_000

    return ConfirmedItemEvidence(
        action=action,
        participant_id=1,
        item_id=item_id,
        sequence=sequence,
        frame_index=1,
        event_index=sequence,
        timestamp_ms=(situation_start - age_ms),
        age_ms_at_situation_start=(age_ms),
        source_event_sha256=(f"{sequence:064x}"),
    )


def _context(
    *evidence: ConfirmedItemEvidence,
) -> SituationItemEvidenceContext:
    return SituationItemEvidenceContext(
        context_id="d" * 64,
        match_id="KR_1",
        record_id="e" * 64,
        situation_id="f" * 64,
        situation_start_timestamp_ms=(500_000),
        max_events_per_participant=5,
        participants=(
            ParticipantItemEvidence(
                participant_id=1,
                evidence=tuple(evidence),
            ),
        ),
    )


def test_major_completed_purchase_is_high() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3071,
            gold_total=3000,
            from_item_ids=(
                1036,
                3067,
            ),
            into_item_ids=(),
            depth=3,
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3071,
                age_ms=30_000,
            )
        ),
        resolver=resolver,
    )

    evaluation = result.evaluations[0]

    assert evaluation.tier == ItemPowerSpikeTier.HIGH

    assert evaluation.commentary_candidate is True


def test_control_ward_is_not_candidate() -> None:
    resolver = _resolver(
        _metadata(
            item_id=2055,
            gold_total=75,
            tags=(
                "Consumable",
                "Vision",
            ),
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=2055,
                age_ms=10_000,
            )
        ),
        resolver=resolver,
    )

    evaluation = result.evaluations[0]

    assert evaluation.commentary_candidate is False

    assert ItemPowerSpikeReason.UTILITY_ITEM in evaluation.reasons


def test_trinket_is_not_candidate() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3340,
            gold_total=0,
            tags=(
                "Trinket",
                "Vision",
            ),
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3340,
                age_ms=10_000,
            )
        ),
        resolver=resolver,
    )

    assert result.evaluations[0].commentary_candidate is False


def test_recent_purchase_gets_bonus() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3001,
            gold_total=1300,
            from_item_ids=(1001,),
            into_item_ids=(),
            depth=2,
        )
    )

    recent = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3001,
                age_ms=30_000,
            )
        ),
        resolver=resolver,
    )

    old = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3001,
                age_ms=400_000,
            )
        ),
        resolver=resolver,
    )

    assert recent.evaluations[0].score > old.evaluations[0].score


def test_non_purchase_is_not_evaluated() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3071,
            gold_total=3000,
            from_item_ids=(1036,),
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3071,
                age_ms=30_000,
                action=(ItemEvidenceAction.SOLD),
            )
        ),
        resolver=resolver,
    )

    assert result.evaluations == ()

    assert result.top_signals == ()


def test_missing_metadata_is_rejected() -> None:
    resolver = _resolver(
        _metadata(
            item_id=1001,
            gold_total=300,
        )
    )

    try:
        build_item_power_spike_context(
            item_context=_context(
                _evidence(
                    item_id=9999,
                    age_ms=30_000,
                )
            ),
            resolver=resolver,
        )

    except ValueError as exc:
        assert "no usable Data Dragon" in str(exc)

    else:
        raise AssertionError("Expected missing metadata to raise ValueError")


def test_top_signals_are_deduplicated() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3071,
            gold_total=3000,
            from_item_ids=(
                1036,
                3067,
            ),
            into_item_ids=(),
            depth=3,
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3071,
                age_ms=60_000,
                sequence=1,
            ),
            _evidence(
                item_id=3071,
                age_ms=20_000,
                sequence=2,
            ),
        ),
        resolver=resolver,
    )

    assert len(result.evaluations) == 2

    assert len(result.top_signals) == 1


def test_top_signal_limit_is_enforced() -> None:
    items = tuple(
        _metadata(
            item_id=(3000 + index),
            gold_total=3000,
            from_item_ids=(
                1001,
                1002,
            ),
            into_item_ids=(),
            depth=3,
        )
        for index in range(4)
    )

    resolver = _resolver(*items)

    evidences = tuple(
        _evidence(
            item_id=(3000 + index),
            age_ms=(10_000 + index * 1000),
            sequence=(index + 1),
        )
        for index in range(4)
    )

    result = build_item_power_spike_context(
        item_context=_context(*evidences),
        resolver=resolver,
        top_signal_limit=2,
    )

    assert len(result.top_signals) == 2


def test_exact_inventory_claim_stays_forbidden() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3071,
            gold_total=3000,
            from_item_ids=(1036,),
        )
    )

    result = build_item_power_spike_context(
        item_context=_context(
            _evidence(
                item_id=3071,
                age_ms=30_000,
            )
        ),
        resolver=resolver,
    )

    assert result.exact_inventory_claim_allowed is False

    assert result.evidence_authority == ("confirmed_timeline_event_plus_ddragon")


def test_context_id_is_deterministic() -> None:
    resolver = _resolver(
        _metadata(
            item_id=3071,
            gold_total=3000,
            from_item_ids=(1036,),
            into_item_ids=(),
            depth=3,
        )
    )

    item_context = _context(
        _evidence(
            item_id=3071,
            age_ms=30_000,
        )
    )

    first = build_item_power_spike_context(
        item_context=item_context,
        resolver=resolver,
    )

    second = build_item_power_spike_context(
        item_context=item_context,
        resolver=resolver,
    )

    assert first.context_id == second.context_id
