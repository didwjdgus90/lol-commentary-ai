from __future__ import annotations

import pytest

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    ParticipantIdentity,
)
from lol_commentary_backend.intelligence.game_state_models import (
    FrameAlignmentMode,
    GameStateSnapshot,
    SituationStateContext,
)
from lol_commentary_backend.intelligence.models import (
    CommentaryCandidate,
    CommentaryCandidateType,
    SalienceBand,
)
from lol_commentary_backend.intelligence.priority_models import (
    MacroContextReference,
    PriorityReason,
    SituationPriority,
    SituationPriorityTier,
)
from lol_commentary_backend.intelligence.record_builder import (
    build_commentary_intelligence_records,
)
from lol_commentary_backend.intelligence.situation_models import (
    SituationKind,
    TemporalSituation,
)


def _participant(
    *,
    participant_id: int,
    team_id: int,
    champion_id: int,
    champion_name: str,
    position: str,
) -> ParticipantIdentity:
    return ParticipantIdentity.model_construct(
        participant_id=(participant_id),
        team_id=team_id,
        champion_id=champion_id,
        champion_name=(champion_name),
        individual_position=(position),
        team_position=(position),
        win=False,
    )


def _candidate(
    *,
    index: int,
    timestamp_ms: int,
    actor_participant_id: int | None = None,
    target_participant_id: int | None = None,
    salience_score: int = 45,
) -> CommentaryCandidate:
    return CommentaryCandidate.model_construct(
        candidate_id=(f"{index + 1:064x}"),
        match_id="KR_1",
        source_sequence=index,
        source_event_sha256=(f"{index + 101:064x}"),
        timestamp_ms=(timestamp_ms),
        candidate_type=(CommentaryCandidateType.CHAMPION_KILL),
        salience_score=(salience_score),
        salience_band=(SalienceBand.MEDIUM),
        actor_participant_id=(actor_participant_id),
        target_participant_id=(target_participant_id),
    )


def _situation(
    *,
    index: int = 0,
    start_timestamp_ms: int = 10_000,
    candidates: tuple[
        CommentaryCandidate,
        ...,
    ]
    | None = None,
    salience: int = 45,
) -> TemporalSituation:
    if candidates is None:
        candidates = (
            _candidate(
                index=index,
                timestamp_ms=(start_timestamp_ms),
                actor_participant_id=1,
                salience_score=(salience),
            ),
        )

    return TemporalSituation.model_construct(
        situation_id=(f"{index + 501:064x}"),
        match_id="KR_1",
        start_timestamp_ms=(start_timestamp_ms),
        end_timestamp_ms=(start_timestamp_ms),
        duration_ms=0,
        situation_kind=(SituationKind.COMBAT),
        max_salience_score=(salience),
        primary_candidates=(candidates),
        semantic_markers=(),
    )


def _snapshot(
    *,
    frame_index: int,
    timestamp_ms: int,
) -> GameStateSnapshot:
    return GameStateSnapshot.model_construct(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        teams=(),
    )


def _context(
    *,
    situation: TemporalSituation,
    before_frame: int = 0,
    after_frame: int = 1,
    before_gold: int = 100,
    after_gold: int = 200,
) -> SituationStateContext:
    return SituationStateContext.model_construct(
        situation_id=(situation.situation_id),
        match_id="KR_1",
        situation_start_timestamp_ms=(situation.start_timestamp_ms),
        situation_end_timestamp_ms=(situation.end_timestamp_ms),
        before_state=_snapshot(
            frame_index=before_frame,
            timestamp_ms=0,
        ),
        after_state=_snapshot(
            frame_index=after_frame,
            timestamp_ms=60_000,
        ),
        alignment_mode=(FrameAlignmentMode.DISTINCT_FRAMES),
        start_to_before_frame_ms=(situation.start_timestamp_ms),
        end_to_after_frame_ms=(60_000 - situation.end_timestamp_ms),
        frame_interval_ms=60_000,
        before_gold_diff_100_minus_200=(before_gold),
        after_gold_diff_100_minus_200=(after_gold),
        interval_gold_diff_change_100_minus_200=(after_gold - before_gold),
        before_leading_team_id=100,
        after_leading_team_id=100,
        leading_team_changed=False,
    )


def _priority(
    *,
    situation: TemporalSituation,
    context: SituationStateContext,
) -> SituationPriority:
    macro = MacroContextReference.model_construct(
        macro_interval_id=("a" * 64),
        match_id="KR_1",
        before_frame_index=(context.before_state.frame_index),
        after_frame_index=(context.after_state.frame_index),
        before_timestamp_ms=(context.before_state.timestamp_ms),
        after_timestamp_ms=(context.after_state.timestamp_ms),
        frame_interval_ms=(context.frame_interval_ms),
        before_gold_diff_100_minus_200=(context.before_gold_diff_100_minus_200),
        after_gold_diff_100_minus_200=(context.after_gold_diff_100_minus_200),
        interval_gold_diff_change_100_minus_200=(context.interval_gold_diff_change_100_minus_200),
        leading_team_changed=(context.leading_team_changed),
        large_macro_gold_change=False,
        shared_situation_count=1,
        anchor_situation_id=(situation.situation_id),
    )

    return SituationPriority.model_construct(
        situation_id=(situation.situation_id),
        match_id="KR_1",
        priority_tier=(SituationPriorityTier.LOW),
        event_salience_score=(situation.max_salience_score),
        situation_member_count=(
            len(situation.primary_candidates) + len(situation.semantic_markers)
        ),
        semantic_marker_count=(len(situation.semantic_markers)),
        macro_context=macro,
        is_macro_anchor=True,
        reasons=(PriorityReason.ROUTINE_EVENT,),
    )


def _participants() -> tuple[
    ParticipantIdentity,
    ...,
]:
    return (
        _participant(
            participant_id=1,
            team_id=100,
            champion_id=266,
            champion_name="Aatrox",
            position="TOP",
        ),
        _participant(
            participant_id=2,
            team_id=200,
            champion_id=122,
            champion_name="Darius",
            position="TOP",
        ),
    )


def test_actor_entity_is_resolved() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    records = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )

    assert len(records) == 1

    assert records[0].referenced_entities[0].champion_name == "Aatrox"


def test_actor_and_target_are_resolved() -> None:
    candidate = _candidate(
        index=0,
        timestamp_ms=10_000,
        actor_participant_id=1,
        target_participant_id=2,
    )

    situation = _situation(candidates=(candidate,))

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    record = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    assert [entity.participant_id for entity in record.referenced_entities] == [
        1,
        2,
    ]


def test_record_contains_no_player_pii() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    record = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    payload = record.model_dump_json().casefold()

    forbidden = (
        "puuid",
        "summoner",
        "game_name",
        "tag_line",
        "riot_id",
        "account_id",
    )

    assert all(value not in payload for value in forbidden)

    assert record.provenance.contains_player_pii is False


def test_unknown_participant_raises() -> None:
    candidate = _candidate(
        index=0,
        timestamp_ms=10_000,
        actor_participant_id=99,
    )

    situation = _situation(candidates=(candidate,))

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    with pytest.raises(
        ValueError,
        match=("unknown participant_id"),
    ):
        build_commentary_intelligence_records(
            match_id="KR_1",
            participants=_participants(),
            situations=(situation,),
            contexts=(context,),
            priorities=(priority,),
        )


def test_missing_game_state_raises() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    with pytest.raises(
        ValueError,
        match=("game-state coverage mismatch"),
    ):
        build_commentary_intelligence_records(
            match_id="KR_1",
            participants=_participants(),
            situations=(situation,),
            contexts=(),
            priorities=(priority,),
        )


def test_missing_priority_raises() -> None:
    situation = _situation()

    context = _context(situation=situation)

    with pytest.raises(
        ValueError,
        match=("priority coverage mismatch"),
    ):
        build_commentary_intelligence_records(
            match_id="KR_1",
            participants=_participants(),
            situations=(situation,),
            contexts=(context,),
            priorities=(),
        )


def test_record_id_is_deterministic() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    first = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    second = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    assert first.record_id == second.record_id


def test_macro_context_is_non_causal() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    record = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    assert record.provenance.macro_interpretation == "shared_macro_context_only"

    assert record.provenance.causal_policy == (
        "do_not_attribute_macro_interval_delta_to_single_situation"
    )


def test_candidate_lineage_is_preserved() -> None:
    situation = _situation()

    context = _context(situation=situation)

    priority = _priority(
        situation=situation,
        context=context,
    )

    record = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(situation,),
        contexts=(context,),
        priorities=(priority,),
    )[0]

    candidate = situation.primary_candidates[0]

    assert candidate.candidate_id in record.provenance.candidate_ids

    assert candidate.source_event_sha256 in record.provenance.source_event_sha256s


def test_records_are_chronological() -> None:
    later = _situation(
        index=1,
        start_timestamp_ms=20_000,
    )

    earlier = _situation(
        index=0,
        start_timestamp_ms=10_000,
    )

    earlier_context = _context(situation=earlier)

    later_context = _context(situation=later)

    earlier_priority = _priority(
        situation=earlier,
        context=earlier_context,
    )

    later_priority = _priority(
        situation=later,
        context=later_context,
    )

    records = build_commentary_intelligence_records(
        match_id="KR_1",
        participants=_participants(),
        situations=(
            later,
            earlier,
        ),
        contexts=(
            later_context,
            earlier_context,
        ),
        priorities=(
            later_priority,
            earlier_priority,
        ),
    )

    assert [record.start_timestamp_ms for record in records] == [
        10_000,
        20_000,
    ]
