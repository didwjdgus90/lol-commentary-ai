from __future__ import annotations

from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
    NormalizedMatch,
    ParticipantFrameSnapshot,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)
from lol_commentary_backend.intelligence.game_state_context import (
    build_situation_state_contexts,
)
from lol_commentary_backend.intelligence.priority_builder import (
    build_situation_priorities,
)
from lol_commentary_backend.intelligence.record_builder import (
    build_commentary_intelligence_records,
)
from lol_commentary_backend.intelligence.situation_clusterer import (
    build_temporal_situations,
)

PII_FRAGMENTS = (
    "puuid",
    "summoner",
    "game_name",
    "gamename",
    "tag_line",
    "tagline",
    "riot_id",
    "account_id",
    "accountid",
    "profile_icon",
)


def _load_match(
    path: Path,
) -> NormalizedMatch:
    return NormalizedMatch.model_validate_json(path.read_text(encoding="utf-8"))


def _load_events(
    path: Path,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    result: list[NormalizedGameEvent] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        result.append(NormalizedGameEvent.model_validate_json(line))

    return tuple(result)


def _load_snapshots(
    path: Path,
) -> tuple[
    ParticipantFrameSnapshot,
    ...,
]:
    result: list[ParticipantFrameSnapshot] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        result.append(ParticipantFrameSnapshot.model_validate_json(line))

    return tuple(result)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    root = repository_root / "data" / "processed" / "riot_api" / "normalized_v1"

    match_directories = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if (
                    path.is_dir()
                    and (path / "match.json").is_file()
                    and (path / "events.jsonl").is_file()
                    and (path / "participant_frames.jsonl").is_file()
                )
            ),
            key=lambda path: path.name,
        )
    )

    if len(match_directories) != 3:
        raise RuntimeError("Expected 3 normalized matches")

    total_candidates = 0
    total_situations = 0
    total_records = 0

    total_candidate_lineage = 0
    total_entity_links = 0
    records_without_entities = 0

    record_ids: set[str] = set()

    macro_interval_ids: set[str] = set()

    referenced_entities: set[
        tuple[
            str,
            int,
        ]
    ] = set()

    tiers: Counter[str] = Counter()

    pii_hits: set[str] = set()

    print("=== COMMENTARY INTELLIGENCE RECORD SMOKE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match = _load_match(directory / "match.json")

        events = _load_events(directory / "events.jsonl")

        snapshots = _load_snapshots(directory / "participant_frames.jsonl")

        candidates = build_commentary_candidates(
            match_id=(directory.name),
            events=events,
        )

        situations = build_temporal_situations(
            match_id=(directory.name),
            candidates=(candidates),
        )

        contexts = build_situation_state_contexts(
            match_id=(directory.name),
            situations=(situations),
            snapshots=snapshots,
            participant_teams={
                participant.participant_id: (participant.team_id)
                for participant in match.participants
            },
        )

        priorities = build_situation_priorities(
            match_id=(directory.name),
            situations=(situations),
            contexts=(contexts),
        )

        records = build_commentary_intelligence_records(
            match_id=(directory.name),
            participants=tuple(match.participants),
            situations=(situations),
            contexts=(contexts),
            priorities=(priorities),
        )

        if len(records) != len(situations):
            raise RuntimeError("Situation/record coverage mismatch")

        match_entity_links = 0

        match_without_entities = 0

        for record in records:
            if record.record_id in record_ids:
                raise RuntimeError("Duplicate record_id")

            record_ids.add(record.record_id)

            macro_interval_ids.add(record.provenance.macro_interval_id)

            tiers[record.priority.priority_tier.value] += 1

            candidate_count = len(record.provenance.candidate_ids)

            total_candidate_lineage += candidate_count

            entity_count = len(record.referenced_entities)

            total_entity_links += entity_count

            match_entity_links += entity_count

            if entity_count == 0:
                records_without_entities += 1
                match_without_entities += 1

            for entity in record.referenced_entities:
                referenced_entities.add(
                    (
                        record.match_id,
                        entity.participant_id,
                    )
                )

            serialized = record.model_dump_json().casefold()

            for fragment in PII_FRAGMENTS:
                if fragment in serialized:
                    pii_hits.add(fragment)

            if record.provenance.contains_player_pii:
                raise RuntimeError("Record unexpectedly marks player PII")

            if record.provenance.macro_interpretation != ("shared_macro_context_only"):
                raise RuntimeError("Invalid macro interpretation")

        total_candidates += len(candidates)

        total_situations += len(situations)

        total_records += len(records)

        print(f"MATCH={directory.name}")

        print(f"  candidates={len(candidates)}")

        print(f"  situations={len(situations)}")

        print(f"  records={len(records)}")

        print(f"  entity_links={match_entity_links}")

        print(f"  records_without_entities={match_without_entities}")

        print()

    if total_candidates != 266:
        raise RuntimeError(f"Expected 266 candidates, got {total_candidates}")

    if total_situations != 125:
        raise RuntimeError(f"Expected 125 situations, got {total_situations}")

    if total_records != 125:
        raise RuntimeError(f"Expected 125 intelligence records, got {total_records}")

    if len(record_ids) != 125:
        raise RuntimeError("Expected 125 unique record IDs")

    if len(macro_interval_ids) != 77:
        raise RuntimeError("Expected 77 unique macro intervals")

    if total_candidate_lineage != 266:
        raise RuntimeError("Candidate lineage coverage must equal 266")

    expected_tiers = {
        "critical": 10,
        "high": 24,
        "low": 44,
        "medium": 47,
    }

    if dict(tiers) != expected_tiers:
        raise RuntimeError(f"Priority distribution changed unexpectedly: {dict(tiers)}")

    if pii_hits:
        raise RuntimeError(f"PII-like fragments found: {sorted(pii_hits)}")

    print("=== AGGREGATE ===")

    print(f"Candidates: {total_candidates}")

    print(f"Situations: {total_situations}")

    print(f"Commentary intelligence records: {total_records}")

    print(f"Unique record IDs: {len(record_ids)}")

    print(f"Unique macro intervals: {len(macro_interval_ids)}")

    print(f"Candidate lineage links: {total_candidate_lineage}")

    print(f"Entity links: {total_entity_links}")

    print(f"Unique referenced participants: {len(referenced_entities)}")

    print(f"Records without participant entities: {records_without_entities}")

    print(f"PII-like hits: {len(pii_hits)}")

    print()

    print("Priority tiers:")

    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")

    print()

    print("COMMENTARY_INTELLIGENCE_RECORD_SMOKE=PASS")


if __name__ == "__main__":
    main()
