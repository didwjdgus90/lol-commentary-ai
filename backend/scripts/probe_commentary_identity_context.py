from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    NormalizedGameEvent,
)
from lol_commentary_backend.intelligence.candidate_builder import (
    build_commentary_candidates,
)

PII_KEY_FRAGMENTS = (
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


SAFE_IDENTITY_KEY_FRAGMENTS = (
    "participant",
    "team",
    "champion",
    "position",
    "lane",
    "role",
)


PARTICIPANT_ID_KEYS = (
    "participant_id",
    "participantId",
)


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _load_events(
    path: Path,
) -> tuple[
    NormalizedGameEvent,
    ...,
]:
    events: list[NormalizedGameEvent] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            event = NormalizedGameEvent.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid normalized event at {path}:{line_number}") from exc

        events.append(event)

    return tuple(events)


def _participant_objects(
    match_payload: dict[str, Any],
) -> tuple[
    dict[str, Any],
    ...,
]:
    raw_participants = match_payload.get("participants")

    if not isinstance(
        raw_participants,
        list,
    ):
        raise ValueError("match.json does not contain a participants list")

    participants: list[dict[str, Any]] = []

    for index, item in enumerate(raw_participants):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Invalid participant object at index {index}")

        participants.append(item)

    return tuple(participants)


def _find_participant_id(
    participant: dict[str, Any],
) -> int:
    for key in PARTICIPANT_ID_KEYS:
        value = participant.get(key)

        if isinstance(
            value,
            int,
        ):
            return value

    raise ValueError("Participant object does not contain participant_id")


def _contains_pii_key(
    key: str,
) -> bool:
    lowered = key.casefold()

    return any(fragment in lowered for fragment in PII_KEY_FRAGMENTS)


def _is_safe_identity_key(
    key: str,
) -> bool:
    if _contains_pii_key(key):
        return False

    lowered = key.casefold()

    return any(fragment in lowered for fragment in SAFE_IDENTITY_KEY_FRAGMENTS)


def _value_type_name(
    value: Any,
) -> str:
    if value is None:
        return "None"

    return type(value).__name__


def _safe_sample(
    participant: dict[str, Any],
) -> dict[str, Any]:
    result: dict[
        str,
        Any,
    ] = {}

    for key, value in sorted(participant.items()):
        if not _is_safe_identity_key(key):
            continue

        if (
            isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
            or value is None
        ):
            result[key] = value

    return result


def _candidate_participant_refs(
    candidate_payload: dict[
        str,
        Any,
    ],
) -> set[int]:
    references: set[int] = set()

    for key, value in candidate_payload.items():
        lowered = key.casefold()

        if "participant" not in lowered:
            continue

        if isinstance(
            value,
            int,
        ):
            if value > 0:
                references.add(value)

            continue

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            for item in value:
                if (
                    isinstance(
                        item,
                        int,
                    )
                    and item > 0
                ):
                    references.add(item)

    return references


def _schema_key(
    participant: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(sorted(participant))


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
                )
            ),
            key=lambda path: path.name,
        )
    )

    if not match_directories:
        raise RuntimeError("No normalized matches found")

    participant_schema_counts: Counter[tuple[str, ...]] = Counter()

    safe_key_coverage: Counter[str] = Counter()

    safe_key_types: dict[
        str,
        Counter[str],
    ] = {}

    total_participants = 0
    total_candidates = 0
    total_candidate_references = 0

    unresolved_references: list[
        tuple[
            str,
            int,
        ]
    ] = []

    pii_like_keys: set[str] = set()

    champion_keys: set[str] = set()

    position_keys: set[str] = set()

    team_keys: set[str] = set()

    samples: list[
        tuple[
            str,
            dict[str, Any],
        ]
    ] = []

    print("=== COMMENTARY IDENTITY CONTEXT PROBE ===")

    print(f"Matches: {len(match_directories)}")

    print()

    for directory in match_directories:
        match_payload = _load_json_object(directory / "match.json")

        participants = _participant_objects(match_payload)

        if len(participants) != 10:
            raise RuntimeError(
                f"Expected 10 participants for {directory.name}, got {len(participants)}"
            )

        participants_by_id: dict[
            int,
            dict[str, Any],
        ] = {}

        for participant in participants:
            participant_id = _find_participant_id(participant)

            if participant_id in participants_by_id:
                raise RuntimeError(
                    f"Duplicate participant ID in {directory.name}: {participant_id}"
                )

            participants_by_id[participant_id] = participant

            participant_schema_counts[_schema_key(participant)] += 1

            total_participants += 1

            for key, value in participant.items():
                lowered = key.casefold()

                if _contains_pii_key(key):
                    pii_like_keys.add(key)

                if "champion" in lowered:
                    champion_keys.add(key)

                if "position" in lowered or "lane" in lowered or "role" in lowered:
                    position_keys.add(key)

                if "team" in lowered:
                    team_keys.add(key)

                if not _is_safe_identity_key(key):
                    continue

                safe_key_coverage[key] += 1

                if key not in safe_key_types:
                    safe_key_types[key] = Counter()

                safe_key_types[key][_value_type_name(value)] += 1

        events = _load_events(directory / "events.jsonl")

        candidates = build_commentary_candidates(
            match_id=(directory.name),
            events=events,
        )

        match_reference_count = 0
        match_unresolved: set[int] = set()

        for candidate in candidates:
            candidate_payload = candidate.model_dump(mode="python")

            references = _candidate_participant_refs(candidate_payload)

            match_reference_count += len(references)

            for participant_id in references:
                if participant_id not in participants_by_id:
                    match_unresolved.add(participant_id)

        total_candidates += len(candidates)

        total_candidate_references += match_reference_count

        for participant_id in sorted(match_unresolved):
            unresolved_references.append(
                (
                    directory.name,
                    participant_id,
                )
            )

        if len(samples) < 3:
            first_participant_id = min(participants_by_id)

            samples.append(
                (
                    directory.name,
                    _safe_sample(participants_by_id[first_participant_id]),
                )
            )

        print(f"MATCH={directory.name}")

        print(f"  participants={len(participants)}")

        print(f"  candidates={len(candidates)}")

        print(f"  candidate_participant_refs={match_reference_count}")

        print(f"  unresolved_participant_refs={len(match_unresolved)}")

        print()

    if total_participants != 30:
        raise RuntimeError(f"Expected 30 normalized participants, got {total_participants}")

    if total_candidates != 266:
        raise RuntimeError(f"Expected 266 commentary candidates, got {total_candidates}")

    if unresolved_references:
        raise RuntimeError("Unresolved participant references found")

    print("=== PARTICIPANT SCHEMA ===")

    print(f"Schema variants: {len(participant_schema_counts)}")

    for index, (
        schema,
        count,
    ) in enumerate(
        participant_schema_counts.most_common(),
        start=1,
    ):
        print(f"Schema #{index}: participants={count}")

        for key in schema:
            print(f"  {key}")

    print()

    print("=== SAFE IDENTITY FIELDS ===")

    for key in sorted(safe_key_coverage):
        type_summary = ", ".join(
            (f"{type_name}:{count}")
            for (
                type_name,
                count,
            ) in sorted(safe_key_types[key].items())
        )

        print(
            f"{key}: coverage={safe_key_coverage[key]}/{total_participants} types=[{type_summary}]"
        )

    print()

    print("Champion-related keys:")

    if champion_keys:
        for key in sorted(champion_keys):
            print(f"  {key}")
    else:
        print("  NONE")

    print()

    print("Position/lane/role keys:")

    if position_keys:
        for key in sorted(position_keys):
            print(f"  {key}")
    else:
        print("  NONE")

    print()

    print("Team-related keys:")

    if team_keys:
        for key in sorted(team_keys):
            print(f"  {key}")
    else:
        print("  NONE")

    print()

    print("PII-like normalized keys:")

    if pii_like_keys:
        for key in sorted(pii_like_keys):
            print(f"  {key}")
    else:
        print("  NONE")

    print()

    print("=== SAFE SAMPLES ===")

    for (
        match_id,
        sample,
    ) in samples:
        print(f"MATCH={match_id}")

        for key, value in sample.items():
            print(f"  {key}={value}")

    print()

    print("=== REFERENCE COVERAGE ===")

    print(f"Participants: {total_participants}")

    print(f"Candidates: {total_candidates}")

    print(f"Candidate participant references: {total_candidate_references}")

    print(f"Unresolved references: {len(unresolved_references)}")

    print()

    print("COMMENTARY_IDENTITY_CONTEXT_PROBE=PASS")


if __name__ == "__main__":
    main()
