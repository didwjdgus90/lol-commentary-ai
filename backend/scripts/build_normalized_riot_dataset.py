from __future__ import annotations

import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.normalization.artifacts import (
    build_normalized_riot_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic "
            "normalized Riot match "
            "artifacts from verified "
            "raw Match-V5 snapshots."
        )
    )

    parser.add_argument(
        "--match-id",
        action="append",
        dest="match_ids",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    match_ids = tuple(args.match_ids) if args.match_ids else None

    print("=== RIOT NORMALIZED DATASET BUILD ===")

    result = build_normalized_riot_dataset(
        repository_root=(repository_root),
        match_ids=match_ids,
    )

    manifest = result.manifest

    print(f"Matches: {manifest.match_count}")

    print(f"Participants: {manifest.total_participant_count}")

    print(f"Participant frames: {manifest.total_participant_frame_count}")

    print(f"Events: {manifest.total_event_count}")

    print(f"Built matches: {result.built_match_count}")

    print(f"Reused matches: {result.reused_match_count}")

    aggregate_action = "built" if (result.aggregate_manifest_changed) else "reused"

    print(f"Aggregate manifest: {aggregate_action}")

    print()

    for reference in manifest.matches:
        print(
            f"MATCH="
            f"{reference.match_id} "
            f"FRAMES="
            f"{reference.participant_frame_count} "
            f"EVENTS="
            f"{reference.event_count}"
        )

    print()

    print("RIOT_NORMALIZED_DATASET=PASS")


if __name__ == "__main__":
    main()
