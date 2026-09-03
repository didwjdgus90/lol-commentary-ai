from __future__ import annotations

import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.riot_api.client import (
    RiotApiClient,
)
from lol_commentary_backend.ingestion.riot_api.collector import (
    collect_riot_match_snapshots,
    parse_riot_id,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Collect Riot Match-V5 match and timeline raw snapshots with SHA256 lineage.")
    )

    parser.add_argument(
        "--riot-id",
        required=True,
        help=("Riot ID in GameName#TagLine format."),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    game_name, tag_line = parse_riot_id(args.riot_id)

    repository_root = Path(__file__).resolve().parents[2]

    print("=== RIOT MATCH RAW COLLECTION ===")

    print(f"Riot ID: {game_name}#{tag_line}")

    print(f"Start: {args.start}")

    print(f"Requested matches: {args.count}")

    print(f"Force refresh: {args.force_refresh}")

    print()

    with RiotApiClient() as client:
        result = collect_riot_match_snapshots(
            client=client,
            repository_root=(repository_root),
            game_name=game_name,
            tag_line=tag_line,
            start=args.start,
            count=args.count,
            force_refresh=(args.force_refresh),
        )

    print("=== COLLECTION RESULT ===")

    print(f"Returned matches: {result.returned_match_count}")

    print(f"Downloaded resources: {result.downloaded_resources}")

    print(f"Reused resources: {result.reused_resources}")

    print(f"Manifest records: {result.manifest_record_count}")

    for match_id in result.match_ids:
        print(f"MATCH={match_id}")

    print()

    print("RIOT_RAW_COLLECTION=PASS")


if __name__ == "__main__":
    main()
