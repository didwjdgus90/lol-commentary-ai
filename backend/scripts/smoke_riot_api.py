from __future__ import annotations

import argparse

from lol_commentary_backend.ingestion.riot_api.client import (
    RiotApiClient,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--game-name",
        required=True,
    )

    parser.add_argument(
        "--tag-line",
        required=True,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=3,
    )

    return parser.parse_args()


def _masked(
    value: str,
) -> str:
    if len(value) <= 12:
        return "***"

    return value[:6] + "..." + value[-4:]


def main() -> None:
    args = _parse_args()

    print("=== RIOT API LIVE SMOKE ===")

    print("Regional routing: ASIA")

    print(f"Riot ID: {args.game_name}#{args.tag_line}")

    print()

    with RiotApiClient() as client:
        account = client.get_account_by_riot_id(
            game_name=args.game_name,
            tag_line=args.tag_line,
        )

        print("ACCOUNT_V1=PASS")

        print(f"PUUID={_masked(account.puuid)}")

        matches = client.get_match_ids_by_puuid(
            puuid=account.puuid,
            count=args.count,
        )

        print(f"MATCH_IDS={len(matches.match_ids)}")

        if not matches.match_ids:
            print("No recent matches were returned.")

            print("RIOT_API_LIVE_SMOKE=PASS")

            return

        latest_match_id = matches.match_ids[0]

        print(f"LATEST_MATCH={latest_match_id}")

        match = client.get_match(latest_match_id)

        metadata = match.get("metadata")

        if not isinstance(
            metadata,
            dict,
        ):
            raise RuntimeError("Match metadata missing")

        if metadata.get("matchId") != latest_match_id:
            raise RuntimeError("Match ID mismatch")

        print("MATCH_DETAIL=PASS")

        timeline = client.get_timeline(latest_match_id)

        timeline_metadata = timeline.get("metadata")

        if not isinstance(
            timeline_metadata,
            dict,
        ):
            raise RuntimeError("Timeline metadata missing")

        if timeline_metadata.get("matchId") != latest_match_id:
            raise RuntimeError("Timeline match ID mismatch")

        info = timeline.get("info")

        if not isinstance(
            info,
            dict,
        ):
            raise RuntimeError("Timeline info missing")

        frames = info.get("frames")

        if not isinstance(
            frames,
            list,
        ):
            raise RuntimeError("Timeline frames missing")

        print("MATCH_TIMELINE=PASS")

        print(f"TIMELINE_FRAMES={len(frames)}")

    print()

    print("RIOT_API_LIVE_SMOKE=PASS")


if __name__ == "__main__":
    main()
