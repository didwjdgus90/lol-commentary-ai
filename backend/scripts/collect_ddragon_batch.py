import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.sources.ddragon import (
    ChampionDetailScope,
    DataDragonLocale,
    collect_ddragon_data,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        action="append",
        dest="versions",
    )

    parser.add_argument(
        "--major",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--start-minor",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-minor",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--locale",
        action="append",
        choices=(
            "ko_KR",
            "en_US",
        ),
        dest="locales",
    )

    parser.add_argument(
        "--champion-detail-scope",
        choices=(
            "none",
            "latest",
            "all",
        ),
        default="latest",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    locales: tuple[
        DataDragonLocale,
        ...,
    ] = tuple(
        args.locales
        or (
            "ko_KR",
            "en_US",
        )
    )

    detail_scope: ChampionDetailScope = args.champion_detail_scope

    explicit_versions = tuple(args.versions) if args.versions else None

    print("=== DATA DRAGON BATCH COLLECTION ===")

    print(f"Explicit versions: {explicit_versions}")

    if explicit_versions is None:
        print(f"Version range: {args.major}.{args.start_minor} -> {args.major}.{args.end_minor}")

    print(f"Locales: {locales}")

    print(f"Champion detail scope: {detail_scope}")

    print(f"Overwrite: {args.overwrite}")

    print()

    summary = collect_ddragon_data(
        repository_root=(repository_root),
        locales=locales,
        explicit_versions=(explicit_versions),
        major=args.major,
        start_minor=(args.start_minor),
        end_minor=(args.end_minor),
        champion_detail_scope=(detail_scope),
        timeout_seconds=(args.timeout),
        overwrite=args.overwrite,
    )

    print("Selected versions:")

    for version in summary.selected_versions:
        print(f"  - {version}")

    print()

    print(f"Locales: {summary.locales}")

    print(f"Champion detail scope: {summary.champion_detail_scope}")

    print(f"Champion detail snapshots: {summary.champion_detail_count}")

    print(f"Downloaded: {summary.downloaded_count}")

    print(f"Reused: {summary.reused_count}")

    print(f"Data Dragon manifest: {summary.ddragon_manifest_path}")

    print(f"Game constants manifest: {summary.game_constants_manifest_path}")

    print()

    print("DATA_DRAGON_BATCH_COLLECTION=PASS")


if __name__ == "__main__":
    main()
