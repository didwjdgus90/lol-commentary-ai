import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.sources.patch_notes import (
    PatchLocale,
    build_patch_sequence,
    collect_patch_notes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-patch",
        default="26.1",
    )

    parser.add_argument(
        "--end-patch",
        default="26.17",
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
        PatchLocale,
        ...,
    ] = tuple(
        args.locales
        or (
            "ko_KR",
            "en_US",
        )
    )

    patches = build_patch_sequence(
        start_patch=(args.start_patch),
        end_patch=(args.end_patch),
    )

    expected_count = len(patches) * len(locales)

    print("=== PATCH NOTE BATCH COLLECTION ===")

    print(f"Patch range: {patches[0]} -> {patches[-1]}")

    print(f"Patch count: {len(patches)}")

    print(f"Locales: {locales}")

    print(f"Expected snapshots: {expected_count}")

    print(f"Overwrite: {args.overwrite}")

    print()

    summary = collect_patch_notes(
        repository_root=(repository_root),
        start_patch=(args.start_patch),
        end_patch=(args.end_patch),
        locales=locales,
        timeout_seconds=(args.timeout),
        overwrite=(args.overwrite),
    )

    for item in summary.items:
        record = item.manifest

        print(f"[{record.patch} {record.locale}]")

        print(f"  ACTION={item.action}")

        print(f"  TITLE={record.title}")

        print(f"  URL={record.source_url}")

        print(f"  FILE={record.file_path}")

        print(f"  BYTES={record.byte_count}")

        print(f"  SHA256={record.content_sha256}")

        print()

    print("=== SUMMARY ===")

    print(f"Requested: {summary.requested_count}")

    print(f"Downloaded: {summary.downloaded_count}")

    print(f"Reused: {summary.reused_count}")

    print(f"Manifest: {summary.manifest_path}")

    if summary.requested_count != expected_count:
        raise RuntimeError("Requested snapshot count mismatch")

    if summary.downloaded_count + summary.reused_count != expected_count:
        raise RuntimeError("Not all requested snapshots were resolved")

    print("PATCH_NOTE_BATCH_COLLECTION=PASS")


if __name__ == "__main__":
    main()
