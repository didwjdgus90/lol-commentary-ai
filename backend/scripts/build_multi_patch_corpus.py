import argparse
from pathlib import Path

from lol_commentary_backend.ingestion.multi_patch.pipeline import (
    build_multi_patch_corpus,
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
        dest="locales",
        choices=(
            "ko_KR",
            "en_US",
        ),
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=600,
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repository_root = Path(__file__).resolve().parents[2]

    locales = tuple(
        args.locales
        or (
            "ko_KR",
            "en_US",
        )
    )

    print("=== MULTI-PATCH CORPUS BUILD ===")

    print(f"Patch range: {args.start_patch} -> {args.end_patch}")

    print(f"Locales: {locales}")

    print(f"Max chunk chars: {args.max_chunk_chars}")

    print()

    manifest = build_multi_patch_corpus(
        repository_root=(repository_root),
        start_patch=(args.start_patch),
        end_patch=(args.end_patch),
        locales=locales,
        max_chunk_chars=(args.max_chunk_chars),
    )

    print(f"Patch count: {manifest.patch_count}")

    print(f"Shard count: {manifest.shard_count}")

    print(f"Total documents: {manifest.total_document_count}")

    print(f"Total chunks: {manifest.total_chunk_count}")

    print()

    for shard in manifest.shards:
        print(f"{shard.patch} {shard.locale}: chunks={shard.chunk_count} sha={shard.corpus_sha256}")

    expected_shards = manifest.patch_count * len(manifest.locales)

    if manifest.shard_count != expected_shards:
        raise RuntimeError(
            f"Shard count mismatch: expected={expected_shards} actual={manifest.shard_count}"
        )

    if manifest.total_chunk_count <= 0:
        raise RuntimeError("No chunks were produced")

    print()

    print("MULTI_PATCH_CORPUS_BUILD=PASS")


if __name__ == "__main__":
    main()
