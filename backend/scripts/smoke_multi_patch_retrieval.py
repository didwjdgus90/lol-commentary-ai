from __future__ import annotations

import argparse
from pathlib import Path

from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
    RetrievalStrategy,
)
from lol_commentary_backend.retrieval.runtime.multi_factory import (
    build_multi_patch_retrieval_service,
    load_multi_patch_scope,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        default=("정수 약탈자 변경"),
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
        "--patch",
        action="append",
        dest="patches",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--mode",
        choices=(
            "auto",
            "primary",
            "fallback",
        ),
        default="auto",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    locales = tuple(args.locales or ("ko_KR",))

    patches = tuple(args.patches) if args.patches else None

    mode = RetrievalMode(args.mode)

    scope = load_multi_patch_scope(
        repository_root=(Path(__file__).resolve().parents[2]),
        locales=locales,
        patches=patches,
    )

    repository_root = Path(__file__).resolve().parents[2]

    enable_primary = mode != RetrievalMode.FALLBACK

    print("=== MULTI-PATCH HYBRID RETRIEVAL ===")

    print(f"Query: {args.query}")

    print(f"Mode: {mode.value}")

    print(f"Locales: {scope.locales}")

    print(f"Patch count: {len(scope.patches)}")

    print(f"Corpus count: {len(scope.corpus_sha256s)}")

    print(f"Unique chunks: {len(scope.chunks)}")

    print()

    service = build_multi_patch_retrieval_service(
        repository_root=(repository_root),
        locales=locales,
        patches=patches,
        enable_primary=(enable_primary),
        requested_device=(args.device),
    )

    try:
        response = service.retrieve(
            args.query,
            top_k=args.top_k,
            mode=mode,
        )

    finally:
        service.close()

    print(f"Strategy: {response.strategy_used.value}")

    print(f"Expanded query: {response.expanded_query}")

    print(f"Fallback used: {response.fallback_used}")

    print(f"Elapsed ms: {response.elapsed_ms:.2f}")

    print()

    if not response.hits:
        raise RuntimeError("Retrieval returned no hits")

    for hit in response.hits:
        print(f"#{hit.rank} score={hit.score:.6f}")

        print(f"  PATCH={hit.patch}")

        print(f"  LOCALE={hit.locale}")

        print(f"  TITLE={hit.title}")

        print(f"  ENTITY={hit.entity_name}")

        print(f"  DENSE_RANK={hit.dense_rank}")

        print(f"  SPARSE_RANK={hit.sparse_rank}")

        print()

    if mode == RetrievalMode.PRIMARY and response.strategy_used != RetrievalStrategy.BGE_ALIAS_RRF:
        raise RuntimeError("PRIMARY mode did not use hybrid RRF")

    if mode == RetrievalMode.FALLBACK and response.strategy_used != RetrievalStrategy.ALIAS_BM25:
        raise RuntimeError("FALLBACK mode did not use Alias BM25")

    print("MULTI_PATCH_HYBRID_RETRIEVAL=PASS")


if __name__ == "__main__":
    main()
