from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from lol_commentary_backend.retrieval.runtime.models import (
    RetrievalMode,
)
from lol_commentary_backend.retrieval.runtime.multi_factory import (
    build_multi_patch_retrieval_service,
)

PATCH = "26.17"

LOCALE = "ko_KR"

TOP_K = 10


@dataclass(
    frozen=True,
    slots=True,
)
class ProbeCase:
    label: str

    query: str

    expected_terms: tuple[str, ...]

    entity_like: bool


CASES = (
    ProbeCase(
        label="champion_yone",
        query="26.17 Yone 변경 사항",
        expected_terms=(
            "요네",
            "yone",
        ),
        entity_like=True,
    ),
    ProbeCase(
        label="item_infinity_edge",
        query=("26.17 무한의 대검 변경 사항"),
        expected_terms=(
            "무한의 대검",
            "infinity edge",
        ),
        entity_like=True,
    ),
    ProbeCase(
        label="item_kraken_slayer",
        query=("26.17 크라켄 학살자 변경 사항"),
        expected_terms=(
            "크라켄 학살자",
            "kraken slayer",
        ),
        entity_like=True,
    ),
    ProbeCase(
        label="system_objective",
        query=("26.17 오브젝트 드래곤 바론 전령 변경 사항"),
        expected_terms=(
            "드래곤",
            "바론",
            "전령",
            "dragon",
            "baron",
            "herald",
        ),
        entity_like=False,
    ),
    ProbeCase(
        label="system_structure",
        query=("26.17 포탑 억제기 구조물 변경 사항"),
        expected_terms=(
            "포탑",
            "억제기",
            "구조물",
            "turret",
            "inhibitor",
        ),
        entity_like=False,
    ),
)


def _normalize(
    value: object,
) -> str:
    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return " ".join(text.casefold().split())


def _matches_terms(
    *,
    text: str,
    terms: tuple[
        str,
        ...,
    ],
) -> bool:
    normalized_text = _normalize(text)

    return any(_normalize(term) in normalized_text for term in terms)


def _chunk_search_text(
    payload: dict[
        str,
        object,
    ],
) -> str:
    relevant_values = (
        payload.get("title"),
        payload.get("entity_name"),
        payload.get("text"),
        payload.get("heading_path"),
        payload.get("section_kind"),
    )

    return json.dumps(
        relevant_values,
        ensure_ascii=False,
        sort_keys=True,
    )


def _find_chunk_files(
    *,
    repository_root: Path,
) -> tuple[
    Path,
    ...,
]:
    shard_root = (
        repository_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "multi_patch_v1"
        / PATCH
        / LOCALE
    )

    if not shard_root.is_dir():
        raise FileNotFoundError(f"Processed RAG shard does not exist: {shard_root}")

    files = tuple(sorted(shard_root.rglob("chunks.jsonl")))

    if not files:
        raise FileNotFoundError(f"No chunks.jsonl found under {shard_root}")

    return files


def _load_chunks(
    *,
    files: tuple[
        Path,
        ...,
    ],
) -> tuple[
    dict[
        str,
        object,
    ],
    ...,
]:
    chunks: list[
        dict[
            str,
            object,
        ]
    ] = []

    for path in files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue

            payload = json.loads(line)

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(f"Chunk must be JSON object: {path}:{line_number}")

            chunks.append(payload)

    if not chunks:
        raise RuntimeError("Processed RAG shard contains no chunks")

    return tuple(chunks)


def _corpus_matches(
    *,
    chunks: tuple[
        dict[
            str,
            object,
        ],
        ...,
    ],
    case: ProbeCase,
) -> tuple[
    dict[
        str,
        object,
    ],
    ...,
]:
    result = []

    for payload in chunks:
        search_text = _chunk_search_text(payload)

        if _matches_terms(
            text=search_text,
            terms=(case.expected_terms),
        ):
            result.append(payload)

    return tuple(result)


def _hit_matches(
    *,
    hit: object,
    case: ProbeCase,
) -> bool:
    title = getattr(
        hit,
        "title",
        None,
    )

    entity_name = getattr(
        hit,
        "entity_name",
        None,
    )

    text = getattr(
        hit,
        "text",
        None,
    )

    heading_path = getattr(
        hit,
        "heading_path",
        None,
    )

    section_kind = getattr(
        hit,
        "section_kind",
        None,
    )

    combined = json.dumps(
        (
            title,
            entity_name,
            text,
            heading_path,
            section_kind,
        ),
        ensure_ascii=False,
    )

    return _matches_terms(
        text=combined,
        terms=(case.expected_terms),
    )


def _text_excerpt(
    value: str,
    *,
    limit: int = 180,
) -> str:
    normalized = " ".join(value.split())

    if len(normalized) <= limit:
        return normalized

    return normalized[:limit] + "..."


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    database_url = os.environ.get("LOL_DATABASE_URL")

    if not database_url:
        raise RuntimeError("LOL_DATABASE_URL is not set")

    chunk_files = _find_chunk_files(repository_root=(repository_root))

    chunks = _load_chunks(files=chunk_files)

    print("=== STEP 45-E-A RETRIEVAL EVIDENCE GROUND-TRUTH PROBE ===")

    print()

    print(f"PATCH={PATCH}")

    print(f"LOCALE={LOCALE}")

    print(f"CHUNK_FILES={len(chunk_files)}")

    print(f"CHUNKS={len(chunks)}")

    print()

    service = build_multi_patch_retrieval_service(
        repository_root=(repository_root),
        locales=(LOCALE,),
        patches=(PATCH,),
        enable_primary=True,
        requested_device="auto",
        database_url=(database_url),
        source_top_n=30,
        rrf_k=60,
    )

    if not service.primary_available:
        raise RuntimeError("Primary retrieval must be available")

    cases_with_corpus_evidence = 0

    cases_without_corpus_evidence = 0

    corpus_present_retrieved_top1 = 0

    corpus_present_retrieved_top3 = 0

    corpus_present_retrieved_top5 = 0

    corpus_present_retrieved_top10 = 0

    for case in CASES:
        matches = _corpus_matches(
            chunks=chunks,
            case=case,
        )

        has_corpus_evidence = bool(matches)

        if has_corpus_evidence:
            cases_with_corpus_evidence += 1

        else:
            cases_without_corpus_evidence += 1

        response = service.retrieve(
            case.query,
            top_k=TOP_K,
            mode=(RetrievalMode.PRIMARY),
        )

        matching_ranks = [
            index
            for index, hit in enumerate(
                response.hits,
                start=1,
            )
            if _hit_matches(
                hit=hit,
                case=case,
            )
        ]

        first_matching_rank = matching_ranks[0] if matching_ranks else None

        if has_corpus_evidence and first_matching_rank is not None:
            if first_matching_rank <= 1:
                corpus_present_retrieved_top1 += 1

            if first_matching_rank <= 3:
                corpus_present_retrieved_top3 += 1

            if first_matching_rank <= 5:
                corpus_present_retrieved_top5 += 1

            if first_matching_rank <= 10:
                corpus_present_retrieved_top10 += 1

        print(f"CASE={case.label}")

        print(f"  query={case.query}")

        print(f"  entity_like={case.entity_like}")

        print(f"  corpus_match_count={len(matches)}")

        print("  corpus_evidence=" + ("PRESENT" if has_corpus_evidence else "ABSENT"))

        print(
            "  first_matching_rank="
            + (str(first_matching_rank) if (first_matching_rank is not None) else "NONE")
        )

        print(f"  retrieval_strategy={response.strategy_used.value}")

        print(f"  hits={len(response.hits)}")

        print()

        print("  --- TOP HITS ---")

        for rank, hit in enumerate(
            response.hits,
            start=1,
        ):
            subject_match = _hit_matches(
                hit=hit,
                case=case,
            )

            print(f"  RANK={rank} match={subject_match}")

            print(f"    section={hit.section_kind}")

            print(f"    title={hit.title}")

            print(f"    entity={hit.entity_name}")

            print(f"    score={hit.score:.6f}")

            print(f"    dense_rank={hit.dense_rank}")

            print(f"    sparse_rank={hit.sparse_rank}")

            print("    text=" + _text_excerpt(hit.text))

        print()

        if matches:
            print("  --- CORPUS MATCH EXAMPLES ---")

            for payload in matches[:3]:
                print(f"    title={payload.get('title')}")

                print(f"    entity={payload.get('entity_name')}")

                print(f"    section={payload.get('section_kind')}")

                text = payload.get(
                    "text",
                    "",
                )

                print("    text=" + _text_excerpt(str(text)))

        print()

    print("=== AGGREGATE ===")

    print(f"cases={len(CASES)}")

    print(f"cases_with_corpus_evidence={cases_with_corpus_evidence}")

    print(f"cases_without_corpus_evidence={cases_without_corpus_evidence}")

    print(f"corpus_present_retrieved_top1={corpus_present_retrieved_top1}")

    print(f"corpus_present_retrieved_top3={corpus_present_retrieved_top3}")

    print(f"corpus_present_retrieved_top5={corpus_present_retrieved_top5}")

    print(f"corpus_present_retrieved_top10={corpus_present_retrieved_top10}")

    print()

    print("STEP_45_E_A_EVIDENCE_PROBE=PASS")


if __name__ == "__main__":
    main()
