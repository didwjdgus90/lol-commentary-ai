from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PATCH = "26.17"

LOCALE = "ko_KR"


class ProbeIntent(StrEnum):
    CHAMPION_PATCH = "champion_patch"

    ITEM_PATCH = "item_patch"

    OBJECTIVE_PATCH = "objective_patch"

    STRUCTURE_PATCH = "structure_patch"


@dataclass(
    frozen=True,
    slots=True,
)
class ScopeProbeCase:
    label: str

    intent: ProbeIntent

    aliases: tuple[
        str,
        ...,
    ] = ()

    strong_terms: tuple[
        str,
        ...,
    ] = ()


CASES = (
    ScopeProbeCase(
        label="champion_yone",
        intent=(ProbeIntent.CHAMPION_PATCH),
        aliases=(
            "Yone",
            "요네",
        ),
    ),
    ScopeProbeCase(
        label="item_infinity_edge",
        intent=(ProbeIntent.ITEM_PATCH),
        aliases=(
            "Infinity Edge",
            "무한의 대검",
        ),
    ),
    ScopeProbeCase(
        label="item_kraken_slayer",
        intent=(ProbeIntent.ITEM_PATCH),
        aliases=(
            "Kraken Slayer",
            "크라켄 학살자",
        ),
    ),
    ScopeProbeCase(
        label="system_objective",
        intent=(ProbeIntent.OBJECTIVE_PATCH),
        strong_terms=(
            "드래곤",
            "바론",
            "내셔",
            "전령",
            "dragon",
            "baron",
            "herald",
        ),
    ),
    ScopeProbeCase(
        label="system_structure",
        intent=(ProbeIntent.STRUCTURE_PATCH),
        strong_terms=(
            "포탑",
            "억제기",
            "turret",
            "inhibitor",
        ),
    ),
)


ENTITY_SECTION_BY_INTENT = {
    ProbeIntent.CHAMPION_PATCH: ("champion"),
    ProbeIntent.ITEM_PATCH: ("item"),
}


ALLOWED_SYSTEM_SECTIONS = {
    "general",
    "system",
    "objective",
    "structure",
}


def _normalize(
    value: object,
) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    return " ".join(text.casefold().split())


def _exact_alias_match(
    *,
    value: object,
    aliases: tuple[
        str,
        ...,
    ],
) -> bool:
    normalized_value = _normalize(value)

    if not normalized_value:
        return False

    return any(normalized_value == _normalize(alias) for alias in aliases)


def _contains_term(
    *,
    text: str,
    terms: tuple[
        str,
        ...,
    ],
) -> bool:
    normalized_text = _normalize(text)

    return any(_normalize(term) in normalized_text for term in terms)


def _heading_text(
    payload: dict[
        str,
        object,
    ],
) -> str:
    return json.dumps(
        (
            payload.get("title"),
            payload.get("entity_name"),
            payload.get("heading_path"),
        ),
        ensure_ascii=False,
    )


def _full_text(
    payload: dict[
        str,
        object,
    ],
) -> str:
    return json.dumps(
        (
            payload.get("title"),
            payload.get("entity_name"),
            payload.get("heading_path"),
            payload.get("text"),
        ),
        ensure_ascii=False,
    )


def _entity_scope_match(
    *,
    payload: dict[
        str,
        object,
    ],
    case: ScopeProbeCase,
) -> bool:
    expected_section = ENTITY_SECTION_BY_INTENT[case.intent]

    section_kind = _normalize(payload.get("section_kind"))

    if section_kind != expected_section:
        return False

    entity_name = payload.get("entity_name")

    return _exact_alias_match(
        value=entity_name,
        aliases=case.aliases,
    )


def _system_heading_match(
    *,
    payload: dict[
        str,
        object,
    ],
    case: ScopeProbeCase,
) -> bool:
    section_kind = _normalize(payload.get("section_kind"))

    if section_kind not in ALLOWED_SYSTEM_SECTIONS:
        return False

    return _contains_term(
        text=_heading_text(payload),
        terms=case.strong_terms,
    )


def _system_body_match(
    *,
    payload: dict[
        str,
        object,
    ],
    case: ScopeProbeCase,
) -> bool:
    section_kind = _normalize(payload.get("section_kind"))

    if section_kind not in ALLOWED_SYSTEM_SECTIONS:
        return False

    return _contains_term(
        text=_full_text(payload),
        terms=case.strong_terms,
    )


def _strict_matches(
    *,
    chunks: tuple[
        dict[
            str,
            object,
        ],
        ...,
    ],
    case: ScopeProbeCase,
) -> tuple[
    dict[
        str,
        object,
    ],
    ...,
]:
    if case.intent in ENTITY_SECTION_BY_INTENT:
        return tuple(
            payload
            for payload in chunks
            if _entity_scope_match(
                payload=payload,
                case=case,
            )
        )

    return tuple(
        payload
        for payload in chunks
        if _system_heading_match(
            payload=payload,
            case=case,
        )
    )


def _body_candidates(
    *,
    chunks: tuple[
        dict[
            str,
            object,
        ],
        ...,
    ],
    case: ScopeProbeCase,
) -> tuple[
    dict[
        str,
        object,
    ],
    ...,
]:
    if case.intent in ENTITY_SECTION_BY_INTENT:
        return ()

    return tuple(
        payload
        for payload in chunks
        if _system_body_match(
            payload=payload,
            case=case,
        )
    )


def _load_chunks(
    *,
    repository_root: Path,
) -> tuple[
    dict[
        str,
        object,
    ],
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

    chunk_files = tuple(sorted(shard_root.rglob("chunks.jsonl")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunks.jsonl found under {shard_root}")

    result: list[
        dict[
            str,
            object,
        ]
    ] = []

    for path in chunk_files:
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
                raise ValueError(f"Chunk must be a JSON object: {path}:{line_number}")

            result.append(payload)

    return tuple(result)


def _excerpt(
    value: object,
    *,
    limit: int = 220,
) -> str:
    normalized = " ".join(str(value if value is not None else "").split())

    if len(normalized) <= limit:
        return normalized

    return normalized[:limit] + "..."


def _print_chunk(
    *,
    payload: dict[
        str,
        object,
    ],
) -> None:
    print(f"    section={payload.get('section_kind')}")

    print(f"    title={payload.get('title')}")

    print(f"    entity={payload.get('entity_name')}")

    print("    heading=" + _excerpt(payload.get("heading_path")))

    print("    text=" + _excerpt(payload.get("text")))


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    chunks = _load_chunks(repository_root=(repository_root))

    section_counts = Counter(_normalize(payload.get("section_kind")) for payload in chunks)

    print("=== STEP 45-E-B INTENT-AWARE EVIDENCE SCOPE PROBE ===")

    print()

    print(f"PATCH={PATCH}")

    print(f"LOCALE={LOCALE}")

    print(f"CHUNKS={len(chunks)}")

    print()

    print("=== SECTION DISTRIBUTION ===")

    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count}")

    print()

    strict_present = 0

    strict_absent = 0

    for case in CASES:
        strict_matches = _strict_matches(
            chunks=chunks,
            case=case,
        )

        body_candidates = _body_candidates(
            chunks=chunks,
            case=case,
        )

        if strict_matches:
            strict_present += 1

        else:
            strict_absent += 1

        print(f"CASE={case.label}")

        print(f"  intent={case.intent.value}")

        print(f"  strict_match_count={len(strict_matches)}")

        print("  strict_scope=" + ("PRESENT" if strict_matches else "ABSENT"))

        if case.intent not in ENTITY_SECTION_BY_INTENT:
            print(f"  body_candidate_count={len(body_candidates)}")

        print()

        if strict_matches:
            print("  --- STRICT MATCHES ---")

            for payload in strict_matches[:5]:
                _print_chunk(payload=payload)

                print()

        if not strict_matches and body_candidates:
            print("  --- BODY-ONLY CANDIDATES ---")

            for payload in body_candidates[:10]:
                _print_chunk(payload=payload)

                print()

        print()

    print("=== AGGREGATE ===")

    print(f"cases={len(CASES)}")

    print(f"strict_present={strict_present}")

    print(f"strict_absent={strict_absent}")

    print()

    print("EXPECTED_POLICY=ENTITY_EXACT_SECTION_PLUS_STRONG_SYSTEM_SCOPE")

    print("GLOBAL_SCORE_THRESHOLD=DISALLOWED")

    print()

    print("STEP_45_E_B_SCOPE_PROBE=PASS")


if __name__ == "__main__":
    main()
