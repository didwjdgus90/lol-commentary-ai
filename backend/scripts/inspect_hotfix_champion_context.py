import sys
from collections import Counter
from pathlib import Path

from lol_commentary_backend.ingestion.patch_note_entity_resolution.catalog import (
    load_entity_catalog,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    PatchEntityType,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
    PatchSection,
    SectionKind,
)

DDRAGON_VERSION = "16.1.1"
DDRAGON_LOCALE = "ko_KR"


def _configure_utf8_output() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)

    if callable(reconfigure):
        reconfigure(
            encoding="utf-8",
            errors="replace",
        )


def _walk(
    section: PatchSection,
) -> list[PatchSection]:
    rows = [section]

    for child in section.children:
        rows.extend(_walk(child))

    return rows


def _candidate_types(
    title: str,
    catalog,
) -> tuple[str, ...]:
    candidates = catalog.exact_candidates(title)

    return tuple(sorted({candidate.entity_type.value for candidate in candidates}))


def _candidate_ids(
    title: str,
    catalog,
) -> tuple[str, ...]:
    return tuple(candidate.entity_id for candidate in catalog.exact_candidates(title))


def _safe_champion_anchor(
    section: PatchSection,
    catalog,
) -> bool:
    if section.level != 3:
        return False

    candidates = catalog.exact_candidates(
        section.title,
        entity_type=PatchEntityType.CHAMPION,
    )

    return len(candidates) == 1


def main() -> None:
    _configure_utf8_output()

    repository_root = Path(__file__).resolve().parents[2]

    parsed_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "parsed.json"
    )

    catalog_dir = (
        repository_root / "data" / "raw" / "data_dragon" / DDRAGON_VERSION / DDRAGON_LOCALE
    )

    if not parsed_path.is_file():
        raise FileNotFoundError(f"Parsed document not found: {parsed_path}")

    document = PatchNoteDocument.model_validate_json(parsed_path.read_text(encoding="utf-8"))
    catalog = load_entity_catalog(catalog_dir)

    hotfix_sections = [
        section for section in document.sections if section.kind == SectionKind.HOTFIX
    ]

    if not hotfix_sections:
        raise ValueError("Hotfix section not found")

    print("=== HOTFIX CHAMPION CONTEXT PROBE ===")
    print(f"Patch: {document.patch}")
    print(f"Hotfix top-level sections: {len(hotfix_sections)}")
    print()

    total_nodes = 0
    champion_anchors = 0
    item_titles = 0
    ambiguous_entity_titles = 0
    non_entity_titles = 0

    for hotfix_index, hotfix in enumerate(
        hotfix_sections,
        start=1,
    ):
        print("=" * 110)
        print(f"HOTFIX #{hotfix_index}: {hotfix.title!r}")
        print()

        nodes = hotfix.children
        current_champion: str | None = None

        for index, section in enumerate(nodes):
            total_nodes += 1

            candidate_types = _candidate_types(
                section.title,
                catalog,
            )
            candidate_ids = _candidate_ids(
                section.title,
                catalog,
            )

            is_anchor = _safe_champion_anchor(
                section,
                catalog,
            )

            if is_anchor:
                current_champion = section.title
                champion_anchors += 1

            if "item" in candidate_types:
                item_titles += 1

            if len(candidate_types) > 1:
                ambiguous_entity_titles += 1

            if not candidate_types:
                non_entity_titles += 1

            print(f"{index:02d}. L{section.level} title={section.title!r}")
            print(f"    kind={section.kind.value!r} entity_name={section.entity_name!r}")
            print(f"    candidate_types={candidate_types!r} candidate_ids={candidate_ids!r}")
            print(f"    champion_anchor={is_anchor} active_champion={current_champion!r}")
            print(
                f"    paragraphs={len(section.paragraphs)} "
                f"changes={len(section.changes)} "
                f"children={len(section.children)}"
            )

            if section.changes:
                for change in section.changes:
                    print(f"      change: {change.raw_text}")

            if section.children:
                for child in section.children:
                    child_types = _candidate_types(
                        child.title,
                        catalog,
                    )
                    print(
                        "      child: "
                        f"L{child.level} "
                        f"{child.title!r} "
                        f"candidate_types="
                        f"{child_types!r}"
                    )

            print()

    print("=" * 110)
    print("SUMMARY")
    print(f"hotfix child nodes: {total_nodes}")
    print(f"safe champion anchors: {champion_anchors}")
    print(f"titles matching item catalog: {item_titles}")
    print(f"titles matching multiple entity types: {ambiguous_entity_titles}")
    print(f"titles matching no entity: {non_entity_titles}")
    print()

    print("=== TRANSITION SUMMARY ===")

    transition_counter: Counter[tuple[str, str]] = Counter()

    for hotfix in hotfix_sections:
        previous_class: str | None = None

        for section in hotfix.children:
            types = _candidate_types(
                section.title,
                catalog,
            )

            if _safe_champion_anchor(
                section,
                catalog,
            ):
                current_class = "champion_anchor"
            elif "item" in types:
                current_class = "item_title"
            elif section.level == 4:
                current_class = "detail_h4"
            elif section.level == 3:
                current_class = "other_h3"
            else:
                current_class = f"level_{section.level}"

            if previous_class is not None:
                transition_counter[(previous_class, current_class)] += 1

            previous_class = current_class

    for (
        previous_class,
        current_class,
    ), count in transition_counter.most_common():
        print(f"{previous_class} -> {current_class}: {count}")

    print()
    print(
        "IMPORTANT: this probe does not assign champion "
        "context. It only exposes the ordered boundaries "
        "needed to design a safe rule."
    )


if __name__ == "__main__":
    main()
