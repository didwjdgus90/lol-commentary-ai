import json
from collections import Counter
from pathlib import Path
from statistics import median

from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolvedPatchRecord,
)
from lol_commentary_backend.retrieval.documents.patch_note_builder import (
    build_patch_rag_documents,
)


def _percentile(
    values: list[int],
    percentile: float,
) -> int:
    if not values:
        return 0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)

    return ordered[index]


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    input_path = (
        repository_root / "data" / "processed" / "patch_notes" / "26.1" / "ko_kr" / "resolved.jsonl"
    )

    output_dir = repository_root / "data" / "processed" / "rag" / "patch_notes" / "26.1" / "ko_kr"

    output_path = output_dir / "documents.jsonl"

    records: list[ResolvedPatchRecord] = []

    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            records.append(ResolvedPatchRecord.model_validate_json(line))

    documents = build_patch_rag_documents(records)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for document in documents:
            file.write(
                json.dumps(
                    document.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    lengths = [len(document.retrieval_text) for document in documents]

    resolution_counts = Counter(document.resolution_method.value for document in documents)

    entity_counts = Counter(document.entity_type.value for document in documents)

    print(f"RAG documents saved: {output_path}")
    print(f"Documents: {len(documents)}")
    print(f"Unique document IDs: {len({document.document_id for document in documents})}")
    print()

    print("Entity types:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")

    print()
    print("Resolution methods:")
    for method, count in resolution_counts.items():
        print(f"  {method}: {count}")

    print()
    print("Retrieval text length (characters):")

    if lengths:
        print(f"  min: {min(lengths)}")
        print(f"  median: {int(median(lengths))}")
        print(f"  p95: {_percentile(lengths, 0.95)}")
        print(f"  max: {max(lengths)}")
    else:
        print("  no documents")


if __name__ == "__main__":
    main()
