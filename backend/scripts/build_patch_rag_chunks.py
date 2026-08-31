import json
from collections import Counter
from pathlib import Path
from statistics import median

from lol_commentary_backend.retrieval.chunks.patch_note_chunker import (
    DEFAULT_MAX_CHUNK_CHARS,
    build_patch_rag_chunks_for_documents,
)
from lol_commentary_backend.retrieval.documents.models import (
    PatchRagDocument,
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
        repository_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "documents.jsonl"
    )

    output_path = input_path.with_name("chunks.jsonl")

    documents: list[PatchRagDocument] = []

    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            documents.append(PatchRagDocument.model_validate_json(line))

    chunks = build_patch_rag_chunks_for_documents(
        documents,
        max_chars=DEFAULT_MAX_CHUNK_CHARS,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    chunk_lengths = [chunk.char_count for chunk in chunks]

    chunks_per_document = Counter(chunk.document_id for chunk in chunks)

    strategy_counts = Counter(chunk.strategy.value for chunk in chunks)

    split_documents = sum(count > 1 for count in chunks_per_document.values())

    print(f"RAG chunks saved: {output_path}")
    print(f"Max chunk chars: {DEFAULT_MAX_CHUNK_CHARS}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Unique chunk IDs: {len({chunk.chunk_id for chunk in chunks})}")
    print(f"Documents represented: {len(chunks_per_document)}")
    print(f"Unsplit documents: {len(documents) - split_documents}")
    print(f"Split documents: {split_documents}")
    print(f"Max chunks per document: {max(chunks_per_document.values(), default=0)}")

    print()
    print("Chunk strategies:")
    for strategy, count in strategy_counts.items():
        print(f"  {strategy}: {count}")

    print()
    print("Chunk text length (characters):")

    if chunk_lengths:
        print(f"  min: {min(chunk_lengths)}")
        print(f"  median: {int(median(chunk_lengths))}")
        print(f"  p95: {_percentile(chunk_lengths, 0.95)}")
        print(f"  max: {max(chunk_lengths)}")
    else:
        print("  no chunks")


if __name__ == "__main__":
    main()
