import json
from collections import Counter
from pathlib import Path

from lol_commentary_backend.retrieval.evaluation.builder import (
    build_retrieval_eval_cases,
    corpus_sha256,
    load_patch_rag_chunks,
    load_retrieval_eval_seeds,
)


def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]

    seed_path = backend_root / "evaluation" / "retrieval" / "patch_26_1_v1_seed.jsonl"

    chunks_path = (
        project_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "chunks.jsonl"
    )

    output_dir = project_root / "data" / "processed" / "evaluation" / "retrieval"
    output_path = output_dir / "patch_26_1_v1.jsonl"

    seeds = load_retrieval_eval_seeds(seed_path)
    chunks = load_patch_rag_chunks(chunks_path)

    corpus_hash = corpus_sha256(chunks_path)

    cases = build_retrieval_eval_cases(
        seeds,
        chunks,
        corpus_hash=corpus_hash,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for case in cases:
            file.write(
                json.dumps(
                    case.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    languages = Counter(case.language.value for case in cases)
    query_types = Counter(case.query_type.value for case in cases)
    difficulties = Counter(case.difficulty.value for case in cases)
    granularities = Counter(case.target_granularity.value for case in cases)

    print("=== RETRIEVAL EVALUATION DATASET v1 ===")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Cases: {len(cases)}")
    print(f"Unique query IDs: {len({case.query_id for case in cases})}")
    target_document_ids = {
        document_id for case in cases for document_id in case.relevant_document_ids
    }

    print(f"Unique target documents: {len(target_document_ids)}")
    print(f"Corpus SHA-256: {corpus_hash}")
    print()

    print("Languages:")
    for key, value in languages.items():
        print(f"  {key}: {value}")

    print()
    print("Query types:")
    for key, value in query_types.items():
        print(f"  {key}: {value}")

    print()
    print("Difficulty:")
    for key, value in difficulties.items():
        print(f"  {key}: {value}")

    print()
    print("Target granularity:")
    for key, value in granularities.items():
        print(f"  {key}: {value}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
