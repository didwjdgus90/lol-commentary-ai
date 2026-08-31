from hashlib import sha256
from pathlib import Path

from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalSeed,
    TargetGranularity,
)

RETRIEVAL_EVAL_DATASET_VERSION = "0.1.0"


def corpus_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_retrieval_eval_seeds(
    path: Path,
) -> list[RetrievalEvalSeed]:
    seeds: list[RetrievalEvalSeed] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                seed = RetrievalEvalSeed.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid retrieval evaluation seed at line {line_number}"
                ) from exc

            seeds.append(seed)

    _validate_seed_uniqueness(seeds)

    return seeds


def load_patch_rag_chunks(
    path: Path,
) -> list[PatchRagChunk]:
    chunks: list[PatchRagChunk] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                chunk = PatchRagChunk.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"Invalid RAG chunk JSONL at line {line_number}") from exc

            chunks.append(chunk)

    return chunks


def _validate_seed_uniqueness(
    seeds: list[RetrievalEvalSeed],
) -> None:
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()

    for seed in seeds:
        if seed.query_id in seen_ids:
            raise ValueError(f"Duplicate retrieval query_id: {seed.query_id}")

        normalized_query = " ".join(seed.query.split()).casefold()

        if normalized_query in seen_queries:
            raise ValueError(f"Duplicate retrieval query text: {seed.query}")

        seen_ids.add(seed.query_id)
        seen_queries.add(normalized_query)


def _selector_matches(
    chunk: PatchRagChunk,
    seed: RetrievalEvalSeed,
) -> bool:
    selector = seed.selector

    if chunk.patch != selector.patch:
        return False

    if chunk.title != selector.title:
        return False

    if selector.entity_name is not None and chunk.entity_name != selector.entity_name:
        return False

    if selector.section_kind is not None and chunk.section_kind != selector.section_kind:
        return False

    return True


def _resolve_seed(
    seed: RetrievalEvalSeed,
    chunks: list[PatchRagChunk],
    *,
    corpus_hash: str,
) -> RetrievalEvalCase:
    document_candidates = [chunk for chunk in chunks if _selector_matches(chunk, seed)]

    if not document_candidates:
        raise ValueError(f"{seed.query_id}: target selector matched no chunks")

    document_ids = {chunk.document_id for chunk in document_candidates}

    if len(document_ids) != 1:
        raise ValueError(f"{seed.query_id}: target selector matched multiple documents")

    required_text = seed.selector.required_text_contains

    if required_text is None:
        relevant_chunks = document_candidates
        granularity = TargetGranularity.DOCUMENT
    else:
        relevant_chunks = [chunk for chunk in document_candidates if required_text in chunk.text]

        if not relevant_chunks:
            raise ValueError(
                f"{seed.query_id}: required answer text was not found in target document"
            )

        granularity = TargetGranularity.ANSWER_CHUNK

    return RetrievalEvalCase(
        dataset_version=(RETRIEVAL_EVAL_DATASET_VERSION),
        query_id=seed.query_id,
        query=seed.query,
        language=seed.language,
        query_type=seed.query_type,
        difficulty=seed.difficulty,
        target_granularity=granularity,
        relevant_document_ids=sorted({chunk.document_id for chunk in relevant_chunks}),
        relevant_chunk_ids=[
            chunk.chunk_id
            for chunk in sorted(
                relevant_chunks,
                key=lambda item: item.chunk_index,
            )
        ],
        relevant_source_record_ids=sorted({chunk.source_record_id for chunk in relevant_chunks}),
        selector=seed.selector,
        rationale=seed.rationale,
        corpus_sha256=corpus_hash,
    )


def build_retrieval_eval_cases(
    seeds: list[RetrievalEvalSeed],
    chunks: list[PatchRagChunk],
    *,
    corpus_hash: str,
) -> list[RetrievalEvalCase]:
    _validate_seed_uniqueness(seeds)

    cases = [
        _resolve_seed(
            seed,
            chunks,
            corpus_hash=corpus_hash,
        )
        for seed in seeds
    ]

    return cases
