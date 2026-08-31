import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)

from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.embeddings.models import (
    EMBEDDING_CANDIDATES,
    EmbeddingCandidate,
)
from lol_commentary_backend.retrieval.embeddings.probe_models import (
    TokenizerProbeResult,
    TokenLengthStats,
)
from lol_commentary_backend.retrieval.embeddings.query import (
    format_embedding_query,
)

SAMPLE_QUERIES = (
    "트린다미어 E 스킬이 26.1에서 얼마나 상향됐어?",
    "26.1 정수 약탈자 가격 변경",
    "헬리아의 메아리 체력 몇으로 바뀌었어?",
    "Aphelios passive hotfix 26.1",
    "구원 가격이 어떻게 바뀌었어?",
    "군단의 방패 삭제됐어?",
    "ARAM 아수라장 요새파괴자 비활성화",
    "퀸 기본 능력치 변경",
)


def _percentile(
    values: Sequence[int],
    percentile: float,
) -> int:
    if not values:
        return 0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)

    return ordered[index]


def _stats(
    values: list[int],
) -> TokenLengthStats:
    if not values:
        return TokenLengthStats(
            count=0,
            minimum=0,
            median=0,
            p95=0,
            maximum=0,
        )

    return TokenLengthStats(
        count=len(values),
        minimum=min(values),
        median=int(median(values)),
        p95=_percentile(values, 0.95),
        maximum=max(values),
    )


def _token_count(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> int:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False,
    )

    return len(token_ids)


def _tokenizer_model_max_length(
    tokenizer: PreTrainedTokenizerBase,
    candidate: EmbeddingCandidate,
) -> int:
    value = tokenizer.model_max_length

    # Hugging Face tokenizers sometimes use an extremely
    # large sentinel value to mean "not explicitly bounded".
    if value > 10_000_000:
        return candidate.declared_max_tokens

    return int(value)


def _probe_candidate(
    candidate: EmbeddingCandidate,
    chunks: list[PatchRagChunk],
) -> TokenizerProbeResult:
    tokenizer = AutoTokenizer.from_pretrained(
        candidate.model_id,
    )

    chunk_counts = [_token_count(tokenizer, chunk.text) for chunk in chunks]

    formatted_queries = [
        format_embedding_query(
            raw_query,
            candidate,
        )
        for raw_query in SAMPLE_QUERIES
    ]

    query_counts = [_token_count(tokenizer, query) for query in formatted_queries]

    over_limit = sum(count > candidate.experiment_max_tokens for count in chunk_counts)

    return TokenizerProbeResult(
        candidate_key=candidate.key,
        model_id=candidate.model_id,
        corpus_chunks=len(chunks),
        declared_max_tokens=(candidate.declared_max_tokens),
        experiment_max_tokens=(candidate.experiment_max_tokens),
        tokenizer_model_max_length=(
            _tokenizer_model_max_length(
                tokenizer,
                candidate,
            )
        ),
        chunk_stats=_stats(chunk_counts),
        query_stats=_stats(query_counts),
        chunks_over_256=sum(count > 256 for count in chunk_counts),
        chunks_over_512=sum(count > 512 for count in chunk_counts),
        chunks_over_1024=sum(count > 1024 for count in chunk_counts),
        chunks_over_8192=sum(count > 8192 for count in chunk_counts),
        chunks_over_experiment_limit=over_limit,
        compatible_with_current_corpus=(over_limit == 0),
    )


def _print_result(
    result: TokenizerProbeResult,
) -> None:
    print("=" * 100)
    print(f"Candidate: {result.candidate_key}")
    print(f"Model: {result.model_id}")
    print(f"Declared max tokens: {result.declared_max_tokens}")
    print(f"Experiment max tokens: {result.experiment_max_tokens}")
    print(f"Tokenizer model_max_length: {result.tokenizer_model_max_length}")
    print()

    print("Chunk tokens:")
    print(
        f"  min={result.chunk_stats.minimum} "
        f"median={result.chunk_stats.median} "
        f"p95={result.chunk_stats.p95} "
        f"max={result.chunk_stats.maximum}"
    )
    print(
        f"  >256={result.chunks_over_256} "
        f">512={result.chunks_over_512} "
        f">1024={result.chunks_over_1024} "
        f">8192={result.chunks_over_8192}"
    )
    print(f"  over configured limit={result.chunks_over_experiment_limit}")
    print()

    print("Sample query tokens:")
    print(
        f"  min={result.query_stats.minimum} "
        f"median={result.query_stats.median} "
        f"p95={result.query_stats.p95} "
        f"max={result.query_stats.maximum}"
    )
    print(f"Compatible with current corpus: {result.compatible_with_current_corpus}")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    chunks_path = (
        repository_root
        / "data"
        / "processed"
        / "rag"
        / "patch_notes"
        / "26.1"
        / "ko_kr"
        / "chunks.jsonl"
    )

    output_dir = repository_root / "data" / "processed" / "evaluation" / "embedding_tokenizers"
    output_path = output_dir / "step32_tokenizer_probe.json"

    chunks: list[PatchRagChunk] = []

    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunks.append(PatchRagChunk.model_validate_json(line))

    print("=== MULTI-MODEL EMBEDDING TOKENIZER EXPERIMENT ===")
    print(f"Corpus chunks: {len(chunks)}")
    print("Only tokenizer files are loaded in this step; model weights are not loaded.")
    print()

    results = [
        _probe_candidate(
            candidate,
            chunks,
        )
        for candidate in EMBEDDING_CANDIDATES
        if candidate.experiment_enabled
    ]

    for result in results:
        _print_result(result)
        print()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Experiment result saved: {output_path}")


if __name__ == "__main__":
    main()
