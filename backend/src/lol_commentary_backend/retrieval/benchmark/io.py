import json
from hashlib import sha256
from pathlib import Path

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalBenchmarkRun,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_chunks(
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
                chunks.append(PatchRagChunk.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid chunk JSONL at line {line_number}") from exc

    return chunks


def load_eval_cases(
    path: Path,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                cases.append(RetrievalEvalCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid evaluation JSONL at line {line_number}") from exc

    return cases


def validate_corpus_lineage(
    cases: list[RetrievalEvalCase],
    chunks_path: Path,
) -> str:
    current_hash = file_sha256(chunks_path)

    mismatched = [case.query_id for case in cases if case.corpus_sha256 != current_hash]

    if mismatched:
        raise ValueError(
            "Evaluation dataset was built from a "
            "different chunk corpus. Rebuild Step 33. "
            f"First mismatch: {mismatched[0]}"
        )

    return current_hash


def save_benchmark_run(
    run: RetrievalBenchmarkRun,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            run.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_benchmark_run(
    path: Path,
) -> RetrievalBenchmarkRun:
    return RetrievalBenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))
