import json
from pathlib import Path

import sentence_transformers
import transformers

from lol_commentary_backend.retrieval.baseline.decision import (
    build_retrieval_baseline_decision,
)
from lol_commentary_backend.retrieval.baseline.models import (
    ReproducibilityCheck,
)
from lol_commentary_backend.retrieval.benchmark.io import (
    load_benchmark_run,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    RerankerBenchmarkRun,
)


def _load_hybrid(
    path: Path,
) -> HybridRetrievalRun:
    return HybridRetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def _load_reranker(
    path: Path,
) -> RerankerBenchmarkRun:
    return RerankerBenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    retrieval_root = project_root / "data" / "processed" / "evaluation" / "retrieval"

    step38_dir = retrieval_root / "step38"
    step39_dir = retrieval_root / "step39"

    check = ReproducibilityCheck.model_validate_json(
        (step39_dir / "reproducibility_check.json").read_text(encoding="utf-8")
    )

    if not check.metrics_match:
        raise ValueError("Reproducibility gate did not pass")

    alias_bm25 = load_benchmark_run(step39_dir / "alias_bm25_current_env.json")

    hybrid = _load_hybrid(step39_dir / "rrf_bge_m3_alias_bm25_current_env.json")

    reranker_runs = [
        _load_reranker(step38_dir / "gte_multilingual_reranker_base.json"),
        _load_reranker(step38_dir / "bge_reranker_v2_m3.json"),
        _load_reranker(step38_dir / "qwen3_reranker_06b.json"),
    ]

    decision = build_retrieval_baseline_decision(
        alias_bm25=alias_bm25,
        bge_alias_hybrid=hybrid,
        reranker_runs=reranker_runs,
        sentence_transformers_version=(sentence_transformers.__version__),
        transformers_version=(transformers.__version__),
    )

    processed_path = step39_dir / "retrieval_baseline_v2.json"

    versioned_dir = Path(__file__).resolve().parents[1] / "evaluation" / "retrieval"
    versioned_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    versioned_path = versioned_dir / "retrieval_baseline_v2.json"

    payload = json.dumps(
        decision.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )

    processed_path.write_text(
        payload,
        encoding="utf-8",
    )
    versioned_path.write_text(
        payload,
        encoding="utf-8",
    )

    print("=== RETRIEVAL BASELINE v2 FREEZE ===")
    print(f"Primary: {decision.primary.key}")
    print(f"  Hit@1: {decision.primary.hit_at_1:.4f}")
    print(f"  Hit@3: {decision.primary.hit_at_3:.4f}")
    print(f"  Recall@5: {decision.primary.recall_at_5:.4f}")
    print(f"  MRR: {decision.primary.mrr:.4f}")
    print()

    print(f"Fast fallback: {decision.fast_fallback.key}")
    print(f"  Hit@1: {decision.fast_fallback.hit_at_1:.4f}")
    print(f"  Recall@5: {decision.fast_fallback.recall_at_5:.4f}")
    print()

    print("Rejected rerankers:")
    for key in decision.rejected_rerankers:
        print(f"  {key}")

    print()
    print("Pinned inference environment:")
    print(f"  sentence-transformers={decision.sentence_transformers_version}")
    print(f"  transformers={decision.transformers_version}")

    print()
    print(f"Versioned baseline: {versioned_path}")
    print(f"Processed copy: {processed_path}")


if __name__ == "__main__":
    main()
