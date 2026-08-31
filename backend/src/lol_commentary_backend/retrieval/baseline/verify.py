from math import isclose

from lol_commentary_backend.retrieval.baseline.models import (
    ReproducibilityCheck,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)

REPRODUCIBILITY_CHECK_VERSION = "0.1.0"


def build_reproducibility_check(
    *,
    reference: HybridRetrievalRun,
    reproduced: HybridRetrievalRun,
    sentence_transformers_version: str,
    transformers_version: str,
    tolerance: float = 1e-12,
) -> ReproducibilityCheck:
    if reference.corpus_sha256 != reproduced.corpus_sha256:
        raise ValueError("Reference and reproduced runs used different corpora")

    metric_pairs = (
        (
            reference.metrics.hit_at_1,
            reproduced.metrics.hit_at_1,
        ),
        (
            reference.metrics.hit_at_3,
            reproduced.metrics.hit_at_3,
        ),
        (
            reference.metrics.recall_at_5,
            reproduced.metrics.recall_at_5,
        ),
        (
            reference.metrics.mrr,
            reproduced.metrics.mrr,
        ),
    )

    matches = all(
        isclose(
            reference_value,
            reproduced_value,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for reference_value, reproduced_value in metric_pairs
    )

    return ReproducibilityCheck(
        check_version=(REPRODUCIBILITY_CHECK_VERSION),
        engine_key=reproduced.hybrid_key,
        corpus_sha256=reproduced.corpus_sha256,
        reference_hit_at_1=(reference.metrics.hit_at_1),
        reproduced_hit_at_1=(reproduced.metrics.hit_at_1),
        reference_hit_at_3=(reference.metrics.hit_at_3),
        reproduced_hit_at_3=(reproduced.metrics.hit_at_3),
        reference_recall_at_5=(reference.metrics.recall_at_5),
        reproduced_recall_at_5=(reproduced.metrics.recall_at_5),
        reference_mrr=reference.metrics.mrr,
        reproduced_mrr=reproduced.metrics.mrr,
        metrics_match=matches,
        sentence_transformers_version=(sentence_transformers_version),
        transformers_version=transformers_version,
    )
