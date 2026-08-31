from collections.abc import Sequence

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalBenchmarkRun,
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.diagnostics.models import (
    EngineRankDiagnostic,
    FailureCategory,
    FusionEffect,
    FusionQueryDiagnostic,
    QueryFailureDiagnostic,
    RetrievalDiagnosticReport,
)
from lol_commentary_backend.retrieval.hybrid.models import (
    HybridRetrievalRun,
)

DIAGNOSTIC_REPORT_VERSION = "0.1.0"


def _result_index(
    results: Sequence[RetrievalQueryResult],
) -> dict[str, RetrievalQueryResult]:
    return {result.query_id: result for result in results}


def _validate_standalone_runs(
    runs: Sequence[RetrievalBenchmarkRun],
) -> None:
    if not runs:
        raise ValueError("At least one benchmark run is required")

    corpus_hashes = {run.corpus_sha256 for run in runs}

    if len(corpus_hashes) != 1:
        raise ValueError("Standalone runs used different corpora")

    query_id_sets = [{result.query_id for result in run.query_results} for run in runs]

    first = query_id_sets[0]

    if any(query_ids != first for query_ids in query_id_sets[1:]):
        raise ValueError("Standalone runs contain different query IDs")


def _engine_diagnostic(
    run: RetrievalBenchmarkRun,
    result: RetrievalQueryResult,
) -> EngineRankDiagnostic:
    top_1 = result.top_results[0] if result.top_results else None

    return EngineRankDiagnostic(
        engine=run.engine_key,
        first_relevant_rank=(result.first_relevant_rank),
        hit_at_1=result.hit_at_1,
        hit_at_5=result.hit_at_5,
        relevant_in_top_10=(
            result.first_relevant_rank is not None and result.first_relevant_rank <= 10
        ),
        top_1_title=(top_1.title if top_1 is not None else None),
        top_1_entity_name=(top_1.entity_name if top_1 is not None else None),
    )


def diagnose_queries(
    runs: Sequence[RetrievalBenchmarkRun],
) -> list[QueryFailureDiagnostic]:
    _validate_standalone_runs(runs)

    indexed = {run.engine_key: _result_index(run.query_results) for run in runs}

    query_ids = sorted(next(iter(indexed.values())))

    diagnostics: list[QueryFailureDiagnostic] = []

    for query_id in query_ids:
        source_results = [
            (
                run,
                indexed[run.engine_key][query_id],
            )
            for run in runs
        ]

        ranks = [
            result.first_relevant_rank
            for _, result in source_results
            if result.first_relevant_rank is not None
        ]

        best_rank = min(ranks) if ranks else None

        if best_rank == 1:
            category = FailureCategory.SOLVED_TOP1
        elif best_rank is not None and best_rank <= 10:
            category = FailureCategory.RERANKABLE_TOP10
        else:
            category = FailureCategory.CANDIDATE_RECALL_FAILURE

        best_engines = [
            run.engine_key
            for run, result in source_results
            if result.first_relevant_rank == best_rank
        ]

        reference = source_results[0][1]

        diagnostics.append(
            QueryFailureDiagnostic(
                query_id=query_id,
                query=reference.query,
                language=reference.language,
                query_type=reference.query_type,
                difficulty=reference.difficulty,
                category=category,
                best_relevant_rank=best_rank,
                best_engines=best_engines,
                engine_diagnostics=[
                    _engine_diagnostic(
                        run,
                        result,
                    )
                    for run, result in source_results
                ],
            )
        )

    return diagnostics


def diagnose_fusion(
    hybrid: HybridRetrievalRun,
    dense: RetrievalBenchmarkRun,
    sparse: RetrievalBenchmarkRun,
) -> list[FusionQueryDiagnostic]:
    if hybrid.corpus_sha256 != dense.corpus_sha256 or hybrid.corpus_sha256 != sparse.corpus_sha256:
        raise ValueError("Fusion and source runs used different corpora")

    hybrid_by_query = _result_index(hybrid.query_results)
    dense_by_query = _result_index(dense.query_results)
    sparse_by_query = _result_index(sparse.query_results)

    query_ids = set(hybrid_by_query)

    if query_ids != set(dense_by_query) or query_ids != set(sparse_by_query):
        raise ValueError("Fusion and source runs contain different query IDs")

    diagnostics: list[FusionQueryDiagnostic] = []

    for query_id in sorted(query_ids):
        hybrid_result = hybrid_by_query[query_id]
        dense_result = dense_by_query[query_id]
        sparse_result = sparse_by_query[query_id]

        source_hit = dense_result.hit_at_1 or sparse_result.hit_at_1

        if hybrid_result.hit_at_1 and source_hit:
            effect = FusionEffect.PRESERVED_SUCCESS
        elif hybrid_result.hit_at_1 and not source_hit:
            effect = FusionEffect.NEW_TOP1_SUCCESS
        elif not hybrid_result.hit_at_1 and source_hit:
            effect = FusionEffect.LOST_SOURCE_TOP1
        else:
            effect = FusionEffect.STILL_MISS_TOP1

        diagnostics.append(
            FusionQueryDiagnostic(
                query_id=query_id,
                query=hybrid_result.query,
                dense_hit_at_1=(dense_result.hit_at_1),
                sparse_hit_at_1=(sparse_result.hit_at_1),
                hybrid_hit_at_1=(hybrid_result.hit_at_1),
                dense_first_relevant_rank=(dense_result.first_relevant_rank),
                sparse_first_relevant_rank=(sparse_result.first_relevant_rank),
                hybrid_first_relevant_rank=(hybrid_result.first_relevant_rank),
                effect=effect,
            )
        )

    return diagnostics


def build_diagnostic_report(
    standalone_runs: Sequence[RetrievalBenchmarkRun],
    *,
    hybrid: HybridRetrievalRun,
    hybrid_dense_source: RetrievalBenchmarkRun,
    hybrid_sparse_source: RetrievalBenchmarkRun,
) -> RetrievalDiagnosticReport:
    query_diagnostics = diagnose_queries(standalone_runs)

    fusion_diagnostics = diagnose_fusion(
        hybrid,
        hybrid_dense_source,
        hybrid_sparse_source,
    )

    return RetrievalDiagnosticReport(
        report_version=(DIAGNOSTIC_REPORT_VERSION),
        corpus_sha256=(standalone_runs[0].corpus_sha256),
        cases=len(query_diagnostics),
        solved_top1=[
            item.query_id
            for item in query_diagnostics
            if item.category == FailureCategory.SOLVED_TOP1
        ],
        rerankable_top10=[
            item.query_id
            for item in query_diagnostics
            if item.category == FailureCategory.RERANKABLE_TOP10
        ],
        candidate_recall_failures=[
            item.query_id
            for item in query_diagnostics
            if item.category == FailureCategory.CANDIDATE_RECALL_FAILURE
        ],
        query_diagnostics=query_diagnostics,
        fusion_diagnostics=fusion_diagnostics,
    )
