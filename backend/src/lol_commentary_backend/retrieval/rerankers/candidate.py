from collections.abc import Sequence

from lol_commentary_backend.retrieval.benchmark.models import (
    RetrievalQueryResult,
)
from lol_commentary_backend.retrieval.evaluation.models import (
    RetrievalEvalCase,
)
from lol_commentary_backend.retrieval.rerankers.models import (
    CandidateSourceCoverage,
)


def _result_index(
    results: Sequence[RetrievalQueryResult],
) -> dict[str, RetrievalQueryResult]:
    return {result.query_id: result for result in results}


def analyze_candidate_coverage(
    *,
    source_key: str,
    results: Sequence[RetrievalQueryResult],
    cases: Sequence[RetrievalEvalCase],
    top_n: int,
) -> CandidateSourceCoverage:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    result_by_query = _result_index(results)
    case_by_query = {case.query_id: case for case in cases}

    if set(result_by_query) != set(case_by_query):
        raise ValueError("Candidate source and evaluation cases contain different query IDs")

    missing_query_ids: list[str] = []

    for query_id in sorted(case_by_query):
        case = case_by_query[query_id]
        result = result_by_query[query_id]

        candidate_ids = {item.chunk_id for item in result.top_results[:top_n]}

        relevant_ids = set(case.relevant_chunk_ids)

        if not relevant_ids.issubset(candidate_ids):
            missing_query_ids.append(query_id)

    cases_count = len(cases)
    full_gold_cases = cases_count - len(missing_query_ids)

    coverage_rate = 0.0 if cases_count == 0 else full_gold_cases / cases_count

    return CandidateSourceCoverage(
        source_key=source_key,
        top_n=top_n,
        cases=cases_count,
        full_gold_cases=full_gold_cases,
        coverage_rate=coverage_rate,
        missing_query_ids=missing_query_ids,
    )


def require_full_candidate_coverage(
    coverage: CandidateSourceCoverage,
) -> None:
    if coverage.coverage_rate != 1.0:
        raise ValueError(
            "Candidate source does not have "
            "100% Gold coverage. "
            f"Missing queries: "
            f"{coverage.missing_query_ids}"
        )
