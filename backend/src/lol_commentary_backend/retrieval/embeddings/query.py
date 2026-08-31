from lol_commentary_backend.retrieval.embeddings.models import (
    EmbeddingCandidate,
)

LOL_PATCH_RETRIEVAL_TASK = (
    "Given a League of Legends patch-related query in Korean or English, "
    "retrieve the most relevant official patch-note passage that answers "
    "the query."
)


def format_embedding_query(
    query: str,
    candidate: EmbeddingCandidate,
) -> str:
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("query must not be empty")

    if candidate.query_mode == "qwen_instruct":
        return f"Instruct: {LOL_PATCH_RETRIEVAL_TASK}\nQuery:{normalized_query}"

    if candidate.query_mode == "e5_instruct":
        return f"Instruct: {LOL_PATCH_RETRIEVAL_TASK}\nQuery: {normalized_query}"

    return normalized_query
