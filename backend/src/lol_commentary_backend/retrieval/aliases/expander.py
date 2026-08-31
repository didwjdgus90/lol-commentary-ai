import re

from lol_commentary_backend.retrieval.aliases.models import (
    BilingualAliasCatalog,
    QueryExpansion,
)


def _contains_alias(
    query: str,
    alias: str,
) -> bool:
    if not alias:
        return False

    if alias.isascii():
        pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"

        return (
            re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )
            is not None
        )

    return alias in query


def expand_bilingual_entity_query(
    query: str,
    catalog: BilingualAliasCatalog,
) -> QueryExpansion:
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("query must not be empty")

    additions: list[str] = []
    matched_keys: list[str] = []

    ordered_aliases = sorted(
        catalog.aliases,
        key=lambda item: max(
            len(item.ko_name),
            len(item.en_name),
        ),
        reverse=True,
    )

    for record in ordered_aliases:
        ko_match = _contains_alias(
            normalized_query,
            record.ko_name,
        )
        en_match = _contains_alias(
            normalized_query,
            record.en_name,
        )

        if not ko_match and not en_match:
            continue

        counterpart: str | None = None

        if en_match and not _contains_alias(
            normalized_query,
            record.ko_name,
        ):
            counterpart = record.ko_name
        elif ko_match and not _contains_alias(
            normalized_query,
            record.en_name,
        ):
            counterpart = record.en_name

        if counterpart is None:
            continue

        if any(counterpart.casefold() == existing.casefold() for existing in additions):
            continue

        additions.append(counterpart)
        matched_keys.append(f"{record.entity_type.value}:{record.entity_key}")

    if additions:
        expanded_query = f"{normalized_query} | " + " | ".join(additions)
    else:
        expanded_query = normalized_query

    return QueryExpansion(
        original_query=normalized_query,
        expanded_query=expanded_query,
        matched_entity_keys=matched_keys,
        added_aliases=additions,
        changed=bool(additions),
    )
