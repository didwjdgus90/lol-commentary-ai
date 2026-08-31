from __future__ import annotations

import os
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DATABASE_URL_ENV = "LOL_DATABASE_URL"


def resolve_database_url(
    explicit_url: str | None = None,
) -> str:
    if explicit_url is not None:
        url = explicit_url.strip()

        if url:
            return url

    url = os.getenv(
        DATABASE_URL_ENV,
        "",
    ).strip()

    if not url:
        raise RuntimeError(
            f"{DATABASE_URL_ENV} is not set. "
            "Provide the PostgreSQL connection URL "
            "through the environment."
        )

    return url


def connect_database(
    database_url: str | None = None,
) -> psycopg.Connection[Any]:
    connection = psycopg.connect(resolve_database_url(database_url))

    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise

    return connection


def _configure_vector_connection(
    connection: psycopg.Connection[Any],
) -> None:
    register_vector(connection)


def build_vector_pool(
    database_url: str | None = None,
    *,
    min_size: int = 1,
    max_size: int = 4,
) -> ConnectionPool:
    if min_size <= 0:
        raise ValueError("min_size must be positive")

    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    pool = ConnectionPool(
        conninfo=resolve_database_url(database_url),
        min_size=min_size,
        max_size=max_size,
        open=False,
        configure=(_configure_vector_connection),
        name="lol-retrieval-pgvector",
    )

    pool.open(wait=True)

    return pool
