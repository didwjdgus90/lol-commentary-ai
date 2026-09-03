import argparse

import httpx

from lol_commentary_backend.ingestion.sources.models import (
    SourceFormat,
)
from lol_commentary_backend.ingestion.sources.registry import (
    list_data_sources,
)

USER_AGENT = "lol-commentary-ai/data-source-probe-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
    )

    return parser.parse_args()


def _validate_response(
    *,
    source_id: str,
    response_format: SourceFormat,
    response: httpx.Response,
) -> None:
    response.raise_for_status()

    if not response.content:
        raise RuntimeError(f"{source_id}: empty response")

    if response_format == SourceFormat.JSON:
        try:
            response.json()
        except ValueError as exc:
            raise RuntimeError(f"{source_id}: invalid JSON") from exc


def main() -> None:
    args = _parse_args()

    sources = list_data_sources(probe_enabled=True)

    print("=== RIOT PUBLIC DATA SOURCE PROBE ===")

    print(f"Probe source count: {len(sources)}")

    print()

    failures: list[str] = []

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=True,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
    ) as client:
        for source in sources:
            print(f"[{source.source_id}]")

            print(f"  URL: {source.url_template}")

            try:
                response = client.get(source.url_template)

                _validate_response(
                    source_id=(source.source_id),
                    response_format=(source.response_format),
                    response=response,
                )

            except (
                httpx.HTTPError,
                RuntimeError,
            ) as exc:
                failures.append(source.source_id)

                print("  RESULT=FAIL")

                print(f"  ERROR={type(exc).__name__}: {exc}")

                print()

                continue

            content_type = response.headers.get(
                "content-type",
                "",
            )

            print(f"  STATUS={response.status_code}")

            print(f"  CONTENT_TYPE={content_type}")

            print(f"  BYTES={len(response.content)}")

            print("  RESULT=PASS")

            print()

    if failures:
        raise RuntimeError(f"Public data source probe failed: {failures}")

    print("RIOT_PUBLIC_DATA_SOURCE_PROBE=PASS")


if __name__ == "__main__":
    main()
