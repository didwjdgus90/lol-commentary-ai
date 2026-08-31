from html import escape

from lol_commentary_backend.rag.context.models import (
    ContextBundle,
    ContextEvidence,
)

FORMAT_VERSION = "evidence_xml_v1"


def _clean_text(
    value: str,
) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

    cleaned = "".join(
        character for character in normalized if (character in {"\n", "\t"} or ord(character) >= 32)
    )

    return cleaned.strip()


def _xml_text(
    value: str,
) -> str:
    return escape(
        _clean_text(value),
        quote=False,
    )


def _xml_attribute(
    value: str,
) -> str:
    return escape(
        _clean_text(value),
        quote=True,
    )


def _format_heading_path(
    evidence: ContextEvidence,
) -> str:
    return " > ".join(_xml_text(part) for part in evidence.heading_path)


def _format_evidence(
    evidence: ContextEvidence,
) -> str:
    entity_name = _xml_text(evidence.entity_name) if evidence.entity_name else ""

    return "\n".join(
        [
            (
                "  <evidence "
                f'id="{_xml_attribute(evidence.citation_id)}" '
                f'source_rank="{evidence.source_rank}">'
            ),
            (f"    <patch>{_xml_text(evidence.patch)}</patch>"),
            (f"    <locale>{_xml_text(evidence.locale)}</locale>"),
            (f"    <section_kind>{_xml_text(evidence.section_kind)}</section_kind>"),
            (f"    <title>{_xml_text(evidence.title)}</title>"),
            (f"    <entity>{entity_name}</entity>"),
            (f"    <heading_path>{_format_heading_path(evidence)}</heading_path>"),
            (f"    <source_url>{_xml_text(evidence.source_url)}</source_url>"),
            (f"    <chunk_id>{_xml_text(evidence.chunk_id)}</chunk_id>"),
            "    <content>",
            (
                "      "
                + _xml_text(evidence.text).replace(
                    "\n",
                    "\n      ",
                )
            ),
            "    </content>",
            "  </evidence>",
        ]
    )


def format_context_for_prompt(
    bundle: ContextBundle,
) -> str:
    evidence_blocks = [_format_evidence(evidence) for evidence in bundle.evidence]

    lines = [
        (
            "<retrieval_context "
            'schema_version="1" '
            f'format_version="{FORMAT_VERSION}" '
            f'evidence_count="{bundle.selected_count}">'
        ),
        "  <retrieval_metadata>",
        (f"    <query>{_xml_text(bundle.query)}</query>"),
        (f"    <expanded_query>{_xml_text(bundle.expanded_query)}</expanded_query>"),
        (f"    <strategy>{_xml_text(bundle.strategy_used.value)}</strategy>"),
        "  </retrieval_metadata>",
    ]

    if evidence_blocks:
        lines.extend(
            [
                "  <evidence_list>",
                *evidence_blocks,
                "  </evidence_list>",
            ]
        )
    else:
        lines.append("  <evidence_list></evidence_list>")

    lines.append("</retrieval_context>")

    return "\n".join(lines)
