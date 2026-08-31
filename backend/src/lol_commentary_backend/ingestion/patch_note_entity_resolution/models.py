from enum import StrEnum

from pydantic import Field

from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
)


class ResolutionMethod(StrEnum):
    BASELINE_EXACT = "baseline_exact"
    TITLE_EXACT = "title_exact"
    MAP_EXACT = "map_exact"
    EVIDENCE_EXACT = "evidence_exact"
    CONTEXT_EXACT = "context_exact"
    REMOVED_FROM_TARGET_MAP = "removed_from_target_map"
    MAP_INCOMPATIBLE = "map_incompatible"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class ResolvedPatchRecord(NormalizedPatchRecord):
    ddragon_version: str = Field(min_length=1)
    entity_id: str | None = None
    entity_key: str | None = None
    resolution_method: ResolutionMethod
    resolution_candidate_count: int = Field(ge=0)
    resolution_original_candidate_count: int | None = Field(default=None, ge=0)
    target_map_id: str | None = None
    resolution_evidence_fields: list[str] = Field(default_factory=list)
    context_method: str | None = None
    context_confidence: str | None = None
    context_anchor_title: str | None = None
    context_evidence: list[str] = Field(default_factory=list)
    entity_source_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
