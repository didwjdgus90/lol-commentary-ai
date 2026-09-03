from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from selectolax.lexbor import (
    LexborHTMLParser,
)

from lol_commentary_backend.ingestion.entity_catalog.models import (
    CanonicalEntity,
    EntityType,
)
from lol_commentary_backend.ingestion.entity_catalog.resolver import (
    EntityCatalogResolver,
    ResolutionStatus,
)
from lol_commentary_backend.ingestion.multi_patch.models import (
    CorpusShardManifest,
    CorpusShardReference,
    MultiPatchCorpusManifest,
)
from lol_commentary_backend.ingestion.patch_note_entity_resolution.models import (
    ResolutionMethod,
    ResolvedPatchRecord,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.models import (
    NormalizedPatchRecord,
    PatchEntityType,
)
from lol_commentary_backend.ingestion.patch_note_normalizer.normalizer import (
    normalize_patch_note_document,
)
from lol_commentary_backend.ingestion.patch_note_parser.models import (
    PatchNoteDocument,
)
from lol_commentary_backend.ingestion.patch_note_parser.parser import (
    parse_riot_patch_sections,
)
from lol_commentary_backend.ingestion.sources.patch_notes import (
    PatchNoteManifestRecord,
    build_patch_sequence,
)
from lol_commentary_backend.ingestion.version_mapping.models import (
    PatchDataDragonMapping,
)
from lol_commentary_backend.retrieval.chunks.models import (
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.chunks.patch_note_chunker import (
    DEFAULT_MAX_CHUNK_CHARS,
    build_patch_rag_chunks_for_documents,
)
from lol_commentary_backend.retrieval.documents.models import (
    PatchRagDocument,
)
from lol_commentary_backend.retrieval.documents.patch_note_builder import (
    build_patch_rag_documents,
)

SUMMONERS_RIFT_MAP_ID = "11"

_STANDARD_ITEM_SECTION_NAMES = {
    "신규 아이템",
    "복귀 아이템",
    "업데이트된 아이템",
    "items",
    "item changes",
    "new items",
    "returning items",
    "updated items",
}


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_raw_file(
    *,
    path: Path,
    record: PatchNoteManifestRecord,
) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"Raw patch file missing: {path}")

    content = path.read_bytes()

    if len(content) != record.byte_count:
        raise ValueError(f"Raw patch byte count mismatch: {record.patch} {record.locale}")

    actual_sha256 = hashlib.sha256(content).hexdigest()

    if actual_sha256 != record.content_sha256:
        raise ValueError(f"Raw patch SHA mismatch: {record.patch} {record.locale}")

    return content


def _extract_title(
    html: bytes,
) -> str:
    tree = LexborHTMLParser(html)

    heading = tree.css_first("h1")

    if heading is None:
        raise ValueError("Patch note h1 not found")

    title = " ".join(
        heading.text(
            separator=" ",
            strip=True,
        ).split()
    )

    if not title:
        raise ValueError("Patch note h1 is empty")

    return title


def _read_patch_manifest(
    path: Path,
) -> tuple[
    PatchNoteManifestRecord,
    ...,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Patch manifest missing: {path}")

    records = tuple(
        PatchNoteManifestRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    if not records:
        raise ValueError("Patch manifest is empty")

    return records


def _read_mappings(
    path: Path,
) -> tuple[
    PatchDataDragonMapping,
    ...,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Patch/Data Dragon mapping missing: {path}")

    mappings = tuple(
        PatchDataDragonMapping.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    if not mappings:
        raise ValueError("Patch/Data Dragon mapping is empty")

    return mappings


def _read_entities(
    path: Path,
) -> tuple[
    CanonicalEntity,
    ...,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Entity file missing: {path}")

    entities = tuple(
        CanonicalEntity.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    if not entities:
        raise ValueError("Entity catalog is empty")

    return entities


def _patch_entity_type(
    entity_type: EntityType,
) -> PatchEntityType:
    if entity_type == EntityType.CHAMPION:
        return PatchEntityType.CHAMPION

    if entity_type == EntityType.ITEM:
        return PatchEntityType.ITEM

    raise ValueError(f"Unsupported entity type: {entity_type}")


def _resolver_entity_type(
    entity_type: PatchEntityType,
) -> EntityType | None:
    if entity_type == PatchEntityType.CHAMPION:
        return EntityType.CHAMPION

    if entity_type == PatchEntityType.ITEM:
        return EntityType.ITEM

    return None


def _normalized_heading(
    value: str,
) -> str:
    return " ".join(value.casefold().split())


def _is_standard_item_section(
    record: NormalizedPatchRecord,
) -> bool:
    if not record.heading_path:
        return False

    first = _normalized_heading(record.heading_path[0])

    return first in _STANDARD_ITEM_SECTION_NAMES


def _is_game_mode_record(
    record: NormalizedPatchRecord,
) -> bool:
    if record.section_kind == "game_mode":
        return True

    if not record.heading_path:
        return False

    first = _normalized_heading(record.heading_path[0])

    return "arena" in first or "aram" in first or "게임 모드" in first or "무작위 총력전" in first


def _preferred_entity_name(
    *,
    entity: CanonicalEntity,
    ddragon_version: str,
    locale: str,
    fallback: str,
) -> tuple[
    str,
    str | None,
]:
    matching = [
        observation
        for observation in entity.observations
        if (observation.ddragon_version == ddragon_version and observation.locale == locale)
    ]

    if not matching:
        return (
            fallback,
            None,
        )

    chosen = matching[0]

    return (
        chosen.name,
        chosen.source_sha256,
    )


def _resolved_payload(
    *,
    record: NormalizedPatchRecord,
    ddragon_version: str,
    entity: CanonicalEntity | None,
    method: ResolutionMethod,
    candidate_count: int,
    original_candidate_count: int,
    target_map_id: str | None,
) -> ResolvedPatchRecord:
    payload = record.model_dump(mode="python")

    entity_source_sha256: str | None = None

    if entity is not None:
        fallback_name = record.entity_name or record.title

        (
            localized_name,
            entity_source_sha256,
        ) = _preferred_entity_name(
            entity=entity,
            ddragon_version=(ddragon_version),
            locale=record.locale,
            fallback=fallback_name,
        )

        payload["entity_type"] = _patch_entity_type(entity.entity_type)

        payload["entity_name"] = localized_name

        if entity.entity_type == EntityType.CHAMPION:
            payload["entity_id"] = entity.riot_id

            payload["entity_key"] = entity.riot_key

        else:
            payload["entity_id"] = entity.riot_key

            payload["entity_key"] = None

    else:
        payload["entity_id"] = None
        payload["entity_key"] = None

    payload.update(
        {
            "ddragon_version": (ddragon_version),
            "resolution_method": method,
            "resolution_candidate_count": (candidate_count),
            "resolution_original_candidate_count": (original_candidate_count),
            "target_map_id": (target_map_id),
            "resolution_evidence_fields": [],
            "entity_source_sha256": (entity_source_sha256),
        }
    )

    return ResolvedPatchRecord.model_validate(payload)


def _contains_removed_signal(
    record: NormalizedPatchRecord,
) -> bool:
    texts = [
        *record.changes,
        *record.paragraphs,
    ]

    for text in texts:
        normalized = text.casefold()

        if "삭제되었습니다" in text or "removed" in normalized:
            return True

    return False


def _resolve_record(
    *,
    record: NormalizedPatchRecord,
    resolver: EntityCatalogResolver,
    entities_by_uid: dict[
        str,
        CanonicalEntity,
    ],
    ddragon_version: str,
) -> ResolvedPatchRecord:
    query = record.entity_name or record.title

    requested_type = _resolver_entity_type(record.entity_type)

    base = resolver.resolve(
        query,
        patch=record.patch,
        entity_type=requested_type,
    )

    original_candidate_count = len(base.candidate_entity_uids)

    final = base

    target_map_id: str | None = None

    should_try_standard_map = not _is_game_mode_record(record) and (
        requested_type == EntityType.ITEM or _is_standard_item_section(record)
    )

    if (
        not should_try_standard_map
        and base.status == ResolutionStatus.AMBIGUOUS
        and base.candidate_entity_uids
        and not _is_game_mode_record(record)
    ):
        candidate_types = {
            entities_by_uid[entity_uid].entity_type
            for entity_uid in base.candidate_entity_uids
            if entity_uid in entities_by_uid
        }

        if candidate_types == {EntityType.ITEM}:
            should_try_standard_map = True

            requested_type = EntityType.ITEM

    if should_try_standard_map:
        target_map_id = SUMMONERS_RIFT_MAP_ID

        final = resolver.resolve(
            query,
            patch=record.patch,
            entity_type=(requested_type or EntityType.ITEM),
            map_id=target_map_id,
        )

    if final.status == ResolutionStatus.RESOLVED and final.selected_entity_uid is not None:
        entity = entities_by_uid[final.selected_entity_uid]

        if target_map_id is not None:
            method = ResolutionMethod.MAP_EXACT

        elif record.entity_name:
            method = ResolutionMethod.BASELINE_EXACT

        else:
            method = ResolutionMethod.TITLE_EXACT

        return _resolved_payload(
            record=record,
            ddragon_version=(ddragon_version),
            entity=entity,
            method=method,
            candidate_count=1,
            original_candidate_count=(
                max(
                    original_candidate_count,
                    1,
                )
            ),
            target_map_id=(target_map_id),
        )

    if (
        target_map_id is not None
        and original_candidate_count > 0
        and not final.candidate_entity_uids
    ):
        method = (
            ResolutionMethod.REMOVED_FROM_TARGET_MAP
            if _contains_removed_signal(record)
            else ResolutionMethod.MAP_INCOMPATIBLE
        )

    elif final.status == ResolutionStatus.AMBIGUOUS:
        method = ResolutionMethod.AMBIGUOUS

    else:
        method = ResolutionMethod.UNRESOLVED

    return _resolved_payload(
        record=record,
        ddragon_version=(ddragon_version),
        entity=None,
        method=method,
        candidate_count=len(final.candidate_entity_uids),
        original_candidate_count=(original_candidate_count),
        target_map_id=(target_map_id),
    )


def _canonical_jsonl_line(
    model: BaseModel,
) -> str:
    payload = model.model_dump(mode="json")

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_jsonl(
    *,
    path: Path,
    models: Sequence[BaseModel],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = "\n".join(_canonical_jsonl_line(model) for model in models) + "\n"

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(path)

    return _sha256_file(path)


def _write_json(
    *,
    path: Path,
    model: BaseModel,
) -> None:
    payload = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        payload,
        encoding="utf-8",
    )

    temporary.replace(path)


def build_multi_patch_corpus(
    *,
    repository_root: Path,
    start_patch: str = "26.1",
    end_patch: str = "26.17",
    locales: tuple[
        str,
        ...,
    ] = (
        "ko_KR",
        "en_US",
    ),
    max_chunk_chars: int = (DEFAULT_MAX_CHUNK_CHARS),
) -> MultiPatchCorpusManifest:
    patches = build_patch_sequence(
        start_patch=start_patch,
        end_patch=end_patch,
    )

    patch_manifest_path = (
        repository_root / "data" / "manifests" / "patch_notes" / "raw_sources.jsonl"
    )

    mapping_path = (
        repository_root / "data" / "processed" / "version_mapping" / "v1" / "patch_ddragon.jsonl"
    )

    entities_path = repository_root / "data" / "processed" / "entities" / "v1" / "entities.jsonl"

    aliases_path = repository_root / "data" / "processed" / "entities" / "v1" / "aliases.jsonl"

    entity_metadata_path = (
        repository_root / "data" / "processed" / "entities" / "v1" / "metadata.json"
    )

    raw_records = _read_patch_manifest(patch_manifest_path)

    mappings = _read_mappings(mapping_path)

    entities = _read_entities(entities_path)

    entities_by_uid = {entity.entity_uid: entity for entity in entities}

    mapping_by_patch = {mapping.patch: mapping for mapping in mappings}

    raw_by_key = {
        (
            record.patch,
            record.locale,
        ): record
        for record in raw_records
    }

    resolver = EntityCatalogResolver.from_files(
        entities_path=entities_path,
        aliases_path=aliases_path,
        mappings_path=mapping_path,
    )

    output_root = repository_root / "data" / "processed" / "rag" / "patch_notes" / "multi_patch_v1"

    shard_references: list[CorpusShardReference] = []

    total_document_count = 0
    total_chunk_count = 0

    for patch in patches:
        mapping = mapping_by_patch.get(patch)

        if mapping is None:
            raise ValueError(f"Missing version mapping for patch {patch}")

        for locale in locales:
            raw = raw_by_key.get(
                (
                    patch,
                    locale,
                )
            )

            if raw is None:
                raise ValueError(f"Missing raw patch snapshot: {patch} {locale}")

            raw_path = repository_root / raw.file_path

            html = _verify_raw_file(
                path=raw_path,
                record=raw,
            )

            title = _extract_title(html)

            document = PatchNoteDocument(
                patch=patch,
                locale=locale,
                source_url=(raw.source_url),
                fetched_at=(raw.snapshot_at),
                source_sha256=(raw.content_sha256),
                source_size_bytes=(raw.byte_count),
                collector_version=("patch-note-batch-v1"),
                title=title,
                sections=(parse_riot_patch_sections(html)),
            )

            normalized = normalize_patch_note_document(document)

            resolved = [
                _resolve_record(
                    record=record,
                    resolver=resolver,
                    entities_by_uid=(entities_by_uid),
                    ddragon_version=(mapping.ddragon_version),
                )
                for record in normalized
            ]

            rag_documents: list[PatchRagDocument] = build_patch_rag_documents(resolved)

            chunks: list[PatchRagChunk] = build_patch_rag_chunks_for_documents(
                rag_documents,
                max_chars=(max_chunk_chars),
            )

            if not normalized:
                raise RuntimeError(f"No normalized records for {patch} {locale}")

            if not rag_documents:
                raise RuntimeError(f"No RAG documents for {patch} {locale}")

            if not chunks:
                raise RuntimeError(f"No RAG chunks for {patch} {locale}")

            chunk_ids = [chunk.chunk_id for chunk in chunks]

            if len(chunk_ids) != len(set(chunk_ids)):
                raise RuntimeError(f"Duplicate chunk IDs in {patch} {locale}")

            shard_dir = output_root / patch / locale

            normalized_path = shard_dir / "normalized.jsonl"

            resolved_path = shard_dir / "resolved.jsonl"

            documents_path = shard_dir / "documents.jsonl"

            chunks_path = shard_dir / "chunks.jsonl"

            shard_manifest_path = shard_dir / "manifest.json"

            normalized_sha = _write_jsonl(
                path=normalized_path,
                models=list(normalized),
            )

            resolved_sha = _write_jsonl(
                path=resolved_path,
                models=list(resolved),
            )

            documents_sha = _write_jsonl(
                path=documents_path,
                models=list(rag_documents),
            )

            chunks_sha = _write_jsonl(
                path=chunks_path,
                models=list(chunks),
            )

            resolved_entity_count = sum(record.entity_id is not None for record in resolved)

            ambiguous_count = sum(
                record.resolution_method == ResolutionMethod.AMBIGUOUS for record in resolved
            )

            unresolved_count = sum(
                record.resolution_method
                in {
                    ResolutionMethod.UNRESOLVED,
                    ResolutionMethod.MAP_INCOMPATIBLE,
                }
                for record in resolved
            )

            shard_manifest = CorpusShardManifest(
                patch=patch,
                locale=locale,
                ddragon_version=(mapping.ddragon_version),
                raw_source_sha256=(raw.content_sha256),
                normalized_count=(len(normalized)),
                resolved_count=(len(resolved)),
                resolved_entity_count=(resolved_entity_count),
                ambiguous_count=(ambiguous_count),
                unresolved_count=(unresolved_count),
                document_count=(len(rag_documents)),
                chunk_count=(len(chunks)),
                max_chunk_chars=(max_chunk_chars),
                normalized_sha256=(normalized_sha),
                resolved_sha256=(resolved_sha),
                documents_sha256=(documents_sha),
                corpus_sha256=(chunks_sha),
            )

            _write_json(
                path=(shard_manifest_path),
                model=(shard_manifest),
            )

            relative_manifest = shard_manifest_path.relative_to(repository_root).as_posix()

            shard_references.append(
                CorpusShardReference(
                    patch=patch,
                    locale=locale,
                    manifest_path=(relative_manifest),
                    corpus_sha256=(chunks_sha),
                    chunk_count=(len(chunks)),
                )
            )

            total_document_count += len(rag_documents)

            total_chunk_count += len(chunks)

    aggregate = MultiPatchCorpusManifest(
        start_patch=start_patch,
        end_patch=end_patch,
        locales=locales,
        patch_count=len(patches),
        shard_count=len(shard_references),
        total_document_count=(total_document_count),
        total_chunk_count=(total_chunk_count),
        patch_manifest_sha256=(_sha256_file(patch_manifest_path)),
        version_mapping_sha256=(_sha256_file(mapping_path)),
        entity_catalog_sha256=(_sha256_file(entity_metadata_path)),
        shards=tuple(shard_references),
    )

    _write_json(
        path=(output_root / "manifest.json"),
        model=aggregate,
    )

    return aggregate
