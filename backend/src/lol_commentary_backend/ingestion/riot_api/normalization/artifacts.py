from __future__ import annotations

import json
from collections.abc import (
    Iterable,
)
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lol_commentary_backend.ingestion.riot_api.collector import (
    canonical_json_bytes,
)
from lol_commentary_backend.ingestion.riot_api.normalization.artifact_models import (
    NORMALIZED_DATASET_VERSION,
    NormalizedArtifactFile,
    NormalizedArtifactType,
    NormalizedDatasetBuildResult,
    NormalizedDatasetManifest,
    NormalizedMatchArtifactManifest,
    NormalizedMatchReference,
)
from lol_commentary_backend.ingestion.riot_api.normalization.models import (
    RIOT_NORMALIZER_VERSION,
)
from lol_commentary_backend.ingestion.riot_api.normalization.normalizer import (
    normalize_match_timeline,
)
from lol_commentary_backend.ingestion.riot_api.raw_models import (
    RiotRawManifestRecord,
    RiotRawResourceType,
)


def _sha256_bytes(
    payload: bytes,
) -> str:
    return sha256(payload).hexdigest()


def _file_sha256(
    path: Path,
) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative_path(
    *,
    repository_root: Path,
    path: Path,
) -> str:
    return path.relative_to(repository_root).as_posix()


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"Expected JSON object: {path}")

    return payload


def _jsonl_bytes(
    records: Iterable[BaseModel],
) -> bytes:
    lines = [
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
        for record in records
    ]

    text = "\n".join(lines) + ("\n" if lines else "")

    return text.encode("utf-8")


def _write_if_changed(
    *,
    path: Path,
    payload: bytes,
) -> bool:
    if path.is_file() and path.read_bytes() == payload:
        return False

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_bytes(payload)

    temporary.replace(path)

    return True


def _load_raw_manifest(
    path: Path,
) -> dict[
    str,
    RiotRawManifestRecord,
]:
    if not path.is_file():
        raise FileNotFoundError(f"Riot raw manifest missing: {path}")

    records: dict[
        str,
        RiotRawManifestRecord,
    ] = {}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = RiotRawManifestRecord.model_validate_json(line)

        except ValueError as exc:
            raise ValueError(f"Invalid Riot raw manifest at line {line_number}") from exc

        if record.resource_key in records:
            raise ValueError(f"Duplicate raw resource key: {record.resource_key}")

        records[record.resource_key] = record

    return records


def _require_raw_record(
    *,
    records: dict[
        str,
        RiotRawManifestRecord,
    ],
    resource_key: str,
    resource_type: (RiotRawResourceType),
) -> RiotRawManifestRecord:
    record = records.get(resource_key)

    if record is None:
        raise ValueError(f"Raw manifest record missing: {resource_key}")

    if record.resource_type != resource_type:
        raise ValueError(f"Unexpected raw resource type for {resource_key}")

    return record


def _validate_raw_record(
    *,
    repository_root: Path,
    record: RiotRawManifestRecord,
) -> Path:
    path = repository_root / record.file_path

    if not path.is_file():
        raise FileNotFoundError(f"Raw source file missing: {path}")

    if path.stat().st_size != record.size_bytes:
        raise ValueError(f"Raw source size mismatch: {record.resource_key}")

    if _file_sha256(path) != record.sha256:
        raise ValueError(f"Raw source SHA256 mismatch: {record.resource_key}")

    return path


def _artifact_file(
    *,
    repository_root: Path,
    artifact_type: (NormalizedArtifactType),
    path: Path,
    payload: bytes,
    record_count: int,
) -> NormalizedArtifactFile:
    return NormalizedArtifactFile(
        artifact_type=artifact_type,
        file_path=_relative_path(
            repository_root=(repository_root),
            path=path,
        ),
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        record_count=record_count,
    )


def _build_one_match(
    *,
    repository_root: Path,
    output_root: Path,
    raw_records: dict[
        str,
        RiotRawManifestRecord,
    ],
    match_id: str,
) -> tuple[
    NormalizedMatchReference,
    bool,
]:
    match_record = _require_raw_record(
        records=raw_records,
        resource_key=(f"match:{match_id}"),
        resource_type=(RiotRawResourceType.MATCH),
    )

    timeline_record = _require_raw_record(
        records=raw_records,
        resource_key=(f"timeline:{match_id}"),
        resource_type=(RiotRawResourceType.TIMELINE),
    )

    match_path = _validate_raw_record(
        repository_root=(repository_root),
        record=match_record,
    )

    timeline_path = _validate_raw_record(
        repository_root=(repository_root),
        record=timeline_record,
    )

    match_payload = _load_json_object(match_path)

    timeline_payload = _load_json_object(timeline_path)

    bundle = normalize_match_timeline(
        match=match_payload,
        timeline=timeline_payload,
    )

    if bundle.match.match_id != match_id:
        raise ValueError("Normalized match ID mismatch")

    if bundle.match.source_match_sha256 != match_record.sha256:
        raise ValueError("Normalizer match SHA does not match raw manifest")

    if bundle.match.source_timeline_sha256 != timeline_record.sha256:
        raise ValueError("Normalizer timeline SHA does not match raw manifest")

    match_output_root = output_root / match_id

    normalized_match_path = match_output_root / "match.json"

    participant_frames_path = match_output_root / "participant_frames.jsonl"

    events_path = match_output_root / "events.jsonl"

    manifest_path = match_output_root / "manifest.json"

    match_bytes = canonical_json_bytes(bundle.match.model_dump(mode="json"))

    ordered_frames = tuple(
        sorted(
            bundle.participant_frames,
            key=lambda item: (
                item.frame_index,
                item.participant_id,
            ),
        )
    )

    frame_bytes = _jsonl_bytes(ordered_frames)

    ordered_events = tuple(
        sorted(
            bundle.events,
            key=lambda item: (item.sequence,),
        )
    )

    event_bytes = _jsonl_bytes(ordered_events)

    changed = False

    changed |= _write_if_changed(
        path=normalized_match_path,
        payload=match_bytes,
    )

    changed |= _write_if_changed(
        path=participant_frames_path,
        payload=frame_bytes,
    )

    changed |= _write_if_changed(
        path=events_path,
        payload=event_bytes,
    )

    files = (
        _artifact_file(
            repository_root=(repository_root),
            artifact_type=(NormalizedArtifactType.MATCH),
            path=normalized_match_path,
            payload=match_bytes,
            record_count=1,
        ),
        _artifact_file(
            repository_root=(repository_root),
            artifact_type=(NormalizedArtifactType.PARTICIPANT_FRAMES),
            path=participant_frames_path,
            payload=frame_bytes,
            record_count=len(ordered_frames),
        ),
        _artifact_file(
            repository_root=(repository_root),
            artifact_type=(NormalizedArtifactType.EVENTS),
            path=events_path,
            payload=event_bytes,
            record_count=len(ordered_events),
        ),
    )

    manifest = NormalizedMatchArtifactManifest(
        normalizer_version=(RIOT_NORMALIZER_VERSION),
        match_id=match_id,
        source_match_file_path=(match_record.file_path),
        source_timeline_file_path=(timeline_record.file_path),
        source_match_sha256=(match_record.sha256),
        source_timeline_sha256=(timeline_record.sha256),
        participant_count=len(bundle.match.participants),
        participant_frame_count=len(ordered_frames),
        event_count=len(ordered_events),
        files=files,
    )

    manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))

    changed |= _write_if_changed(
        path=manifest_path,
        payload=manifest_bytes,
    )

    reference = NormalizedMatchReference(
        match_id=match_id,
        manifest_path=_relative_path(
            repository_root=(repository_root),
            path=manifest_path,
        ),
        manifest_sha256=(_sha256_bytes(manifest_bytes)),
        participant_count=(manifest.participant_count),
        participant_frame_count=(manifest.participant_frame_count),
        event_count=(manifest.event_count),
    )

    return (
        reference,
        changed,
    )


def build_normalized_riot_dataset(
    *,
    repository_root: Path,
    match_ids: (tuple[str, ...] | None) = None,
) -> NormalizedDatasetBuildResult:
    raw_manifest_path = repository_root / "data" / "manifests" / "riot_api" / "raw_sources.jsonl"

    raw_records = _load_raw_manifest(raw_manifest_path)

    available_match_ids = tuple(
        sorted(
            record.match_id
            for record in (raw_records.values())
            if (record.resource_type == RiotRawResourceType.MATCH and record.match_id is not None)
        )
    )

    if match_ids is None:
        selected_match_ids = available_match_ids

    else:
        selected_match_ids = tuple(
            dict.fromkeys(match_id.strip() for match_id in match_ids if match_id.strip())
        )

    if not selected_match_ids:
        raise ValueError("No Riot matches selected")

    output_root = repository_root / "data" / "processed" / "riot_api" / NORMALIZED_DATASET_VERSION

    references: list[NormalizedMatchReference] = []

    built_count = 0
    reused_count = 0

    for match_id in selected_match_ids:
        reference, changed = _build_one_match(
            repository_root=(repository_root),
            output_root=output_root,
            raw_records=raw_records,
            match_id=match_id,
        )

        references.append(reference)

        if changed:
            built_count += 1

        else:
            reused_count += 1

    ordered_references = tuple(
        sorted(
            references,
            key=lambda item: item.match_id,
        )
    )

    aggregate = NormalizedDatasetManifest(
        normalizer_version=(RIOT_NORMALIZER_VERSION),
        match_count=len(ordered_references),
        total_participant_count=sum(item.participant_count for item in ordered_references),
        total_participant_frame_count=sum(
            item.participant_frame_count for item in ordered_references
        ),
        total_event_count=sum(item.event_count for item in ordered_references),
        matches=ordered_references,
    )

    aggregate_path = output_root / "manifest.json"

    aggregate_bytes = canonical_json_bytes(aggregate.model_dump(mode="json"))

    aggregate_changed = _write_if_changed(
        path=aggregate_path,
        payload=aggregate_bytes,
    )

    return NormalizedDatasetBuildResult(
        built_match_count=(built_count),
        reused_match_count=(reused_count),
        aggregate_manifest_changed=(aggregate_changed),
        manifest=aggregate,
    )
