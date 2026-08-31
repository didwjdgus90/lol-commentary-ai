from hashlib import sha256

from lol_commentary_backend.retrieval.chunks.models import (
    ChunkStrategy,
    PatchRagChunk,
)
from lol_commentary_backend.retrieval.documents.models import (
    PatchRagDocument,
)

PATCH_NOTE_CHUNKER_VERSION = "0.1.0"
DEFAULT_MAX_CHUNK_CHARS = 600
_MIN_MAX_CHUNK_CHARS = 200


def _context_prefix(document: PatchRagDocument) -> str:
    parts = [f"{document.patch} 패치"]

    if document.entity_name is not None:
        parts.append(document.entity_name)

    parts.append(document.title)

    heading = " > ".join(document.heading_path)

    return f"{' | '.join(parts)}\n\n{heading}"


def _semantic_blocks(
    document: PatchRagDocument,
) -> list[str]:
    blocks: list[str] = []

    blocks.extend(paragraph.strip() for paragraph in document.paragraphs if paragraph.strip())

    blocks.extend(f"- {change.strip()}" for change in document.changes if change.strip())

    return blocks


def _split_long_text(
    text: str,
    *,
    max_chars: int,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()

    if not words:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    parts: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""

            parts.extend(
                word[index : index + max_chars]
                for index in range(
                    0,
                    len(word),
                    max_chars,
                )
            )
            continue

        candidate = word if not current else f"{current} {word}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        parts.append(current)
        current = word

    if current:
        parts.append(current)

    return parts


def _chunk_bodies(
    blocks: list[str],
    *,
    max_body_chars: int,
) -> list[str]:
    expanded: list[str] = []

    for block in blocks:
        expanded.extend(
            _split_long_text(
                block,
                max_chars=max_body_chars,
            )
        )

    bodies: list[str] = []
    current = ""

    for block in expanded:
        candidate = block if not current else f"{current}\n\n{block}"

        if len(candidate) <= max_body_chars:
            current = candidate
            continue

        if current:
            bodies.append(current)

        current = block

    if current:
        bodies.append(current)

    return bodies


def _chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    text: str,
) -> str:
    payload = f"{PATCH_NOTE_CHUNKER_VERSION}\n{document_id}\n{chunk_index}\n{text}"

    return sha256(payload.encode("utf-8")).hexdigest()


def _build_chunk(
    document: PatchRagDocument,
    *,
    text: str,
    chunk_index: int,
    chunk_count: int,
    strategy: ChunkStrategy,
) -> PatchRagChunk:
    return PatchRagChunk(
        chunker_version=PATCH_NOTE_CHUNKER_VERSION,
        chunk_id=_chunk_id(
            document_id=document.document_id,
            chunk_index=chunk_index,
            text=text,
        ),
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        document_id=document.document_id,
        document_content_sha256=(document.content_sha256),
        source_record_id=document.source_record_id,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        char_count=len(text),
        strategy=strategy,
        patch=document.patch,
        locale=document.locale,
        source_url=document.source_url,
        source_sha256=document.source_sha256,
        ddragon_version=document.ddragon_version,
        section_kind=document.section_kind,
        entity_type=document.entity_type,
        entity_name=document.entity_name,
        entity_id=document.entity_id,
        entity_key=document.entity_key,
        entity_resolved=document.entity_resolved,
        resolution_method=document.resolution_method,
        removed_from_target_map=(document.removed_from_target_map),
        target_map_id=document.target_map_id,
        heading_path=list(document.heading_path),
        title=document.title,
        text=text,
    )


def build_patch_rag_chunks(
    document: PatchRagDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[PatchRagChunk]:
    if max_chars < _MIN_MAX_CHUNK_CHARS:
        raise ValueError(f"max_chars must be >= {_MIN_MAX_CHUNK_CHARS}")

    if len(document.retrieval_text) <= max_chars:
        return [
            _build_chunk(
                document,
                text=document.retrieval_text,
                chunk_index=0,
                chunk_count=1,
                strategy=(ChunkStrategy.SINGLE_DOCUMENT),
            )
        ]

    prefix = _context_prefix(document)
    separator = "\n\n"

    max_body_chars = max_chars - len(prefix) - len(separator)

    if max_body_chars <= 0:
        raise ValueError(f"Document context prefix is too large for max_chars={max_chars}")

    blocks = _semantic_blocks(document)

    if not blocks:
        blocks = [document.retrieval_text]

    bodies = _chunk_bodies(
        blocks,
        max_body_chars=max_body_chars,
    )

    texts = [f"{prefix}{separator}{body}" for body in bodies]

    chunk_count = len(texts)

    return [
        _build_chunk(
            document,
            text=text,
            chunk_index=index,
            chunk_count=chunk_count,
            strategy=ChunkStrategy.SEMANTIC_SPLIT,
        )
        for index, text in enumerate(texts)
    ]


def build_patch_rag_chunks_for_documents(
    documents: list[PatchRagDocument],
    *,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[PatchRagChunk]:
    chunks: list[PatchRagChunk] = []

    for document in documents:
        chunks.extend(
            build_patch_rag_chunks(
                document,
                max_chars=max_chars,
            )
        )

    return chunks
