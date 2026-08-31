BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 1. RETRIEVAL CORPUS
-- ============================================================
--
-- 하나의 검색 corpus 버전을 표현한다.
--
-- corpus_sha256이 다르면
-- 검색 대상 chunk 파일 자체가 달라졌다는 의미다.
--

CREATE TABLE IF NOT EXISTS retrieval_corpora (
    corpus_sha256 TEXT PRIMARY KEY,

    patch TEXT NOT NULL,
    locale TEXT NOT NULL,

    chunker_version TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT retrieval_corpora_sha256_format
        CHECK (corpus_sha256 ~ '^[0-9a-f]{64}$'),

    CONSTRAINT retrieval_corpora_chunk_count_positive
        CHECK (chunk_count > 0)
);


-- ============================================================
-- 2. RAG CHUNKS
-- ============================================================
--
-- 실제 PatchRagChunk의 text와 metadata를 저장한다.
--
-- embedding은 이 테이블에 넣지 않는다.
-- chunk와 embedding의 생명주기가 다르기 때문이다.
--

CREATE TABLE IF NOT EXISTS rag_chunks (
    corpus_sha256 TEXT NOT NULL,

    schema_version SMALLINT NOT NULL,
    chunker_version TEXT NOT NULL,

    chunk_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,

    document_id TEXT NOT NULL,
    document_content_sha256 TEXT NOT NULL,
    source_record_id TEXT NOT NULL,

    chunk_index INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,

    strategy TEXT NOT NULL,

    patch TEXT NOT NULL,
    locale TEXT NOT NULL,

    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,

    ddragon_version TEXT NOT NULL,

    section_kind TEXT NOT NULL,

    entity_type TEXT NOT NULL,
    entity_name TEXT,
    entity_id TEXT,
    entity_key TEXT,

    entity_resolved BOOLEAN NOT NULL,

    resolution_method TEXT NOT NULL,

    removed_from_target_map BOOLEAN NOT NULL,
    target_map_id TEXT,

    heading_path TEXT[] NOT NULL,

    title TEXT NOT NULL,
    chunk_text TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        corpus_sha256,
        chunk_id
    ),

    CONSTRAINT rag_chunks_corpus_fk
        FOREIGN KEY (corpus_sha256)
        REFERENCES retrieval_corpora (corpus_sha256)
        ON DELETE CASCADE,

    CONSTRAINT rag_chunks_chunk_id_format
        CHECK (chunk_id ~ '^[0-9a-f]{64}$'),

    CONSTRAINT rag_chunks_content_sha256_format
        CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),

    CONSTRAINT rag_chunks_document_id_format
        CHECK (document_id ~ '^[0-9a-f]{64}$'),

    CONSTRAINT rag_chunks_document_content_sha256_format
        CHECK (
            document_content_sha256
            ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT rag_chunks_source_record_id_format
        CHECK (
            source_record_id
            ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT rag_chunks_source_sha256_format
        CHECK (
            source_sha256
            ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT rag_chunks_chunk_index_nonnegative
        CHECK (chunk_index >= 0),

    CONSTRAINT rag_chunks_chunk_count_positive
        CHECK (chunk_count > 0),

    CONSTRAINT rag_chunks_char_count_positive
        CHECK (char_count > 0),

    CONSTRAINT rag_chunks_heading_path_nonempty
        CHECK (cardinality(heading_path) > 0)
);


-- ============================================================
-- 3. RAG CHUNK EMBEDDINGS
-- ============================================================
--
-- 같은 chunk라도 embedding model은 바뀔 수 있다.
--
-- 따라서 embedding을 별도 테이블로 분리한다.
--
-- 현재 frozen production baseline:
--
-- BAAI/bge-m3
-- dimension = 1024
-- normalize_embeddings = True
--

CREATE TABLE IF NOT EXISTS rag_chunk_embeddings (
    corpus_sha256 TEXT NOT NULL,
    chunk_id TEXT NOT NULL,

    embedding_model_id TEXT NOT NULL,

    embedding_dimension INTEGER NOT NULL,
    normalized BOOLEAN NOT NULL,

    embedding VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        corpus_sha256,
        chunk_id,
        embedding_model_id
    ),

    CONSTRAINT rag_chunk_embeddings_chunk_fk
        FOREIGN KEY (
            corpus_sha256,
            chunk_id
        )
        REFERENCES rag_chunks (
            corpus_sha256,
            chunk_id
        )
        ON DELETE CASCADE,

    CONSTRAINT rag_chunk_embeddings_dimension
        CHECK (embedding_dimension = 1024)
);


-- ============================================================
-- 4. NORMAL LOOKUP INDEXES
-- ============================================================
--
-- 아직 HNSW는 만들지 않는다.
--
-- 현재 corpus가 185 chunks뿐이므로
-- 먼저 exact vector search를 기준으로 검증한다.
--

CREATE INDEX IF NOT EXISTS idx_rag_chunks_patch_locale
    ON rag_chunks (
        patch,
        locale
    );

CREATE INDEX IF NOT EXISTS idx_rag_chunks_entity_key
    ON rag_chunks (
        corpus_sha256,
        entity_key
    );

CREATE INDEX IF NOT EXISTS idx_rag_embeddings_model
    ON rag_chunk_embeddings (
        corpus_sha256,
        embedding_model_id
    );


COMMIT;