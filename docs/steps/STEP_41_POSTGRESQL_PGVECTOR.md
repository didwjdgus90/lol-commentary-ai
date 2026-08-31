# Step 41 — PostgreSQL + pgvector Persistent Dense Retrieval

## 1. Step 목표

Step 41의 목표는 기존 Production RetrievalService에서 사용하던 **메모리 기반 NumPy Dense Search**를 **PostgreSQL + pgvector 기반 Persistent Vector Search**로 교체하는 것이다.

기존 구조는 애플리케이션이 시작될 때마다 185개의 RAG chunk 전체를 BGE-M3로 다시 embedding한 뒤 NumPy 배열을 메모리에 올려 검색했다.

기존 구조:

```text
Application Start
    ↓
185 PatchRagChunk 로드
    ↓
BGE-M3 corpus embedding
    ↓
NumPy matrix
    ↓
Query embedding
    ↓
Matrix similarity search
```

Step 41 이후 구조:

```text
Offline / Indexing
185 PatchRagChunk
    ↓
BGE-M3 embedding
    ↓
PostgreSQL vector(1024)에 영구 저장

Runtime
Query
    ↓
BGE-M3 query embedding
    ↓
PostgreSQL / pgvector
    ↓
Exact cosine search
    ↓
Dense candidates
    ↓
Alias BM25
    ↓
RRF
```

핵심은 **검색 알고리즘 자체를 바꾸는 것이 아니라 Dense embedding 저장소를 메모리에서 PostgreSQL로 교체하는 것**이다.

---

# 2. Step 시작 전 상태

Step 39에서 Production Retrieval Baseline v2를 고정했다.

Primary retrieval:

```text
BGE-M3
+
Bilingual Alias BM25
+
Reciprocal Rank Fusion
```

Strategy key:

```text
rrf_bge_m3_alias_bm25
```

Fast fallback:

```text
alias_bm25
```

현재 frozen corpus:

```text
Patch: 26.1
Locale: ko_kr
Chunks: 185
```

Frozen corpus SHA-256:

```text
85d1c93a86fdc54a8ea8ea79cd33b62df62ab39464a7dc70e848db369ef20a41
```

Dense model:

```text
BAAI/bge-m3
```

Embedding dimension:

```text
1024
```

Corpus/query embedding:

```text
normalize_embeddings=True
```

---

# 3. 기존 구조의 문제

기존 `BgeDenseIndex`는 애플리케이션 시작 시 다음 작업을 수행했다.

```text
chunks.jsonl
    ↓
185개의 chunk.text
    ↓
SentenceTransformer.encode()
    ↓
185 × 1024 NumPy matrix
```

CPU 환경에서는 corpus embedding에 약 50~65초가 걸렸다.

문제점은 다음과 같다.

### 3.1 서버 재시작 비용

서버를 재시작할 때마다 동일한 corpus를 다시 embedding한다.

하지만 Patch 26.1 chunk가 변경되지 않았다면 이 계산은 매번 반복할 필요가 없다.

### 3.2 프로세스 메모리에 검색 데이터를 의존

Embedding corpus가 Python 프로세스 메모리에만 존재한다.

따라서 프로세스가 종료되면 embedding도 사라진다.

### 3.3 향후 corpus 확장 문제

현재는 185 chunks이지만 이후 여러 patch를 누적하면 수천~수만 개의 embedding을 관리해야 한다.

Persistent vector storage가 필요하다.

---

# 4. 기술 선택

## 4.1 PostgreSQL 18

개발 환경은 PostgreSQL 18을 사용했다.

기존 Windows PostgreSQL과 충돌하지 않도록 Docker PostgreSQL은 host port `5433`을 사용한다.

```text
Host localhost:5433
    ↓
Docker PostgreSQL:5432
```

---

## 4.2 pgvector 0.8.6

Docker image:

```text
pgvector/pgvector:0.8.6-pg18-bookworm
```

확인된 extension:

```text
vector | 0.8.6
```

pgvector를 사용하면 PostgreSQL에 다음 컬럼을 저장할 수 있다.

```sql
embedding VECTOR(1024)
```

그리고 cosine distance 검색에 다음 연산자를 사용할 수 있다.

```sql
<=>
```

---

## 4.3 Psycopg 3

Python dependency:

```text
psycopg==3.3.4
psycopg-binary==3.3.4
psycopg-pool==3.3.1
pgvector==0.5.0
```

ORM이 필요한 단계가 아니므로 SQLAlchemy는 도입하지 않았다.

현재 요구사항은 다음 정도다.

```text
connection pool
SQL 실행
vector parameter 전달
transaction
upsert
```

따라서 Psycopg 3를 직접 사용하는 편이 구조가 단순하다.

---

# 5. 왜 HNSW를 사용하지 않았는가

Step 41에서는 HNSW 또는 IVFFlat ANN index를 만들지 않았다.

현재 corpus:

```text
185 chunks
```

뿐이기 때문이다.

이번 Step의 핵심 목적은:

```text
기존 NumPy exact search
        ↓
pgvector exact search
```

의 동등성을 증명하는 것이다.

Approximate Nearest Neighbor index를 먼저 사용하면 storage 변경과 검색 알고리즘 변경이 동시에 발생한다.

그러면 결과 차이가 발생했을 때 원인을 분리하기 어렵다.

따라서 현재는 exact cosine search를 사용한다.

향후 corpus가 커지면 별도 benchmark를 통해 다음을 비교해야 한다.

```text
Exact
vs
HNSW

Recall
Latency
Memory
Index build time
```

---

# 6. Docker 환경 구성

Repository root:

```text
compose.pgvector.yml
```

주요 구성:

```text
PostgreSQL 18
pgvector 0.8.6
host port 5433
named volume
healthcheck
```

PostgreSQL 18 Docker volume은 다음 위치를 사용한다.

```text
/var/lib/postgresql
```

최종 container 상태:

```text
lol-commentary-pgvector
Up
healthy
```

---

# 7. pgvector extension 검증

실제 DB에서:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

를 수행했다.

확인 결과:

```text
extname | extversion
vector  | 0.8.6
```

Vector cosine 연산 smoke:

```sql
SELECT
    '[1,0,0]'::vector(3)
    <=>
    '[1,0,0]'::vector(3);
```

결과:

```text
0
```

동일한 vector의 cosine distance가 0으로 정상 계산됐다.

---

# 8. DB Schema

Step 41에서는 데이터를 세 계층으로 분리했다.

```text
retrieval_corpora
        ↓
rag_chunks
        ↓
rag_chunk_embeddings
```

---

## 8.1 retrieval_corpora

하나의 retrieval corpus 버전을 관리한다.

핵심 key:

```text
corpus_sha256
```

저장 정보:

```text
patch
locale
chunker_version
chunk_count
created_at
```

Corpus SHA가 달라지면 retrieval corpus의 실제 내용이 달라졌다는 뜻이다.

---

## 8.2 rag_chunks

실제 `PatchRagChunk` metadata와 text를 저장한다.

예:

```text
chunk_id
content_sha256
document_id
patch
locale
entity information
heading_path
title
chunk_text
```

Primary key:

```text
(corpus_sha256, chunk_id)
```

같은 chunk ID라도 서로 다른 corpus lineage를 구분할 수 있게 했다.

---

## 8.3 rag_chunk_embeddings

Embedding은 chunk와 별도 테이블로 분리했다.

이유는 chunk와 embedding model의 생명주기가 다르기 때문이다.

하나의 chunk에 앞으로 다음처럼 여러 embedding을 가질 수 있다.

```text
chunk
 ├─ BGE-M3
 ├─ future model A
 └─ future model B
```

Primary key:

```text
(
    corpus_sha256,
    chunk_id,
    embedding_model_id
)
```

현재 embedding:

```text
BAAI/bge-m3
vector(1024)
normalized=True
```

---

# 9. Corpus Ingestion Pipeline

파일:

```text
backend/scripts/index_pgvector_corpus.py
```

Storage implementation:

```text
backend/src/lol_commentary_backend/retrieval/storage/
```

Pipeline:

```text
chunks.jsonl
    ↓
file_sha256()
    ↓
Frozen baseline SHA 확인
    ↓
185 PatchRagChunk 로드
    ↓
BGE-M3
    ↓
185 × 1024 normalized embeddings
    ↓
PostgreSQL transaction
    ↓
retrieval_corpora
rag_chunks
rag_chunk_embeddings
```

---

# 10. Data Lineage Guard

DB에 데이터를 넣기 전에 현재 chunk 파일 SHA를 frozen baseline과 비교한다.

```text
current corpus SHA
        ==
retrieval_baseline_v2 corpus SHA
```

다르면 ingestion을 중단한다.

목적:

```text
평가하지 않은 새로운 corpus
        ↓
실수로 production vector DB 적재
```

를 방지하는 것이다.

---

# 11. Embedding Validation

DB 적재 전 다음을 검사한다.

```text
2D matrix 여부
row count == chunk count
dimension == 1024
NaN 없음
Inf 없음
normalized=True이면 L2 norm ≈ 1
```

즉 잘못된 embedding matrix를 데이터베이스에 저장하지 않는다.

---

# 12. 실제 적재 결과

최종 DB row count:

```text
retrieval_corpora      1
rag_chunks             185
rag_chunk_embeddings   185
```

즉 현재 frozen corpus 185개 모두에 BGE-M3 vector가 존재한다.

---

# 13. PgVectorDenseIndex

신규 runtime:

```text
backend/src/lol_commentary_backend/retrieval/runtime/pgvector_dense.py
```

역할:

```text
Query
    ↓
BGE-M3
    ↓
normalized 1024 vector
    ↓
PostgreSQL
    ↓
embedding <=> query
    ↓
cosine distance ASC
    ↓
Top-N
```

SQL 개념:

```sql
ORDER BY embedding <=> query_vector ASC
LIMIT top_n
```

---

# 14. Distance와 기존 score 계약

pgvector `<=>` 결과는 cosine **distance**다.

즉:

```text
작을수록 좋음
```

기존 DenseRetriever score는:

```text
클수록 좋음
```

계약이었다.

따라서 runtime에서는:

```text
score = 1 - cosine_distance
```

로 변환한다.

이렇게 하면 기존 `RetrievalService` 상위 계층의 계약을 변경하지 않아도 된다.

---

# 15. Connection Pool

Runtime DB 연결에는 Psycopg `ConnectionPool`을 사용한다.

구조:

```text
RetrievalService
    ↓
PgVectorDenseIndex
    ↓
ConnectionPool
    ├─ connection
    ├─ connection
    └─ ...
```

매 query마다 PostgreSQL connection을 새로 만드는 비용을 피하기 위한 구조다.

서비스 종료 시:

```text
RetrievalService.close()
    ↓
PgVectorDenseIndex.close()
    ↓
ConnectionPool.close()
```

가 실행된다.

`close()`는 idempotent하게 설계해 여러 번 호출돼도 실제 resource 종료는 한 번만 수행한다.

---

# 16. 가장 중요한 검증 — NumPy ↔ pgvector Parity

Storage backend 변경에서 가장 중요한 검증은 검색 결과가 실제로 동일한지 확인하는 것이다.

비교 대상:

```text
LEFT
BGE-M3
→ NumPy exact search

RIGHT
BGE-M3
→ PostgreSQL pgvector exact cosine
```

동일한 모델 객체와 동일한 corpus를 사용했다.

Gold Query:

```text
24 queries
```

Top-N:

```text
10
```

실제 결과:

```text
Exact Top-N order:        24/24
Same Top-N candidate set: 24/24

Maximum score delta:
0.0000001422

Allowed tolerance:
0.0000100000

Score tolerance PASS:
True

Ranking mismatches:
[]

Candidate mismatches:
[]
```

즉 24개 query 모두 Top-10 순서까지 완전히 동일했다.

---

# 17. Dense Retrieval Metrics Parity

NumPy:

```text
Hit@1     0.7500
Hit@3     0.8333
Recall@5  0.8750
MRR       0.8067
```

pgvector:

```text
Hit@1     0.7500
Hit@3     0.8333
Recall@5  0.8750
MRR       0.8067
```

결과:

```text
Metrics match: True
PGVECTOR_DENSE_PARITY=PASS
```

따라서 Step 41에서는 다음을 증명했다.

> Storage backend를 NumPy에서 PostgreSQL + pgvector로 변경했지만 Dense retrieval ranking과 평가 metric은 변경되지 않았다.

---

# 18. Production Factory 전환

기존 factory:

```text
BgeDenseIndex.build()
```

변경 후:

```text
PgVectorDenseIndex.build()
```

상위 `RetrievalService`는 변경된 storage 구현을 직접 알 필요가 없다.

기존 Protocol:

```text
DenseRetriever.search()
```

계약을 그대로 사용하기 때문이다.

구조:

```text
RetrievalService
        ↓
DenseRetriever Protocol
        ↓
PgVectorDenseIndex
```

이는 storage implementation을 application logic에서 분리한 구조다.

---

# 19. Production Runtime Smoke

Query:

```text
Essence Reaver AD nerf hotfix
```

Alias expansion:

```text
Essence Reaver AD nerf hotfix | 정수 약탈자
```

Primary 결과:

```text
Requested mode:
primary

Strategy:
rrf_bge_m3_alias_bm25

Primary available:
True

Fallback used:
False
```

실제 top result:

```text
정수 약탈자
```

내용에는 hotfix 변경:

```text
총가격 2,900 → 3,050
공격력 55 → 50
```

이 포함됐다.

Smoke:

```text
RETRIEVAL_SERVICE_SMOKE=PASS
```

---

# 20. Startup 동작 변화

기존 Production startup에서는 185 corpus embedding 때문에 다음과 같은 batch 진행이 나타났다.

```text
Batches 24/24
```

pgvector factory 전환 후 production primary startup에서는 이 corpus embedding 과정이 사라졌다.

현재 startup은 주로:

```text
BGE-M3 model load
DB connection pool
stored embedding validation
```

으로 구성된다.

최근 smoke:

```text
Startup ≈ 11.0 seconds
Retrieval ≈ 355 ms
```

CPU 개발 환경 측정값이므로 production SLA로 해석하면 안 된다.

하지만 corpus 전체 embedding이 runtime startup에서 제거됐다는 것은 확인됐다.

---

# 21. Fallback 독립성

DB URL을 제거한 상태에서도:

```text
mode=fallback
```

은 정상 동작했다.

결과:

```text
Strategy:
alias_bm25

Primary available:
False

Fallback used:
False

RETRIEVAL_SERVICE_SMOKE=PASS
```

즉 PostgreSQL 장애 또는 Dense backend 사용 불가 상황에서도 명시적인 BM25 fallback 자체는 DB에 의존하지 않는다.

---

# 22. Environment / Secret 관리

실제 로컬 설정:

```text
.env
```

공유용 설정 명세:

```text
.env.example
```

Git policy:

```text
.env
.env.*
```

는 ignore한다.

단:

```text
!.env.example
```

로 `.env.example`은 Git에 포함한다.

검증 결과:

```text
ENV_IGNORED=True
ENV_EXAMPLE_TRACKABLE=True
```

또한 실제 개발 비밀번호 문자열에 대해:

```text
git grep
```

결과가 없음을 확인했다.

즉 tracked 파일에 실제 비밀번호가 포함되지 않았다.

---

# 23. load_dev_env.ps1

새 PowerShell을 열 때마다 다음을 직접 작성할 필요가 없어졌다.

```text
LOL_POSTGRES_PASSWORD
LOL_DATABASE_URL
```

대신:

```text
.env
    ↓
load_dev_env.ps1
    ↓
LOL_POSTGRES_*
LOL_DATABASE_URL
```

로 현재 Process 환경을 구성한다.

검증:

```text
POWERSHELL_PARSE=PASS

DATABASE_URL_SET=True
POSTGRES_PASSWORD_SET=True

ENV_DB_CONNECTION=PASS
```

또한 `.env`가 기본값:

```text
LOL_POSTGRES_PASSWORD=change-me
```

인 경우 script가 의도적으로 실행을 차단하는 것도 확인했다.

---

# 24. Compose Environment Contract

Compose는 `.env`를 사용한다.

검증:

```text
COMPOSE_CONFIG_EXIT=0
```

Container:

```text
healthy
```

DB:

```text
corpora     1
chunks      185
embeddings  185
```

---

# 25. 테스트

Step 41-7 시점 전체 Python regression:

```text
163 passed
```

추가된 주요 테스트:

```text
test_pgvector_ingestion.py
test_pgvector_dense.py
test_pgvector_parity.py
test_retrieval_service_lifecycle.py
```

검증 범위:

```text
Corpus manifest
Embedding validation
Cosine distance → score conversion
Dense parity comparison
Resource lifecycle
기존 RetrievalService regression
```

정적 검사:

```text
ruff check
PASS

ruff format --check
PASS

ty check
PASS
```

---

# 26. 발생한 주요 오류와 해결

## 26.1 Docker Engine 미실행

오류:

```text
dockerDesktopLinuxEngine
The system cannot find the file specified
```

원인:

```text
Docker Desktop Engine이 실행되지 않음
```

해결:

```text
Docker Desktop 시작
docker info readiness 확인
```

---

## 26.2 PostgreSQL 18 Docker volume 경로

초기:

```text
/var/lib/postgresql/data
```

대신 PostgreSQL 18용:

```text
/var/lib/postgresql
```

로 수정했다.

---

## 26.3 PowerShell Python quoting

한 줄 Python 명령에서 SQL quote가 깨져:

```text
SyntaxError
```

가 발생했다.

해결:

```text
PowerShell here-string
    ↓
uv run python -
```

형식으로 변경했다.

---

## 26.4 LOL_DATABASE_URL 누락

새 PowerShell에서는 Process 환경변수가 사라져:

```text
RuntimeError:
LOL_DATABASE_URL is not set
```

이 발생했다.

이 문제를 해결하기 위해:

```text
.env
+
load_dev_env.ps1
```

구조를 만들었다.

---

## 26.5 load_dev_env PowerShell Parser Error

초기 조건:

```text
value
-eq
"change-me"
```

형태가 PowerShell parser에서 실패했다.

조건식을 한 줄의 명확한 expression으로 변경해 해결했다.

최종:

```text
POWERSHELL_PARSE=PASS
```

---

# 27. 현재 Production Retrieval Architecture

최종 구조:

```text
User Query
    ↓
RetrievalService
    │
    ├─────────────────────────────┐
    ↓                             ↓
PgVectorDenseIndex          AliasBm25Index
    ↓                             ↓
BGE-M3 Query Embedding      bilingual alias expansion
    ↓                             ↓
PostgreSQL pgvector              BM25
    ↓                             ↓
Dense Top-N                 Sparse Top-N
    └──────────────┬──────────────┘
                   ↓
                  RRF
                   ↓
             Retrieval Hits
```

Storage:

```text
PostgreSQL 18
    ↓
pgvector 0.8.6
    ↓
retrieval_corpora
rag_chunks
rag_chunk_embeddings
```

---

# 28. Verified Facts / Design Choices / Future Work

## Verified Facts

다음은 실제 실행 결과로 검증됐다.

```text
Docker PostgreSQL healthy
pgvector extension 0.8.6
vector(1024)
185 chunks stored
185 embeddings stored
24/24 exact dense ranking parity
Dense metrics identical
163 Python tests passed
Primary RetrievalService smoke passed
Fallback without DB URL passed
.env ignored
.env.example trackable
Compose config valid
Psycopg DB connection passed
```

---

## Design Choices

다음은 의도적으로 선택한 설계다.

```text
Psycopg 3 direct SQL
SQLAlchemy 미사용
Exact cosine search
HNSW 미사용
corpus / chunk / embedding 분리
ConnectionPool 사용
Frozen corpus SHA gate
.env secret 분리
```

---

## 아직 검증하지 않은 것

다음은 Step 41 범위 밖이다.

```text
HNSW 성능
수천~수만 chunk scalability
PostgreSQL production hosting
AWS RDS/Aurora 선택
connection pool production sizing
FastAPI lifespan integration
multi-patch corpus routing
embedding cache
GPU inference server
query embedding service 분리
```

이 항목들은 이후 단계에서 별도 benchmark와 설계가 필요하다.

---

# 29. Step 41 최종 판단

Step 41에서 목표했던:

```text
Startup corpus embedding 제거
Persistent vector storage
pgvector exact dense retrieval
기존 ranking 보존
Production RetrievalService integration
Fallback 보존
Configuration / secret separation
```

을 모두 구현했다.

가장 중요한 regression evidence:

```text
24 / 24 exact Top-10 order parity
Maximum score delta = 0.0000001422
Metrics match = True
```

따라서:

```text
STEP_41_POSTGRESQL_PGVECTOR=VERIFIED_COMPLETE
```

로 판단한다.

---

# 30. 다음 단계

다음 단계부터 Retrieval 결과를 실제 LLM에게 넘기기 위한 상위 RAG 계층으로 이동할 수 있다.

주요 후보:

```text
Context Builder
    ↓
retrieval 결과 정제
중복 제거
score / candidate filtering
token budget
evidence formatting
    ↓
Prompt Builder
    ↓
LLM commentary generation
```

특히 현재 Dense 검색이 일부 일반적인 "버그 수정" chunk를 높은 순위로 가져오는 사례가 존재한다.

이 문제는 frozen retrieval baseline을 즉시 변경하기보다 다음 상위 계층에서:

```text
candidate filtering
context selection
entity-aware selection
token budget
```

을 설계하면서 다루는 것이 안전하다.

Step 41은 Storage/Runtime 전환 단계였으며 retrieval baseline 자체는 보존했다.
