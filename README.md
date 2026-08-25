# LoL Commentary AI

League of Legends 데이터를 기반으로 경기 상황을 분석하고, RAG·머신러닝·LLM을 결합해 AI 해설을 생성하는 개인 프로젝트입니다.

## 목표

```text
Riot / LoL Data
        ↓
Data Ingestion
        ↓
Preprocessing
        ↓
RAG + ML
        ↓
Prompt Engineering
        ↓
LLM
        ↓
AI Commentary
        ↓
FastAPI
        ↓
React
        ↓
Docker / AWS
```

현재는 실제 서비스 개발을 가정해 개발 환경, 코드 품질 검사, 테스트, CI 기반부터 단계적으로 구축하고 있습니다.

## 현재 구현 상태

- Python 3.13
- uv 기반 dependency 관리
- FastAPI backend
- `/health/live`
- `/health/ready`
- Ruff lint / format
- ty type check
- pytest
- pytest-cov
- GitHub Actions CI 구성

현재 Health API 테스트는 2개이며, 작성된 backend 코드 범위에서 테스트 coverage 100%를 확인했습니다.

## Backend 실행

```powershell
cd backend
uv sync --frozen --dev
uv run fastapi dev --port 8001
```

개발 서버 실행 후:

```text
http://127.0.0.1:8001/health/live
http://127.0.0.1:8001/health/ready
http://127.0.0.1:8001/docs
```

를 확인할 수 있습니다.

> 로컬 환경에서 8000번 포트를 다른 서버가 사용할 수 있어 현재 개발 예시는 8001번 포트를 사용합니다.

## 품질 검사

```powershell
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov=lol_commentary_backend --cov-report=term-missing
```

## 예정 기술 스택

### Backend
- Python
- FastAPI
- Pydantic
- PostgreSQL
- pgvector

### Data
- Riot / LoL official data
- Polars
- Parquet
- DuckDB

### RAG
- Dense retrieval
- Sparse/BM25 retrieval
- Metadata filtering
- RRF
- Reranking
- Retrieval evaluation

### Machine Learning
- Logistic Regression baseline
- XGBoost / LightGBM
- Calibration
- SHAP
- MLflow

### LLM
- Prompt Engineering
- Structured output
- Grounding
- RAG context
- LoRA / QLoRA fine-tuning

### Frontend / Infra
- React
- TypeScript
- Docker
- GitHub Actions
- AWS
- Terraform
- OpenTelemetry

## 개발 원칙

1. 단순히 실행되는 코드보다 검증 가능한 코드를 작성합니다.
2. 최신 정보와 패치 정보는 RAG로 관리합니다.
3. Fine-tuning은 최신 사실 암기보다 해설 스타일과 출력 행동 학습에 사용합니다.
4. 데이터 원본과 전처리 결과를 분리합니다.
5. 재현 가능한 dependency와 데이터 lineage를 관리합니다.
6. 기능 추가 전에 테스트와 품질 검사를 함께 고려합니다.
7. 초기에는 modular monolith 구조를 유지하고, 필요할 때만 분리합니다.

## 개발 로드맵

- [x] Python / uv 환경 구축
- [x] FastAPI backend 초기화
- [x] Health Check API
- [x] Ruff / ty / pytest
- [x] Test coverage 확인
- [x] GitHub Actions CI 구성
- [ ] Riot Patch Notes 수집
- [ ] Raw data 저장 및 metadata/hash 관리
- [ ] Patch Notes parsing
- [ ] Data Dragon 정규화
- [ ] RAG baseline
- [ ] RAG evaluation
- [ ] Match / Timeline ingestion
- [ ] Win probability ML
- [ ] Prompt Engineering
- [ ] LLM commentary
- [ ] Fine-tuning
- [ ] React frontend
- [ ] Docker
- [ ] AWS deployment
- [ ] Observability

## License

License는 프로젝트 공개 범위와 Riot 데이터/콘텐츠 사용 조건을 검토한 뒤 결정할 예정입니다.
