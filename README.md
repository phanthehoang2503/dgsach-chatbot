# BookStreet RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot and REST API for querying information about **Đường Sách (Book Street)** and its book catalog, built with **FastAPI**, **PostgreSQL (`pgvector`)**, and **Ollama / Google Gemini**.

## Features

- **Vector Search with `pgvector`**: Cosine similarity search (`<=>`) with HNSW/vector indexing and configurable `MAX_DISTANCE` threshold filtering.
- **Pluggable Embeddings**: Supports local embeddings via **Ollama** (`qwen3-embedding:0.6b` by default) or cloud embeddings via **Google Gemini**, with embedding fingerprint validation to prevent index/model mismatches.
- **Zero-Downtime Data Ingestion**: Stages embeddings in a temporary table and atomically swaps into `documents` (`src/migration.py`).
- **Multiple Interfaces**:
  - **REST API** (`api.py`) built with FastAPI, including health check, chat endpoint, and API-key-protected document management.
  - **Interactive CLI** (`main.py`) for local terminal Q&A.
  - **Evaluation Suite** (`evaluate.py`) for retrieval hit-rate and answer quality benchmarking.
- **Safe Integration Testing**: Built-in guardrails (`tests/conftest.py`) to prevent accidental destruction of non-test databases.

## Tech Stack

- **Language**: Python >= 3.11
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Web Framework**: FastAPI + Uvicorn
- **Database**: PostgreSQL 16 + `pgvector` (`psycopg` v3)
- **LLM & Embeddings**: Ollama, Google GenAI SDK (`google-genai`)
- **Testing & Linting**: `pytest`, `httpx`, `ruff`

## Project Structure

```text
.
├── api.py                        # FastAPI application & endpoints
├── main.py                       # Interactive CLI chatbot
├── ingest.py                     # Data ingestion & document management
├── evaluate.py                   # Retrieval & answer evaluation script
├── init.sql                      # PostgreSQL pgvector extension init script
├── docker-compose.example.yml    # Example Docker Compose (pgvector + Adminer)
├── .env.example                  # Environment variable template
├── pyproject.toml                # Project metadata & dependencies
├── data/
│   └── data.txt                  # Source knowledge base (one document per line)
├── src/
│   ├── database.py               # DB connection & schema initialization
│   ├── embeddings.py             # Ollama & Gemini embedder implementations
│   ├── llm.py                    # Prompting & LLM answer generation
│   ├── migration.py              # Atomic staging-to-production table swap
│   └── retriever.py              # Vector search & fingerprint verification
└── tests/
    ├── conftest.py               # Test database safety checks
    ├── unit/                     # Unit tests (mocked DB/AI providers)
    ├── integration/              # Integration tests (requires test DB)
    └── e2e/                      # End-to-end tests
```

## Prerequisites

- **Python** `>= 3.11`
- **uv** (`pip install uv` or via official installer)
- **Docker & Docker Compose** (for PostgreSQL + `pgvector`)
- **Ollama** (running locally, if using local LLM/embeddings) and/or a **Gemini API key**

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd AIC_prj_1
uv sync
```

### 2. Configure Environment Variables

Copy the example environment file and update the values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5434` | PostgreSQL port |
| `DB_NAME` | `vectordb` | Target database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASS` | — | Database password |
| `TEST_DB_NAME` | — | Optional dedicated test database name (must contain `test`) |
| `ALLOW_DESTRUCTIVE_TESTS` | `false` | Set `true` or `1` to bypass test DB name check |
| `MAX_DISTANCE` | `0.5` | Maximum cosine distance threshold for retrieved chunks |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider (`ollama` or `gemini`) |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |
| `VECTOR_DIM` | `1024` | Embedding vector dimensionality |
| `LLM_MODEL` | `qwen3:4b` | Ollama chat model for answer generation |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature for LLM |
| `GEMINI_API_KEY` | — | Google Gemini API key (required when `EMBEDDING_PROVIDER=gemini`) |
| `ADMIN_API_KEY` | — | Secret key for `x-api-key` header on `/api/v1/documents` |
| `OLLAMA_TIMEOUT` | `60` | Ollama request timeout in seconds (chat and embeddings) |
| `GEMINI_TIMEOUT_MS` | `30000` | Gemini HTTP timeout in milliseconds |
| `DB_CONNECT_TIMEOUT` | `5` | PostgreSQL connection timeout in seconds |
| `DB_STATEMENT_TIMEOUT_MS` | `10000` | PostgreSQL statement timeout in ms for API queries |
| `CHAT_RATE_LIMIT` | `10/minute` | Rate limit per IP on `POST /api/v1/chat` |
| `LLM_MAX_CONCURRENCY` | `4` | Maximum concurrent LLM calls |

### 3. Start PostgreSQL & pgvector

Copy `docker-compose.example.yml` to `docker-compose.yml` (adjust credentials if needed) and start the services:

```bash
cp docker-compose.example.yml docker-compose.yml
docker compose up -d
```

This starts:
- **PostgreSQL 16 + pgvector** on `localhost:5434`
- **Adminer** (web DB UI) on `http://localhost:8080`

### 4. Pull Ollama Models (if using Ollama)

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
```

### 5. Ingest Knowledge Base

Load `data/data.txt` into the `documents` table:

```bash
uv run python ingest.py
```

## Usage

### Run the FastAPI Server

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs will be available at `http://localhost:8000/docs`.

#### API Endpoints

- **`GET /health`** — Readiness check verifying PostgreSQL connectivity.
- **`POST /api/v1/chat`** — Query the RAG chatbot.
  ```bash
  curl -X POST http://localhost:8000/api/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "Thứ Bảy đường sách Lục Bình mở cửa đến mấy giờ?", "top_k": 5}'
  ```
- **`POST /api/v1/documents`** *(Admin)* — Add a new document to both PostgreSQL and `data/data.txt`.
  ```bash
  curl -X POST http://localhost:8000/api/v1/documents \
    -H "Content-Type: application/json" \
    -H "x-api-key: your_admin_api_key_here" \
    -d '{"content": "Sách: Ví Dụ Mới. Tác giả: Nguyễn Văn A. Giá: 85.000đ."}'
  ```
- **`GET /api/v1/documents?limit=50&offset=0`** *(Admin)* — List stored documents.
  ```bash
  curl http://localhost:8000/api/v1/documents?limit=20&offset=0 \
    -H "x-api-key: your_admin_api_key_here"
  ```

### Run the Interactive CLI

```bash
uv run python main.py
```

Type your question in Vietnamese or English. Enter `exit`, `quit`, or `q` to stop.

### Run Evaluation

Benchmark retrieval hit-rate and end-to-end answer accuracy against `CASES` in `evaluate.py`:

```bash
uv run python evaluate.py
```

## Testing & Code Quality

### Run Unit Tests

```bash
uv run pytest tests/unit
```

### Run Integration Tests

Integration tests drop/recreate tables, so `tests/conftest.py` enforces that `DB_NAME` (or `TEST_DB_NAME`) starts with `test_`, ends with `_test`, or equals `test`:

```bash
TEST_DB_NAME=vectordb_test uv run pytest tests/integration
```

### Linting

```bash
uv run ruff check .
```

## Author

- **The Hoang** — [phanthehoang2503@gmail.com](mailto:phanthehoang2503@gmail.com)
