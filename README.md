# BookStreet RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying information about books and activities at Ho Chi Minh City's Book Street (Đường Sách TP.HCM).

The system runs entirely on local infrastructure using Ollama for LLM inference and PostgreSQL with pgvector for vector storage - Might implements factory method for switching between AI providers on the feature.
---

## Tech Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| LLM | [Ollama](https://ollama.com/) | `qwen3:4b` |
| Embedding | [Ollama](https://ollama.com/) | `qwen3-embedding:0.6b` (1024-dim) |
| Vector DB | [PostgreSQL 16](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) | Cosine distance (`<=>`) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Auto-generated Swagger UI at `/docs` |
| DB Admin | [Adminer](https://www.adminer.org/) | Web UI on port 8080 |

---

## Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.com/download)

### 1. Clone & pull models

```bash
git clone https://github.com/phanthehoang2503/dgsach-chatbot.git
cd dgsach-chatbot

ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b (for lower end pc testing purpose)
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed:

```ini
# PostgreSQL
DB_HOST=localhost
DB_PORT=5434
DB_NAME=vectordb
DB_USER=postgres
DB_PASS=your_password_here

# AI Provider
OLLAMA_HOST=http://127.0.0.1:11434
EMBEDDING_MODEL=qwen3-embedding:0.6b
VECTOR_DIM=1024
LLM_MODEL=qwen3:4b
LLM_TEMPERATURE=0.2
```

### 3. Start the database

```bash
cp docker-compose.example.yml docker-compose.yml
# Update POSTGRES_PASSWORD in docker-compose.yml to match DB_PASS
docker compose up -d
```

Adminer is available at `http://localhost:8080`.

### 4. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate  # or: conda activate your_env_name
pip install -r requirements.txt
```

### 5. Ingest data

```bash
python ingest.py
```

This reads chunks from `data/data.txt`, embeds them into a staging table, then performs an atomic table swap to replace the production `documents` table — ensuring zero downtime if the process fails midway.

---

## Usage

### CLI (Interactive)

```bash
python main.py
```

Type a question, get an answer. Type `exit` to quit.

### REST API

```bash
uvicorn api:app --reload --port 8000
```

**Swagger UI**: `http://localhost:8000/docs`

**Endpoints**:

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/chat` | Send a question, get RAG-powered answer |

**Example request**:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Đường sách có quán cà phê nào không?", "top_k": 2}'
```

**Example response**:

```json
{
  "query": "Đường sách có quán cà phê nào không?",
  "ans": "Quán cà phê Đỗ Phủ tại đường sách phục vụ cà phê bơ rất ngon...",
  "sources": [
    "Quán cà phê Đỗ Phủ tại đường sách phục vụ cà phê bơ rất ngon."
  ]
}
```

---

## Testing

**Unit tests** (no Docker/Ollama required):

```bash
python -m unittest discover tests/unit
```

**Integration tests** (requires running PostgreSQL + Ollama):

```powershell
# PowerShell
$env:ALLOW_DESTRUCTIVE_TESTS="1"
python -m unittest discover tests/intergration
```

---

## Project Structure

```text
dgsach-chatbot/
├── data/
│   └── data.txt                    # Example knowledge base (book street info)
├── src/
│   ├── database.py                 # DB connection & schema init
│   ├── migration.py                # Atomic staging table swap
│   ├── embeddings.py               # Ollama embedding client
│   ├── retriever.py                # Dense retriever (for searching documents)
│   └── llm.py                      # Prompt template & LLM generation
├── tests/
│   ├── unit/
│   ├── intergration/
│   └── e2e/
├── api.py                          # FastAPI endpoints
├── ingest.py                       # Data ingestion pipeline
├── main.py                         # CLI chat loop
├── .env.example
├── docker-compose.example.yml
└── requirements.txt
```

