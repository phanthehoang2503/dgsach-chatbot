import logging
import os
import secrets
from contextlib import contextmanager
from typing import Annotated

import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, StringConstraints

from ingest import add_document, list_documents
from src.database import get_connection
from src.llm import generate_answer
from src.retriever import retrieve_ctx

load_dotenv()

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="BookStreet RAG API", version="1.1.0")


QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]
DocumentText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class ChatReq(BaseModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=10, description="Số lượng ngữ cảnh tìm kiếm")


class ChatRes(BaseModel):
    query: str
    ans: str
    sources: list[str]


class DocumentReq(BaseModel):
    content: DocumentText


class DocumentRes(BaseModel):
    id: int
    content: str


def require_admin(x_api_key: str = Header(default="")):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Admin API key is not configured",
        )
    if not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key không hợp lệ")


@contextmanager
def _handle_endpoint_errors(unexpected_msg: str):
    try:
        yield
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except (psycopg.Error, ConnectionError, TimeoutError) as exc:
        logger.exception(unexpected_msg)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Dịch vụ tạm thời không khả dụng",
        ) from exc
    except Exception as exc:
        logger.exception(unexpected_msg)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, unexpected_msg
        ) from exc


@app.get("/health")
def health_check():
    try:
        with (
            get_connection(register_pgvector=False) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1;")
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Database is unreachable"
        ) from exc


@app.post("/api/v1/chat", response_model=ChatRes)
def chat_endpoint(req: ChatReq):
    with _handle_endpoint_errors("Không thể xử lý câu hỏi"):
        sources = retrieve_ctx(req.query, top_k=req.top_k)
        answer = generate_answer(req.query, sources)
        return ChatRes(query=req.query, ans=answer, sources=sources)


@app.post(
    "/api/v1/documents",
    response_model=DocumentRes,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def create_document(req: DocumentReq):
    with _handle_endpoint_errors("Không thể thêm tài liệu lúc này"):
        doc_id, content = add_document(req.content)
        return DocumentRes(id=doc_id, content=content)


@app.get(
    "/api/v1/documents",
    response_model=list[DocumentRes],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin)],
)
def get_documents(
    limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)
):
    with _handle_endpoint_errors("Không thể đọc tài liệu lúc này"):
        rows = list_documents(limit, offset)
        return [DocumentRes(id=doc_id, content=content) for doc_id, content in rows]