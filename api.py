import logging
import os
import secrets
from contextlib import contextmanager
from typing import Annotated

import httpx
import ollama
import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ingest import add_document, list_documents
from src.database import get_connection
from src.errors import (
    IndexNotReadyError,
    InvalidInputError,
    UpstreamUnavailableError,
)
from src.llm import generate_answer
from src.retriever import retrieve_ctx

load_dotenv()

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="BookStreet RAG API", version="1.1.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

MAX_BODY_BYTES = 16 * 1024


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body quá lớn"},
                )
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Content-Length không hợp lệ"},
            )
    return await call_next(request)


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


UPSTREAM_ERRORS = (
    psycopg.Error,
    httpx.HTTPError,
    ollama.ResponseError,
    ConnectionError,
    TimeoutError,
    UpstreamUnavailableError,
)


@contextmanager
def _handle_endpoint_errors(unexpected_msg: str):
    try:
        yield
    except HTTPException:
        raise
    except InvalidInputError as exc:
        logger.info("Invalid input: %s", exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except IndexNotReadyError as exc:
        logger.exception("Index not ready")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Hệ thống đang cập nhật dữ liệu, vui lòng thử lại sau",
        ) from exc
    except UPSTREAM_ERRORS as exc:
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
            get_connection(
                register_pgvector=False,
                statement_timeout_ms=int(os.getenv("DB_STATEMENT_TIMEOUT_MS") or 10000),
            ) as conn,
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
@limiter.limit(lambda: os.getenv("CHAT_RATE_LIMIT") or "10/minute")
def chat_endpoint(request: Request, req: ChatReq):
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
