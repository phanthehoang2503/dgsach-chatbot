import logging
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, StringConstraints

from ingest import add_document, list_documents
from src.llm import generate_answer
from src.retriever import retrieve_ctx

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
    if not expected or not secrets.compare_digest(
        x_api_key.encode(), expected.encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key không hợp lệ")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatRes)
def chat_endpoint(req: ChatReq):
    try:
        sources = retrieve_ctx(req.query, top_k=req.top_k)
        answer = generate_answer(req.query, sources)
        return ChatRes(query=req.query, ans=answer, sources=sources)
    except Exception:
        logger.exception("Lỗi khi xử lý chat")
        raise HTTPException(500, "Không thể xử lý câu hỏi")


@app.post(
    "/api/v1/documents",
    response_model=DocumentRes,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def create_document(req: DocumentReq):
    try:
        doc_id = add_document(req.content)
        return DocumentRes(id=doc_id, content=" ".join(req.content.split()))
    except Exception:
        logger.exception("Lỗi khi thêm tài liệu")
        raise HTTPException(500, "Không thể thêm tài liệu lúc này")


@app.get(
    "/api/v1/documents",
    response_model=list[DocumentRes],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin)],
)
def get_documents(
    limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)
):
    try:
        rows = list_documents(limit, offset)
        return [DocumentRes(id=doc_id, content=content) for doc_id, content in rows]
    except Exception:
        logger.exception("Lỗi khi đọc tài liệu")
        raise HTTPException(500, "Không thể đọc tài liệu lúc này")