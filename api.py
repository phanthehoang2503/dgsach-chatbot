from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.retriever import retrieve_ctx
from src.llm import generate_answer

app = FastAPI(title="BookStreet RAG API", version="1.0.0")


class ChatReq(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    top_k: int = Field(default=2, ge=1, le=10, description="Số lượng ctx tiềm kiếm")


class ChatRes(BaseModel):
    query: str
    ans: str
    sources: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatRes)
def chat_endpoint(req: ChatReq):
    try:
        sources = retrieve_ctx(req.query, top_k=req.top_k)
        answer = generate_answer(req.query, sources)
        return ChatRes(query=req.query, ans=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
