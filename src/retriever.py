from src.embeddings import get_embedding
from src.database import get_connection
from src.embeddings import get_embedding as _default_embed
from typing import Callable, Optional


def retrieve_ctx(
    query: str, top_k: int = 2, embed_fn: Optional[Callable[[str], list[float]]] = None
) -> list[str]:
    """Tìm kiếm các đoạn tài liệu liên quan nhất với câu hỏi từ vector database."""
    embed = embed_fn or _default_embed
    query_vec = embed(query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                    SELECT content, embedding <=> %s::vector AS distance
                    FROM documents
                    ORDER BY distance ASC
                    LIMIT %s;
                """,
                (query_vec, top_k),
            )
            rows = cur.fetchall()

    return [r[0] for r in rows]
