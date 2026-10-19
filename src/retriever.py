from src.database import get_connection
from src.embeddings import get_embedding


def retrieve_ctx(query: str, top_k: int = 2) -> list[str]:
    """Tìm kiếm các đoạn tài liệu liên quan nhất với câu hỏi từ vector database."""
    query_vec = get_embedding(query)

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
