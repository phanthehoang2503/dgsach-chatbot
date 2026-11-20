from src.embeddings import get_embedder
import os
from src.database import get_connection
from typing import Callable, Optional


def retrieve_with_scores(
    query: str, top_k: int = 5, embed_fn: Optional[Callable[[str], list[float]]] = None
) -> list[tuple[str, float]]:
    """Trả về các đoạn gần nhất kèm cosine distance."""
    embedder = None if embed_fn else get_embedder()
    embed = embed_fn or embedder.embed_query
    query_vec = embed(query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            if embedder:
                assert_index_matches(cur, embedder.fingerprint)
            cur.execute(
                """
                    SELECT content, embedding <=> %s::vector AS distance
                    FROM documents
                    ORDER BY distance ASC
                    LIMIT %s;
                """,
                (query_vec, top_k),
            )
            return cur.fetchall()


def retrieve_ctx(
    query: str, top_k: int = 5, embed_fn: Optional[Callable[[str], list[float]]] = None
) -> list[str]:
    max_distance = float(os.getenv("MAX_DISTANCE", "0.5"))
    rows = retrieve_with_scores(query, top_k, embed_fn)
    return [content for content, distance in rows if distance <= max_distance]


def assert_index_matches(cur, expected: str):
    cur.execute("SELECT obj_description('documents'::regclass, 'pg_class');")
    stored = cur.fetchone()[0]
    if stored != expected:
        raise RuntimeError(
            f"Bảng documents dùng '{stored}', cấu hình hiện tại là '{expected}'. "
            "Hãy chạy lại ingest.py."
        )
