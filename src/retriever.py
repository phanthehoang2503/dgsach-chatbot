import os
from collections.abc import Callable

from psycopg import errors as pg_errors

from src.database import get_connection
from src.embeddings import get_embedder
from src.errors import ConfigError, IndexNotReadyError, InvalidInputError


def _get_max_distance(default: float = 0.5) -> float:
    raw = os.getenv("MAX_DISTANCE")
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"MAX_DISTANCE không hợp lệ: {raw!r}") from exc


def retrieve_with_scores(
    query: str,
    top_k: int = 5,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[tuple[str, float]]:
    """Trả về các đoạn gần nhất kèm cosine distance."""
    if top_k <= 0:
        raise InvalidInputError(f"top_k phải lớn hơn 0, nhận được: {top_k}")

    embedder = None if embed_fn else get_embedder()
    embed = embed_fn or embedder.embed_query
    query_vec = embed(query)

    with get_connection() as conn, conn.cursor() as cur:
        if embedder:
            assert_index_matches(cur, embedder.fingerprint)
        cur.execute(
            """
                SELECT content, embedding <=> %s::vector AS distance
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT %s;
            """,
            (query_vec, top_k),
        )
        return cur.fetchall()


def retrieve_ctx(
    query: str,
    top_k: int = 5,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[str]:
    max_distance = _get_max_distance()
    rows = retrieve_with_scores(query, top_k, embed_fn)
    return [
        content
        for content, distance in rows
        if distance is not None and distance <= max_distance
    ]


_verified_fingerprints: set[str] = set()


def reset_index_check_cache() -> None:
    """Xóa cache kiểm tra fingerprint của bảng documents."""
    _verified_fingerprints.clear()


def assert_index_matches(cur, expected: str):
    if expected in _verified_fingerprints:
        return
    try:
        cur.execute("SELECT obj_description('documents'::regclass, 'pg_class');")
        row = cur.fetchone()
    except pg_errors.UndefinedTable as exc:
        raise IndexNotReadyError(
            "Bảng 'documents' chưa tồn tại trong database. "
            "Hãy chạy `ingest.py` để tạo bảng và nạp dữ liệu."
        ) from exc
    stored = row[0] if row else None
    if stored != expected:
        raise IndexNotReadyError(
            f"Bảng documents dùng '{stored}', cấu hình hiện tại là '{expected}'. "
            "Hãy chạy lại ingest.py."
        )
    _verified_fingerprints.add(expected)
