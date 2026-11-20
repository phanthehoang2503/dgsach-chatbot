import os
from typing import Literal

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql

load_dotenv()


def get_connection(
    register_pgvector: bool = True,
    autocommit: bool = True,
    statement_timeout_ms: int | None = None,
    role: Literal["app", "ingest"] = "app",
):
    """Tạo và cấu hình kết nối tới PostgreSQL/pgvector."""
    if role == "ingest":
        user = os.getenv("INGEST_DB_USER") or os.getenv("DB_USER")
        password = os.getenv("INGEST_DB_PASS") or os.getenv("DB_PASS")
    else:
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")

    kwargs: dict[str, object] = {
        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT") or 5),
    }
    if statement_timeout_ms:
        kwargs["options"] = f"-c statement_timeout={int(statement_timeout_ms)}"
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=user,
        password=password,
        autocommit=autocommit,
        **kwargs,
    )
    if register_pgvector:
        register_vector(conn)
    return conn


def init_db(
    vector_dim: int = 1024, drop_old: bool = False, table_name: str = "documents"
):
    """Tạo bảng documents nếu chưa tồn tại"""
    dim = int(vector_dim)
    table_id = sql.Identifier(table_name)
    index_id = sql.Identifier(f"idx_{table_name}_embedding_hnsw")
    with (
        get_connection(register_pgvector=False, role="ingest") as conn,
        conn.cursor() as cur,
    ):
        # 1: đảm bảo extension vector của postgresql được tạo
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # 2: tạo / làm mới bảng và index HNSW
        if drop_old:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(table_id))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                  id serial PRIMARY KEY,
                  content text NOT NULL,
                  embedding vector({}) NOT NULL
                );
                """
            ).format(table_id, sql.Literal(dim))
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} "
                "USING hnsw (embedding vector_cosine_ops);"
            ).format(index_id, table_id)
        )
