import os
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection(register_pgvector: bool = True):
    """Tạo và cấu hình kết nối tới PostgreSQL/pgvector."""
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        autocommit=True,
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
    # 1: đảm bảo extension vector của postgresql được tạo
    with get_connection(register_pgvector=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # 2: tạo / làm mới bảng
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            if drop_old:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(table_id))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                      id serial PRIMARY KEY,
                      content text,
                      embedding vector({})
                    );
                    """
                ).format(table_id, sql.Literal(dim))
            )
