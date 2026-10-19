import os
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection():
    """Tạo và cấu hình kết nối tới PostgreSQL/pgvector."""
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        autocommit=True,
    )
    register_vector(conn)
    return conn


def init_db(vector_dim: int = 1024, drop_old: bool = False):
    """Tạo bảng documents nếu chưa tồn tại (hoặc làm mới nếu drop_old=True)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            if drop_old:
                cur.execute("DROP TABLE IF EXISTS documents;")

            cur.execute(
                f"""
                    CREATE TABLE IF NOT EXISTS documents (
                      id serial PRIMARY KEY,
                      content text,
                      embedding vector({vector_dim})
                    );
                  """
            )
