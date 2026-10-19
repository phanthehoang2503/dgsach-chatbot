import os
import psycopg
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
    if register_vector:
        register_vector(conn)
    return conn


def init_db(vector_dim: int = 1024, drop_old: bool = False):
    """Tạo bảng documents nếu chưa tồn tại"""
    # 1: đảm bảo extension vector của postgresql được tạo
    with get_connection(register_pgvector=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2: tạo/ làm mới bảng
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
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


# Chỉ cập nhật db khi ingest hoàn tất, phòng hờ sập kết nối dẫn đến xóa mất toàn bộ tài liệu được mã hóa
def swap_tables(
    staging_table: str = "document_staging", target_table: str = "documents"
):
    with psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5434"),
        dbname=os.getenv("DB_NAME", "vectordb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres"),
        autocommit=False,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                    DROP TABLE IF EXISTS {target_table}_old;
                    ALTER TABLE IF EXISTs {target_table} RENAME TO {target_table}_old;
                    ALTER TABLE {staging_table} RENAME TO {target_table};
                    DROP TABLE IF EXISTS {target_table}_old;
                """
            )
            conn.commit()
