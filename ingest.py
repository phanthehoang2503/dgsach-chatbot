import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from psycopg import sql

from src.database import get_connection, init_db
from src.embeddings import get_embedder
from src.migration import swap_table
from src.retriever import assert_index_matches

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = str(
    BASE_DIR / "data" / "data.txt"
    if (BASE_DIR / "data" / "data.txt").exists()
    else BASE_DIR / "data.txt"
)
VECTOR_DIM = int(os.getenv("VECTOR_DIM") or "1024")  # VD: qwen3-embedding:0.6b
_doc_write_lock = threading.Lock()


def load_chunks(file_path: str) -> list[str]:
    """Đọc dữ liệu từ file và lọc các dòng trống."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.read().split("\n") if line.strip()]
    return chunks


def ingest_data(file_path: str = DATA_FILE):
    """Quy trình nạp dữ liệu vào vector database."""
    print("Ingestion...")

    # 1. Đọc dữ liệu
    chunks = load_chunks(file_path)
    total = len(chunks)
    print(f"1. đã đọc {total} đoạn văn bản từ {file_path}.")

    # 2. Khởi tạo / làm sạch dữ liệu
    staging_table = f"documents_staging_{uuid.uuid4().hex[:8]}"
    print(f"2. đang chuẩn bị bảng {staging_table} trong db")
    init_db(vector_dim=VECTOR_DIM, drop_old=True, table_name=staging_table)

    # Embedding và lưu vào DB
    print("3. đang tạo embedding và staging...")
    insert_stmt = sql.SQL(
        "INSERT INTO {} (content, embedding) VALUES (%s, %s);"
    ).format(sql.Identifier(staging_table))
    comment_stmt = sql.SQL("COMMENT ON TABLE {} IS {};")
    try:
        embedder = get_embedder()
        with get_connection() as conn, conn.cursor() as cur:
            for idx, chunk in enumerate(chunks, 1):
                vec = embedder.embed_document(chunk)
                cur.execute(insert_stmt, (chunk, vec))
                print(f" -> [{idx}/{total}] đã nạp: {chunk[:40]}...")
            cur.execute(
                comment_stmt.format(
                    sql.Identifier(staging_table),
                    sql.Literal(embedder.fingerprint),
                )
            )
        print(f"Hoàn tất nạp {total} đoạn vào vector database.")

        # commit staging
        print("4. Commit staging...")
        swap_table(staging_table=staging_table, target_table="documents")
        print(f"Đã nạp {total} đoạn vào vector db thành công.")
    except Exception as e:
        print(f"Lỗi trong quá trình embedding/nạp: {e}")
        try:
            with (
                get_connection(register_pgvector=False) as conn,
                conn.cursor() as cur,
            ):
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {};").format(
                        sql.Identifier(staging_table)
                    )
                )
        except Exception as cleanup_err:  # noqa: BLE001
            print(f"Không thể dọn bảng staging {staging_table}: {cleanup_err}")
        raise


def add_document(text: str, file_path: str = DATA_FILE) -> tuple[int, str]:
    """Thêm một tài liệu vào DB và data.txt để không mất khi ingest lại."""
    content = " ".join(text.split())  # data.txt quy ước mỗi dòng là một tài liệu
    embedder = get_embedder()
    vec = embedder.embed_document(content)

    with _doc_write_lock:
        with get_connection() as conn, conn.cursor() as cur:
            assert_index_matches(cur, embedder.fingerprint)
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s) RETURNING id;",
                (content, vec),
            )
            doc_id = cur.fetchone()[0]

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")
    return doc_id, content


def list_documents(limit: int = 50, offset: int = 0) -> list[tuple[int, str]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM documents ORDER BY id LIMIT %s OFFSET %s;",
            (limit, offset),
        )
        return cur.fetchall()


if __name__ == "__main__":
    ingest_data()
