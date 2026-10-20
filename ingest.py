import os
import uuid
from src.migration import swap_table
from src.database import get_connection, init_db
from src.embeddings import get_embedding
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = (
    os.path.join("data", "data.txt")
    if os.path.exists(os.path.join("data", "data.txt"))
    else "data.txt"
)
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))  # VD: qwen3-embedding:0.6b


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
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for idx, chunk in enumerate(chunks, 1):
                    vec = get_embedding(chunk)
                    cur.execute(
                        f"INSERT INTO {staging_table} (content, embedding) VALUES (%s, %s);",
                        (chunk, vec),
                    )
                    print(f" -> [{idx}/{total}] đã nạp: {chunk[:40]}...")
        print(f"Hoàn tất nạp {total} đoạn vào vector database.")
    except Exception as e:
        print(f"Lỗi trong quá trình embedding/nạp: {e}")
        raise

    # commit staging
    print("4. Commit staging...")
    swap_table(staging_table=staging_table, target_table="documents")
    print(f"Đã nạp {total} đoạn vào vector db thành công.")


if __name__ == "__main__":
    ingest_data()
