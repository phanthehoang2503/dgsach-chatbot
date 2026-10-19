import os
from src.database import get_connection, init_db
from src.embeddings import get_embedding

DATA_FILE = "data.txt"
VECTOR_DIM = 1024  # luôn luôn phải khớp với số chiều của model embedding


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

    # Đọc dữ liệu
    chunks = load_chunks(file_path)
    total = len(chunks)
    print(f"1. đã đọc {total} đoạn văn bản từ {file_path}.")

    # Khởi tạo / làm sạch dữ liệu
    print("2. đang chuẩn bị bảng document trong db")
    init_db(vector_dim=VECTOR_DIM, drop_old=True)

    # Embedding và lưu vào DB
    print("3. đang tạo embedding và lưu vào db")
    with get_connection() as conn:
        with conn.cursor() as cur:
            for idx, chunk in enumerate(chunks, 1):
                vec = get_embedding(chunk)
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
                    (chunk, vec),
                )
                print(f" -> [{idx}/{total}] đã nạp: {chunk[:40]}...")
    print(f"Hoàn tất nạp {total} đoạn vào vector database.")


if __name__ == "__main__":
    ingest_data()
