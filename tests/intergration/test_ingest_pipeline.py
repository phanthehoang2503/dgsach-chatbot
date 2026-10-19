import os
import unittest
from src.database import get_connection, init_db
from src.embeddings import get_embedding


def validate_test_database():
    """Chỉ cho phép test xóa bảng trên database test."""
    test_db = os.getenv("TEST_DB_NAME")
    if test_db:
        os.environ["DB_NAME"] = test_db

    db_name = os.getenv("DB_NAME", "")

    is_safe = (
        db_name.startswith("test_")
        or db_name.endswith("_test")
        or db_name == "test"
        or os.getenv("ALLOW_DESTRUCTIVE_TESTS", "").lower() in ("true", "1")
    )
    if not is_safe:
        raise unittest.SkipTest(
            f"Database '{db_name}' không phải database test an toàn. "
            "Đặt ALLOW_DESTRUCTIVE_TESTS=1 nếu muốn bỏ qua."
        )


class TestIngestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validate_test_database()
        init_db(vector_dim=1024, drop_old=True)

    def test_embed_and_insert_flow(self):
        sample_doc = "Sách Lập trình Python căn bản tại Đường sách TP.HCM"

        # 1. Sinh vector thật từ Ollama
        vec = get_embedding(sample_doc)
        self.assertEqual(len(vec), 1024)

        # 2. Ghi trực tiếp vào PostgreSQL
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s) RETURNING id;",
                    (sample_doc, vec),
                )
                inserted_id = cur.fetchone()[0]
                self.assertIsNotNone(inserted_id)

                # 3. Query lại từ DB để đảm bảo pgvector lưu đúng
                cur.execute(
                    "SELECT content, embedding FROM documents WHERE id = %s;",
                    (inserted_id,),
                )
                row = cur.fetchone()
                self.assertEqual(row[0], sample_doc)
                self.assertIsNotNone(row[1])


if __name__ == "__main__":
    unittest.main()
