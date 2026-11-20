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


class TestVectorSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validate_test_database()
        init_db(vector_dim=1024, drop_old=True)
        cls.docs = [
            "Quán cà phê Đỗ Phủ tại đường sách phục vụ cà phê bơ rất ngon.",
            "Gian hàng sách Nhã Nam đang có chương trình giảm giá sách thiếu nhi 30%.",
        ]
        with get_connection() as conn:
            with conn.cursor() as cur:
                for doc in cls.docs:
                    vec = get_embedding(doc)
                    cur.execute(
                        "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
                        (doc, vec),
                    )

    def test_similarity_search(self):
        query = "Ở đâu bán đồ uống ngon?"
        query_vec = get_embedding(query)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, embedding <=> %s::vector AS distance
                    FROM documents
                    ORDER BY distance ASC
                    LIMIT 1;
                    """,
                    (query_vec,),
                )
                best_match = cur.fetchone()[0]

                # Đoạn văn về quán cà phê phải có khoảng cách gần hơn đoạn về giảm giá sách
                self.assertIn("cà phê", best_match.lower())


if __name__ == "__main__":
    unittest.main()
