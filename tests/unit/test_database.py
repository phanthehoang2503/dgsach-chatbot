import unittest
from src.database import get_connection, init_db


class TestDatabase(unittest.TestCase):
    def test_connection(self):
        """Kiểm tra có kết nối được tới PostgreSQL không."""
        conn = get_connection()
        self.assertIsNotNone(conn)
        conn.close()

    def test_init_db(self):
        """Kiểm tra tạo bảng documents thành công."""
        init_db(vector_dim=1024, drop_old=False)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'documents'
                    );
                """
                )
                exists = cur.fetchone()[0]
                self.assertTrue(exists, "Bảng 'documents' chưa được tạo!")


if __name__ == "__main__":
    unittest.main()
