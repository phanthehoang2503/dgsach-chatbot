import unittest
import os
from ingest import load_chunks


class TestIngest(unittest.TestCase):
    def test_load_chunks(self):
        data_path = (
            os.path.join("data", "data.txt")
            if os.path.exists(os.path.join("data", "data.txt"))
            else "data.txt"
        )
        chunks = load_chunks(data_path)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        # Đảm bảo không có chunk nào là chuỗi rỗng
        for c in chunks:
            self.assertTrue(len(c.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
