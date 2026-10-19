import unittest
from src.embeddings import get_embedding


class TestEmbeddings(unittest.TestCase):
    def test_get_embedding_output(self):
        sample_text = "Đường sách Nguyễn Văn Bình"
        vec = get_embedding(sample_text)

        # 1. Kết quả phải là một list
        self.assertIsInstance(vec, list)
        # 2. Không được rỗng
        self.assertGreater(len(vec), 0)
        # 3. Phải đúng 1024 chiều (qwen3-embedding:0.6B)
        self.assertEqual(len(vec), 1024)
        # 4. Phần tử bên trong phải là float
        self.assertIsInstance(vec[0], float)


if __name__ == "__main__":
    unittest.main()
