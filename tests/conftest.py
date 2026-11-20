import os
import unittest


def validate_test_database() -> None:
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
