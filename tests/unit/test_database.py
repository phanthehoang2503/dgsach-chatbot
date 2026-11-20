from unittest.mock import MagicMock, patch

from src.database import get_connection, init_db


def test_get_connection_registers_pgvector_by_default():
    mock_conn = MagicMock()
    with (
        patch("src.database.psycopg.connect", return_value=mock_conn) as mock_connect,
        patch("src.database.register_vector") as mock_register,
    ):
        conn = get_connection()

    assert conn is mock_conn
    assert mock_connect.call_args.kwargs["autocommit"] is True
    mock_register.assert_called_once_with(mock_conn)


def test_get_connection_skips_pgvector_when_disabled():
    mock_conn = MagicMock()
    with (
        patch("src.database.psycopg.connect", return_value=mock_conn) as mock_connect,
        patch("src.database.register_vector") as mock_register,
    ):
        conn = get_connection(register_pgvector=False, autocommit=False)

    assert conn is mock_conn
    assert mock_connect.call_args.kwargs["autocommit"] is False
    mock_register.assert_not_called()


def test_init_db_uses_single_connection_and_creates_not_null_table_and_hnsw_index():
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("src.database.get_connection", return_value=mock_conn) as mock_get_conn:
        init_db(vector_dim=768, drop_old=True, table_name="documents")

    mock_get_conn.assert_called_once_with(register_pgvector=False)
    executed = [
        call.args[0].as_string(None)
        if hasattr(call.args[0], "as_string")
        else call.args[0]
        for call in mock_cur.execute.call_args_list
    ]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in executed)
    assert any('DROP TABLE IF EXISTS "documents"' in s for s in executed)
    assert any(
        'CREATE TABLE IF NOT EXISTS "documents"' in s
        and "content text NOT NULL" in s
        and "embedding vector(768) NOT NULL" in s
        for s in executed
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_documents_embedding_hnsw"' in s
        and "USING hnsw (embedding vector_cosine_ops)" in s
        for s in executed
    )
