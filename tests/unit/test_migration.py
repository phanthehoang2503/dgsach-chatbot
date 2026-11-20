from unittest.mock import MagicMock, patch

from src.migration import swap_table


def test_swap_table_reuses_connection_and_renames_index_and_sequence():
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("src.migration.get_connection", return_value=mock_conn) as mock_get_conn:
        swap_table(staging_table="documents_staging_abc", target_table="documents")

    mock_get_conn.assert_called_once_with(
        autocommit=False, register_pgvector=False, role="ingest"
    )
    executed = [
        call.args[0].as_string(None)
        if hasattr(call.args[0], "as_string")
        else call.args[0]
        for call in mock_cur.execute.call_args_list
    ]
    assert any(
        'ALTER TABLE IF EXISTS "documents" RENAME TO "documents_old"' in s
        for s in executed
    )
    assert any(
        'ALTER TABLE "documents_staging_abc" RENAME TO "documents"' in s
        for s in executed
    )
    assert any(
        'ALTER INDEX IF EXISTS "documents_staging_abc_pkey" RENAME TO "documents_pkey"'
        in s
        for s in executed
    )
    assert any(
        'ALTER SEQUENCE IF EXISTS "documents_staging_abc_id_seq" RENAME TO "documents_id_seq"'
        in s
        for s in executed
    )
    assert any(
        'DROP INDEX IF EXISTS "idx_documents_staging_abc_embedding_hnsw"' in s
        for s in executed
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_documents_embedding_hnsw" ON "documents"' in s
        for s in executed
    )
    mock_conn.commit.assert_called_once()
