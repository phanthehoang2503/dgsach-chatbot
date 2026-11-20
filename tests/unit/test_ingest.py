from unittest.mock import MagicMock, patch

import pytest

from ingest import add_document, ingest_data, load_chunks


def test_load_chunks_reads_and_strips_non_empty_lines(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("  Sách A  \n\n  \nSách B\n", encoding="utf-8")

    chunks = load_chunks(str(data_file))
    assert chunks == ["Sách A", "Sách B"]


def test_load_chunks_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(str(tmp_path / "missing.txt"))


def test_add_document_normalizes_whitespace_and_persists(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("Existing doc", encoding="utf-8")

    mock_embedder = MagicMock()
    mock_embedder.fingerprint = "ollama:m:4"
    mock_embedder.embed_document.return_value = [0.1, 0.2, 0.3, 0.4]

    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = (42,)
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with (
        patch("ingest.get_embedder", return_value=mock_embedder),
        patch("ingest.get_connection", return_value=mock_conn),
        patch("ingest.assert_index_matches") as mock_assert,
    ):
        doc_id, content = add_document("  Hello   \n  world  ", file_path=str(data_file))

    assert doc_id == 42
    assert content == "Hello world"
    mock_assert.assert_called_once_with(mock_cur, "ollama:m:4")
    assert "Hello world" in data_file.read_text(encoding="utf-8")


def test_ingest_data_cleans_up_staging_table_on_failure(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("Sách Python\n", encoding="utf-8")

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with (
        patch("ingest.init_db"),
        patch("ingest.get_embedder", side_effect=RuntimeError("embed failed")),
        patch("ingest.get_connection", return_value=mock_conn),
        pytest.raises(RuntimeError, match="embed failed"),
    ):
        ingest_data(str(data_file))

    executed = [
        call.args[0].as_string(None)
        if hasattr(call.args[0], "as_string")
        else call.args[0]
        for call in mock_cur.execute.call_args_list
    ]
    assert any('DROP TABLE IF EXISTS "documents_staging_' in s for s in executed)
