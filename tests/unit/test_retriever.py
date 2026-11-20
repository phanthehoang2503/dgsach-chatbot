from unittest.mock import MagicMock, patch

import pytest
from psycopg import errors as pg_errors

from src.retriever import (
    assert_index_matches,
    reset_index_check_cache,
    retrieve_ctx,
    retrieve_with_scores,
)


@pytest.fixture(autouse=True)
def clear_index_cache():
    reset_index_check_cache()
    yield
    reset_index_check_cache()


@pytest.mark.parametrize("bad_top_k", [0, -1, -10])
def test_retrieve_with_scores_rejects_non_positive_top_k(bad_top_k):
    with pytest.raises(ValueError, match="top_k"):
        retrieve_with_scores("sach", top_k=bad_top_k, embed_fn=lambda _: [0.1])


def test_retrieve_with_scores_filters_null_embeddings():
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [("doc 1", 0.2)]
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("src.retriever.get_connection", return_value=mock_conn):
        rows = retrieve_with_scores("query", top_k=3, embed_fn=lambda _: [0.1, 0.2])

    assert rows == [("doc 1", 0.2)]
    sql_text = mock_cur.execute.call_args.args[0]
    assert "WHERE embedding IS NOT NULL" in sql_text


def test_retrieve_ctx_handles_none_distance_and_empty_max_distance(monkeypatch):
    monkeypatch.setenv("MAX_DISTANCE", "")
    rows = [("doc_none", None), ("doc_ok", 0.4), ("doc_far", 0.8)]

    with patch("src.retriever.retrieve_with_scores", return_value=rows):
        ctx = retrieve_ctx("query", top_k=3, embed_fn=lambda _: [0.1])

    assert ctx == ["doc_ok"]


def test_retrieve_ctx_rejects_invalid_max_distance(monkeypatch):
    monkeypatch.setenv("MAX_DISTANCE", "invalid")
    with pytest.raises(ValueError, match="MAX_DISTANCE"):
        retrieve_ctx("query", top_k=3, embed_fn=lambda _: [0.1])


def test_assert_index_matches_raises_actionable_error_on_missing_table():
    mock_cur = MagicMock()
    mock_cur.execute.side_effect = pg_errors.UndefinedTable('relation "documents" does not exist')

    with pytest.raises(RuntimeError, match="ingest.py"):
        assert_index_matches(mock_cur, "ollama:m:1024")


def test_assert_index_matches_raises_on_mismatch():
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ("ollama:other:512",)

    with pytest.raises(RuntimeError, match="ingest.py"):
        assert_index_matches(mock_cur, "ollama:m:1024")


def test_assert_index_matches_caches_and_resets():
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = ("ollama:m:1024",)

    assert_index_matches(mock_cur, "ollama:m:1024")
    assert_index_matches(mock_cur, "ollama:m:1024")
    assert mock_cur.execute.call_count == 1

    reset_index_check_cache()
    assert_index_matches(mock_cur, "ollama:m:1024")
    assert mock_cur.execute.call_count == 2
