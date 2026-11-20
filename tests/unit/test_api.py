from unittest.mock import MagicMock, patch

import psycopg
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_returns_200_when_db_reachable():
    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    with patch("api.get_connection", return_value=mock_conn):
        res = client.get("/health")

    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    mock_cur.execute.assert_called_once_with("SELECT 1;")


def test_health_returns_503_when_db_unreachable():
    with patch("api.get_connection", side_effect=psycopg.OperationalError("down")):
        res = client.get("/health")

    assert res.status_code == 503


def test_admin_endpoints_fail_closed_when_key_missing_or_wrong(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    res_missing = client.get("/api/v1/documents")
    assert res_missing.status_code == 503

    monkeypatch.setenv("ADMIN_API_KEY", "")
    res_empty = client.post("/api/v1/documents", json={"content": "Sách mới"})
    assert res_empty.status_code == 503

    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")
    res_wrong = client.get("/api/v1/documents", headers={"X-API-Key": "wrong"})
    assert res_wrong.status_code == 401


def test_create_and_list_documents_with_valid_admin_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-key")

    with patch("api.add_document", return_value=(7, "Sách mới")) as mock_add:
        res_post = client.post(
            "/api/v1/documents",
            json={"content": "  Sách   mới  "},
            headers={"X-API-Key": "secret-key"},
        )

    assert res_post.status_code == 201
    assert res_post.json() == {"id": 7, "content": "Sách mới"}
    mock_add.assert_called_once_with("Sách   mới")

    with patch("api.list_documents", return_value=[(7, "Sách mới")]):
        res_get = client.get("/api/v1/documents", headers={"X-API-Key": "secret-key"})

    assert res_get.status_code == 200
    assert res_get.json() == [{"id": 7, "content": "Sách mới"}]


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (ValueError("bad input"), 400),
        (RuntimeError("index mismatch"), 409),
        (psycopg.OperationalError("db down"), 503),
        (ZeroDivisionError("unexpected"), 500),
    ],
)
def test_chat_endpoint_maps_errors(exc, expected_status):
    with patch("api.retrieve_ctx", side_effect=exc):
        res = client.post("/api/v1/chat", json={"query": "Có sách gì?", "top_k": 3})

    assert res.status_code == expected_status
