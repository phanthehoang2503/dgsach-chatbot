from unittest.mock import MagicMock, patch

import httpx
import ollama
import psycopg
import pytest
from fastapi.testclient import TestClient

from api import app, limiter
from src.errors import (
    ConfigError,
    IndexNotReadyError,
    InvalidInputError,
    UpstreamUnavailableError,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    yield
    limiter.reset()


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
        (InvalidInputError("bad input"), 400),
        (IndexNotReadyError("index mismatch"), 503),
        (ConfigError("bad config"), 500),
        (ValueError("generic value error"), 500),
        (RuntimeError("generic runtime error"), 500),
        (httpx.ConnectError("ollama down"), 503),
        (ollama.ResponseError("model error"), 503),
        (psycopg.OperationalError("db down"), 503),
        (UpstreamUnavailableError("llm overloaded"), 503),
        (ZeroDivisionError("unexpected"), 500),
    ],
)
def test_chat_endpoint_maps_errors(exc, expected_status):
    with patch("api.retrieve_ctx", side_effect=exc):
        res = client.post("/api/v1/chat", json={"query": "Có sách gì?", "top_k": 3})

    assert res.status_code == expected_status


def test_chat_endpoint_does_not_leak_internal_details():
    err = IndexNotReadyError(
        "Bảng documents dùng 'ollama:qwen3-embedding:0.6b:1024', cấu hình hiện tại là 'gemini:..."
    )
    with patch("api.retrieve_ctx", side_effect=err):
        res = client.post("/api/v1/chat", json={"query": "Có sách gì?", "top_k": 3})

    assert res.status_code == 503
    assert "ollama" not in res.text
    assert "1024" not in res.text


def test_chat_endpoint_rate_limit(monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT", "2/minute")
    limiter.reset()
    with (
        patch("api.retrieve_ctx", return_value=["Doc 1"]),
        patch("api.generate_answer", return_value="Answer"),
    ):
        assert client.post("/api/v1/chat", json={"query": "Hỏi 1"}).status_code == 200
        assert client.post("/api/v1/chat", json={"query": "Hỏi 2"}).status_code == 200
        res = client.post("/api/v1/chat", json={"query": "Hỏi 3"})
    assert res.status_code == 429


def test_request_body_over_16kb_returns_413():
    payload = '{"query": "' + ("a" * (20 * 1024)) + '"}'
    res = client.post(
        "/api/v1/chat",
        content=payload,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 413
