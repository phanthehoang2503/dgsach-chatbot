from unittest.mock import MagicMock, patch

import pytest

from src.errors import UpstreamUnavailableError
from src.llm import SYSTEM_PROMPT, generate_answer, get_ollama_client


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Đường sách ở đâu?", "Không tìm thấy thông tin trong tài liệu."),
        ("duong sach mo cua may gio?", "Không tìm thấy thông tin trong tài liệu."),
        ("What’s “Clean Code”?", "I couldn't find this in the documents."),
        ("Where is the book street?", "I couldn't find this in the documents."),
    ],
)
def test_generate_answer_empty_context_language_fallback(query, expected):
    assert generate_answer(query, []) == expected


def test_generate_answer_wraps_xml_and_handles_empty_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_TEMPERATURE", "")

    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "Sách hay tại đường sách."}}

    with patch("src.llm.get_ollama_client", return_value=mock_client):
        ans = generate_answer("Có sách gì?", ["Doc 1", "Doc 2"])

    assert ans == "Sách hay tại đường sách."
    kwargs = mock_client.chat.call_args.kwargs
    assert kwargs["model"] == "qwen3:4b"
    assert kwargs["options"] == {"temperature": 0.2}
    messages = kwargs["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "<context>\nDoc 1\nDoc 2\n</context>" in messages[1]["content"]
    assert "<question>\nCó sách gì?\n</question>" in messages[1]["content"]


def test_generate_answer_rejects_invalid_temperature(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "not-a-float")
    with pytest.raises(ValueError, match="LLM_TEMPERATURE"):
        generate_answer("Có sách gì?", ["Doc 1"])


def test_get_ollama_client_timeout(monkeypatch):
    get_ollama_client.cache_clear()
    try:
        monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
        with patch("src.llm.ollama.Client") as mock_cls:
            get_ollama_client()
        assert mock_cls.call_args.kwargs["timeout"] == 60.0

        get_ollama_client.cache_clear()
        monkeypatch.setenv("OLLAMA_TIMEOUT", "15")
        with patch("src.llm.ollama.Client") as mock_cls:
            get_ollama_client()
        assert mock_cls.call_args.kwargs["timeout"] == 15.0
    finally:
        get_ollama_client.cache_clear()


def test_generate_answer_raises_when_llm_slots_exhausted():
    mock_sem = MagicMock()
    mock_sem.acquire.return_value = False
    with (
        patch("src.llm._llm_slots", mock_sem),
        pytest.raises(UpstreamUnavailableError),
    ):
        generate_answer("Có sách gì?", ["Doc 1"])
