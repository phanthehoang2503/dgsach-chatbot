from unittest.mock import patch

import pytest

from src.llm import SYSTEM_PROMPT, generate_answer


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

    with patch("src.llm.ollama_client.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Sách hay tại đường sách."}}
        ans = generate_answer("Có sách gì?", ["Doc 1", "Doc 2"])

    assert ans == "Sách hay tại đường sách."
    kwargs = mock_chat.call_args.kwargs
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
