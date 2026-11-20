from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import OllamaEmbedder, get_embedder, get_embedding


@pytest.fixture(autouse=True)
def clear_embedder_cache():
    get_embedder.cache_clear()
    yield
    get_embedder.cache_clear()


def test_ollama_embedder_validates_dimension():
    mock_client = MagicMock()
    mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

    with patch("src.embeddings.ollama.Client", return_value=mock_client):
        embedder = OllamaEmbedder(model="qwen3-embedding:0.6b", dim=3)
        assert embedder.embed_query("hello") == [0.1, 0.2, 0.3]

        embedder_bad = OllamaEmbedder(model="qwen3-embedding:0.6b", dim=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            embedder_bad.embed_document("hello")


def test_get_embedder_normalizes_provider_and_handles_empty_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "  OLLAMA  ")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("VECTOR_DIM", "")

    with patch("src.embeddings.ollama.Client"):
        embedder = get_embedder()

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "qwen3-embedding:0.6b"
    assert embedder.dim == 1024


def test_get_embedder_rejects_invalid_provider_or_dim(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "invalid-provider")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        get_embedder()

    get_embedder.cache_clear()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_DIM", "not-an-int")
    with pytest.raises(ValueError, match="VECTOR_DIM"):
        get_embedder()


def test_get_embedding_delegates_to_embedder():
    mock_embedder = MagicMock()
    mock_embedder.embed_document.return_value = [0.5] * 1024

    with patch("src.embeddings.get_embedder", return_value=mock_embedder):
        vec = get_embedding("Đường sách Nguyễn Văn Bình")

    assert len(vec) == 1024
    mock_embedder.embed_document.assert_called_once_with("Đường sách Nguyễn Văn Bình")
