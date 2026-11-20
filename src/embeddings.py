import os
from functools import lru_cache
from typing import Protocol

import ollama
from dotenv import load_dotenv

from src.errors import ConfigError

load_dotenv()


class Embedder(Protocol):
    fingerprint: str

    def embed_document(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, model: str, dim: int):
        self.client = ollama.Client(
            host=os.getenv("OLLAMA_HOST"),
            timeout=_parse_float_env("OLLAMA_TIMEOUT", 60.0),
        )
        self.model = model
        self.dim = dim
        self.fingerprint = f"ollama:{model}:{dim}"

    def _embed(self, text: str) -> list[float]:
        embedding = self.client.embeddings(model=self.model, prompt=text)["embedding"]
        if len(embedding) != self.dim:
            raise ConfigError(
                f"Ollama embedding dimension mismatch for model '{self.model}': "
                f"expected {self.dim}, got {len(embedding)}"
            )
        return embedding

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    def __init__(self, model: str, dim: int):
        from google import genai  # chỉ cần cài khi dùng Gemini
        from google.genai import types

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=_parse_int_env("GEMINI_TIMEOUT_MS", 30000)
            ),
        )
        self.model = model
        self.dim = dim
        self.fingerprint = f"gemini:{model}:{dim}"

    def _embed(self, text: str, task_type: str) -> list[float]:
        from google.genai import types

        res = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.dim
            ),
        )
        return res.embeddings[0].values

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")


PROVIDERS = {"ollama": OllamaEmbedder, "gemini": GeminiEmbedder}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} không hợp lệ: {raw!r}") from exc


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} không hợp lệ: {raw!r}") from exc


@lru_cache
def get_embedder() -> Embedder:
    raw_provider = os.getenv("EMBEDDING_PROVIDER")
    provider = (
        raw_provider.strip().lower()
        if raw_provider and raw_provider.strip()
        else "ollama"
    )
    if provider not in PROVIDERS:
        raise ConfigError(f"EMBEDDING_PROVIDER không hợp lệ: {provider}")
    raw_model = os.getenv("EMBEDDING_MODEL")
    model = (
        raw_model.strip() if raw_model and raw_model.strip() else "qwen3-embedding:0.6b"
    )
    return PROVIDERS[provider](
        model=model,
        dim=_parse_int_env("VECTOR_DIM", 1024),
    )


def get_embedding(text: str) -> list[float]:
    """Giữ lại cho các test hiện có."""
    return get_embedder().embed_document(text)
