import os
from functools import lru_cache
from typing import Protocol

import ollama
from dotenv import load_dotenv

load_dotenv(override=True)


class Embedder(Protocol):
    fingerprint: str

    def embed_document(self, text: str) -> list[float]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    def __init__(self, model: str, dim: int):
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST"))
        self.model = model
        self.fingerprint = f"ollama:{model}:{dim}"

    def _embed(self, text: str) -> list[float]:
        return self.client.embeddings(model=self.model, prompt=text)["embedding"]

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    def __init__(self, model: str, dim: int):
        from google import genai  # chỉ cần cài khi dùng Gemini

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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


@lru_cache
def get_embedder() -> Embedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    if provider not in PROVIDERS:
        raise ValueError(f"EMBEDDING_PROVIDER không hợp lệ: {provider}")
    return PROVIDERS[provider](
        model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        dim=int(os.getenv("VECTOR_DIM", "1024")),
    )


def get_embedding(text: str) -> list[float]:
    """Giữ lại cho các test hiện có."""
    return get_embedder().embed_document(text)
