import os
import ollama
from dotenv import load_dotenv

load_dotenv(override=True)

ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST"))


def get_embedding(text: str, model: str = None) -> list[float]:
    """Sinh vector embedding từ Ollama."""
    model = model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    res = ollama_client.embeddings(model=model, prompt=text)
    return res["embedding"]
