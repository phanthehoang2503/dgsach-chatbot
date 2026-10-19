import ollama

ollama_client = ollama.Client(host="http://127.0.0.1:11434")


def get_embedding(text: str, model: str = "qwen3-embedding:0.6b") -> list[float]:
    """Sinh vector embedding từ Ollama."""
    res = ollama_client.embeddings(model=model, prompt=text)
    return res["embedding"]
