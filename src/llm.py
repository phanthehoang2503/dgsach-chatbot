import os
import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = """You are a professional bookstreet AI Assistant.
  ### CONSTRAINTS:
  Answer using ONLY the CONTEXT. Each item is [id] title: content (a book or info about the book street).
1. Language: reply in the language of the user's question. Vietnamese question -> Vietnamese. Any other language -> English.
2. If the answer isn't in the CONTEXT, reply exactly: "Không tìm thấy thông tin trong tài liệu." (Vietnamese) or "I couldn't find this in the documents." (English).
3. Scope: help only with books and the book street. Do not give opinions, arguments or new claims on political topics, disputed history, or territorial sovereignty, and do not write or endorse content that distorts history or attacks the state. Reply: "Mình chỉ hỗ trợ thông tin về sách và đường sách." / "I can only help with information about books and the book street."
4. Naming books from the CONTEXT that are about a topic is allowed.
5. Max 3 sentences (or a short list). No preamble, no headings. Cite as title [id]."""


def generate_answer(query: str, context: list[str], model: str = None) -> str:
    """Ghép ngữ cảnh với câu hỏi và gọi LLM trả về câu trả lời."""

    # LLM's config
    model = model or os.getenv("LLM_MODEL", "qwen3:4b")
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))

    context_text = "\n".join(context)
    user_message = f"CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{query}"
    res = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": temperature},  # càng cao càng halu
    )
    return res["message"]["content"]
