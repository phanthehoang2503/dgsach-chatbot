import os
import psycopg
import ollama
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from google import genai
from google.genai import types

load_dotenv()

# khởi tạo client
client = genai.Client()
ollama_client = ollama.Client(host="http://127.0.0.1:11434")

# embedding
# def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
#     res = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=text,
#         config=types.EmbedContentConfig(
#             task_type=task_type,
#             output_dimensionality=768,  # khớp với embeddinggemma (768 chiều)
#         ),
#     )
#     return res.embeddings[0].values


def get_embedding(text: str) -> list[float]:
    res = ollama_client.embeddings(model="qwen3-embedding:0.6B", prompt=text)
    return res["embedding"]


# kết nối db vs lưu vector
conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    autocommit=True,
)
register_vector(conn)

with conn.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS documents;")
    cur.execute(
        """
  CREATE TABLE IF NOT EXISTS documents (
            id serial PRIMARY KEY,
            content text,
            embedding vector(1024)
            );
  """
    )
    cur.execute("DELETE FROM documents;")

    with open("data.txt", "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f.read().split("\n") if line.strip()]

    print("Đang tạo embedding và lưu vào vectordb")
    for chunk in chunks:
        vec = get_embedding(chunk)
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s);", (chunk, vec)
        )
    print(f"Đã nạp {len(chunks)} đoạn vào vectorDB")

    # truy xuat vector
    user_query = input()
    # question_vec = get_embedding(user_query, task_type="RETRIEVAL_QUERY")
    question_vec = get_embedding(user_query)

    cur.execute(
        """
    SELECT content, embedding <=> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT 2
  """,
        (question_vec,),
    )
    rows = cur.fetchall()
    retrieved_ctx = [r[0] for r in rows]
    ctx_text = "\n".join(retrieved_ctx)
    print("Đoạn tìm thấy từ vectorDB")
    print(ctx_text)
conn.close()


# Response
SYSTEM_PROMPT = """You are a professional bookstreet AI Assistant.
  ### CONSTRAINTS:
  Answer using ONLY the CONTEXT. Each item is [id] title: content (a book or info about the book street).
1. Language: reply in the language of the user's question. Vietnamese question -> Vietnamese. Any other language -> English.
2. If the answer isn't in the CONTEXT, reply exactly: "Không tìm thấy thông tin trong tài liệu." (Vietnamese) or "I couldn't find this in the documents." (English).
3. Scope: help only with books and the book street. Do not give opinions, arguments or new claims on political topics, disputed history, or territorial sovereignty, and do not write or endorse content that distorts history or attacks the state. Reply: "Mình chỉ hỗ trợ thông tin về sách và đường sách." / "I can only help with information about books and the book street."
4. Naming books from the CONTEXT that are about a topic is allowed.
5. Max 3 sentences (or a short list). No preamble, no headings. Cite as title [id]."""
USER_MESSAGE = f"""CONTEXT:
  {ctx_text}
USER QUESTION:
  {user_query}"""

# res = client.models.generate_content(
#     model="gemini-3.8-flash",
#     contents=USER_MESSAGE,
#     config=types.GenerateContentConfig(
#         system_instruction=SYSTEM_PROMPT,
#         temperature=0.2,  # Đặt thấp để bám chặt vào tài liệu
#     )
# )

res = ollama_client.chat(
    model="qwen3:4b",  # đổi thành model bạn đã pull
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ],
    options={"temperature": 0.2},  # thấp để bám chặt vào tài liệu
)

print(f"Chatbox: \n")
print(res["message"]["content"])
