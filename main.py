import sys
from src.retriever import retrieve_ctx
from src.llm import generate_answer


def main():
    print("=" * 60)
    print("CHATBOT ĐƯỜNG SÁCH & TÌM KIẾM SÁCH (RAG)")
    print("Nhập câu hỏi của bạn. Gõ 'exit', 'quit' hoặc 'q' để thoát.")
    print("=" * 60)

    while True:
        try:
            query = input("\n Bạn: ").strip()
            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("Tạm biệt và hẹn gặp lại.")
                break

            # 1. Truy xuất
            ctx = retrieve_ctx(query, top_k=2)

            # 2. LLM generate câu trả lời
            print("Đang xử lý câu trả lời...")
            ans = generate_answer(query, ctx)

            print(f"Chat bot: \n{ans}")
        except KeyboardInterrupt:
            print(f"\n Bái bai")
            sys.exit(0)
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    main()
