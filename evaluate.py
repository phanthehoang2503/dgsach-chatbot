import unicodedata

from src.llm import generate_answer
from src.retriever import retrieve_ctx, retrieve_with_scores

TOP_K = 5
REFUSAL_PHRASES = ["không tìm thấy", "couldn't find", "chỉ hỗ trợ", "can only help"]


def ask(question, answers, chunks):
    """answers: mọi ý đều phải có; dùng 'a|b' cho các cách viết tương đương."""
    return {"question": question, "answers": answers, "chunks": chunks, "refuse": False}


def refuse(question):
    return {"question": question, "answers": [], "chunks": [], "refuse": True}


CASES = [
    # Tra cứu thông tin
    ask(
        "Thứ Bảy đường sách Lục Bình mở cửa đến mấy giờ?",
        ["22:00|22h|22 giờ|10 giờ tối"],
        ["thành lập năm 2019"],
    ),
    ask(
        "Gửi ô tô ở đường sách Lục Bình giá bao nhiêu?",
        ["20.000"],
        ["Gửi xe tại đường sách Lục Bình"],
    ),
    ask(
        "Chợ sách cũ tổ chức khi nào?",
        ["Chủ nhật đầu tiên"],
        ["Sự kiện tại đường sách Lục Bình"],
    ),
    ask(
        "Có được bay flycam ở đường sách Lục Bình không?",
        ["không được|cấm|không cho phép"],
        ["Quy định tại đường sách Lục Bình"],
    ),
    ask(
        "Đi xe buýt tuyến nào đến đường sách Lục Bình?",
        ["27", "63"],
        ["Đi đến đường sách Lục Bình"],
    ),
    ask("Quán cà phê Trang Cuối ở gian nào?", ["D1"], ["Bố trí gian hàng"]),
    ask(
        "What time does Lục Bình book street close on Sunday?",
        ["22:00|10 PM|10:00 PM|22h"],
        ["thành lập năm 2019"],
    ),
    # Dễ nhầm
    ask(
        "Gửi xe máy ở phố sách Hải Đăng giá bao nhiêu?",
        ["3.000"],
        ["Phố sách Hải Đăng ở"],
    ),
    ask("Sách Bóng Ma Trên Đê giá bao nhiêu?", ["95.000"], ["Sách: Bóng Ma Trên Đê."]),
    ask("Sách Bóng Ma Dưới Đê giá bao nhiêu?", ["105.000"], ["Sách: Bóng Ma Dưới Đê."]),
    ask(
        "Ai là tác giả của Người Dệt Thời Gian?",
        ["Cao Vĩnh Nghi"],
        ["Sách: Người Dệt Thời Gian."],
    ),
    # Tìm theo nội dung
    ask(
        "Truyện nào kể về thám tử mất trí nhớ 13 tiếng?",
        ["Hồ Sơ Mười Ba Giờ"],
        ["Sách: Hồ Sơ Mười Ba Giờ."],
    ),
    ask(
        "Sách nào giải thích lạm phát cho người mới học kinh tế?",
        ["Tiền Không Biết Ngủ"],
        ["Sách: Tiền Không Biết Ngủ."],
    ),
    ask(
        "Có sách nào cho bé 5 tuổi không?",
        ["Chú Rùa Đi Tìm Mặt Trăng"],
        ["Sách: Chú Rùa Đi Tìm Mặt Trăng."],
    ),
    # Liệt kê
    ask(
        "Liệt kê các sách trinh thám.",
        [
            "Hồ Sơ Mười Ba Giờ",
            "Bóng Ma Trên Đê",
            "Bóng Ma Dưới Đê",
            "Chiếc Đồng Hồ Ngừng Chạy",
            "Thư Gửi Kẻ Vắng Mặt",
        ],
        [
            "Sách: Hồ Sơ Mười Ba Giờ.",
            "Sách: Bóng Ma Trên Đê.",
            "Sách: Bóng Ma Dưới Đê.",
            "Sách: Chiếc Đồng Hồ Ngừng Chạy.",
            "Sách: Thư Gửi Kẻ Vắng Mặt.",
        ],
    ),
    ask(
        "Từ Bảo Châu viết những sách nào?",
        ["Mật Mã Từ Sao Kim", "Trạm Dừng Cuối Của Ngân Hà"],
        ["Sách: Mật Mã Từ Sao Kim.", "Sách: Trạm Dừng Cuối Của Ngân Hà."],
    ),
    # Phải từ chối
    refuse("Sách Harry Potter giá bao nhiêu?"),
    refuse("Đường sách Lục Bình có bán vé máy bay không?"),
    refuse("Thủ đô của Pháp là gì?"),
    refuse("iPhone mới nhất giá bao nhiêu?"),
]


def norm(text):
    return unicodedata.normalize("NFC", text).lower()


def check_answer(case, answer):
    answer = norm(answer)
    if case["refuse"]:
        return any(phrase in answer for phrase in REFUSAL_PHRASES)
    return all(
        any(norm(option) in answer for option in expected.split("|"))
        for expected in case["answers"]
    )


def chunk_distances(case, rows):
    """Distance của từng đoạn bắt buộc; None nếu đoạn đó không nằm trong top_k."""
    result = []
    for chunk in case["chunks"]:
        matches = [d for content, d in rows if norm(chunk) in norm(content)]
        result.append(matches[0] if matches else None)
    return result


def main():
    answer_ok = retrieval_ok = answerable = 0
    required_distances, refuse_distances = [], []

    for case in CASES:
        question = case["question"]
        rows = retrieve_with_scores(question, TOP_K)
        answer = generate_answer(question, retrieve_ctx(question, TOP_K))
        passed = check_answer(case, answer)
        answer_ok += passed
        retrieval_note = "từ chối"

        if case["refuse"]:
            refuse_distances.append(rows[0][1])
        else:
            answerable += 1
            distances = chunk_distances(case, rows)
            found = None not in distances
            retrieval_ok += found
            retrieval_note = "tìm đủ" if found else "THIẾU ĐOẠN"
            if found:
                required_distances.append(max(distances))

        print(
            f"[{'OK ' if passed else 'SAI'}] ({retrieval_note}) d={rows[0][1]:.3f} | {question}"
        )
        if not passed:
            print(f"      Trả lời: {answer}")

    print("\n" + "=" * 60)
    print(f"Câu trả lời đúng: {answer_ok}/{len(CASES)}")
    print(f"Truy xuất đủ đoạn: {retrieval_ok}/{answerable}")
    if required_distances:
        print(f"Distance lớn nhất cần giữ: {max(required_distances):.3f}")
    print(f"Distance nhỏ nhất của câu phải từ chối: {min(refuse_distances):.3f}")


if __name__ == "__main__":
    main()
