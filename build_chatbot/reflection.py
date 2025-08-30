from together import Together
import os
import re

class Reflection:
    def __init__(self, client: Together, model_name: str, window_size: int = 3):
        self.client = client
        self.model_name = model_name
        self.window_size = window_size

    def rewrite(self, history, question: str) -> str:
        try:
            recent_history = history[-self.window_size:]
            history_text = "\n".join(
                [f"{h['role'].capitalize()}: {h['content']}" for h in recent_history]
            )
            prompt = f"""
Bạn là trợ lý tư vấn nha khoa My Auris. 
Nhiệm vụ của bạn: viết lại câu hỏi của người dùng sao cho ngắn gọn và rõ ràng, dựa trên lịch sử hội thoại trước đó.

Lịch sử hội thoại:
{history_text}

Câu hỏi mới: "{question}"

Viết lại câu hỏi (chỉ trả về câu hỏi đã viết lại, không thêm gì khác).
"""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=None
            )

            rewritten = response.choices[0].message.content
            cleaned = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()

            print("\n========== REFLECTION DEBUG ==========")
            print(f"Original Q: {question}")
            print(f"History: {history_text}")
            print(f"Rewritten Q: {cleaned}")
            print("=====================================\n")

            return cleaned
        except Exception as e:
            return question
