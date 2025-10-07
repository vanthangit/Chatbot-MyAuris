import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from embedding.embedding import EmbeddingStore
from router.router import Route, SemanticRouter
from router.samplequestions import serviceSample, chitchatSample
from reflection.reflection import Reflection
from google import genai
from router.wrapper import BGEM3Wrapper

app = Flask(__name__)
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../chatbot-myauris-auth.json"

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "chatbot-myauris")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_ID = "gemini-2.0-flash"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../crawl_data/data/clean/myauris_allqa.jsonl"))
INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../faiss_index.bin"))

# Khởi tạo EmbeddingStore dùng BGEM3FlagModel
embedding = EmbeddingStore(DATA_PATH, INDEX_PATH)

# Reflection
reflection = Reflection(client, MODEL_ID)

# Router setup
serviceRoute = Route("service", serviceSample)
chitchatRoute = Route("chitchat", chitchatSample)
embedding_wrapper = BGEM3Wrapper(embedding.model)
semantic_router = SemanticRouter(embedding_wrapper, [serviceRoute, chitchatRoute])

# -------------------
# Hàm generate answer
# -------------------
def generate_answer(query, docs):
    context = "\n".join(docs) or "Không tìm thấy thông tin liên quan."
    prompt = f"""Bạn là trợ lý tư vấn nha khoa của My Auris. Trả lời đầy đủ, đúng câu hỏi bằng tiếng Việt. Nếu câu hỏi không liên quan đến nha khoa, thông báo rằng bạn chỉ hỗ trợ tư vấn nha khoa:
Thông tin: {context}
Câu hỏi: {query}
Trả lời:"""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Lỗi khi gọi Vertex AI: {str(e)}"

def generate_chitchat_answer(query):
    prompt = f"""Bạn là trợ lý tư vấn nha khoa My Auris. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt.
Nếu câu hỏi liên quan đến dịch vụ nha khoa, bạn phải nói sẽ chuyển cho bộ phận chuyên môn.
Câu hỏi: {query}
Trả lời:"""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Lỗi khi gọi Vertex AI (chitchat): {str(e)}"

# -------------------
# API endpoint
# -------------------
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get("message", {}).get("content", "")
        history = data.get("history", [])

        if not user_query:
            return jsonify({"status": "error", "message": "Missing message content"}), 400

        # Reflection step
        rewritten_query = reflection.rewrite(history, user_query)

        # Route decision
        score, route = semantic_router.guide(rewritten_query)

        if route == "service":
            top_docs = embedding.retrieve(rewritten_query)
            answer = generate_answer(rewritten_query, top_docs)
        else:
            answer = generate_chitchat_answer(rewritten_query)

        return jsonify({"status": "success", "content": answer, "route": route, "score": float(score)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)