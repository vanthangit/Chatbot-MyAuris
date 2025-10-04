import os
import re
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from router import Route, SemanticRouter
from samplequestions import serviceSample, chitchatSample
from reflection import Reflection

# Vertex AI
from google import genai
from rich import print as rich_print

app = Flask(__name__)
load_dotenv()

# Google Cloud config
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "chatbot-myauris")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_ID = "gemini-2.0-flash"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Đường dẫn tới dữ liệu và FAISS index
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../crawl_data/data/clean/myauris_allqa.jsonl"))
INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "faiss_index"))

# Load dữ liệu
docs = []
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                docs.append(item["text"])
            except json.JSONDecodeError:
                continue
else:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

documents = [Document(page_content=doc) for doc in docs]

# Khởi tạo embedding và FAISS
embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
if os.path.exists(INDEX_PATH):
    vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INDEX_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Router setup
serviceRoute = Route("service", serviceSample)
chitchatRoute = Route("chitchat", chitchatSample)
semantic_router = SemanticRouter(embedding_model, [serviceRoute, chitchatRoute])

# Reflection vẫn giữ (nếu bạn muốn dùng LLM để rewrite query, có thể thay bằng Vertex AI luôn)
reflection = Reflection(client, MODEL_ID)

def generate_answer(query, docs):
    try:
        context = "\n".join([doc.page_content for doc in docs]) or "Không tìm thấy thông tin liên quan."
        prompt = f"""Bạn là trợ lý tư vấn nha khoa của My Auris. Trả lời đầy đủ, đúng câu hỏi bằng tiếng Việt. Nếu câu hỏi không liên quan đến nha khoa, thông báo rằng bạn chỉ hỗ trợ tư vấn nha khoa:
Thông tin: {context}
Câu hỏi: {query}
Trả lời:"""

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )

        answer = response.text.strip()
        cleaned_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return cleaned_answer
    except Exception as e:
        return f"Lỗi khi gọi Vertex AI: {str(e)}"

def generate_chitchat_answer(query):
    try:
        prompt = f"""Bạn là trợ lý tư vấn nha khoa My Auris. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt.
Nếu câu hỏi liên quan đến dịch vụ nha khoa, bạn có thể nói sẽ chuyển cho bộ phận chuyên môn.
Câu hỏi: {query}
Trả lời:"""

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )

        answer = response.text.strip()
        cleaned_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return cleaned_answer
    except Exception as e:
        return f"Lỗi khi gọi Vertex AI (chitchat): {str(e)}"


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get("message", {}).get("content", "")
        history = data.get("history", [])

        # Reflection step
        rewritten_query = reflection.rewrite(history, user_query)

        if not user_query:
            return jsonify({"status": "error", "message": "Missing message content"}), 400

        # Route decision
        score, route = semantic_router.guide(rewritten_query)

        if route == "service":
            top_docs = retriever.get_relevant_documents(rewritten_query)
            answer = generate_answer(rewritten_query, top_docs)
        else:  # chitchat
            answer = generate_chitchat_answer(rewritten_query)

        return jsonify({"status": "success", "content": answer, "route": route, "score": float(score)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
