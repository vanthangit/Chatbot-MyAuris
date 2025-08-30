# 🤖 Chatbot AI - Nha khoa My Auris

## 📌 Giới thiệu
Dự án xây dựng **Chatbot AI cho Nha khoa My Auris** với mục tiêu tự động trả lời các câu hỏi thường gặp (FAQ) của khách hàng. Chatbot sử dụng mô hình ngôn ngữ lớn (LLM) kết hợp với kỹ thuật **RAG (Retrieval-Augmented Generation)** để cung cấp câu trả lời chính xác dựa trên dữ liệu dịch vụ nha khoa.

---

## 🏗️ Kiến trúc hệ thống
Pipeline chính của hệ thống bao gồm:

1. **Crawl dữ liệu**  
   - Thu thập dữ liệu dịch vụ từ website My Auris bằng `Selenium` + `BeautifulSoup`.
   - Lưu dữ liệu thô (raw) dưới dạng `.json` hoặc `.csv`.

2. **Xử lý & Làm sạch dữ liệu**  
   - Chuẩn hóa văn bản, loại bỏ ký tự thừa.
   - Chuẩn hóa intent (FAQ, chitchat, service).
   - Lưu dữ liệu sạch vào `data/processed`.

3. **Embedding & Lưu trữ vector**  
   - Sử dụng SentenceTransformers để sinh embedding tiếng Việt.
   - Lưu vào **FAISS vector database**.

4. **RAG Pipeline**  
   - Khi người dùng hỏi → Truy xuất dữ liệu liên quan từ FAISS.  
   - LLM (gọi qua API TogetherAI) sinh câu trả lời dựa trên context.  

5. **API & Frontend**  
   - Backend: Flask/FastAPI → expose endpoint `/chat`.  
   - Frontend: Streamlit UI để demo chatbot.  

---

## ⚙️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.9
- **Frameworks & Libraries:**
  - `Selenium`, `BeautifulSoup4` → Crawl dữ liệu
  - `pandas`, `re` → Xử lý dữ liệu
  - `SentenceTransformers`, `FAISS` → Embedding + Vector DB
  - `LangChain` → Quản lý RAG pipeline
  - `Flask`/`FastAPI` → API backend
  - `Streamlit` → Giao diện chatbot
- **LLM Provider:** TogetherAI API (DeepSeek R1 / Llama / GPT)

---



