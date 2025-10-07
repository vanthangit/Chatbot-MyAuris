import streamlit as st
import requests
import uuid
import re

# Thiết lập cấu hình trang
st.set_page_config(page_title="Chatbot Nha Khoa MyAuris", page_icon="🦷")
st.title("🦷 Chatbot Tư Vấn Nha Khoa - MyAuris")

# Tạo session_id duy nhất cho mỗi phiên người dùng
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Khởi tạo lịch sử hội thoại với giới hạn
MAX_HISTORY = 50
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nhập và kiểm tra URL backend
if "flask_api_url" not in st.session_state:
    st.session_state.flask_api_url = "http://127.0.0.1:5000"

# Kiểm tra tính hợp lệ của URL backend
def is_valid_url(url):
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return re.match(pattern, url) is not None

if st.session_state.flask_api_url:
    if is_valid_url(st.session_state.flask_api_url):
        try:
            response = requests.get(f"{st.session_state.flask_api_url}/", timeout=5)
            if response.status_code == 200:
                st.success(f"Đã kết nối backend: {st.session_state.flask_api_url}/chat")
            else:
                st.error(f"URL backend không phản hồi. Mã lỗi: {response.status_code}")
        except requests.RequestException:
            st.error("Không thể kết nối đến backend. Vui lòng kiểm tra URL.")
    else:
        st.error("URL backend không hợp lệ. Vui lòng nhập URL đúng định dạng (VD: http://localhost:5000).")


# Hiển thị lịch sử hội thoại
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if st.session_state.flask_api_url and is_valid_url(st.session_state.flask_api_url):
    if user_input := st.chat_input("Hỏi chatbot nha khoa..."):
        # Thêm tin nhắn người dùng vào lịch sử
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Gửi truy vấn tới backend
        with st.chat_message("assistant"):
            with st.spinner("Đang trả lời..."):
                try:
                    payload = {
                        "message": {"content": user_input},
                        "sessionId": st.session_state.session_id,
                        "history": st.session_state.chat_history
                    }
                    response = requests.post(
                        f"{st.session_state.flask_api_url}/chat",
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    if result.get("status") == "success":
                        assistant_reply = result.get("content", "Không có phản hồi.")
                    else:
                        assistant_reply = f"Lỗi từ backend: {result.get('message', 'Không xác định')}"
                except requests.RequestException as e:
                    assistant_reply = f"Không thể kết nối backend: {str(e)}"

            st.markdown(assistant_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        # Giới hạn lịch sử hội thoại
        if len(st.session_state.chat_history) > MAX_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]