import logging
import os
from pathlib import Path
import sys
from typing import Optional
import requests
import streamlit as st

# 1. CẤU HÌNH & IMPORT LOGIC GỐC
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from project_config import get_config
    from embedding import MedicalHybridSearch
    from query_optimizer import optimize_query_pipeline
    CONFIG = get_config()
except ImportError as e:
    st.error(f"❌ Không tìm thấy các file logic: {e}")
    st.stop()

# 2. CÁC HÀM LOGIC (GIỮ NGUYÊN)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_lm_studio_url() -> str:
    env_value = os.getenv("LM_STUDIO_URL")
    if env_value: return env_value
    return CONFIG.lm_studio.url

@st.cache_resource
def load_medical_search_engine():
    search_engine = MedicalHybridSearch(
        output_dir=str(CONFIG.paths.search_index_dir),
        abbreviation_file=str(CONFIG.paths.embedding_abbrev_map),
    )
    search_engine.load_index()
    return search_engine

def build_context_from_results(results: list) -> str:
    return "\n".join([f"Tài liệu {i+1}:\nTiêu đề: {d.get('title')}\nNội dung: {d.get('page_content')}\n" for i, d in enumerate(results)])

def generate_response_with_lm_studio(user_query: str, context: str) -> str:
    try:
        url = _get_lm_studio_url()
        payload = {
            "model": CONFIG.lm_studio.model,
            "messages": [
                {"role": "system", "content": f"Bạn là bác sĩ. Trả lời dựa trên context:\n{context}"},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.3
        }
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return "❌ Không thể kết nối với LM Studio. Vui lòng kiểm tra server."

# 3. GIAO DIỆN UI TRẮNG CHUYÊN NGHIỆP
st.set_page_config(page_title="Trợ lý Y tế AI", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Playfair+Display:wght@700&display=swap');
    .stApp { background-color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .main-title { font-family: 'Playfair Display', serif; font-size: 2.5rem; color: #1E293B; text-align: center; margin-top: -50px; }
    .sub-title { color: #64748B; text-align: center; margin-bottom: 40px; font-weight: 300; }
    .stChatMessage { border: 1px solid #E2E8F0 !important; border-radius: 12px !important; margin-bottom: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

# 4. MAIN APP
def main():
    st.markdown('<h1 class="main-title">Hệ thống Trợ lý Y tế AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Trợ lý phân tích y khoa chuyên nghiệp</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.title("Hệ thống")
        if st.button("Xóa lịch sử"):
            st.session_state.messages = []
            st.rerun()

    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                for s in msg["sources"]:
                    with st.expander(f"Nguồn: {s['title']}"):
                        st.write(s['page_content'])

    # Nhập liệu
    if user_input := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # KHỞI TẠO BIẾN ĐỂ LƯU KẾT QUẢ NGOÀI KHỐI STATUS
        ai_response = ""
        final_sources = []

        # Xử lý RAG
        with st.status("Đang xử lý...", expanded=True) as status:
            try:
                search_engine = load_medical_search_engine()
                optimized = optimize_query_pipeline(user_input, search_engine.abbreviation_map, use_lm_studio=True, lm_studio_url=_get_lm_studio_url())
                
                results = search_engine.hybrid_search(optimized, top_k=CONFIG.app.top_k)
                
                if results:
                    context = build_context_from_results(results)
                    ai_response = generate_response_with_lm_studio(optimized, context)
                    final_sources = results[:1]
                    status.update(label="Phân tích hoàn tất!", state="complete")
                else:
                    ai_response = "Xin lỗi, tôi không tìm thấy tài liệu liên quan."
                    status.update(label="Không có dữ liệu", state="error")
            except Exception as e:
                status.update(label="Lỗi hệ thống", state="error")
                ai_response = f"Lỗi: {str(e)}"

        # HIỂN THỊ KẾT QUẢ (NGOÀI KHỐI STATUS ĐỂ TRÁNH LỖI NESTED EXPANDER)
        if ai_response:
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                for s in final_sources:
                    with st.expander(f"📚 Tài liệu tham khảo: {s['title']}"):
                        st.info(s['page_content'])
            
            st.session_state.messages.append({
                "role": "assistant", "content": ai_response, "sources": final_sources
            })

if __name__ == "__main__":
    main()