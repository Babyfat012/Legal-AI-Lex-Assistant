import streamlit as st
import httpx
import os
import sys
import asyncio
from datetime import datetime
import pandas as pd
import uuid

# --- Fix PYTHONPATH ---
# Thêm thư mục gốc (rag-demo) vào sys.path để có thể import từ 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.database import IngestLog, SessionLocal, init_db
from sqlalchemy.orm import Session

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "admin123")

# Page Config
st.set_page_config(
    page_title="Lex Assistant | Admin Dashboard",
    page_icon="⚖️",
    layout="wide",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Cards/Containers */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-12w0qpk {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.4);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
    }
    
    /* Success/Error Messages */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

async def log_ingest(file_name: str, status: str, chunk_count: int = 0, elapsed: int = 0, error: str = None):
    db = SessionLocal()
    try:
        log = IngestLog(
            file_name=file_name,
            status=status,
            chunk_count=chunk_count,
            elapsed_secs=elapsed,
            error_msg=error,
            upload_at=datetime.utcnow()
        )
        db.add(log)
        db.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        db.close()

async def ingest_file_api(file, recreate: bool, source_url: str = ""):
    start_time = datetime.utcnow()
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            files = {"file": (file.name, file.getvalue(), file.type)}
            data = {
                "recreate_collection": str(recreate).lower(),
                "source_url": source_url,
            }
            response = await client.post(
                f"{API_BASE_URL}/api/v1/ingest/upload",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                await log_ingest(
                    file_name=file.name,
                    status="success",
                    chunk_count=result.get("chunks_stored", 0),
                    elapsed=(datetime.utcnow() - start_time).seconds
                )
                return True, result
            else:
                error_msg = response.text
                await log_ingest(file_name=file.name, status="failed", error=error_msg)
                return False, error_msg
    except Exception as e:
        await log_ingest(file_name=file.name, status="failed", error=str(e))
        return False, str(e)

async def delete_file_api(file_name: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_BASE_URL}/api/v1/collection/documents/{file_name}")
            if response.status_code == 200:
                db = SessionLocal()
                log = db.query(IngestLog).filter(IngestLog.file_name == file_name).first()
                if log:
                    log.status = "deleted"
                    db.commit()
                db.close()
                return True
            return False
    except Exception as e:
        st.error(f"Error deleting: {e}")
        return False

# --- Auth ---
def check_admin_auth() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h1 style='text-align: center;'>🔐 Admin Login</h1>", unsafe_allow_html=True)
            secret = st.text_input("Mật khẩu quản trị", type="password")
            if st.button("Đăng nhập"):
                if secret == ADMIN_SECRET:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Sai mật khẩu!")
        return False
    return True

# --- Tabs ---
def render_upload_tab():
    st.markdown("## 📤 Ingest Tài Liệu Mới")
    st.info("Hỗ trợ các định dạng .pdf và .docx. Dữ liệu sẽ được xử lý qua pipeline Agentic RAG.")
    
    uploaded_files = st.file_uploader(
        "Kéo thả files vào đây",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    source_url = st.text_input(
        "🔗 URL nguồn gốc của tài liệu",
        placeholder="https://thuvienphapluat.vn/van-ban/...",
        help=(
            "URL trang web chứa nội dung tài liệu (VD: Thư viện Pháp luật). "
            "Chatbot sẽ dùng URL này để tạo link highlight trực tiếp đến đoạn văn liên quan "
            "khi trả lời câu hỏi (#:~:text=). "
            "Để trống nếu không có URL."
        ),
    )
    
    recreate = st.checkbox(
        "⚠️ Làm mới Collection (XÓA TOÀN BỘ dữ liệu cũ trước khi ingest)",
        value=False,
        help="Chỉ áp dụng cho file đầu tiên trong batch."
    )
    
    if uploaded_files and st.button("🚀 Bắt đầu Ingest"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Đang xử lý: {file.name}...")
            is_success, info = asyncio.run(
                ingest_file_api(file, recreate and (i == 0), source_url=source_url.strip())
            )
            
            if is_success:
                st.success(f"✅ **{file.name}**: {info['chunks_stored']} chunks stored.")
            else:
                st.error(f"❌ **{file.name}**: {info}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Hoàn thành!")
        st.balloons()

def render_manage_tab():
    st.markdown("## 📋 Danh Sách Tài Liệu")
    
    db = SessionLocal()
    logs = db.query(IngestLog).order_by(IngestLog.upload_at.desc()).all()
    db.close()
    
    if not logs:
        st.write("Chưa có dữ liệu.")
        return
    
    df = pd.DataFrame([
        {
            "Tên File": l.file_name,
            "Ngày Upload": l.upload_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Chunks": l.chunk_count,
            "Trạng Thái": l.status,
            "Thời Gian (s)": l.elapsed_secs,
            "Error": l.error_msg
        } for l in logs
    ])
    
    # Filter
    search = st.text_input("🔍 Tìm kiếm theo tên file")
    if search:
        df = df[df["Tên File"].str.contains(search, case=False)]
    
    st.dataframe(df, use_container_width=True)
    
    # Simple Actions Row
    st.markdown("### 🛠 Thao tác")
    selected_file = st.selectbox("Chọn file để xóa", ["--"] + df["Tên File"].tolist())
    if selected_file != "--":
        if st.warning(f"Bạn có chắc chắn muốn xóa tất cả chunks của '{selected_file}'?"):
            if st.button("XÁC NHẬN XÓA"):
                if asyncio.run(delete_file_api(selected_file)):
                    st.success(f"Đã xóa {selected_file}")
                    st.rerun()

def render_collection_tab():
    st.markdown("## 📊 Thông Số Qdrant Collection")
    
    try:
        response = httpx.get(f"{API_BASE_URL}/api/v1/collection/info")
        if response.status_code == 200:
            info = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📦 Tổng Points", f"{info['points_count']:,}")
            col2.metric("📐 Dimension", info["dimension"])
            col3.metric("🚦 Status", info["status"].upper())
            col4.metric("📊 Segments", info["segments_count"])
            
            st.markdown("### Vector Configurations")
            st.json(info)
        else:
            st.error("Không thể lấy thông tin từ Backend.")
    except Exception as e:
        st.error(f"Lỗi kết nối Backend: {e}")

# --- Main ---
def main():
    # Initialize DB table
    init_db()
    
    st.sidebar.markdown("<h1 style='text-align: center; color: #38bdf8;'>Admin Panel</h1>", unsafe_allow_html=True)
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1053/1053331.png", width=100) # Placeholder for legal icon
    
    if not check_admin_auth():
        return
    
    st.sidebar.divider()
    menu = ["📤 Upload Tài Liệu", "📋 Quản Lý Tài Liệu", "📊 Thống Kê Collection"]
    choice = st.sidebar.radio("Điều hướng", menu)
    
    if st.sidebar.button("🚪 Đăng xuất"):
        st.session_state.authenticated = False
        st.rerun()

    if choice == "📤 Upload Tài Liệu":
        render_upload_tab()
    elif choice == "📋 Quản Lý Tài Liệu":
        render_manage_tab()
    elif choice == "📊 Thống Kê Collection":
        render_collection_tab()

if __name__ == "__main__":
    main()
