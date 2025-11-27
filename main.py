"""
Chatbot Hành Chính Việt Nam - RAG System với FAISS và Gradio
"""
import os
import gradio as gr
from typing import List, Tuple, Dict, Optional
import logging
from dotenv import load_dotenv
import shutil

from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.reranker import Reranker
from utils.database import Database
from utils.auth import AuthManager
from utils.natural_language import is_natural_question, get_natural_response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_STORAGE_DIR = "pdfs"
FIXED_FILES_DIR = "fixed_pdfs"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(FIXED_FILES_DIR, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

pdf_processor = PDFProcessor(chunk_size=400, overlap=100)
vector_store = VectorStore()
reranker = Reranker()

# Khởi tạo database và auth
try:
    database = Database()
    auth_manager = AuthManager(database)
    logger.info("Đã khởi tạo database và auth manager")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo database: {str(e)}")
    database = None
    auth_manager = None


def get_llm_client():
    """Khởi tạo LLM client (Groq, Together, hoặc OpenRouter)"""
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("Đã kết nối Groq API")
            return client, "groq", "llama-3.3-70b-versatile"
        except Exception as e:
            logger.warning(f"Không thể kết nối Groq: {str(e)}")
    
    logger.warning("Chưa cấu hình API key cho LLM. Vui lòng thêm GROQ_API_KEY vào file .env")
    return None, None, None


llm_client, llm_provider, llm_model = get_llm_client()


def generate_answer(query: str, context_chunks: List[Dict], selected_file: Optional[str] = None) -> str:
    """
    Sinh câu trả lời từ LLM dựa trên context (cải thiện để tăng độ chính xác và đầy đủ)
    
    Args:
        query: Câu hỏi của người dùng
        context_chunks: Các chunk context liên quan
        selected_file: File được chọn (nếu có)
        
    Returns:
        Câu trả lời với định dạng markdown
    """
    if not context_chunks:
        return "Trong các tài liệu đã upload chưa có thông tin về nội dung này."
    
    context_by_file = {}
    for chunk in context_chunks:
        filename = chunk['filename']
        page = chunk.get('page_number', 0)
        key = f"{filename}_page_{page}"
        if key not in context_by_file:
            context_by_file[key] = {
                "filename": filename,
                "page": page,
                "texts": []
            }
        context_by_file[key]["texts"].append(chunk['text'])
    
    sorted_keys = sorted(context_by_file.keys(), key=lambda k: (context_by_file[k]['filename'], context_by_file[k]['page']))
    
    context_parts = []
    for key in sorted_keys:
        data = context_by_file[key]
        combined_text = " ".join(data["texts"])
        combined_text = " ".join(combined_text.split())
        context_parts.append(combined_text)
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    file_context = f" (trong file: {selected_file})" if selected_file else ""
    prompt = f"""Bạn là trợ lý hành chính Việt Nam cực kỳ chính xác và chuyên nghiệp. 
Nhiệm vụ của bạn là trả lời câu hỏi dựa HOÀN TOÀN vào các tài liệu tham khảo được cung cấp bên dưới.

TÀI LIỆU THAM KHẢO{file_context}:
{context_text}

CÂU HỎI: {query}

YÊU CẦU TRẢ LỜI (QUAN TRỌNG - PHẢI TUÂN THỦ):
1. **ĐỌC KỸ TOÀN BỘ TÀI LIỆU THAM KHẢO**: Phân tích tất cả các đoạn văn bản được cung cấp, đặc biệt chú ý đến các câu văn hoàn chỉnh và các đoạn liên quan. Nội dung có thể được phân chia giữa các phần khác nhau, hãy kết hợp tất cả thông tin liên quan.

2. **TRẢ LỜI ĐẦY ĐỦ - KHÔNG ĐƯỢC CẮT CỤT**: 
   - Nếu trong tài liệu có câu như "được quy định như sau:" hoặc "bao gồm:" thì BẮT BUỘC phải liệt kê đầy đủ nội dung tiếp theo.
   - Nếu có danh sách, bảng, hoặc các mục liệt kê, phải trích dẫn ĐẦY ĐỦ tất cả các mục.
   - KHÔNG được dừng lại ở giữa chừng, KHÔNG được để câu trả lời bị cắt cụt.
   - Nếu thông tin dài, vẫn phải trích dẫn đầy đủ, có thể chia thành nhiều đoạn.
   - Kết hợp thông tin từ các phần khác nhau của tài liệu nếu chúng liên quan đến cùng một chủ đề.

3. **SỬ DỤNG ĐỊNH DẠNG MARKDOWN ĐỂ LÀM ĐẸP**:
   - Sử dụng **bold** cho các tiêu đề và điểm quan trọng: **Tiêu đề**
   - Sử dụng *italic* cho nhấn mạnh: *nhấn mạnh*
   - Sử dụng danh sách có dấu đầu dòng (-) hoặc đánh số (1., 2., 3.) cho các mục liệt kê
   - Sử dụng > cho trích dẫn quan trọng
   - Sử dụng `code` cho các số, mã, hoặc thuật ngữ kỹ thuật
   - Chia thành các đoạn văn rõ ràng với khoảng trắng giữa các đoạn

4. **CẤU TRÚC TRẢ LỜI**:
   - Bắt đầu với một câu tóm tắt ngắn gọn (nếu phù hợp)
   - Trình bày thông tin theo cấu trúc logic, có thể chia thành các phần nhỏ với tiêu đề phụ
   - Sử dụng danh sách để liệt kê các điểm quan trọng
   - Kết hợp thông tin từ nhiều phần của tài liệu một cách mạch lạc

5. **NGÔN NGỮ**: Sử dụng ngôn ngữ hành chính chuẩn mực, rõ ràng, dễ hiểu.

6. **GIỚI HẠN**: 
   - KHÔNG được tự bịa thêm thông tin bên ngoài tài liệu.
   - KHÔNG được nói "dựa trên kiến thức của tôi" hoặc các cụm từ tương tự.
   - KHÔNG được thêm trích dẫn nguồn dạng "[Tên file - Trang X]" vào câu trả lời.
   - Nếu không tìm thấy thông tin chính xác trong tài liệu, hãy trả lời: "Trong các tài liệu đã upload chưa có thông tin về nội dung này."

**LƯU Ý ĐẶC BIỆT**: Đảm bảo rằng câu trả lời của bạn HOÀN CHỈNH và ĐẦY ĐỦ. Nếu trong tài liệu có câu dẫn như "như sau:", "bao gồm:", "cụ thể:", v.v., bạn PHẢI trích dẫn đầy đủ nội dung tiếp theo, không được dừng lại ở đó. Hãy kết hợp thông tin từ các phần khác nhau của tài liệu nếu chúng cùng đề cập đến chủ đề được hỏi.

Hãy trả lời một cách chi tiết, đầy đủ và có định dạng đẹp:
"""
    
    if llm_client is None:
        return f"""⚠️ Chưa cấu hình LLM API key. Đây là thông tin tìm được từ tài liệu:

{context_text}

Vui lòng thêm GROQ_API_KEY vào file .env để chatbot có thể trả lời tự động."""
    
    try:
        if llm_provider in ["groq"]:
            try:
                response = llm_client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4096  # Tăng max_tokens lên 4096 để đảm bảo trả lời đầy đủ
                )
                answer = response.choices[0].message.content
                if answer:
                    answer_clean = answer.strip()
                    incomplete_patterns = [
                        answer_clean.endswith('như sau:'),
                        answer_clean.endswith('như sau'),
                        answer_clean.endswith('bao gồm:'),
                        answer_clean.endswith('bao gồm'),
                        answer_clean.endswith('cụ thể:'),
                        answer_clean.endswith('cụ thể'),
                        answer_clean.endswith('gồm:'),
                        (answer_clean.endswith(':') and len(answer_clean.split('\n')) < 3)  # Kết thúc bằng : nhưng quá ngắn
                    ]
                    
                    if any(incomplete_patterns):
                        logger.warning("Phát hiện câu trả lời có thể bị cắt cụt, thử lại với max_tokens cao hơn...")
                        try:
                            response = llm_client.chat.completions.create(
                                model=llm_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.1,
                                max_tokens=8192  # Tăng lên 8192 nếu cần
                            )
                            new_answer = response.choices[0].message.content
                            if len(new_answer) > len(answer):
                                answer = new_answer
                                logger.info("Đã lấy được câu trả lời đầy đủ hơn")
                        except Exception as retry_error:
                            logger.warning(f"Không thể retry với max_tokens cao hơn: {str(retry_error)}")
                
                return answer
            except Exception as model_error:
                if llm_provider == "groq":
                    logger.warning(f"Model {llm_model} không khả dụng, thử model dự phòng...")
                    fallback_models = ["mistral-saba-24b", "llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
                    for fallback_model in fallback_models:
                        try:
                            logger.info(f"Thử model dự phòng: {fallback_model}")
                            response = llm_client.chat.completions.create(
                                model=fallback_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.1,
                                max_tokens=4096
                            )
                            logger.info(f"Thành công với model: {fallback_model}")
                            answer = response.choices[0].message.content
                            return answer
                        except Exception as e2:
                            logger.warning(f"Model {fallback_model} cũng không khả dụng: {str(e2)}")
                            continue
                    logger.error(f"Tất cả models đều không khả dụng")
                    raise model_error
                else:
                    raise model_error
        else:
            return f"⚠️ LLM provider không được hỗ trợ. Thông tin từ tài liệu:\n\n{context_text}"
    except Exception as e:
        logger.error(f"Lỗi khi gọi LLM: {str(e)}")
        return f"⚠️ Lỗi khi tạo câu trả lời: {str(e)}\n\nThông tin từ tài liệu:\n\n{context_text}"


def process_pdfs(files: List, progress=gr.Progress()):
    """
    Xử lý nhiều file PDF với progress bar
    
    Args:
        files: List các file PDF upload
        progress: Gradio progress tracker
    """
    if not files:
        gr.Error("Vui lòng chọn ít nhất một file PDF")
        return
    
    try:
        if progress:
            progress(0, desc="Đang sao chép file...")
        pdf_paths = []
        for i, file in enumerate(files):
            filename = os.path.basename(file.name)
            dest_path = os.path.join(PDF_STORAGE_DIR, filename)
            shutil.copy(file.name, dest_path)
            pdf_paths.append(dest_path)
            if progress:
                progress((i + 1) / (len(files) * 3), desc=f"Đã sao chép {i + 1}/{len(files)} file...")
        
        logger.info(f"Đang xử lý {len(pdf_paths)} file PDF...")
        if progress:
            progress(0.33, desc=f"Đang xử lý {len(pdf_paths)} file PDF...")
        
        all_chunks, pages_info = pdf_processor.process_multiple_pdfs(pdf_paths)
        
        if not all_chunks:
            gr.Error("Không thể trích xuất văn bản từ các file PDF")
            return
        
        if progress:
            progress(0.66, desc="Đang tạo embeddings và lưu vào vector store...")
        vector_store.add_documents(all_chunks)
        
        if progress:
            progress(1.0, desc="Hoàn tất!")
        
        total_pages = sum(pages_info.values())
        files_summary = "\n".join([f"  • {name}: {pages} trang" 
                                   for name, pages in pages_info.items()])
        
        success_msg = f"Đã xử lý xong {len(pdf_paths)} tài liệu, tổng cộng {total_pages} trang. Bạn có thể đặt câu hỏi ngay!"
        gr.Success(success_msg)
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý PDF: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")


def get_uploaded_files() -> Tuple[str, List[str]]:
    """Lấy danh sách các file đã upload và danh sách tên file cho dropdown"""
    stats = vector_store.get_stats()
    
    if stats["total_files"] == 0:
        return "Chưa có file nào được upload.", []
    
    files_list = "\n".join([f"📄 {filename}: {count} chunks" 
                           for filename, count in stats["files"].items()])
    
    display_text = f"""- Tổng số tài liệu: {stats['total_files']}
- Tổng số chunks: {stats['total_chunks']}
{files_list}"""
    
    file_names = list(stats["files"].keys())
    return display_text, file_names


def delete_file(filename: str) -> Tuple[str, gr.Dropdown]:
    """Xóa một file cụ thể"""
    if not filename or not filename.strip():
        gr.Error("Vui lòng chọn file cần xóa")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
    
    try:
        vector_store.delete_by_filename(filename)
        
        pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        display, file_names = get_uploaded_files()
        gr.Success(f"Đã xóa file: {filename}")
        return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
    except Exception as e:
        logger.error(f"Lỗi khi xóa file: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)


def clear_all_documents() -> Tuple[str, gr.Dropdown]:
    """Xóa toàn bộ tài liệu"""
    try:
        vector_store.clear_all()
        
        for filename in os.listdir(PDF_STORAGE_DIR):
            file_path = os.path.join(PDF_STORAGE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        display, file_names = get_uploaded_files()
        gr.Success("Đã xóa toàn bộ tài liệu")
        return display, gr.Dropdown(choices=file_names, value=None)
    except Exception as e:
        logger.error(f"Lỗi khi xóa tài liệu: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=None)


def chat_interface_fn(message, history, session_id: Optional[str] = None, selected_file: Optional[str] = None, chat_session_id: Optional[str] = None):
    """
    Hàm xử lý chat cho Gradio ChatInterface
    
    Args:
        message: Câu hỏi
        history: Lịch sử chat
        session_id: Session ID của user (nếu đã đăng nhập)
        selected_file: File được chọn để hỏi (nếu có)
        chat_session_id: ID của chat session hiện tại
    """
    if not message.strip():
        return ""
    
    # Session ID đã được xử lý ở wrapper, không cần tạo mới ở đây
    # Chỉ cần đảm bảo chat_session_id được truyền vào đúng

    
    natural_response = get_natural_response(message)
    if natural_response:
        if session_id and database:
            user = auth_manager.get_user_from_session(session_id)
            if user:
                # Đảm bảo có chat_session_id trước khi lưu
                if not chat_session_id:
                    chat_session_id = database.create_chat_session(user["user_id"])
                database.save_chat_message(user["user_id"], message, natural_response, selected_file, chat_session_id)
                if chat_session_id:
                    database.update_session(chat_session_id, title=message)
        return natural_response
    
    stats = vector_store.get_stats()
    if stats["total_chunks"] == 0:
        return "⚠️ Chưa có tài liệu nào được upload. Vui lòng upload file PDF trước khi đặt câu hỏi."
    
    try:
        logger.info(f"Đang tìm kiếm câu trả lời cho: {message} (file: {selected_file})")
        
        search_results = vector_store.search(message, top_k=30, filename=selected_file)
        
        if not search_results:
            response = "Không tìm thấy thông tin liên quan trong các tài liệu đã upload."
            if selected_file:
                response += f" (đã tìm trong file: {selected_file})"
            
            if session_id and database:
                user = auth_manager.get_user_from_session(session_id)
                if user:
                    # Đảm bảo có chat_session_id trước khi lưu
                    if not chat_session_id:
                        chat_session_id = database.create_chat_session(user["user_id"])
                    database.save_chat_message(user["user_id"], message, response, selected_file, chat_session_id)
                    if chat_session_id:
                        # Cập nhật tiêu đề session bằng câu hỏi mới nhất
                        database.update_session(chat_session_id, title=message)
            
            return response
        
        expanded_results = vector_store.get_adjacent_chunks(search_results, page_range=2)
        
        reranked_results = reranker.rerank(message, expanded_results, top_k=15)
        
        answer = generate_answer(message, reranked_results, selected_file)
        
        if session_id and database:
            user = auth_manager.get_user_from_session(session_id)
            if user:
                # Đảm bảo có chat_session_id trước khi lưu
                if not chat_session_id:
                    chat_session_id = database.create_chat_session(user["user_id"])
                database.save_chat_message(user["user_id"], message, answer, selected_file, chat_session_id)
                if chat_session_id:
                    # Cập nhật tiêu đề session bằng câu hỏi mới nhất
                    database.update_session(chat_session_id, title=message)
        
        return answer
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
        return f"Lỗi: {str(e)}"


def create_chat_interface(session_id_state):
    """Tạo chat interface với session state"""
    def chat_fn(message, history):
        session_id = session_id_state.value if hasattr(session_id_state, 'value') else None
        selected_file = session_id_state.selected_file if hasattr(session_id_state, 'selected_file') else None
        return chat_interface_fn(message, history, session_id, selected_file)
    return chat_fn


def login_fn(email, password, session_state):
    """Xử lý đăng nhập với validation và toast thông báo chi tiết"""
    if not auth_manager:
        raise gr.Error("Hệ thống database chưa được khởi tạo. Vui lòng liên hệ quản trị viên.")
    
    email = email.strip() if email else ""
    password = password.strip() if password else ""
    
    if not email:
        raise gr.Error("Vui lòng nhập email của bạn")
    
    if "@" not in email or "." not in email.split("@")[-1]:
        raise gr.Error("Email không hợp lệ. Vui lòng nhập đúng định dạng email (ví dụ: user@example.com)")
    
    if not password:
        raise gr.Error("Vui lòng nhập mật khẩu của bạn")
    
    if len(password) < 6:
        raise gr.Error("Mật khẩu phải có ít nhất 6 ký tự")
    
    result = auth_manager.login(email, password)
    if result["success"]:
        if not isinstance(session_state, dict):
            session_state = {}
        session_state["value"] = result["session_id"]
        session_state["user"] = result["user"]
        session_state["selected_file"] = session_state.get("selected_file")
        
        if database:
            chat_session_id = database.create_chat_session(result["user"]["user_id"])
            session_state["chat_session_id"] = chat_session_id
        
        user_info = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
        ">
            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">
                       👋 Xin chào, <span style="color: #ffd700;">{result['user']['username']}</span>
                    </div>
                    <div style="font-size: 13px; opacity: 0.9;">
                        Email: {result['user']['email']}
                    </div>
            </div>
        </div>
        <script>
            if (window.saveSessionToStorage) {{
                window.saveSessionToStorage('{result["session_id"]}');
            }}
        </script>
        """
        
        gr.Success("✅ " + result['message'])
        
        return (
            session_state,
            gr.update(visible=False),  # Ẩn login_header_btn
            gr.update(visible=False),  # Ẩn register_header_btn
            gr.update(value=user_info, visible=True),  # Hiện thông tin user
            gr.update(visible=True),    # Hiện logout button
            gr.update(visible=False),   # Ẩn login_form
            gr.update(visible=False),   # Ẩn register_form
            gr.update(visible=False),   # Ẩn forgot_form
            gr.update(visible=False)    # Ẩn reset_form
        )
    else:
        # Hiển thị toast lỗi cụ thể dựa trên message từ auth_manager
        error_message = result['message']
        raise gr.Error(error_message)



def register_fn(username, email, password, confirm_password, session_state):
    """Xử lý đăng ký và tự động đăng nhập"""
    if not auth_manager:
        gr.Error("Hệ thống database chưa được khởi tạo")
        return (
            session_state,
            gr.update(visible=True),   # login_header_btn
            gr.update(visible=True),   # register_header_btn
            gr.update(visible=False),  # login_status
            gr.update(visible=False),  # logout_btn
            gr.update(visible=True),   # Giữ register_form hiển thị
            gr.update(visible=False),  # login_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=False)   # reset_form
        )
    
    if password != confirm_password:
        gr.Error("Mật khẩu xác nhận không khớp")
        return (
            session_state,
            gr.update(visible=True),   # login_header_btn
            gr.update(visible=True),   # register_header_btn
            gr.update(visible=False),  # login_status
            gr.update(visible=False),  # logout_btn
            gr.update(visible=True),   # Giữ register_form hiển thị
            gr.update(visible=False),  # login_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=False)   # reset_form
        )
    
    result = auth_manager.register(username, email, password)
    if result["success"]:
        gr.Success(result['message'] + " Đang tự động đăng nhập...")
        
        # Tự động đăng nhập sau khi đăng ký
        login_result = auth_manager.login(email, password)
        if login_result["success"]:
            # Tạo dict để lưu state
            if not isinstance(session_state, dict):
                session_state = {}
            session_state["value"] = login_result["session_id"]
            session_state["user"] = login_result["user"]
            session_state["selected_file"] = None
            
            # Tạo chat session mới
            if database:
                chat_session_id = database.create_chat_session(login_result["user"]["user_id"])
                session_state["chat_session_id"] = chat_session_id
            
            user_info = f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px 20px;
                border-radius: 10px;
                color: white;
            ">
                <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">
                           👋 Xin chào, <span style="color: #ffd700;">{login_result['user']['username']}</span>
                        </div>
                        <div style="font-size: 13px; opacity: 0.9;">
                            Email: {login_result['user']['email']}
                        </div>
                </div>
            </div>
            <script>
                if (window.saveSessionToStorage) {{
                    window.saveSessionToStorage('{login_result["session_id"]}');
                }}
            </script>
            """
            
            return (
                session_state,
                gr.update(visible=False),  # Ẩn login_header_btn
                gr.update(visible=False),  # Ẩn register_header_btn
                gr.update(value=user_info, visible=True),  # Hiện thông tin user
                gr.update(visible=True),   # Hiện logout button
                gr.update(visible=False),  # Ẩn register_form
                gr.update(visible=False),  # Ẩn login_form
                gr.update(visible=False),  # Ẩn forgot_form
                gr.update(visible=False)   # Ẩn reset_form
            )
        else:
            # Đăng ký thành công nhưng đăng nhập thất bại (hiếm khi xảy ra)
            return (
                session_state,
                gr.update(visible=True),   # login_header_btn
                gr.update(visible=True),   # register_header_btn
                gr.update(visible=False),  # login_status
                gr.update(visible=False),  # logout_btn
                gr.update(visible=False),  # Ẩn register_form
                gr.update(visible=True),   # Hiện login_form để đăng nhập
                gr.update(visible=False),  # forgot_form
                gr.update(visible=False)   # reset_form
            )
    else:
        gr.Error(result['message'])
        return (
            session_state,
            gr.update(visible=True),   # login_header_btn
            gr.update(visible=True),   # register_header_btn
            gr.update(visible=False),  # login_status
            gr.update(visible=False),  # logout_btn
            gr.update(visible=True),   # Giữ register_form hiển thị
            gr.update(visible=False),  # login_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=False)   # reset_form
        )


def logout_fn(session_state):
    """Xử lý đăng xuất"""
    if isinstance(session_state, dict) and session_state.get("value"):
        auth_manager.logout(session_state["value"])
        session_state["value"] = None
        session_state["user"] = None
        session_state["selected_file"] = None
        session_state["chat_session_id"] = None
    
    logout_html = """
    <script>
        window.clearSessionFromStorage();
    </script>
    """
    
    gr.Success("Đã đăng xuất")
    return (
        session_state,
        gr.update(visible=True),   # Hiện login_header_btn
        gr.update(visible=True),   # Hiện register_header_btn
        gr.update(value=logout_html, visible=False),  # Ẩn thông tin user và clear localStorage
        gr.update(visible=False),  # Ẩn logout button
        gr.update(visible=False),  # Ẩn login_form
        gr.update(visible=False),  # Ẩn register_form
        gr.update(visible=False),  # Ẩn forgot_form
        gr.update(visible=False)   # Ẩn reset_form
    )


def forgot_password_fn(email):
    """Xử lý quên mật khẩu"""
    if not auth_manager:
        gr.Error("Hệ thống database chưa được khởi tạo")
        return
    
    result = auth_manager.request_password_reset(email)
    if "✅" in result["message"] or "thành công" in result["message"].lower():
        gr.Success(result["message"])
    elif "❌" in result["message"] or "lỗi" in result["message"].lower():
        gr.Error(result["message"])
    else:
        gr.Info(result["message"])


def reset_password_fn(token, new_password, confirm_password):
    """Xử lý reset mật khẩu"""
    if not auth_manager:
        gr.Error("Hệ thống database chưa được khởi tạo")
        return
    
    if new_password != confirm_password:
        gr.Error("Mật khẩu xác nhận không khớp")
        return
    
    result = auth_manager.reset_password(token, new_password)
    if result["success"]:
        gr.Success(result['message'])
    else:
        gr.Error(result['message'])


def select_file_fn(filename, session_state):
    """Chọn file để hỏi"""
    # Đảm bảo session_state là dict
    if not isinstance(session_state, dict):
        session_state = {"value": None, "selected_file": None, "user": None}
    
    # Lưu file được chọn (loại bỏ empty string)
    selected = filename if filename and filename.strip() else None
    session_state["selected_file"] = selected
    
    msg = f"✅ Đã chọn file: {selected}" if selected else "✅ Đã bỏ chọn file (sẽ tìm trong tất cả các file)"
    return msg, session_state


def restore_session_from_id(stored_session_id, session_state):
    """Restore session từ session_id đã lưu trong localStorage"""
    if not stored_session_id or not auth_manager:
        return (
            session_state,
            gr.update(visible=True),   # Hiện login_header_btn
            gr.update(visible=True),   # Hiện register_header_btn
            gr.update(visible=False),  # Ẩn thông tin user
            gr.update(visible=False)   # Ẩn logout button
        )
    
    # Kiểm tra session có hợp lệ không
    user = auth_manager.get_user_from_session(stored_session_id)
    if user:
        # Session hợp lệ, restore state
        if not isinstance(session_state, dict):
            session_state = {}
        session_state["value"] = stored_session_id
        session_state["user"] = user
        session_state["selected_file"] = None
        
        # Tạo chat session mới
        if database:
            chat_session_id = database.create_chat_session(user["user_id"])
            session_state["chat_session_id"] = chat_session_id
        
        user_info = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
        ">
            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                    <div style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">
                       👋 Xin chào, <span style="color: #ffd700;">{user['username']}</span>
                    </div>
                    <div style="font-size: 13px; opacity: 0.9;">
                        Email: {user['email']}
                    </div>
            </div>
        </div>
        <script>
            if (window.saveSessionToStorage) {{
                window.saveSessionToStorage('{stored_session_id}');
            }}
        </script>
        """
        
        return (
            session_state,
            gr.update(visible=False),  # Ẩn login_header_btn
            gr.update(visible=False),  # Ẩn register_header_btn
            gr.update(value=user_info, visible=True),  # Hiện thông tin user
            gr.update(visible=True)    # Hiện logout button
        )
    else:
        # Session không hợp lệ, xóa localStorage
        clear_html = """
        <script>
            window.clearSessionFromStorage();
        </script>
        """
        return (
            session_state,
            gr.update(visible=True),   # Hiện login_header_btn
            gr.update(visible=True),   # Hiện register_header_btn
            gr.update(value=clear_html, visible=False),  # Ẩn thông tin user
            gr.update(visible=False)   # Ẩn logout button
        )


def create_new_chat_session(session_state):
    """Tạo chat session mới"""
    if not isinstance(session_state, dict) or not session_state.get("value"):
        gr.Warning("Vui lòng đăng nhập để sử dụng tính năng này")
        return session_state, None
    
    user = auth_manager.get_user_from_session(session_state["value"])
    if not user or not database:
        gr.Warning("Không thể tạo session mới")
        return session_state, None
    
    # Tạo session mới
    chat_session_id = database.create_chat_session(user["user_id"])
    if chat_session_id:
        session_state["chat_session_id"] = chat_session_id
        gr.Success("Đã tạo cuộc trò chuyện mới!")
        return session_state, []  # Clear chat history
    else:
        gr.Error("Không thể tạo cuộc trò chuyện mới")
        return session_state, None


def get_chat_sessions_list(session_state):
    """Lấy danh sách chat sessions kèm thời gian & câu hỏi gần nhất"""
    if not isinstance(session_state, dict) or not session_state.get("value"):
        return "Vui lòng đăng nhập để xem lịch sử chat"
    
    user = auth_manager.get_user_from_session(session_state["value"])
    if not user or not database:
        return "Không thể lấy danh sách chat"
    
    sessions = database.get_chat_sessions(user["user_id"])
    if not sessions:
        return "Chưa có cuộc trò chuyện nào"
    
    def _shorten(text, limit=90):
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit - 3] + "..."
    
    from datetime import datetime, timedelta
    
    session_lines = []
    for session in sessions:
        # Convert UTC to UTC+7 (Vietnam Time)
        utc_time = datetime.fromisoformat(session["updated_at"].replace("Z", "+00:00"))
        vn_time = utc_time + timedelta(hours=7)
        updated_time = vn_time.strftime("%d/%m/%Y %H:%M")
        
        last_message = database.get_last_message_of_session(session["session_id"])
        last_question = last_message["message"] if last_message and last_message.get("message") else "Chưa có câu hỏi nào"
        short_question = _shorten(last_question)
        session_lines.append(f"- [{updated_time}] {short_question}")
    
    result = "\n".join(session_lines)
    print(f"Chat history response:\n{result}")
    return result


def toggle_chat_history_panel(is_visible, session_state):
    """Đảo trạng thái hiển thị của panel lịch sử chat"""
    is_logged_in = isinstance(session_state, dict) and session_state.get("value")
    current = bool(is_visible)
    
    if not is_logged_in:
        gr.Warning("Vui lòng đăng nhập để xem lịch sử chat")
        return current, gr.update(visible=current)
    
    new_state = not current
    return new_state, gr.update(visible=new_state)


with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot Hành Chính Việt Nam") as app:
    gr.HTML("""
    <style>
        textarea[data-testid="textbox"] {
            overflow-y: hidden !important;
        }
        /* Styling cho button Đăng nhập */
        #header-login-btn {
            height: 40px !important;
            padding-left: 24px !important;
            padding-right: 24px !important;
            background-color: var(--primary-500) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: background-color 0.2s ease !important;
        }
        #header-login-btn:hover {
            background-color: var(--primary-600) !important;
        }
        /* Styling cho button Đăng ký */
        #header-register-btn {
            height: 40px !important;
            padding-left: 24px !important;
            padding-right: 24px !important;
            background-color: transparent !important;
            color: var(--primary-500) !important;
            border: 2px solid var(--primary-500) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        #header-register-btn:hover {
            background-color: var(--primary-50) !important;
            border-color: var(--primary-600) !important;
            color: var(--primary-600) !important;
        }
        /* Styling cho button Đăng xuất - giống nút Đăng ký */
        #header-logout-btn {
            height: 40px !important;
            padding-left: 24px !important;
            padding-right: 24px !important;
            background-color: transparent !important;
            color: var(--primary-500) !important;
            border: 2px solid var(--primary-500) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        #header-logout-btn:hover {
            background-color: var(--primary-50) !important;
            border-color: var(--primary-600) !important;
            color: var(--primary-600) !important;
        }
        /* Styling cho label trong các form - transparent background */
        /* Target tất cả label */
        label,
        label *,
        * label,
        * > label {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
        }
        /* Đảm bảo tất cả label trong form có background transparent */
        .form label,
        .form > div > label,
        .form > div > div > label,
        [class*="form"] label,
        [class*="form"] > div > label,
        [class*="form"] > div > div > label,
        .gr-form label,
        .gr-textbox label,
        .gr-textbox > label,
        .gr-textbox > div > label,
        .gr-textbox > div > div > label,
        .gr-textbox > span > label,
        .gr-textbox > span > div > label,
        div[class*="textbox"] label,
        div[class*="textbox"] > label,
        div[class*="textbox"] > div > label,
        div[class*="textbox"] > div > div > label,
        div[class*="textbox"] > span > label,
        div[class*="textbox"] > span > div > label,
        input[type="text"] + label,
        input[type="password"] + label,
        input[type="email"] + label,
        /* Target label trong các form cụ thể */
        .gr-column label,
        .gr-column > div > label,
        .gr-column > div > div > label,
        .gr-column > span > label,
        .gr-column > span > div > label,
        /* Target tất cả label có class */
        label[class],
        label[class*="label"],
        /* Target label trong wrapper */
        .wrap label,
        .wrap > div > label,
        .wrap > div > div > label,
        .wrap > span > label,
        .wrap > span > div > label,
        /* Target label trong block container */
        .block label,
        .block > div > label,
        .block > div > div > label,
        /* Universal selector cho tất cả label */
        [class*="gr-"] label,
        [class*="gr-"] > div > label,
        [class*="gr-"] > div > div > label {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
        }
        /* Override inline styles nếu có */
        label[style*="background"],
        label[style*="background-color"] {
            background: transparent !important;
            background-color: transparent !important;
        }
        #chat-history-btn {
            width: 100%;
            margin-top: 12px;
        }
        #chat-history-panel {
            border: 1px solid var(--border-color-primary);
            border-radius: 10px;
            padding: 16px;
            background: var(--background-fill-secondary);
        }
        /* Target label trong form đăng nhập và đăng ký cụ thể */
        #login_form label,
        #register_form label,
        #forgot_form label,
        #reset_form label,
        [id*="login"] label,
        [id*="register"] label,
        [id*="forgot"] label,
        [id*="reset"] label {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
        }
        /* Force transparent cho tất cả label elements */
        label {
            background: transparent !important;
            background-color: rgba(0, 0, 0, 0) !important;
            background-image: none !important;
        }
        /* Đảm bảo text fields vẫn có border/outline */
        input[type="text"],
        input[type="password"],
        input[type="email"],
        textarea,
        .gr-textbox input,
        .gr-textbox textarea,
        div[class*="textbox"] input,
        div[class*="textbox"] textarea {
            border: 1px solid var(--input-border-color, #ccc) !important;
            outline: none !important;
        }
        /* Focus state cho text fields */
        input[type="text"]:focus,
        input[type="password"]:focus,
        input[type="email"]:focus,
        textarea:focus,
        .gr-textbox input:focus,
        .gr-textbox textarea:focus,
        div[class*="textbox"] input:focus,
        div[class*="textbox"] textarea:focus {
            border-color: var(--primary-500, #0066cc) !important;
            outline: 2px solid var(--primary-100, rgba(0, 102, 204, 0.1)) !important;
            outline-offset: 2px !important;
        }
    </style>
    <script>
        // Đảm bảo tất cả label có background transparent sau khi page load
        document.addEventListener('DOMContentLoaded', function() {
            function makeLabelsTransparent() {
                const labels = document.querySelectorAll('label');
                labels.forEach(function(label) {
                    // Chỉ ảnh hưởng đến label, không ảnh hưởng đến input fields
                    label.style.background = 'transparent';
                    label.style.backgroundColor = 'transparent';
                    label.style.backgroundImage = 'none';
                });
            }
            makeLabelsTransparent();
            // Chạy lại sau khi Gradio render components
            setTimeout(makeLabelsTransparent, 1000);
            setTimeout(makeLabelsTransparent, 2000);
            // Sử dụng MutationObserver để theo dõi thay đổi DOM
            const observer = new MutationObserver(function(mutations) {
                makeLabelsTransparent();
            });
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    </script>
    <script>
        // Lưu và load session từ localStorage
        function saveSessionToStorage(sessionId) {
            if (sessionId) {
                localStorage.setItem('ragviet_session_id', sessionId);
                console.log('Đã lưu session:', sessionId);
            }
        }
        
        function loadSessionFromStorage() {
            const sessionId = localStorage.getItem('ragviet_session_id');
            if (sessionId) {
                console.log('Đã load session:', sessionId);
                return sessionId;
            }
            return null;
        }
        
        function clearSessionFromStorage() {
            localStorage.removeItem('ragviet_session_id');
            console.log('Đã xóa session');
        }
        
        // Expose functions to window
        window.saveSessionToStorage = saveSessionToStorage;
        window.loadSessionFromStorage = loadSessionFromStorage;
        window.clearSessionFromStorage = clearSessionFromStorage;
        
        // Auto-restore session khi load trang
        function tryRestoreSession() {
            const savedSession = loadSessionFromStorage();
            if (savedSession) {
                console.log('Tìm thấy session đã lưu, đang restore...');
                // Thử nhiều selector khác nhau
                const selectors = [
                    '#restore_session_input textarea',
                    '#restore_session_input input',
                    'textarea[data-testid="textbox"]',
                    '.gr-textbox textarea'
                ];
                
                let restoreInput = null;
                for (const selector of selectors) {
                    const elements = document.querySelectorAll(selector);
                    for (const elem of elements) {
                        if (elem.closest('#restore_session_input') || elem.id === 'restore_session_input') {
                            restoreInput = elem;
                            break;
                        }
                    }
                    if (restoreInput) break;
                }
                
                if (restoreInput) {
                    console.log('Đã tìm thấy restore input, đang trigger...');
                    // Set value directly
                    restoreInput.value = savedSession;
                    
                    // Trigger events manually to ensure Gradio catches the change
                    restoreInput.dispatchEvent(new Event('input', { bubbles: true }));
                    restoreInput.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // Try to trigger React/Gradio internal state update if possible (hacky but sometimes needed)
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    if (nativeInputValueSetter) {
                        nativeInputValueSetter.call(restoreInput, savedSession);
                        restoreInput.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                } else {
                    console.log('Chưa tìm thấy restore input, thử lại sau...');
                }
            }
        }
        
        // Thử restore nhiều lần với khoảng thời gian dài hơn
        setTimeout(tryRestoreSession, 1000);
        setTimeout(tryRestoreSession, 2000);
        setTimeout(tryRestoreSession, 3000);
        setTimeout(tryRestoreSession, 5000);
    </script>
    </style>
    """)
    gr.Markdown("""
    # 💻 Chatbot Trả Lời Tự Động Văn Bản Hành Chính Việt Nam
    Upload file PDF hành chính của bạn và đặt câu hỏi - chatbot sẽ trả lời dựa trên nội dung tài liệu!
    
    """)
    
    session_state = gr.State(value={"value": None, "user": None, "selected_file": None, "chat_session_id": None})
    chat_history_visible = gr.State(False)
    
    restore_session_input = gr.Textbox(
        visible=True,
        show_label=False,
        elem_id="restore_session_input",
        interactive=False,
        container=False,
        lines=1,
        placeholder=""
    )
    
    gr.HTML("""
    <style>
        .link-button {
            color: #0066cc !important;
            text-decoration: underline !important;
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 5px !important;
            font-size: inherit !important;
        }
        .auth-section {
            padding: 10px;
            border-radius: 8px;
            background: var(--background-fill-secondary);
        }
        #restore_session_input {
            position: absolute !important;
            left: -9999px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 1px !important;
            width: 1px !important;
            overflow: hidden !important;
        }
    </style>
    """)
    
    with gr.Row(elem_id="header-tabs-row"):
        with gr.Column(scale=0, min_width=300, elem_classes="auth-section"):
            auth_text = gr.Markdown("**Tài khoản:**", elem_id="auth-text", visible=False)
            with gr.Row():
                login_header_btn = gr.Button("Đăng nhập", variant="secondary", size="sm", elem_id="header-login-btn")
                register_header_btn = gr.Button("Đăng ký", variant="secondary", size="sm", elem_id="header-register-btn")
            login_status = gr.Markdown(visible=False, elem_id="login-status")
            logout_btn = gr.Button("Đăng Xuất", variant="secondary", visible=False, size="sm", elem_id="header-logout-btn")
            
            with gr.Column(visible=False) as login_form:
                gr.Markdown("### Đăng Nhập")
                login_email = gr.Textbox(label="Email", placeholder="Nhập email của bạn")
                login_password = gr.Textbox(label="Mật khẩu", type="password", placeholder="Nhập mật khẩu")
                login_btn = gr.Button("Đăng Nhập", variant="primary", size="lg")
                login_links_col = gr.Column()
                with login_links_col:
                    link_forgot_from_login = gr.Button("Quên mật khẩu?", variant="plain", size="sm", elem_classes="link-button")
                    gr.HTML("<div style='text-align: center; margin-top: 10px;'>Chưa có tài khoản? </div>")
                    link_register_from_login = gr.Button("Đăng ký ngay", variant="plain", size="sm", elem_classes="link-button")
            
            with gr.Column(visible=False) as register_form:
                gr.Markdown("### Đăng Ký")
                reg_username = gr.Textbox(label="Tên đăng nhập", placeholder="Nhập tên đăng nhập")
                reg_email = gr.Textbox(label="Email", placeholder="Nhập email của bạn")
                reg_password = gr.Textbox(label="Mật khẩu", type="password", placeholder="Tối thiểu 6 ký tự")
                reg_confirm_password = gr.Textbox(label="Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu")
                reg_btn = gr.Button("Đăng Ký", variant="primary", size="lg")
                reg_links_col = gr.Column()
                with reg_links_col:
                    gr.HTML("<div style='text-align: center; margin-top: 10px;'>Đã có tài khoản? </div>")
                    link_login_from_register = gr.Button("Đăng nhập", variant="plain", size="sm", elem_classes="link-button")
            
            with gr.Column(visible=False) as forgot_form:
                gr.Markdown("### Quên Mật Khẩu")
                forgot_email = gr.Textbox(label="Email", placeholder="Nhập email đã đăng ký")
                forgot_btn = gr.Button("Gửi mã OTP", variant="primary", size="lg")
                forgot_links_col = gr.Column()
                with forgot_links_col:
                    link_login_from_forgot = gr.Button("Quay lại đăng nhập", variant="plain", size="sm", elem_classes="link-button")
                    link_reset_from_forgot = gr.Button("Đã có OTP? Đặt lại mật khẩu", variant="plain", size="sm", elem_classes="link-button")
            
            with gr.Column(visible=False) as reset_form:
                gr.Markdown("### Đặt Lại Mật Khẩu")
                reset_token = gr.Textbox(label="Mã OTP", placeholder="Nhập mã OTP đã nhận")
                reset_new_password = gr.Textbox(label="Mật khẩu mới", type="password", placeholder="Tối thiểu 6 ký tự")
                reset_confirm_password = gr.Textbox(label="Xác nhận mật khẩu mới", type="password", placeholder="Nhập lại mật khẩu")
                reset_btn = gr.Button("Đặt Lại Mật Khẩu", variant="primary", size="lg")
                reset_links_col = gr.Column()
                with reset_links_col:
                    link_login_from_reset = gr.Button("Quay lại đăng nhập", variant="plain", size="sm", elem_classes="link-button")
                    link_forgot_from_reset = gr.Button("Chưa có token? Yêu cầu mới", variant="plain", size="sm", elem_classes="link-button")
        
        with gr.Column(scale=1):
            with gr.Tab("💬 Chat"):
                # File selection dropdown
                gr.Markdown("### Chọn File Để Hỏi (Tùy chọn)")
                gr.Markdown("*Nếu bạn chưa đăng nhập, thì chỉ có thể sử dụng file mẫu có sẵn của chúng tôi. Vui lòng đăng nhập để sử dụng đầy đủ các tính năng nhé!*")
                
                file_selection_dropdown = gr.Dropdown(
                    label="Chọn file",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False
                )
                file_selection_output = gr.Textbox(label="Trạng thái", interactive=False, lines=1)
                
                # Load file list
                def update_file_dropdown():
                    _, file_names = get_uploaded_files()
                    return gr.Dropdown(choices=[""] + file_names, value=None)
                
                file_selection_dropdown.change(
                    select_file_fn,
                    inputs=[file_selection_dropdown, session_state],
                    outputs=[file_selection_output, session_state]
                )
                
                # Chat interface
                def chat_wrapper(message, history, session_state_val):
                    session_id = None
                    selected_file = None
                    chat_session_id = None
                    
                    # session_state_val được truyền vào từ additional_inputs
                    if isinstance(session_state_val, dict):
                        session_id = session_state_val.get("value")
                        selected_file = session_state_val.get("selected_file")
                        chat_session_id = session_state_val.get("chat_session_id")
                    
                    # Nếu user đã đăng nhập nhưng chưa có chat_session_id, tạo session mới
                    if session_id and database and not chat_session_id:
                        user = auth_manager.get_user_from_session(session_id)
                        if user:
                            chat_session_id = database.create_chat_session(user["user_id"])
                            # Cập nhật session_state ngay lập tức (lưu ý: cái này chỉ update local dict, 
                            # không update lại state của Gradio trừ khi return, nhưng ChatInterface không support return state)
                            if isinstance(session_state_val, dict):
                                session_state_val["chat_session_id"] = chat_session_id
                    
                    response = chat_interface_fn(message, history, session_id, selected_file, chat_session_id)
                    
                    return response
                
                chat_interface = gr.ChatInterface(
                    fn=chat_wrapper,
                    additional_inputs=[session_state],
                    title="Chat với RagVietBot",
                    description="Đặt câu hỏi về nội dung các tài liệu đã upload",
                    examples=[
                        ["Tóm tắt nội dung chính của tài liệu", None],
                        ["Các quy định về thủ tục hành chính là gì?", None],
                        ["Thời hạn xử lý hồ sơ là bao lâu?", None]
                    ],
                    cache_examples=False
                )
                
                chat_history_btn = gr.Button("📜 Lịch sử chat", variant="secondary", elem_id="chat-history-btn")
                with gr.Column(visible=False, elem_id="chat-history-panel") as chat_history_panel:
                    gr.Markdown("### Quản Lý Cuộc Trò Chuyện")
                    with gr.Row():
                        new_chat_btn = gr.Button("➕ Tạo Cuộc Trò Chuyện Mới", variant="primary")
                        refresh_sessions_btn = gr.Button("🔄 Làm Mới Danh Sách", variant="secondary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Danh Sách Cuộc Trò Chuyện")
                    
                    sessions_display = gr.Markdown("Vui lòng đăng nhập để xem lịch sử chat")
                
                def refresh_sessions_fn(session_state):
                    return get_chat_sessions_list(session_state)
                
                chat_history_btn.click(
                    toggle_chat_history_panel,
                    inputs=[chat_history_visible, session_state],
                    outputs=[chat_history_visible, chat_history_panel]
                ).then(
                    refresh_sessions_fn,
                    inputs=[session_state],
                    outputs=[sessions_display]
                )
                
                new_chat_btn.click(
                    create_new_chat_session,
                    inputs=[session_state],
                    outputs=[session_state, chat_interface.chatbot]
                ).then(
                    refresh_sessions_fn,
                    inputs=[session_state],
                    outputs=[sessions_display]
                )
                
                refresh_sessions_btn.click(
                    refresh_sessions_fn,
                    inputs=[session_state],
                    outputs=[sessions_display]
                )
                
                app.load(
                    refresh_sessions_fn,
                    inputs=[session_state],
                    outputs=[sessions_display]
                )
            
            with gr.Tab("📁 Quản Lý Tài Liệu"):
                # Kiểm tra đăng nhập để hiển thị upload
                gr.Markdown("### Upload File PDF")
                gr.Markdown("*⚠️ Chỉ người dùng đã đăng nhập mới có thể upload file. Người dùng chưa đăng nhập chỉ có thể sử dụng các file cố định.*")
                
                file_upload = gr.File(
                    label="Chọn file PDF (có thể chọn nhiều file)",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Xử Lý Tài Liệu", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("### Danh Sách Tài Liệu Đã Upload")
                
                with gr.Row():
                    files_display = gr.Textbox(label="Tài liệu hiện có", lines=10, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### Xóa Tài Liệu")
                
                filename_dropdown = gr.Dropdown(
                    label="Chọn file cần xóa",
                    choices=[],
                    interactive=True
                )
                delete_btn = gr.Button("🗑️ Xóa file", variant="stop")
                
                def refresh_files():
                    display, file_names = get_uploaded_files()
                    return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
                
                def check_auth_and_upload(files, session_state):
                    """Kiểm tra đăng nhập trước khi upload"""
                    if not isinstance(session_state, dict) or not session_state.get("value"):
                        gr.Error("Vui lòng đăng nhập để upload file. Người dùng chưa đăng nhập chỉ có thể sử dụng các file cố định.")
                        return
                    process_pdfs(files)
                
                upload_btn.click(
                    check_auth_and_upload,
                    inputs=[file_upload, session_state],
                    outputs=[]
                ).then(
                    refresh_files,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    update_file_dropdown,
                    outputs=[file_selection_dropdown]
                )
                
                app.load(
                    refresh_files,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    update_file_dropdown,
                    outputs=[file_selection_dropdown]
                )
                
                delete_btn.click(
                    delete_file,
                    inputs=filename_dropdown,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    refresh_files,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    update_file_dropdown,
                    outputs=[file_selection_dropdown]
                )
                
                gr.Markdown("---")
                
                clear_all_btn = gr.Button("🗑️ Xóa Toàn Bộ Tài Liệu", variant="stop")
                clear_all_btn.click(
                    clear_all_documents,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    refresh_files,
                    outputs=[files_display, filename_dropdown]
                ).then(
                    update_file_dropdown,
                    outputs=[file_selection_dropdown]
                )
            
            with gr.Tab("ℹ️ Hướng Dẫn"):
                gr.Markdown("""
        ## Hướng Dẫn Sử Dụng
        
        ### 1. Đăng Ký / Đăng Nhập
        - **Đăng ký**: Tạo tài khoản mới với email và mật khẩu
        - **Đăng nhập**: Đăng nhập để sử dụng đầy đủ tính năng
        - **Quên mật khẩu**: Yêu cầu mã OTP và đặt lại mật khẩu
        - **Lưu ý**: Chỉ người dùng đã đăng nhập mới có thể upload file
        
        ### 2. Upload Tài Liệu (Chỉ khi đã đăng nhập)
        - Vào tab **"Quản Lý Tài Liệu"**
        - Chọn một hoặc nhiều file PDF
        - Click **"Xử Lý Tài Liệu"**
        - Đợi hệ thống xử lý (có thể mất vài phút tùy kích thước file)
        
        ### 3. Đặt Câu Hỏi
        - Vào tab **"Chat"**
        - (Tùy chọn) Chọn một file cụ thể để tăng độ chính xác
        - Nhập câu hỏi liên quan đến nội dung tài liệu
        - Chatbot cũng có thể trả lời các câu hỏi tự nhiên như: chào, hello, giới thiệu, etc.
        - Click **"Gửi"** hoặc nhấn Enter
        - Chatbot sẽ tìm kiếm và trả lời dựa trên tài liệu
        
        ### 4. Chọn File Cụ Thể
        - Trong tab Chat, bạn có thể chọn một file cụ thể từ dropdown
        - Khi chọn file, chatbot sẽ chỉ tìm kiếm trong file đó
        - Điều này giúp tăng độ chính xác khi có nhiều file
        
        ### 5. Lịch Sử Chat
        - Lịch sử chat được tự động lưu khi bạn đã đăng nhập


        ### 6. Quản Lý Tài Liệu
        - Xem danh sách file đã upload
        - Xóa từng file cụ thể
        - Xóa toàn bộ để bắt đầu lại
        
        ## Công Nghệ
        - **Vector Database**: FAISS
        - **Embedding Model**: Vietnamese SBERT / SimCSE-VietNamese
        - **Reranker**: BGE Reranker Base
        - **LLM**: Groq (Llama-3.3-70B-Versatile)
        - **Database**: MongoDB
        """)
    
    # Functions to switch forms
    def show_login():
        return (
            gr.update(visible=True),   # login_form
            gr.update(visible=False),  # register_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=False)   # reset_form
        )
    
    def show_register():
        return (
            gr.update(visible=False),  # login_form
            gr.update(visible=True),   # register_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=False)   # reset_form
        )
    
    def show_forgot():
        return (
            gr.update(visible=False),  # login_form
            gr.update(visible=False),  # register_form
            gr.update(visible=True),   # forgot_form
            gr.update(visible=False)   # reset_form
        )
    
    def show_reset():
        return (
            gr.update(visible=False),  # login_form
            gr.update(visible=False),  # register_form
            gr.update(visible=False),  # forgot_form
            gr.update(visible=True)    # reset_form
        )
    
    login_header_btn.click(show_login, outputs=[login_form, register_form, forgot_form, reset_form])
    register_header_btn.click(show_register, outputs=[login_form, register_form, forgot_form, reset_form])
    
    link_register_from_login.click(show_register, outputs=[login_form, register_form, forgot_form, reset_form])
    link_forgot_from_login.click(show_forgot, outputs=[login_form, register_form, forgot_form, reset_form])
    link_login_from_register.click(show_login, outputs=[login_form, register_form, forgot_form, reset_form])
    link_login_from_forgot.click(show_login, outputs=[login_form, register_form, forgot_form, reset_form])
    link_reset_from_forgot.click(show_reset, outputs=[login_form, register_form, forgot_form, reset_form])
    link_login_from_reset.click(show_login, outputs=[login_form, register_form, forgot_form, reset_form])
    link_forgot_from_reset.click(show_forgot, outputs=[login_form, register_form, forgot_form, reset_form])
    
    login_btn.click(
        login_fn,
        inputs=[login_email, login_password, session_state],
        outputs=[session_state, login_header_btn, register_header_btn, login_status, logout_btn, login_form, register_form, forgot_form, reset_form]
    )
    
    logout_btn.click(
        logout_fn,
        inputs=session_state,
        outputs=[session_state, login_header_btn, register_header_btn, login_status, logout_btn, login_form, register_form, forgot_form, reset_form]
    )
    
    reg_btn.click(
        register_fn,
        inputs=[reg_username, reg_email, reg_password, reg_confirm_password, session_state],
        outputs=[session_state, login_header_btn, register_header_btn, login_status, logout_btn, register_form, login_form, forgot_form, reset_form]
    )
    
    forgot_btn.click(
        forgot_password_fn,
        inputs=forgot_email,
        outputs=[]
    )
    
    reset_btn.click(
        reset_password_fn,
        inputs=[reset_token, reset_new_password, reset_confirm_password],
        outputs=[]
    )
    
    restore_session_input.change(
        restore_session_from_id,
        inputs=[restore_session_input, session_state],
        outputs=[session_state, login_header_btn, register_header_btn, login_status, logout_btn]
    )

if __name__ == "__main__":
    logger.info("Khởi động ứng dụng Chatbot Hành Chính Việt Nam...")
    app.launch(server_name="0.0.0.0", share=False)
