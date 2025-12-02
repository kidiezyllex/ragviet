"""
Chatbot Hành Chính Việt Nam - RAG System với FAISS và Gradio
"""
import os
import gradio as gr
from typing import List, Tuple, Dict, Optional
import logging
from dotenv import load_dotenv
import shutil
import json
from utils.natural_language import is_natural_question, get_natural_response
from api_client import (
    api_login, api_register, api_logout, api_forgot_password, api_reset_password,
    api_verify_session, api_chat_send, api_get_chat_sessions, api_create_chat_session,
    api_upload_files, api_get_files, api_delete_file, api_clear_all_files,
    api_get_chat_history
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdfs(files: List, session_state, progress=gr.Progress()):
    """
    Xử lý nhiều file PDF - gọi Django API
    
    Args:
        files: List các file PDF upload
        session_state: Session state để lấy session_id
        progress: Gradio progress tracker
    """
    if not files:
        gr.Error("Vui lòng chọn ít nhất một file PDF")
        return
    
    session_id = None
    if isinstance(session_state, dict):
        session_id = session_state.get("value")
    
    if not session_id:
        gr.Error("Vui lòng đăng nhập để upload file. Người dùng chưa đăng nhập chỉ có thể sử dụng các file cố định.")
        return
    
    try:
        if progress:
            progress(0.5, desc="Đang upload file lên server...")
        
        result = api_upload_files(files, session_id)
        
        if progress:
            progress(1.0, desc="Hoàn tất!")
        
        if result.get("success"):
            gr.Success(result.get("message", "Đã upload file thành công!"))
        else:
            gr.Error(result.get("message", "Lỗi khi upload file"))
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý PDF: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")


def get_uploaded_files() -> Tuple[str, List[str]]:
    """Lấy danh sách các file đã upload - gọi Django API"""
    result = api_get_files()
    
    if not result.get("success") or result.get("total_files", 0) == 0:
        return "Chưa có file nào được upload.", []
    
    files = result.get("files", [])
    files_list = "\n".join([f"📄 {file['filename']}: {file['chunks']} chunks" for file in files])
    
    display_text = f"""- Tổng số tài liệu: {result['total_files']}
- Tổng số chunks: {result['total_chunks']}
{files_list}"""
    
    file_names = [file['filename'] for file in files]
    return display_text, file_names


def delete_file(filename: str) -> Tuple[str, gr.Dropdown]:
    """Xóa một file cụ thể - gọi Django API"""
    if not filename or not filename.strip():
        gr.Error("Vui lòng chọn file cần xóa")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
    
    try:
        result = api_delete_file(filename)
        
        if result.get("success"):
            display, file_names = get_uploaded_files()
            gr.Success(result.get("message", f"Đã xóa file: {filename}"))
            return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
        else:
            gr.Error(result.get("message", "Lỗi khi xóa file"))
            display, file_names = get_uploaded_files()
            return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)
    except Exception as e:
        logger.error(f"Lỗi khi xóa file: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None)


def clear_all_documents() -> Tuple[str, gr.Dropdown]:
    """Xóa toàn bộ tài liệu - gọi Django API"""
    try:
        result = api_clear_all_files()
        
        if result.get("success"):
            display, file_names = get_uploaded_files()
            gr.Success(result.get("message", "Đã xóa toàn bộ tài liệu"))
            return display, gr.Dropdown(choices=file_names, value=None)
        else:
            gr.Error(result.get("message", "Lỗi khi xóa tài liệu"))
            display, file_names = get_uploaded_files()
            return display, gr.Dropdown(choices=file_names, value=None)
    except Exception as e:
        logger.error(f"Lỗi khi xóa tài liệu: {str(e)}")
        gr.Error(f"Lỗi: {str(e)}")
        display, file_names = get_uploaded_files()
        return display, gr.Dropdown(choices=file_names, value=None)


def chat_interface_fn(message, history, session_id: Optional[str] = None, selected_file: Optional[str] = None, chat_session_id: Optional[str] = None):
    """
    Hàm xử lý chat cho Gradio ChatInterface - gọi Django API
    
    Args:
        message: Câu hỏi
        history: Lịch sử chat
        session_id: Session ID của user (nếu đã đăng nhập)
        selected_file: File được chọn để hỏi (nếu có)
        chat_session_id: ID của chat session hiện tại
    """
    if not message.strip():
        return ""
    
    result = api_chat_send(message, session_id, selected_file, chat_session_id)
    
    if result.get("success"):
        new_chat_session_id = result.get("chat_session_id")
        if new_chat_session_id and new_chat_session_id != chat_session_id:
            pass
        
        return result.get("response", "Không có phản hồi")
    else:
        return result.get("response", "Lỗi khi gửi tin nhắn")


def create_chat_interface(session_id_state):
    """Tạo chat interface với session state"""
    def chat_fn(message, history):
        session_id = session_id_state.value if hasattr(session_id_state, 'value') else None
        selected_file = session_id_state.selected_file if hasattr(session_id_state, 'selected_file') else None
        return chat_interface_fn(message, history, session_id, selected_file)
    return chat_fn


def login_fn(email, password, session_state):
    """Xử lý đăng nhập với validation và toast thông báo chi tiết"""
    email = email.strip() if email else ""
    password = password.strip() if password else ""
    
    if not email:
        gr.Error("Vui lòng nhập email của bạn")
        return (
            session_state,
            gr.update(visible=True),    
            gr.update(visible=True),   
            gr.update(visible=False),  
            gr.update(visible=False), 
            gr.update(visible=True),   
            gr.update(visible=False), 
            gr.update(visible=False), 
            gr.update(visible=False)   
        )
    
    if "@" not in email or "." not in email.split("@")[-1]:
        gr.Error("Email không hợp lệ. Vui lòng nhập đúng định dạng email (ví dụ: user@example.com)")
        return (
            session_state,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    if not password:
        gr.Error("Vui lòng nhập mật khẩu của bạn")
        return (
            session_state,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    if len(password) < 6:
        gr.Error("Mật khẩu phải có ít nhất 6 ký tự")
        return (
            session_state,
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    try:
        result = api_login(email, password)
        if result.get("success"):
            if not isinstance(session_state, dict):
                session_state = {}
            session_state["value"] = result["session_id"]
            session_state["user"] = result["user"]
            session_state["selected_file"] = session_state.get("selected_file")
            session_state["chat_session_id"] = result.get("chat_session_id")
            
            access_token = result.get("access_token", result["session_id"])
            user_info_json = json.dumps(result['user'])
            
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
                    window.saveSessionToStorage('{result["session_id"]}', '{access_token}', {user_info_json});
                }}
            </script>
            """
            
            gr.Success("✅ " + result.get('message', 'Đăng nhập thành công!'))
            
            return (
                session_state,
                gr.update(visible=False),  
                gr.update(visible=False),  
                gr.update(value=user_info, visible=True),  
                gr.update(visible=True),      
                gr.update(visible=False),  
                gr.update(visible=False),  
                gr.update(visible=False),  
                gr.update(visible=False)  
            )
        else:
            error_message = result.get('message', 'Đăng nhập thất bại')
            gr.Error(error_message)
            
            return (
                session_state,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    except Exception as e:
        error_message = f"Lỗi kết nối: {str(e)}"
        gr.Error(error_message)
        
        return (
            session_state,
            gr.update(visible=True),   
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )



def register_fn(username, email, password, confirm_password, session_state):
    """Xử lý đăng ký và tự động đăng nhập"""
    if password != confirm_password:
        gr.Error("Mật khẩu xác nhận không khớp")
        return (
            session_state,
            gr.update(visible=True),  
            gr.update(visible=True),  
            gr.update(visible=False),  
            gr.update(visible=False),  
            gr.update(visible=True),  
            gr.update(visible=False),  
            gr.update(visible=False),  
        )
    
    result = api_register(username, email, password, confirm_password)
    if result["success"]:
        gr.Success(result.get('message', 'Đăng ký thành công!') + " Đang tự động đăng nhập...")
        
        login_result = result
        if login_result.get("success") and "user" in login_result:
            if not isinstance(session_state, dict):
                session_state = {}
            session_state["value"] = login_result["session_id"]
            session_state["user"] = login_result["user"]
            session_state["selected_file"] = None
            session_state["chat_session_id"] = login_result.get("chat_session_id")
            
            access_token = login_result.get("access_token", login_result["session_id"])
            user_info_json = json.dumps(login_result['user'])
            
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
                    window.saveSessionToStorage('{login_result["session_id"]}', '{access_token}', {user_info_json});
                }}
            </script>
            """
            
            return (
                session_state,
                gr.update(visible=False),  
                gr.update(visible=False),  
                gr.update(value=user_info, visible=True),  
                gr.update(visible=True),  
                gr.update(visible=False),  
                gr.update(visible=False),  
            )
        else:
            return (
                session_state,
                gr.update(visible=True),    
                gr.update(visible=True),    
                gr.update(visible=False),  
                gr.update(visible=False),  
            )
    else:
        gr.Error(result['message'])
        return (
            session_state,
            gr.update(visible=True),  
            gr.update(visible=True),  
            gr.update(visible=False),  
            gr.update(visible=False),  
            gr.update(visible=True),  
            gr.update(visible=False),  
            gr.update(visible=False),  
        )


def logout_fn(session_state):
    """Xử lý đăng xuất"""
    if isinstance(session_state, dict) and session_state.get("value"):
        api_logout(session_state["value"])
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
        gr.update(visible=True),  
        gr.update(visible=True),  
        gr.update(value=logout_html, visible=False),  
        gr.update(visible=False),  
        gr.update(visible=False),  
    )


def forgot_password_fn(email):
    """Xử lý quên mật khẩu"""
    result = api_forgot_password(email)
    if "✅" in result["message"] or "thành công" in result["message"].lower():
        gr.Success(result["message"])
    elif "❌" in result["message"] or "lỗi" in result["message"].lower():
        gr.Error(result["message"])
    else:
        gr.Info(result["message"])


def reset_password_fn(token, new_password, confirm_password):
    """Xử lý reset mật khẩu"""
    if new_password != confirm_password:
        gr.Error("Mật khẩu xác nhận không khớp")
        return
    
    result = api_reset_password(token, new_password, confirm_password)
    if result["success"]:
        gr.Success(result['message'])
    else:
        gr.Error(result['message'])


def select_file_fn(filename, session_state):
    """Chọn file để hỏi"""
    if not isinstance(session_state, dict):
        session_state = {"value": None, "selected_file": None, "user": None}
    
    selected = filename if filename and filename.strip() else None
    session_state["selected_file"] = selected
    
    msg = f"✅ Đã chọn file: {selected}" if selected else "✅ Đã bỏ chọn file (sẽ tìm trong tất cả các file)"
    return msg, session_state


def restore_session_from_id(stored_session_id, session_state, is_restoring):
    """Restore session từ session_id đã lưu trong localStorage"""
    if not stored_session_id or not stored_session_id.strip():
        return (
            session_state,
            gr.update(visible=False),  
            gr.update(visible=True),   
            gr.update(visible=True),   
            gr.update(visible=False),  
            gr.update(visible=False)   
        )
    
    try:
        result = api_verify_session(stored_session_id)
        if result.get("success") and result.get("valid"):
            user = result.get("user")
            if user:
                if not isinstance(session_state, dict):
                    session_state = {}
                session_state["value"] = stored_session_id
                session_state["user"] = user
                session_state["selected_file"] = None
                session_state["chat_session_id"] = result.get("chat_session_id")
                access_token = stored_session_id
                
                user_info_json = json.dumps(user)
                
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
                        window.saveSessionToStorage('{stored_session_id}', '{access_token}', {user_info_json});
                    }}
                </script>
                """
                
                return (
                    session_state,
                    gr.update(visible=False),
                    gr.update(visible=False),  
                    gr.update(visible=False),  
                    gr.update(value=user_info, visible=True),  
                    gr.update(visible=True)   
                )
        
        clear_html = """
        <script>
            window.clearSessionFromStorage();
        </script>
        """
        return (
            session_state,
            gr.update(visible=False),
            gr.update(visible=True),  
            gr.update(visible=True),  
            gr.update(value=clear_html, visible=False),  
            gr.update(visible=False)  
        )
    except Exception as e:
        logger.error(f"Lỗi khi restore session: {str(e)}")
        clear_html = """
        <script>
            window.clearSessionFromStorage();
        </script>
        """
        return (
            session_state,
            gr.update(visible=False),  
            gr.update(visible=True),  
            gr.update(visible=True),  
            gr.update(value=clear_html, visible=False),  
            gr.update(visible=False)  
        )


def create_new_chat_session(session_state):
    """Tạo chat session mới - gọi Django API"""
    if not isinstance(session_state, dict) or not session_state.get("value"):
        gr.Warning("Vui lòng đăng nhập để sử dụng tính năng này")
        return session_state, None
    
    session_id = session_state["value"]
    result = api_create_chat_session(session_id)
    
    if result.get("success"):
        chat_session_id = result.get("chat_session_id")
        if chat_session_id:
            session_state["chat_session_id"] = chat_session_id
            gr.Success(result.get("message", "Đã tạo cuộc trò chuyện mới!"))
            return session_state, []  # Clear chat history
        else:
            gr.Error("Không thể tạo cuộc trò chuyện mới")
            return session_state, None
    else:
        gr.Error(result.get("message", "Không thể tạo cuộc trò chuyện mới"))
        return session_state, None


def get_chat_sessions_list(session_state):
    """Lấy danh sách chat sessions - gọi Django API với button Load Chat"""
    if not isinstance(session_state, dict) or not session_state.get("value"):
        return "Vui lòng đăng nhập để xem lịch sử chat"
    
    session_id = session_state["value"]
    result = api_get_chat_sessions(session_id)
    
    if not result.get("success"):
        return result.get("message", "Không thể lấy danh sách chat")
    
    sessions = result.get("sessions", [])
    if not sessions:
        return "Chưa có cuộc trò chuyện nào"
    
    html_parts = []
    for idx, session in enumerate(sessions):
        chat_session_id = session.get("session_id", "")
        updated_time = session.get("updated_at", "")
        last_question = session.get("last_question", "Chưa có câu hỏi nào")
        
        display_question = last_question[:50] + "..." if len(last_question) > 50 else last_question
        
        html_parts.append(f"""
        <div class="chat-session-item" style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            margin: 8px 0;
            background: var(--background-fill-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color-primary);
        ">
            <div style="flex: 1;">
                <div style="font-weight: 500; margin-bottom: 4px;">{display_question}</div>
                <div style="font-size: 12px; color: var(--body-text-color-subdued);">{updated_time}</div>
            </div>
            <button 
                class="load-chat-btn" 
                data-session-id="{chat_session_id}"
                style="
                    padding: 8px 16px;
                    background: var(--primary-500);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    transition: background 0.2s;
                "
                onmouseover="this.style.background='var(--primary-600)'"
                onmouseout="this.style.background='var(--primary-500)'"
            >
                📥 Load Chat
            </button>
        </div>
        """)
    
    html_content = f"""
    <div class="chat-sessions-list">
        {''.join(html_parts)}
    </div>
    <script>
        (function() {{
            // Handle click on Load Chat buttons
            function handleLoadChatClick(e) {{
                const btn = e.target.classList.contains('load-chat-btn') 
                    ? e.target 
                    : e.target.closest('.load-chat-btn');
                
                if (!btn) return;
                
                const sessionId = btn.getAttribute('data-session-id');
                if (!sessionId) return;
                
                // Tìm input với nhiều cách
                let loadChatInput = document.querySelector('#load_chat_session_input textarea') || 
                                 document.querySelector('#load_chat_session_input input') ||
                                 document.querySelector('textarea#load_chat_session_input') ||
                                 document.querySelector('input#load_chat_session_input');
                
                if (!loadChatInput) {{
                    // Thử tìm bằng data-testid
                    const allInputs = document.querySelectorAll('textarea[data-testid="textbox"], input[data-testid="textbox"]');
                    for (const input of allInputs) {{
                        if (input.closest('#load_chat_session_input')) {{
                            loadChatInput = input;
                            break;
                        }}
                    }}
                }}
                
                if (loadChatInput) {{
                    loadChatInput.value = sessionId;
                    // Trigger events
                    loadChatInput.dispatchEvent(new Event('input', {{ bubbles: true, cancelable: true }}));
                    loadChatInput.dispatchEvent(new Event('change', {{ bubbles: true, cancelable: true }}));
                    
                    // Thử dùng native setter
                    try {{
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                            window.HTMLTextAreaElement?.prototype || window.HTMLInputElement?.prototype, 
                            "value"
                        )?.set;
                        if (nativeInputValueSetter) {{
                            nativeInputValueSetter.call(loadChatInput, sessionId);
                            loadChatInput.dispatchEvent(new Event('input', {{ bubbles: true, cancelable: true }}));
                        }}
                    }} catch (e) {{
                        console.log('Không thể dùng native setter:', e);
                    }}
                }} else {{
                    console.warn('Không tìm thấy load_chat_session_input');
                }}
            }}
            
            // Attach event listener
            document.addEventListener('click', handleLoadChatClick);
        }})();
    </script>
    """
    
    return html_content


def load_chat_session(chat_session_id, session_state):
    """Load chat history từ một chat session và trả về history cho ChatInterface"""
    if not chat_session_id or not chat_session_id.strip():
        return session_state, None, gr.update(value="")
    
    if not isinstance(session_state, dict) or not session_state.get("value"):
        gr.Warning("Vui lòng đăng nhập để load chat")
        return session_state, None, gr.update(value="")
    
    session_id = session_state["value"]
    
    # Gọi API để lấy chat history
    result = api_get_chat_history(chat_session_id, session_id)
    
    if not result.get("success"):
        gr.Error(result.get("message", "Không thể load chat history"))
        return session_state, None, gr.update(value="")
    
    messages = result.get("messages", [])
    if not messages:
        gr.Info("Chat session này chưa có tin nhắn nào")
        # Vẫn cập nhật chat_session_id để tiếp tục chat trong session này
        session_state["chat_session_id"] = chat_session_id
        return session_state, [], gr.update(value="")
    
    # Chuyển đổi messages thành format của Gradio ChatInterface (messages format)
    # Format mới: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    history = []
    for msg in messages:
        user_msg = msg.get("message", "")
        bot_msg = msg.get("response", "")
        if user_msg and bot_msg:
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_msg})
    
    # Cập nhật chat_session_id trong session_state
    session_state["chat_session_id"] = chat_session_id
    
    gr.Success(f"Đã load {len(history)} tin nhắn từ chat session")
    return session_state, history, gr.update(value="")


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
        function saveSessionToStorage(sessionId, accessToken, userInfo) {
            if (sessionId) {
                localStorage.setItem('ragviet_session_id', sessionId);
                if (accessToken) {
                    localStorage.setItem('ragviet_access_token', accessToken);
                }
                if (userInfo) {
                    localStorage.setItem('ragviet_user_info', JSON.stringify(userInfo));
                }
                console.log('Đã lưu session:', sessionId);
                console.log('Đã lưu access_token:', accessToken);
                console.log('Đã lưu user_info:', userInfo);
            }
        }
        
        function loadSessionFromStorage() {
            const sessionId = localStorage.getItem('ragviet_session_id');
            const accessToken = localStorage.getItem('ragviet_access_token');
            const userInfoStr = localStorage.getItem('ragviet_user_info');
            
            if (sessionId) {
                console.log('Đã load session:', sessionId);
                console.log('Đã load access_token:', accessToken);
                if (userInfoStr) {
                    try {
                        const userInfo = JSON.parse(userInfoStr);
                        console.log('Đã load user_info:', userInfo);
                    } catch (e) {
                        console.error('Lỗi parse user_info:', e);
                    }
                }
                return sessionId;
            }
            return null;
        }
        
        function clearSessionFromStorage() {
            localStorage.removeItem('ragviet_session_id');
            localStorage.removeItem('ragviet_access_token');
            localStorage.removeItem('ragviet_user_info');
            console.log('Đã xóa session và token');
        }
        
        function getAccessToken() {
            return localStorage.getItem('ragviet_access_token');
        }
        
        function getUserInfo() {
            const userInfoStr = localStorage.getItem('ragviet_user_info');
            if (userInfoStr) {
                try {
                    return JSON.parse(userInfoStr);
                } catch (e) {
                    return null;
                }
            }
            return null;
        }
        
        // Expose functions to window
        window.saveSessionToStorage = saveSessionToStorage;
        window.loadSessionFromStorage = loadSessionFromStorage;
        window.clearSessionFromStorage = clearSessionFromStorage;
        window.getAccessToken = getAccessToken;
        window.getUserInfo = getUserInfo;
        
        // Auto-restore session khi load trang - hiển thị profile ngay từ localStorage
        let restoreAttempts = 0;
        const MAX_RESTORE_ATTEMPTS = 20;
        let hasRestoredFromLocalStorage = false;
        
        // Hàm hiển thị profile từ localStorage ngay lập tức (không cần đợi API)
        function showProfileFromLocalStorage() {
            if (hasRestoredFromLocalStorage) return;
            
            const savedSession = loadSessionFromStorage();
            const userInfo = getUserInfo();
            
            if (savedSession && userInfo) {
                hasRestoredFromLocalStorage = true;
                console.log('✅ Hiển thị profile từ localStorage ngay lập tức:', userInfo);
                
                // Tạo HTML cho profile
                const userInfoHtml = `
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 15px 20px;
                        border-radius: 10px;
                        color: white;
                    ">
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 5px;">
                               👋 Xin chào, <span style="color: #ffd700;">${userInfo.username || 'User'}</span>
                            </div>
                            <div style="font-size: 13px; opacity: 0.9;">
                                Email: ${userInfo.email || ''}
                            </div>
                        </div>
                    </div>
                `;
                
                // Tìm và cập nhật UI ngay lập tức
                const loginStatus = document.querySelector('#login-status');
                const loginHeaderBtn = document.querySelector('#header-login-btn');
                const registerHeaderBtn = document.querySelector('#header-register-btn');
                const logoutBtn = document.querySelector('#header-logout-btn');
                const restoreLoading = document.querySelector('#restore-loading');
                
                if (loginStatus) {
                    loginStatus.innerHTML = userInfoHtml;
                    loginStatus.style.display = 'block';
                }
                if (loginHeaderBtn) loginHeaderBtn.style.display = 'none';
                if (registerHeaderBtn) registerHeaderBtn.style.display = 'none';
                if (logoutBtn) logoutBtn.style.display = 'block';
                if (restoreLoading) restoreLoading.style.display = 'none';
            }
        }
        
        function tryRestoreSession() {
            restoreAttempts++;
            const savedSession = loadSessionFromStorage();
            
            if (!savedSession) {
                if (restoreAttempts === 1) {
                    console.log('Không tìm thấy session đã lưu');
                    // Ẩn loading nếu không có session
                    const restoreLoading = document.querySelector('#restore-loading');
                    if (restoreLoading) restoreLoading.style.display = 'none';
                }
                return;
            }
            
            // Hiển thị profile từ localStorage ngay lập tức
            if (restoreAttempts === 1) {
                showProfileFromLocalStorage();
            }
            
            console.log(`[Attempt ${restoreAttempts}] Tìm thấy session đã lưu, đang verify với API...`, savedSession);
            
            // Tìm restore input với nhiều cách khác nhau
            let restoreInput = null;
            
            // Cách 1: Tìm trực tiếp bằng ID
            restoreInput = document.querySelector('#restore_session_input textarea') || 
                         document.querySelector('#restore_session_input input') ||
                         document.querySelector('textarea#restore_session_input') ||
                         document.querySelector('input#restore_session_input');
            
            // Cách 2: Tìm bằng data-testid
            if (!restoreInput) {
                const allTextareas = document.querySelectorAll('textarea[data-testid="textbox"]');
                for (const textarea of allTextareas) {
                    if (textarea.closest('#restore_session_input')) {
                        restoreInput = textarea;
                        break;
                    }
                }
            }
            
            // Cách 3: Tìm bằng class và parent
            if (!restoreInput) {
                const allInputs = document.querySelectorAll('.gr-textbox textarea, .gr-textbox input');
                for (const input of allInputs) {
                    if (input.closest('#restore_session_input')) {
                        restoreInput = input;
                        break;
                    }
                }
            }
            
            if (restoreInput) {
                console.log('✅ Đã tìm thấy restore input, đang set value...');
                
                // Set value với nhiều cách để đảm bảo Gradio nhận được
                restoreInput.value = savedSession;
                
                // Trigger nhiều loại events
                const events = ['input', 'change', 'keyup', 'keydown', 'paste'];
                events.forEach(eventType => {
                    restoreInput.dispatchEvent(new Event(eventType, { bubbles: true, cancelable: true }));
                });
                
                // Thử dùng native setter
                try {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement?.prototype || window.HTMLInputElement?.prototype, 
                        "value"
                    )?.set;
                    if (nativeInputValueSetter) {
                        nativeInputValueSetter.call(restoreInput, savedSession);
                        restoreInput.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                    }
                } catch (e) {
                    console.log('Không thể dùng native setter:', e);
                }
                
                // Focus và blur để trigger
                restoreInput.focus();
                setTimeout(() => {
                    restoreInput.blur();
                    console.log('✅ Đã trigger restore với value:', savedSession);
                }, 100);
                
                return true; // Thành công
            } else {
                if (restoreAttempts < MAX_RESTORE_ATTEMPTS) {
                    console.log(`[Attempt ${restoreAttempts}] Chưa tìm thấy restore input, thử lại sau...`);
                } else {
                    console.warn('⚠️ Đã thử restore quá nhiều lần mà không tìm thấy input');
                }
                return false; // Chưa thành công
            }
        }
        
        // Hiển thị loading và profile từ localStorage ngay lập tức
        function initRestore() {
            const savedSession = loadSessionFromStorage();
            if (savedSession) {
                // Hiển thị loading
                const restoreLoading = document.querySelector('#restore-loading');
                if (restoreLoading) {
                    restoreLoading.style.display = 'block';
                }
                
                // Hiển thị profile từ localStorage ngay
                showProfileFromLocalStorage();
            }
        }
        
        // Thử restore ngay khi DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                initRestore();
                setTimeout(tryRestoreSession, 100);
            });
        } else {
            initRestore();
            setTimeout(tryRestoreSession, 100);
        }
        
        // Thử restore nhiều lần với interval
        const restoreInterval = setInterval(() => {
            if (tryRestoreSession() || restoreAttempts >= MAX_RESTORE_ATTEMPTS) {
                clearInterval(restoreInterval);
            }
        }, 500);
        
        // Cleanup sau 10 giây
        setTimeout(() => {
            clearInterval(restoreInterval);
        }, 10000);
        
        // Thử restore khi Gradio app đã load xong
        // Sử dụng window.addEventListener để lắng nghe khi Gradio ready
        window.addEventListener('load', () => {
            setTimeout(tryRestoreSession, 500);
        });
        
        // Nếu có Gradio API, sử dụng nó
        if (window.gradio_config) {
            setTimeout(tryRestoreSession, 1000);
        }
        
        // Sử dụng MutationObserver để theo dõi khi restore input được thêm vào DOM
        const restoreObserver = new MutationObserver(function(mutations) {
            const savedSession = loadSessionFromStorage();
            if (savedSession && restoreAttempts < MAX_RESTORE_ATTEMPTS) {
                const restoreInput = document.querySelector('#restore_session_input textarea, #restore_session_input input');
                if (restoreInput && restoreInput.value !== savedSession) {
                    console.log('🔍 MutationObserver: Phát hiện restore input mới, đang set value...');
                    restoreInput.value = savedSession;
                    restoreInput.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
                    restoreInput.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
                    restoreAttempts = MAX_RESTORE_ATTEMPTS; // Đánh dấu đã thử
                }
            }
        });
        
        // Bắt đầu observe
        restoreObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    </script>
    </style>
    """)
    gr.Markdown("""
    # 💻 Chatbot Trả Lời Tự Động Văn Bản Hành Chính Việt Nam
    Upload file PDF hành chính của bạn và đặt câu hỏi - chatbot sẽ trả lời dựa trên nội dung tài liệu!
    
    """)
    
    session_state = gr.State(value={"value": None, "user": None, "selected_file": None, "chat_session_id": None})
    chat_history_visible = gr.State(False)
    is_restoring_session = gr.State(False)  # State để track đang restore
    
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
        #restore_session_input,
        #load_chat_session_input {
            position: absolute !important;
            left: -9999px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 1px !important;
            width: 1px !important;
            overflow: hidden !important;
        }
        /* Loading spinner */
        .spinner {
            border: 3px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top: 3px solid var(--primary-500, #0066cc);
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #restore-loading {
            padding: 15px;
            text-align: center;
            background: var(--background-fill-secondary);
            border-radius: 8px;
            margin-bottom: 10px;
        }
    </style>
    """)
    
    with gr.Row(elem_id="header-tabs-row"):
        with gr.Column(scale=0, min_width=300, elem_classes="auth-section"):
            auth_text = gr.Markdown("**Tài khoản:**", elem_id="auth-text", visible=False)
            restore_loading = gr.Markdown(
                visible=False,
                elem_id="restore-loading",
                value="<div style='text-align: center; padding: 10px;'><div class='spinner'></div><br/>Đang khôi phục phiên đăng nhập...</div>"
            )
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
                
                def update_file_dropdown():
                    _, file_names = get_uploaded_files()
                    return gr.Dropdown(choices=[""] + file_names, value=None)
                
                file_selection_dropdown.change(
                    select_file_fn,
                    inputs=[file_selection_dropdown, session_state],
                    outputs=[file_selection_output, session_state]
                )
                
                def chat_wrapper(message, history, session_state_val):
                    session_id = None
                    selected_file = None
                    chat_session_id = None
                    
                    if isinstance(session_state_val, dict):
                        session_id = session_state_val.get("value")
                        selected_file = session_state_val.get("selected_file")
                        chat_session_id = session_state_val.get("chat_session_id")
                    
                    if session_id and not chat_session_id:
                        create_result = api_create_chat_session(session_id)
                        if create_result.get("success"):
                            chat_session_id = create_result.get("chat_session_id")
                            # Cập nhật session_state ngay lập tức (lưu ý: cái này chỉ update local dict, 
                            # không update lại state của Gradio trừ khi return, nhưng ChatInterface không support return state)
                            if isinstance(session_state_val, dict):
                                session_state_val["chat_session_id"] = chat_session_id
                    
                    response = chat_interface_fn(message, history, session_id, selected_file, chat_session_id)
                    
                    return response
                
                chatbot = gr.Chatbot(type="messages", label="Chat với RagVietBot")
                
                chat_interface = gr.ChatInterface(
                    fn=chat_wrapper,
                    additional_inputs=[session_state],
                    chatbot=chatbot,
                    title="Chat với RagVietBot",
                    description="Đặt câu hỏi về nội dung các tài liệu đã upload",
                    examples=[
                        ["Tóm tắt nội dung chính của tài liệu", None],
                        ["Các quy định về thủ tục hành chính là gì?", None],
                        ["Thời hạn xử lý hồ sơ là bao lâu?", None]
                    ],
                    cache_examples=False
                )
                
                load_chat_session_input = gr.Textbox(
                    visible=False,
                    show_label=False,
                    elem_id="load_chat_session_input",
                    interactive=False,
                    container=False,
                    lines=1,
                    placeholder=""
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
                
                load_chat_session_input.change(
                    load_chat_session,
                    inputs=[load_chat_session_input, session_state],
                    outputs=[session_state, chat_interface.chatbot, load_chat_session_input]
                )
            
            with gr.Tab("📁 Quản Lý Tài Liệu"):
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
                    process_pdfs(files, session_state)
                
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
            gr.update(visible=True),   
            gr.update(visible=False),  
            gr.update(visible=False),  
            gr.update(visible=False)  
        )
    
    def show_register():
        return (
            gr.update(visible=False),  
            gr.update(visible=True),  
            gr.update(visible=False),  
            gr.update(visible=False)  
        )
    
    def show_forgot():
        return (
            gr.update(visible=False),  
            gr.update(visible=False),  
            gr.update(visible=True),  
            gr.update(visible=False)  
        )
    
    def show_reset():
        return (
            gr.update(visible=False), 
            gr.update(visible=False),  
            gr.update(visible=False),  
            gr.update(visible=True)   
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
        inputs=[restore_session_input, session_state, is_restoring_session],
        outputs=[session_state, restore_loading, login_header_btn, register_header_btn, login_status, logout_btn]
    )
    
    def on_app_load():
        """Callback khi app load - JavaScript sẽ tự động trigger restore"""
        pass
    
    app.load(
        fn=on_app_load,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    logger.info("Khởi động ứng dụng Chatbot Hành Chính Việt Nam...")
    app.launch(server_name="0.0.0.0", share=True)
