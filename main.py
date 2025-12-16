import json
import os
import tempfile
from types import SimpleNamespace
import asyncio
from typing import List, Optional, Tuple
import logging

from dotenv import load_dotenv
from nicegui import app, ui, context

logger = logging.getLogger(__name__)

from api_client import (
    api_chat_send,
    api_clear_all_files,
    api_create_chat_session,
    api_delete_file,
    api_forgot_password,
    api_get_chat_history,
    api_get_chat_sessions,
    api_get_files,
    api_login,
    api_logout,
    api_register,
    api_reset_password,
    api_upload_files,
    api_verify_session,
    api_view_file,
)

load_dotenv()

STORAGE_SECRET = os.getenv("STORAGE_SECRET", "ragviet-dev-secret")
app.storage.secret = STORAGE_SECRET
ui.add_head_html("""
<style>
.nicegui-content{padding:0!important;}
.q-message-text strong { font-weight: bold; }
.math-formula {
    font-family: 'Times New Roman', serif;
    font-style: italic;
    margin: 0.5em 0;
    padding: 0.5em;
    background: #f5f5f5;
    border-radius: 4px;
    white-space: pre-wrap;
    font-size: 1.1em;
}
blockquote {
    border-left: 3px solid #ccc;
    padding-left: 1em;
    margin: 0.5em 0;
    color: #666;
}
</style>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
};
</script>
""", shared=True)

class SessionState:
    def __init__(self):
        self.session_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self.user: Optional[dict] = None
        self.selected_file: Optional[str] = None
        self.chat_session_id: Optional[str] = None
        self.pending_load_history: Optional[str] = None  # Chat session ID cần load

    @property
    def is_logged_in(self) -> bool:
        return bool(self.session_id)


session_state = SessionState()


def _get_user_store():
    """
    Lấy storage gắn với client (server-side, không phụ thuộc browser dict).
    Dùng user-level storage để tránh lỗi "response has been built".
    """
    try:
        # Thử lấy từ context.client trước
        if hasattr(context, "client") and context.client:
            client_store = getattr(context.client, "storage", None)
            if client_store:
                user = getattr(client_store, "user", None)
                if user is not None:
                    return user
        # Fallback về app.storage.user
        app_store = getattr(app, "storage", None)
        if app_store:
            user = getattr(app_store, "user", None)
            if user is not None:
                return user
    except Exception:
        pass
    return None


def save_session_to_storage():
    """Lưu session vào storage server-side (user storage) để reload không mất."""
    user_store = _get_user_store()
    if not user_store:
        return
    user_store["session_id"] = session_state.session_id
    user_store["access_token"] = session_state.access_token
    user_store["user"] = session_state.user


def clear_session_storage():
    user_store = _get_user_store()
    if not user_store:
        return
    for key in ("session_id", "access_token", "user"):
        user_store.pop(key, None)

def restore_session_from_storage():
    """Khôi phục session từ local storage nếu còn hợp lệ."""
    if session_state.is_logged_in:
        return True
    
    user_store = _get_user_store()
    if not user_store:
        return False
    stored_session = user_store.get("session_id")
    if not stored_session:
        return False
    verify = api_verify_session(stored_session)
    if verify.get("success") and verify.get("valid"):
        session_state.session_id = stored_session
        session_state.access_token = stored_session
        session_state.user = verify.get("user")
        session_state.chat_session_id = verify.get("chat_session_id")
        print(f"DEBUG: Restored session. chat_session_id={session_state.chat_session_id}")
        return True
    clear_session_storage()
    return False

def notify_success(msg: str, notify_type: str = "positive"):
    ui.notify(msg, type=notify_type)


def notify_error(msg: str):
    ui.notify(msg, type="negative")


def require_login() -> bool:
    if not session_state.is_logged_in:
        notify_error("Vui lòng đăng nhập để sử dụng tính năng này")
        return False
    return True

def require_auth():
    """Kiểm tra đăng nhập và redirect về /login nếu chưa đăng nhập."""
    restore_session_from_storage()
    
    if not session_state.is_logged_in:
        ui.add_head_html(
            '<script>window.location.href = "/login";</script>',
            shared=False
        )
        ui.label("Đang chuyển đến trang đăng nhập...").classes("text-center p-4")
        return False
    return True

def refresh_files_list() -> Tuple[str, List[str]]:
    result = api_get_files(session_state.session_id)
    if not result.get("success") or result.get("total_files", 0) == 0:
        return "Chưa có file nào được upload.", []
    files = result.get("files", [])
    files_list = "\n".join(
        [f"📄 {file['filename']}: {file['chunks']} chunks" for file in files]
    )
    display_text = (
        f"- Tổng số tài liệu: {result['total_files']}\n"
        f"- Tổng số chunks: {result['total_chunks']}\n"
        f"{files_list}"
    )
    file_names = [file["filename"] for file in files]
    return display_text, file_names


async def upload_temp_files(upload_event) -> bool:
    """Nhận UploadEvent (có thể 1 hoặc nhiều file) và gọi API upload."""
    if not require_login():
        return False

    incoming = []
    
    logger.info(f"=== UPLOAD EVENT DEBUG ===")
    logger.info(f"Type: {type(upload_event)}")
    if hasattr(upload_event, "__dict__"):
        logger.info(f"Dict: {upload_event.__dict__}")
    if hasattr(upload_event, "__class__"):
        logger.info(f"Class: {upload_event.__class__}")
        logger.info(f"Class attributes: {[x for x in dir(upload_event.__class__) if not x.startswith('_')]}")
    logger.info(f"All attributes: {[x for x in dir(upload_event) if not x.startswith('_')]}")
    
    # Thử nhiều cách để lấy files
    if hasattr(upload_event, "files") and upload_event.files:
        incoming = upload_event.files if isinstance(upload_event.files, list) else [upload_event.files]
        logger.info(f"Got files from .files attribute: {len(incoming)} files")
    elif hasattr(upload_event, "file") and upload_event.file:
        incoming = [upload_event.file] if not isinstance(upload_event.file, list) else upload_event.file
        logger.info(f"Got files from .file attribute: {len(incoming)} files")
    elif isinstance(upload_event, list):
        incoming = upload_event
        logger.info(f"Upload event is a list: {len(incoming)} items")
    elif hasattr(upload_event, "__iter__") and not isinstance(upload_event, str):
        try:
            incoming = list(upload_event)
            logger.info(f"Upload event is iterable: {len(incoming)} items")
        except:
            incoming = [upload_event]
            logger.info(f"Could not iterate, treating as single item")
    else:
        incoming = [upload_event]
        logger.info(f"Treating upload event as single item")

    logger.info(f"Received upload event with {len(incoming)} file(s)")
    logger.info(f"Upload event type: {type(upload_event)}")
    logger.info(f"Upload event attributes: {dir(upload_event) if hasattr(upload_event, '__dict__') else 'N/A'}")

    temp_wrappers: List[SimpleNamespace] = []
    try:
        for idx, f in enumerate(incoming):
            logger.info(f"Processing file {idx+1}/{len(incoming)}")
            logger.info(f"File object type: {type(f)}")
            logger.info(f"File object attributes: {dir(f) if hasattr(f, '__dict__') else 'N/A'}")
            
            # Lấy tên file gốc - thử nhiều cách
            original_name = None
            if hasattr(f, "name"):
                original_name = f.name
                logger.info(f"Got name from .name: {original_name}")
            elif hasattr(f, "filename"):
                original_name = f.filename
                logger.info(f"Got name from .filename: {original_name}")
            elif isinstance(f, dict):
                original_name = f.get("name") or f.get("filename")
                logger.info(f"Got name from dict: {original_name}")
            elif hasattr(f, "__dict__"):
                # Thử lấy từ __dict__
                original_name = getattr(f, "__dict__", {}).get("name") or getattr(f, "__dict__", {}).get("filename")
                logger.info(f"Got name from __dict__: {original_name}")
            
            if not original_name:
                original_name = "upload.pdf"
                logger.warning(f"Using default name: {original_name}")
            
            logger.info(f"Final file name: {original_name}")
            
            # Lấy nội dung file - thử nhiều cách
            content = None
            file_path = None
            
            # Cách 1: Kiểm tra xem có phải là file path string không
            if isinstance(f, str) and os.path.exists(f):
                file_path = f
                logger.info(f"File is a path string: {file_path}")
            
            # Cách 2: Đọc từ content attribute
            elif hasattr(f, "content"):
                try:
                    content = f.content
                    if content:
                        logger.info(f"Read content from .content attribute: {len(content) if isinstance(content, bytes) else 'not bytes'} bytes")
                    else:
                        logger.warning("Content attribute exists but is None/empty")
                except Exception as e:
                    logger.warning(f"Error reading .content: {e}")
            
            # Cách 3: Đọc từ file object (có thể là coroutine)
            elif hasattr(f, "read"):
                try:
                    if hasattr(f, "seek"):
                        f.seek(0)
                    # Kiểm tra xem read() có phải là coroutine không
                    read_result = f.read()
                    if asyncio.iscoroutine(read_result):
                        content = await read_result
                        logger.info(f"Read content from async .read(): {len(content) if content else 0} bytes")
                    else:
                        content = read_result
                        logger.info(f"Read content from sync .read(): {len(content) if content else 0} bytes")
                    if hasattr(f, "seek"):
                        f.seek(0)  # Reset để có thể đọc lại
                except Exception as e:
                    logger.warning(f"Could not read from file object: {e}")
            
            # Cách 4: Đọc từ path attribute
            if content is None and file_path is None:
                if hasattr(f, "path"):
                    file_path = f.path
                    logger.info(f"Got path from .path: {file_path}")
                elif isinstance(f, dict):
                    file_path = f.get("path")
                    logger.info(f"Got path from dict: {file_path}")
                elif hasattr(f, "__dict__"):
                    file_path = getattr(f, "__dict__", {}).get("path")
                    logger.info(f"Got path from __dict__: {file_path}")
            
            # Đọc từ file path nếu có
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as file_handle:
                        content = file_handle.read()
                    logger.info(f"Read content from path {file_path}: {len(content)} bytes")
                except Exception as e:
                    logger.error(f"Could not read from path {file_path}: {e}")
            
            # Cách 5: NiceGUI có thể lưu file trong thư mục tạm
            if content is None:
                # Thử tìm trong thư mục upload của NiceGUI
                possible_paths = [
                    getattr(f, "path", None),
                    getattr(f, "file_path", None),
                    getattr(f, "tmp_path", None),
                ]
                for pp in possible_paths:
                    if pp and os.path.exists(pp):
                        try:
                            with open(pp, 'rb') as file_handle:
                                content = file_handle.read()
                            logger.info(f"Read content from possible path {pp}: {len(content)} bytes")
                            break
                        except:
                            pass
            
            if content is None or (isinstance(content, bytes) and len(content) == 0):
                logger.error(f"Không thể đọc nội dung file: {original_name}")
                logger.error(f"File object: {f}")
                logger.error(f"File path: {file_path}")
                continue
            
            # Tạo file tạm với tên gốc
            file_ext = os.path.splitext(original_name)[-1] or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, prefix="ragviet_")
            try:
                if isinstance(content, bytes):
                    tmp.write(content)
                elif hasattr(content, "read"):
                    tmp.write(content.read())
                else:
                    tmp.write(str(content).encode())
                tmp.flush()
                tmp.close()
                
                logger.info(f"Created temp file: {tmp.name} for {original_name}")
                
                # Lưu cả path và tên gốc
                temp_wrappers.append(SimpleNamespace(
                    path=tmp.name, 
                    name=original_name  # Lưu tên gốc để API biết tên file
                ))
            except Exception as e:
                logger.error(f"Error writing temp file: {e}")
                try:
                    tmp.close()
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
                except:
                    pass
                continue

        if not temp_wrappers:
            logger.error("No valid files to upload")
            notify_error("Không tìm thấy file để upload")
            return False

        logger.info(f"Uploading {len(temp_wrappers)} file(s) to API...")
        result = api_upload_files(temp_wrappers, session_state.session_id)
        
        if result.get("success"):
            message = result.get("message", "Đã upload file thành công!")
            # Nếu có warning (file không có text nhưng vẫn upload được)
            if result.get("warning"):
                notify_success(message, notify_type="warning")
            else:
                notify_success(message)
            logger.info("Upload successful, returning True for refresh")
            return True
        else:
            error_msg = result.get("message", "Lỗi khi upload file")
            logger.error(f"Upload failed: {error_msg}")
            notify_error(error_msg)
            return False
    except Exception as e:
        logger.error(f"Exception in upload_temp_files: {e}", exc_info=True)
        notify_error(f"Lỗi upload: {e}")
        return False
    finally:
        # Xóa temp files sau khi upload xong
        for t in temp_wrappers:
            try:
                if os.path.exists(t.path):
                    os.remove(t.path)
                    logger.info(f"Deleted temp file: {t.path}")
            except Exception as e:
                logger.warning(f"Không thể xóa temp file {t.path}: {e}")

def handle_login(email: str, password: str):
    email = (email or "").strip()
    password = (password or "").strip()
    if not email or not password:
        notify_error("Vui lòng nhập email và mật khẩu")
        return
    result = api_login(email, password)
    if result.get("success"):
        session_state.session_id = result["session_id"]
        session_state.access_token = result.get("access_token", result["session_id"])
        session_state.user = result.get("user")
        session_state.chat_session_id = result.get("chat_session_id")
        save_session_to_storage()
        notify_success(result.get("message", "Đăng nhập thành công"))
        ui.navigate.to("/")
    else:
        status_code = result.get("status_code")
        msg = (
            result.get("message")
            or result.get("detail")
            or result.get("response")
            or (f"{status_code} Unauthorized" if status_code == 401 else None)
            or "Đăng nhập thất bại"
        )
        notify_error(msg)


def handle_register(username: str, email: str, password: str, confirm: str):
    result = api_register(username, email, password, confirm)
    if result.get("success"):
        notify_success(result.get("message", "Đăng ký thành công"))
        ui.navigate.to("/")
    else:
        notify_error(result.get("message", "Đăng ký thất bại"))


def handle_logout():
    if session_state.session_id:
        try:
            api_logout(session_state.session_id)
        except Exception:
            pass
    session_state.session_id = None
    session_state.user = None
    session_state.selected_file = None
    session_state.chat_session_id = None
    clear_session_storage()
    notify_success("Đã đăng xuất")
    ui.navigate.to("/login")


# -------------------------
# UI building blocks
# -------------------------
def render_navbar():
    # Đảm bảo khôi phục session cho mỗi lần render navbar
    restore_session_from_storage()
    with ui.header().classes("items-center justify-between px-4"):
        ui.label("RAGViet").classes("text-xl font-bold")
        with ui.row().classes("items-center gap-2"):
            ui.button("Trang chủ", on_click=lambda: ui.navigate.to("/")).props("flat")
            ui.button("Chat", on_click=lambda: ui.navigate.to("/chat")).props("flat")
            ui.button("Tài liệu", on_click=lambda: ui.navigate.to("/documents")).props("flat")
            if session_state.is_logged_in:
                ui.button(
                    session_state.user.get("username") if session_state.user else "Đã đăng nhập",
                    on_click=lambda: ui.navigate.to("/profile"),
                ).props("outline")
                ui.button("Đăng xuất", color="negative", on_click=handle_logout)
            else:
                ui.button("Đăng nhập", color="primary", on_click=lambda: ui.navigate.to("/login"))
                ui.button("Đăng ký", on_click=lambda: ui.navigate.to("/register")).props("outline")


def render_files_summary(target_markdown):
    text, _ = refresh_files_list()
    target_markdown.set_content(text)


def render_sidebar(include_file_select: bool = True):
    """Sidebar (1/4 width) chứa upload, danh sách tài liệu, chọn file để chat."""
    text, file_names = refresh_files_list()
    file_select = None

    with ui.column().classes(
        "bg-gray-50 border-r h-screen p-4 gap-3 shrink-0 justify-between"
    ).style("width:25%;max-width:25%;min-width:260px; display: flex; flex-direction: column"):
        def refresh_lists():
            """Refresh danh sách files và cập nhật dropdown"""
            try:
                new_text, new_files = refresh_files_list()
                if include_file_select and file_select is not None:
                    new_options = ["Tất cả"] + new_files
                    file_select.options = new_options
                    # Giữ nguyên giá trị hiện tại nếu vẫn còn trong options
                    current_value = file_select.value
                    if current_value and current_value not in new_options:
                        file_select.value = "Tất cả"
                    logger.info(f"Updated file_select with {len(new_files)} files")
                return new_files
            except Exception as e:
                logger.error(f"Error refreshing lists: {e}", exc_info=True)
                return []

        async def handle_upload(e):
            """Xử lý upload và refresh sau khi thành công"""
            try:
                result = await upload_temp_files(e)
                if result:      # Upload thành công
                    await asyncio.sleep(1.0)
                    max_retries = 5
                    for retry in range(max_retries):
                        new_files = refresh_lists()
                        if new_files:  # Có files rồi
                            logger.info(f"Successfully refreshed file list after {retry + 1} attempts")
                            # Force update UI
                            if file_select is not None:
                                file_select.update()
                            break
                        await asyncio.sleep(0.3)
                    else:
                        logger.warning("File list refresh completed but no files found")
            except Exception as ex:
                logger.error(f"Error in handle_upload: {ex}", exc_info=True)
                notify_error(f"Lỗi khi xử lý upload: {ex}")

        # Section chung cho Select và Upload
        with ui.column().classes("gap-3 w-full"):
            if include_file_select:
                file_select = ui.select(
                    options=["Tất cả"] + file_names,
                    value="Tất cả",
                    label="Chọn tài liệu để chat",
                ).props("clearable dense").classes("w-full").style("font-size: 1rem")
            else:
                file_select = None
            
            ui.upload(
                label="Upload tài liệu PDF",
                multiple=True,
                on_upload=handle_upload,
            ).props("color=primary flat no-thumbnails").classes("w-full").style("margin-top: 16px")

        ui.separator()
        
        # Section lịch sử chat trong sidebar
        with ui.card().classes("w-full shadow-none border p-3 gap-2"):
            ui.label("📜 Lịch sử chat").classes("text-sm font-semibold mb-2")
            chat_history_sidebar = ui.select(
                options={}, 
                label="Chọn cuộc trò chuyện", 
                value=None
            ).props("clearable dense").classes("w-full").style("font-size: 0.85rem")
            
            def refresh_sidebar_history():
                """Refresh chat history trong sidebar"""
                try:
                    sessions_result = api_get_chat_sessions(session_state.session_id)
                    if sessions_result.get("success"):
                        sessions = sessions_result.get("sessions", [])
                        options = {}
                        for session in sessions:
                            s_id = session.get("session_id")
                            if not s_id:
                                continue
                            title = session.get("title", "Chat không có tiêu đề")
                            time_str = session.get("updated_at") or session.get("created_at", "")
                            
                            display_text = f"{title[:30]}..." if len(title) > 30 else title
                            if time_str:
                                display_text += f" ({time_str})"
                            options[s_id] = display_text
                        
                        chat_history_sidebar.options = options
                        
                        # Priority: pending load history -> current chat session
                        target_id = session_state.pending_load_history or session_state.chat_session_id
                        print(f"DEBUG: Refresh sidebar. pending={session_state.pending_load_history}, current={session_state.chat_session_id}, target={target_id}")
                        if target_id and target_id in options:
                            if chat_history_sidebar.value != target_id:
                                chat_history_sidebar.value = target_id
                        
                        chat_history_sidebar.update()
                except Exception as e:
                    logger.error(f"Error refreshing sidebar history: {e}")
            
            def on_sidebar_history_change(e):
                val = e.value
                current = session_state.pending_load_history or session_state.chat_session_id
                print(f"DEBUG: Sidebar change event. Val={val}, Current={current}, Equal={val==current}")
                if val and val != current:
                    # Set flag để load history khi trang load
                    session_state.pending_load_history = val
                    # Navigate về trang chủ
                    ui.navigate.to("/")
            
            chat_history_sidebar.on_value_change(on_sidebar_history_change)
            refresh_sidebar_history()
        
        ui.separator()
        with ui.card().classes("w-full shadow-none border p-3 gap-2"):
            if session_state.is_logged_in and session_state.user:
                ui.label(f"👤 {session_state.user.get('username','')}").classes("text-sm font-semibold")
                ui.label(session_state.user.get("email","")).classes("text-xs text-gray-600")
                ui.button("Hồ sơ", on_click=lambda: ui.navigate.to("/profile")).props("outline").classes("w-full")
                ui.button("Đăng xuất", color="negative", on_click=handle_logout).classes("w-full")
            else:
                ui.button("Đăng nhập", color="primary", on_click=lambda: ui.navigate.to("/login")).classes("w-full")

    return file_select

def render_shell(include_file_select: bool, content_builder):
    """Khung layout 1/4 sidebar - 3/4 main-content."""
    with ui.row().classes("w-full min-h-screen"):
        file_select = render_sidebar(include_file_select=include_file_select)
        with ui.column().classes("min-h-screen p-6 gap-4 bg-white flex-1").style(
            "width:75%;max-width:75%;"
        ):
            content_builder(file_select)


@ui.page("/")
def home_page():
    if not require_auth():
        return
    
    def build_content(file_select):
        # Header cuộc trò chuyện
        with ui.row().classes("w-full items-center justify-between mb-4"):
            conv_label = ui.label("Trò chuyện với: Tất cả tài liệu").classes("text-xl font-semibold")
        
        if file_select:
            def update_conv_label(e):
                name = e.value or "Tất cả"
                if name == "Tất cả":
                    name = "Tất cả tài liệu"
                conv_label.set_text(f"Trò chuyện với: {name}")
                ui.notify(f"Đã chọn tài liệu: {name}", type="positive")
            file_select.on_value_change(update_conv_label)

        msg_input = None
        send_btn = None
        
        with ui.column().classes("w-full gap-2").style("display: flex; flex-direction: column; height: 85vh"):
            chat_log = ui.column().classes("gap-2 flex-1 overflow-auto border rounded p-3 bg-gray-50 w-full").style("display: flex; flex-direction: column; min-height: 0")
            
            # Hàm load chat history
            def load_chat_history(chat_session_id: str):
                """Load lịch sử chat từ một session"""
                if not chat_session_id:
                    return
                
                # Clear chat log hiện tại
                chat_log.clear()
                
                # Lấy lịch sử chat
                history_result = api_get_chat_history(chat_session_id, session_state.session_id)
                print(f"DEBUG: Loaded chat history (session {chat_session_id}): {history_result}")
                
                if history_result.get("success"):
                    messages = history_result.get("messages", [])
                    if messages:
                        for msg in messages:
                            role = msg.get("role", "assistant")
                            content = msg.get("content", "")
                            if content:
                                add_message(role, content)
                        # Set chat_session_id hiện tại
                        session_state.chat_session_id = chat_session_id
                        ui.notify(f"Đã tải {len(messages)} tin nhắn từ lịch sử", type="positive")
                    else:
                        ui.notify("Không có tin nhắn trong session này", type="info")
                else:
                    notify_error(history_result.get("message", "Không thể tải lịch sử chat"))
            
            # Kiểm tra nếu có pending load history từ sidebar
            if session_state.pending_load_history:
                load_session_id = session_state.pending_load_history
                session_state.pending_load_history = None
                # Đợi một chút để UI render xong
                ui.timer(0.3, lambda: load_chat_history(load_session_id), once=True)
            
            def format_text(text: str) -> str:
                """Format text với markdown và MathJax support"""
                import re
                import html
                
                text = re.sub(r'strong>', '<strong>', text)
                text = re.sub(r'</strong>', '</strong>', text)
                
                lines = text.split('\n')
                formatted_lines = []
                in_blockquote = False
                in_math_formula = False
                math_lines = []
                
                math_chars = ['∑', '∫', '=', '≤', '≥', '≠', '±', '×', '÷', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω', 'Δ', 'Ω', '∞']
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Kiểm tra nếu là dòng bắt đầu bằng >
                    if stripped.startswith('>'):
                        content = stripped[1:].strip()
                        
                        # Kiểm tra nếu là công thức toán học
                        is_math = any(char in content for char in math_chars) or \
                                 re.search(r'[a-z]_[a-z]', content) or \
                                 re.search(r'[A-Z][a-z]+[A-Z]', content) or \
                                 (i > 0 and lines[i-1].strip().startswith('>') and in_math_formula)
                        
                        if is_math:
                            # Bắt đầu công thức toán học
                            if not in_math_formula:
                                in_math_formula = True
                                math_lines = []
                            math_lines.append(content)
                        else:
                            # Kết thúc công thức toán học nếu đang trong công thức
                            if in_math_formula:
                                # Render công thức
                                math_content = '\n'.join(math_lines)
                                formatted_lines.append(f'<div class="math-formula">{html.escape(math_content)}</div>')
                                in_math_formula = False
                                math_lines = []
                            
                            # Xử lý blockquote thông thường
                            if not in_blockquote:
                                formatted_lines.append('<blockquote>')
                                in_blockquote = True
                            formatted_lines.append(f'<p>{html.escape(content)}</p>')
                        continue
                    else:
                        # Kết thúc blockquote hoặc công thức
                        if in_math_formula:
                            math_content = '\n'.join(math_lines)
                            formatted_lines.append(f'<div class="math-formula">{html.escape(math_content)}</div>')
                            in_math_formula = False
                            math_lines = []
                        
                        if in_blockquote:
                            formatted_lines.append('</blockquote>')
                            in_blockquote = False
                        
                        formatted_lines.append(line)
                
                # Đóng các blockquote/công thức còn lại
                if in_math_formula:
                    math_content = '\n'.join(math_lines)
                    formatted_lines.append(f'<div class="math-formula">{html.escape(math_content)}</div>')
                if in_blockquote:
                    formatted_lines.append('</blockquote>')
                
                text = '\n'.join(formatted_lines)
                
                # Kiểm tra xem text đã có HTML tags chưa (từ LLM response)
                # Nếu đã có HTML tags hợp lệ, không cần xử lý markdown nữa
                has_html_tags = bool(re.search(r'<(strong|em|ul|li|h[1-6]|blockquote|div|p|code)[^>]*>', text, re.IGNORECASE))
                
                if not has_html_tags:
                    # Chỉ xử lý markdown nếu chưa có HTML tags
                    def replace_bold(match):
                        bold_text = match.group(1)
                        # Nếu đã có <strong> tag thì bỏ qua
                        if '<strong>' in bold_text or '</strong>' in bold_text:
                            return match.group(0)
                        return f'<strong>{html.escape(bold_text)}</strong>'
                    
                    text = re.sub(r'\*\*([^*]+?)\*\*', replace_bold, text)
                    
                    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', text)
                    
                    text = re.sub(r'`([^`]+?)`', r'<code style="background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace;">\1</code>', text)
                    
                    text = re.sub(r'^-\s+(.+)$', r'<li style="margin: 0.3em 0;">\1</li>', text, flags=re.MULTILINE)
                    
                    text = re.sub(r'(<li[^>]*>.*?</li>(?:\s*<li[^>]*>.*?</li>)*)', r'<ul style="margin: 0.5em 0; padding-left: 1.5em;">\1</ul>', text, flags=re.DOTALL)
                    
                    text = re.sub(r'^###\s+(.+)$', r'<h3 style="font-size: 1.2em; font-weight: bold; margin: 1em 0 0.5em 0; color: #333;">\1</h3>', text, flags=re.MULTILINE)
                    text = re.sub(r'^##\s+(.+)$', r'<h2 style="font-size: 1.4em; font-weight: bold; margin: 1.2em 0 0.6em 0; color: #222;">\1</h2>', text, flags=re.MULTILINE)
                    text = re.sub(r'^#\s+(.+)$', r'<h1 style="font-size: 1.6em; font-weight: bold; margin: 1.5em 0 0.8em 0; color: #111;">\1</h1>', text, flags=re.MULTILINE)
                
                paragraphs = re.split(r'\n\s*\n', text)
                formatted_paragraphs = []
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    # Kiểm tra nếu paragraph đã chứa HTML tags hợp lệ
                    has_html_tags = bool(re.search(r'<(strong|em|ul|li|h[1-6]|blockquote|div|p|code)[^>]*>', para, re.IGNORECASE))
                    
                    if has_html_tags:
                        # Nếu đã có HTML tags, chỉ cần thêm vào (không escape)
                        formatted_paragraphs.append(para)
                    elif para.startswith('<') and (para.startswith('<h') or para.startswith('<ul') or para.startswith('<blockquote') or para.startswith('<div')):
                        # Nếu là HTML block element, giữ nguyên
                        formatted_paragraphs.append(para)
                    else:
                        # Nếu là plain text, escape và wrap trong <p>
                        para_escaped = html.escape(para)
                        # Thay \n thành <br> trong paragraph
                        para_escaped = para_escaped.replace('\n', '<br>')
                        formatted_paragraphs.append(f'<p style="margin: 0.5em 0; line-height: 1.6;">{para_escaped}</p>')
                
                formatted = '\n'.join(formatted_paragraphs)
                
                # Clean up multiple <br> tags
                formatted = re.sub(r'<br>\s*<br>+', '<br>', formatted)
                
                return formatted

            def add_message(role: str, text: str):
                with chat_log:
                    if role == "user":
                        msg = ui.chat_message(text, name="Bạn").props("sent")
                        msg.classes("q-message-text q-message-text--sent justify-end")
                        msg.style("height: fit-content; align-self: flex-end; margin-left: auto")
                    else:
                        # Format text với bold cho **text** và dùng chat_message
                        formatted_text = format_text(text)
                        msg = ui.chat_message("", name="Assistant")
                        # Set HTML content vào message text
                        with msg:
                            ui.html(formatted_text, sanitize=False)

            async def ensure_chat_session():
                if not session_state.chat_session_id and session_state.session_id:
                    res = await asyncio.to_thread(
                        api_create_chat_session, session_state.session_id
                    )
                    if res.get("success"):
                        session_state.chat_session_id = res.get("chat_session_id")

            async def send():
                message = (msg_input.value or "").strip()
                if not message:
                    return
                if not require_login():
                    return
                await ensure_chat_session()
                add_message("user", message)
                selected = file_select.value if file_select else None
                # Nếu chọn "Tất cả" hoặc rỗng thì gửi None
                if selected == "Tất cả" or not selected:
                    selected = None
                msg_input.props("disable")
                send_btn.text = "Đang tìm kiếm câu trả lời"
                send_btn.props("loading")
                with chat_log:
                    pending = ui.chat_message("Đang trả lời...", name="Assistant").classes("opacity-70 italic")
                try:
                    resp = await asyncio.to_thread(
                        api_chat_send,
                        message,
                        session_state.session_id,
                        selected_file=selected,
                        chat_session_id=session_state.chat_session_id,
                    )
                    print("chat_response_home:", resp)  # debug log
                    if resp.get("success"):
                        bot = resp.get("response", "Không có phản hồi")
                        session_state.chat_session_id = resp.get("chat_session_id", session_state.chat_session_id)
                        pending.delete()
                        add_message("assistant", bot)
                        # Refresh chat history trong sidebar sau khi có tin nhắn mới
                        if hasattr(session_state, 'refresh_sidebar_history'):
                            session_state.refresh_sidebar_history()
                        ui.notify("Đã nhận câu trả lời", type="positive")
                    else:
                        err = resp.get("message") or resp.get("response") or "Lỗi khi gửi tin nhắn"
                        notify_error(err)
                        pending.delete()
                        add_message("assistant", err)
                finally:
                    msg_input.value = ""
                    msg_input.props(remove="disable")
                    send_btn.text = "Gửi"
                    send_btn.props(remove="loading")

            # Input row fixed ở bottom
            with ui.row().classes("w-full items-stretch gap-2 shrink-0"):
                msg_input = ui.input("Nhập câu hỏi...").props("outlined clearable").classes("flex-1")
                send_btn = ui.button("Gửi", color="primary", on_click=send).style("width: 60px; min-width: 60px; height: 56px; min-height: 56px")

    render_shell(include_file_select=True, content_builder=build_content)


@ui.page("/login")
def login_page():
    render_navbar()
    with ui.row().classes("w-full min-h-screen items-center justify-center bg-gray-50"):
        with ui.column().classes("items-center justify-center gap-4 w-full max-w-md"):
            ui.markdown("## Đăng nhập").classes("self-center")
            with ui.card().classes("gap-3 w-full p-6 shadow-md").style("border: 1px solid #ccc"):
                email = ui.input("Email").classes("w-full")
                password = ui.input("Mật khẩu", password=True).classes("w-full")
                with ui.column().classes("w-full items-center gap-2"):
                    ui.link("Chưa có tài khoản? Đăng ký", "/register")
                    ui.link("Quên mật khẩu?", "/forgot-password")
                ui.button(
                    "Đăng nhập",
                    color="primary",
                    on_click=lambda: handle_login(email.value, password.value),
                ).classes("w-full")


@ui.page("/register")
def register_page():
    render_navbar()
    with ui.row().classes("w-full min-h-screen items-center justify-center bg-gray-50"):
        with ui.column().classes("items-center justify-center gap-4 w-full max-w-md"):
            ui.markdown("## Đăng ký").classes("self-center")
            with ui.card().classes("gap-3 w-full p-6 shadow-md").style("border: 1px solid #ccc"):
                username = ui.input("Tên đăng nhập").classes("w-full")
                email = ui.input("Email").classes("w-full")
                password = ui.input("Mật khẩu", password=True).classes("w-full")
                confirm = ui.input("Xác nhận mật khẩu", password=True).classes("w-full")
                ui.button(
                    "Đăng ký",
                    color="primary",
                    on_click=lambda: handle_register(username.value, email.value, password.value, confirm.value),
                ).classes("w-full")
                with ui.column().classes("w-full items-center"):
                    ui.link("Đã có tài khoản? Đăng nhập", "/login")


@ui.page("/forgot-password")
def forgot_page():
    render_navbar()
    with ui.row().classes("w-full min-h-screen items-center justify-center bg-gray-50"):
        with ui.column().classes("items-center justify-center gap-4 w-full max-w-md"):
            ui.markdown("## Quên mật khẩu").classes("self-center")
            with ui.card().classes("gap-3 w-full p-6 shadow-md").style("border: 1px solid #ccc"):
                email = ui.input("Email đã đăng ký").classes("w-full")

                def submit():
                    res = api_forgot_password(email.value)
                    msg = res.get("message", "Đã gửi yêu cầu")
                    if "thành công" in msg.lower() or "✅" in msg:
                        notify_success(msg)
                    else:
                        notify_error(msg)

                ui.button("Gửi mã OTP", color="primary", on_click=submit).classes("w-full")
                ui.link("Quay lại đăng nhập", "/login")


@ui.page("/reset-password")
def reset_page():
    render_navbar()
    ui.markdown("## Đặt lại mật khẩu")
    token = ui.input("Mã OTP").classes("w-96")
    new_pass = ui.input("Mật khẩu mới", password=True).classes("w-96")
    confirm = ui.input("Xác nhận mật khẩu mới", password=True).classes("w-96")

    def submit():
        if new_pass.value != confirm.value:
            notify_error("Mật khẩu xác nhận không khớp")
            return
        res = api_reset_password(token.value, new_pass.value, confirm.value)
        if res.get("success"):
            notify_success(res.get("message", "Đặt lại mật khẩu thành công"))
            ui.navigate.to("/login")
        else:
            notify_error(res.get("message", "Đặt lại mật khẩu thất bại"))

    ui.button("Đặt lại mật khẩu", color="primary", on_click=submit)
    ui.link("Quay lại đăng nhập", "/login")


@ui.page("/documents")
def documents_page():
    if not require_auth():
        return
    render_navbar()
    ui.markdown("## Quản lý tài liệu")

    files_container = ui.column().classes("w-full gap-2")
    filename_dropdown = ui.select(options=[], label="Chọn file để xóa").props("clearable").classes("w-80")

    def refresh():
        result = api_get_files(session_state.session_id)
        files_container.clear()
        
        if not result.get("success") or result.get("total_files", 0) == 0:
            with files_container:
                ui.label("Chưa có file nào được upload.").classes("text-gray-500")
            filename_dropdown.options = []
            return
        
        files = result.get("files", [])
        filename_dropdown.options = [file["filename"] for file in files]
        
        with files_container:
            ui.markdown(f"### Tổng số: {result['total_files']} tài liệu, {result['total_chunks']} chunks")
            
            for file in files:
                with ui.card().classes("w-full p-4 gap-2"):
                    with ui.row().classes("items-center justify-between w-full"):
                        with ui.column().classes("gap-1"):
                            ui.label(f"📄 {file['filename']}").classes("text-lg font-semibold")
                            ui.label(f"{file['chunks']} chunks").classes("text-sm text-gray-600")
                        
                        with ui.row().classes("gap-2"):
                            # Nút view PDF
                            def view_pdf(fname=file['filename']):
                                view_result = api_view_file(fname, session_state.session_id)
                                if view_result.get("success"):
                                    url = view_result.get("url")
                                    # Mở PDF trong tab mới
                                    ui.run_javascript(f'window.open("{url}", "_blank")')
                                else:
                                    notify_error(view_result.get("message", "Không thể xem file"))
                            
                            ui.button("👁️ Xem PDF", on_click=lambda f=file['filename']: view_pdf(f)).props("outline")
                            
                            # Nút xóa
                            def delete_file(fname=file['filename']):
                                res = api_delete_file(fname, session_state.session_id)
                                if res.get("success"):
                                    notify_success(res.get("message", "Đã xóa file"))
                                    refresh()
                                else:
                                    notify_error(res.get("message", "Không thể xóa file"))
                            
                            ui.button("🗑️ Xóa", color="negative", on_click=lambda f=file['filename']: delete_file(f)).props("outline")

    ui.button("🔄 Làm mới danh sách", on_click=refresh).classes("mb-4")

    ui.markdown("### Upload mới")
    
    async def handle_documents_upload(e):
        """Xử lý upload trong trang documents"""
        try:
            result = await upload_temp_files(e)
            if result:  # Upload thành công
                # Đợi một chút để đảm bảo server đã xử lý xong và lưu vào DB
                await asyncio.sleep(1.5)
                # Retry refresh nếu cần
                for retry in range(3):
                    refresh()
                    await asyncio.sleep(0.5)
                logger.info("Refreshed documents page after upload")
        except Exception as ex:
            logger.error(f"Error in handle_documents_upload: {ex}", exc_info=True)
            notify_error(f"Lỗi khi xử lý upload: {ex}")
    
    ui.upload(
        multiple=True,
        label="Chọn hoặc kéo thả PDF",
        on_upload=handle_documents_upload,
    ).props('accept=".pdf"')

    def delete_selected():
        if not filename_dropdown.value:
            notify_error("Vui lòng chọn file cần xóa")
            return
        res = api_delete_file(filename_dropdown.value, session_state.session_id)
        if res.get("success"):
            notify_success(res.get("message", "Đã xóa file"))
            refresh()
        else:
            notify_error(res.get("message", "Không thể xóa file"))

    def clear_all():
        res = api_clear_all_files(session_state.session_id)
        if res.get("success"):
            notify_success(res.get("message", "Đã xóa toàn bộ tài liệu"))
            refresh()
        else:
            notify_error(res.get("message", "Không thể xóa tài liệu"))

    with ui.row().classes("gap-2 mt-4"):
        ui.button("🗑️ Xóa file đã chọn", color="negative", on_click=delete_selected)
        ui.button("🗑️ Xóa toàn bộ", color="negative", on_click=clear_all)

    refresh()


@ui.page("/chat")
def chat_page():
    if not require_auth():
        return
    # Trang chat đã được gộp vào trang '/', giữ route để tránh 404
    ui.label("Chat hiện đã gộp vào trang Trang chủ. Vui lòng quay lại trang /").classes("p-4")


@ui.page("/profile")
def profile_page():
    if not require_auth():
        return
    render_navbar()
    user = session_state.user or {}
    ui.markdown(
        f"""
        ### Thông tin tài khoản
        - Username: {user.get('username', '')}
        - Email: {user.get('email', '')}
        """
    )
    ui.button("Đăng xuất", color="negative", on_click=handle_logout)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    ui.run(host="0.0.0.0", port=port, reload=False, storage_secret=STORAGE_SECRET)
