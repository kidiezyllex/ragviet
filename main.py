import json
import os
import tempfile
from types import SimpleNamespace
import asyncio
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from nicegui import app, ui, context

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
)

load_dotenv()

STORAGE_SECRET = os.getenv("STORAGE_SECRET", "ragviet-dev-secret")
app.storage.secret = STORAGE_SECRET
ui.add_head_html("""
<style>
.nicegui-content{padding:0!important;}
.q-message-text strong { font-weight: bold; }
</style>
""", shared=True)

class SessionState:
    def __init__(self):
        self.session_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self.user: Optional[dict] = None
        self.selected_file: Optional[str] = None
        self.chat_session_id: Optional[str] = None

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
    # Nếu đã có session thì không cần restore
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
        return True
    clear_session_storage()
    return False


# -------------------------
# UI helpers
# -------------------------
def notify_success(msg: str):
    ui.notify(msg, type="positive")


def notify_error(msg: str):
    ui.notify(msg, type="negative")


def require_login() -> bool:
    if not session_state.is_logged_in:
        notify_error("Vui lòng đăng nhập để sử dụng tính năng này")
        return False
    return True


def require_auth():
    """Kiểm tra đăng nhập và redirect về /login nếu chưa đăng nhập."""
    # Khôi phục session từ storage
    restore_session_from_storage()
    
    # Kiểm tra nếu chưa đăng nhập
    if not session_state.is_logged_in:
        # Sử dụng JavaScript để redirect đảm bảo hoạt động
        ui.add_head_html(
            '<script>window.location.href = "/login";</script>',
            shared=False
        )
        ui.label("Đang chuyển đến trang đăng nhập...").classes("text-center p-4")
        return False
    return True


# -------------------------
# Data helpers
# -------------------------
def refresh_files_list() -> Tuple[str, List[str]]:
    result = api_get_files()
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


def upload_temp_files(upload_event) -> None:
    """Nhận UploadEvent (có thể 1 hoặc nhiều file) và gọi API upload."""
    if not require_login():
        return

    if hasattr(upload_event, "files") and upload_event.files:
        incoming = upload_event.files
    else:
        incoming = [upload_event]

    temp_wrappers: List[SimpleNamespace] = []
    try:
        for f in incoming:
            name = getattr(f, "name", "upload.pdf")
            content = getattr(f, "content", None)
            if content is None:
                continue
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[-1] or ".pdf")
            tmp.write(content if isinstance(content, (bytes, bytearray)) else content.read())
            tmp.flush()
            tmp.close()
            temp_wrappers.append(SimpleNamespace(path=tmp.name, name=tmp.name))

        if not temp_wrappers:
            notify_error("Không tìm thấy file để upload")
            return

        result = api_upload_files(temp_wrappers, session_state.session_id)
        if result.get("success"):
            notify_success(result.get("message", "Đã upload file thành công!"))
        else:
            notify_error(result.get("message", "Lỗi khi upload file"))
    except Exception as e:
        notify_error(f"Lỗi upload: {e}")
    finally:
        for t in temp_wrappers:
            try:
                os.remove(t.path)
            except Exception:
                pass


# -------------------------
# Auth handlers
# -------------------------
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
            new_text, new_files = refresh_files_list()
            if include_file_select and file_select is not None:
                file_select.options = ["Tất cả"] + new_files

        with ui.column().classes("gap-3"):
            ui.upload(
                label="Upload tài liệu PDF",
                multiple=True,
                on_upload=lambda e: (upload_temp_files(e), refresh_lists()),
            ).props("color=primary flat no-thumbnails").classes("w-full")

            if include_file_select:
                file_select = ui.select(
                    options=["Tất cả"] + file_names,
                    value="Tất cả",
                    label="Chọn tài liệu để chat",
                ).props("clearable dense").classes("w-full").style("font-size: 1rem")

            ui.input(
                placeholder="Tìm kiếm tài liệu",
            ).props("dense outlined").classes("w-full")

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


# -------------------------
# Pages
# -------------------------
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
        conv_label = ui.label("Trò chuyện với: Tất cả tài liệu").classes("text-xl font-semibold")
        if file_select:
            def update_conv_label(e):
                name = e.value or "Tất cả"
                if name == "Tất cả":
                    name = "Tất cả tài liệu"
                conv_label.set_text(f"Trò chuyện với: {name}")
                ui.notify(f"Đã chọn tài liệu: {name}", type="positive")
            file_select.on_value_change(update_conv_label)

        # Chat được tích hợp ngay trên trang chủ
        # Container với flex layout: chat log chiếm phần còn lại, input fixed ở bottom
        msg_input = None
        send_btn = None
        
        with ui.column().classes("w-full gap-2").style("display: flex; flex-direction: column; height: calc(100vh - 200px); min-height: 500px"):
            chat_log = ui.column().classes("gap-2 flex-1 overflow-auto border rounded p-3 bg-gray-50 w-full").style("display: flex; flex-direction: column; min-height: 0")
            
            def format_text(text: str) -> str:
                """Format text: **text** thành <strong>text</strong>"""
                import re
                # Thay **text** thành <strong>text</strong>
                formatted = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
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
                            ui.html(formatted_text)

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
            with ui.card().classes("gap-3 w-full p-6 shadow-md"):
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

    files_md = ui.markdown("")
    filename_dropdown = ui.select(options=[], label="Chọn file để xóa").props("clearable").classes("w-80")

    def refresh():
        text, names = refresh_files_list()
        files_md.set_content(text)
        filename_dropdown.options = names

    ui.button("Làm mới danh sách", on_click=refresh)

    ui.markdown("### Upload mới")
    ui.upload(
        multiple=True,
        label="Chọn hoặc kéo thả PDF",
        on_upload=lambda e: (upload_temp_files(e), refresh()),
    ).props('accept=".pdf"')

    def delete_selected():
        if not filename_dropdown.value:
            notify_error("Vui lòng chọn file cần xóa")
            return
        res = api_delete_file(filename_dropdown.value)
        if res.get("success"):
            notify_success(res.get("message", "Đã xóa file"))
            refresh()
        else:
            notify_error(res.get("message", "Không thể xóa file"))

    def clear_all():
        res = api_clear_all_files()
        if res.get("success"):
            notify_success(res.get("message", "Đã xóa toàn bộ tài liệu"))
            refresh()
        else:
            notify_error(res.get("message", "Không thể xóa tài liệu"))

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
