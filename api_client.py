import requests
import json
import os

# API Base URL
API_BASE_URL = os.getenv('DJANGO_API_URL', 'http://localhost:8000/api')


def get_auth_headers(session_id=None, access_token=None):
    """Tạo headers cho authentication"""
    headers = {
        'Content-Type': 'application/json',
    }
    token = access_token or session_id
    if token:
        headers['Authorization'] = f'Bearer {token}'
    return headers


def api_login(email, password):
    """Gọi API đăng nhập"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/login/',
            json={'email': email, 'password': password},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        try:
            data = response.json()
            if isinstance(data, dict):
                data.setdefault("status_code", response.status_code)
            return data
        except:
            return {
                "success": False, 
                "message": f"Lỗi từ server (status {response.status_code})",
                "status_code": response.status_code
            }
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Lỗi: Kết nối timeout. Vui lòng thử lại.", "status_code": None}
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Lỗi: Không thể kết nối đến server. Vui lòng kiểm tra Django backend đã chạy chưa.", "status_code": None}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}", "status_code": None}


def api_register(username, email, password, confirm_password):
    """Gọi API đăng ký"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/register/',
            json={
                'username': username,
                'email': email,
                'password': password,
                'confirm_password': confirm_password
            },
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_logout(session_id):
    """Gọi API đăng xuất"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/logout/',
            json={'session_id': session_id},
            headers=get_auth_headers(session_id)
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_forgot_password(email):
    """Gọi API quên mật khẩu"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/forgot-password/',
            json={'email': email},
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_reset_password(token, new_password, confirm_password):
    """Gọi API reset mật khẩu"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/reset-password/',
            json={
                'token': token,
                'new_password': new_password,
                'confirm_password': confirm_password
            },
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_verify_session(session_id):
    """Gọi API verify session"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/auth/verify-session/',
            json={'session_id': session_id},
            headers=get_auth_headers(session_id),
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            # Đảm bảo luôn có các field cần thiết
            if not isinstance(result, dict):
                return {"success": False, "valid": False}
            return result
        else:
            return {"success": False, "valid": False}
    except requests.exceptions.Timeout:
        return {"success": False, "valid": False, "message": "Lỗi: Kết nối timeout"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "valid": False, "message": "Lỗi: Không thể kết nối đến server"}
    except Exception as e:
        return {"success": False, "valid": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_chat_send(message, session_id, selected_file=None, chat_session_id=None):
    """Gọi API gửi tin nhắn chat"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/chat/send/',
            json={
                'message': message,
                'session_id': session_id,
                'selected_file': selected_file,
                'chat_session_id': chat_session_id
            },
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "response": "Lỗi khi gửi tin nhắn"}
    except Exception as e:
        return {"success": False, "response": f"Lỗi kết nối API: {str(e)}"}


def api_get_chat_sessions(session_id):
    """Gọi API lấy danh sách chat sessions"""
    try:
        response = requests.get(
            f'{API_BASE_URL}/chat/sessions/',
            params={'session_id': session_id},
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "sessions": []}
    except Exception as e:
        return {"success": False, "sessions": [], "message": f"Lỗi kết nối API: {str(e)}"}


def api_create_chat_session(session_id):
    """Gọi API tạo chat session mới"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/chat/sessions/create/',
            json={'session_id': session_id},
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": "Không thể tạo session mới"}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_get_chat_history(chat_session_id, session_id):
    """Gọi API lấy lịch sử chat của một session"""
    try:
        response = requests.get(
            f'{API_BASE_URL}/chat/history/{chat_session_id}/',
            params={'session_id': session_id},
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "messages": []}
    except Exception as e:
        return {"success": False, "messages": [], "message": f"Lỗi kết nối API: {str(e)}"}


def api_upload_files(files, session_id):
    """Gọi API upload files"""
    try:
        files_data = []
        file_handles = []
        
        for file in files:
            # Lấy path từ SimpleNamespace hoặc file object
            file_path = getattr(file, 'path', None) or getattr(file, 'name', None)
            # Lấy tên file gốc (nếu có)
            original_name = getattr(file, 'name', None)
            
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"File không tồn tại: {file_path}")
                continue
            
            try:
                file_handle = open(file_path, 'rb')
                file_handles.append(file_handle)
                
                # Sử dụng tên gốc nếu có, không thì dùng basename của path
                filename = original_name if original_name and original_name != file_path else os.path.basename(file_path)
                files_data.append(('files', (filename, file_handle, 'application/pdf')))
            except Exception as e:
                logger.error(f"Không thể mở file {file_path}: {e}")
                continue
        
        if not files_data:
            return {"success": False, "message": "Không tìm thấy file để upload"}
        
        response = requests.post(
            f'{API_BASE_URL}/files/upload/',
            files=files_data,
            data={'session_id': session_id},
            headers={'Authorization': f'Bearer {session_id}'}
        )
        
        # Đóng files
        for file_handle in file_handles:
            try:
                file_handle.close()
            except:
                pass
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json() if response.content else {"message": f"HTTP {response.status_code}"}
            return {"success": False, **error_msg}
    except Exception as e:
        logger.error(f"Lỗi khi upload files: {e}")
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_get_files(session_id=None):
    """Gọi API lấy danh sách files"""
    try:
        response = requests.get(
            f'{API_BASE_URL}/files/list/',
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "files": []}
    except Exception as e:
        return {"success": False, "files": [], "message": f"Lỗi kết nối API: {str(e)}"}


def api_delete_file(filename, session_id=None):
    """Gọi API xóa file"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/files/delete/',
            json={'filename': filename},
            headers=get_auth_headers(session_id)
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_clear_all_files(session_id=None):
    """Gọi API xóa toàn bộ files"""
    try:
        response = requests.post(
            f'{API_BASE_URL}/files/clear-all/',
            headers=get_auth_headers(session_id)
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_view_file(filename, session_id=None):
    """Gọi API lấy URL để xem file PDF"""
    try:
        response = requests.get(
            f'{API_BASE_URL}/files/view/{filename}/',
            headers=get_auth_headers(session_id)
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": "Không thể lấy URL của file"}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_admin_get_users(session_id):
    """Gọi API admin lấy danh sách tất cả users."""
    try:
        response = requests.get(
            f'{API_BASE_URL}/admin/users/',
            headers=get_auth_headers(session_id),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "users": [], "message": "Không thể lấy danh sách users"}
    except Exception as e:
        return {"success": False, "users": [], "message": f"Lỗi kết nối API: {str(e)}"}


def api_admin_get_files(session_id):
    """Gọi API admin lấy danh sách tất cả tài liệu của mọi user."""
    try:
        response = requests.get(
            f'{API_BASE_URL}/admin/files/',
            headers=get_auth_headers(session_id),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "files": [], "message": "Không thể lấy danh sách tài liệu"}
    except Exception as e:
        return {"success": False, "files": [], "message": f"Lỗi kết nối API: {str(e)}"}


def api_admin_set_user_active(user_id, is_active, session_id):
    """Gọi API admin cập nhật trạng thái active của user."""
    try:
        response = requests.post(
            f'{API_BASE_URL}/admin/users/status/',
            json={"user_id": user_id, "is_active": is_active},
            headers=get_auth_headers(session_id),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": "Không thể cập nhật trạng thái user"}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_admin_delete_user(user_id, session_id):
    """Gọi API admin xóa user và dữ liệu liên quan."""
    try:
        response = requests.post(
            f'{API_BASE_URL}/admin/users/delete/',
            json={"user_id": user_id},
            headers=get_auth_headers(session_id),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": "Không thể xóa user"}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


def api_admin_delete_file(user_id, filename, session_id):
    """Gọi API admin xóa file cụ thể của một user."""
    try:
        response = requests.post(
            f'{API_BASE_URL}/admin/files/delete/',
            json={"user_id": user_id, "filename": filename},
            headers=get_auth_headers(session_id),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": "Không thể xóa file"}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối API: {str(e)}"}


