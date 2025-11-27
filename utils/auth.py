"""
Module xử lý authentication và session management
"""
import secrets
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthManager:
    """Quản lý authentication và sessions"""
    
    def __init__(self, database):
        """
        Khởi tạo AuthManager
        
        Args:
            database: Instance của Database class
        """
        self.db = database
        self.sessions = {}  # {session_id: user_data}
    
    def register(self, username: str, email: str, password: str) -> Dict:
        """
        Đăng ký user mới
        
        Returns:
            Dict với status và message
        """
        if not username or not email or not password:
            return {"success": False, "message": "Vui lòng điền đầy đủ thông tin"}
        
        if len(password) < 6:
            return {"success": False, "message": "Mật khẩu phải có ít nhất 6 ký tự"}
        
        user = self.db.create_user(username, email, password)
        if user:
            return {"success": True, "message": "Đăng ký thành công! Vui lòng đăng nhập."}
        else:
            return {"success": False, "message": "Email hoặc username đã tồn tại"}
    
    def login(self, email: str, password: str) -> Dict:
        """
        Đăng nhập
        
        Returns:
            Dict với status, message và session_id (nếu thành công)
        """
        if not email or not password:
            return {"success": False, "message": "Vui lòng nhập email và mật khẩu"}
        
        user = self.db.verify_password(email, password)
        if user:
            session_id = secrets.token_urlsafe(32)
            user_data = {
                "user_id": str(user["_id"]),
                "username": user["username"],
                "email": user["email"]
            }
            self.sessions[session_id] = user_data
            logger.info(f"User đăng nhập: {user['email']}")
            return {
                "success": True,
                "message": "Đăng nhập thành công!",
                "session_id": session_id,
                "user": user_data
            }
        else:
            return {"success": False, "message": "Email hoặc mật khẩu không đúng"}
    
    def logout(self, session_id: str) -> bool:
        """Đăng xuất"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_user_from_session(self, session_id: Optional[str]) -> Optional[Dict]:
        """Lấy thông tin user từ session"""
        if not session_id:
            return None
        return self.sessions.get(session_id)
    
    def is_authenticated(self, session_id: Optional[str]) -> bool:
        """Kiểm tra user đã đăng nhập chưa"""
        return session_id is not None and session_id in self.sessions
    
    def request_password_reset(self, email: str) -> Dict:
        """
        Yêu cầu reset password
        
        Returns:
            Dict với status và message (không trả về token thực để bảo mật)
        """
        if not email:
            return {"success": False, "message": "Vui lòng nhập email"}
        
        token = self.db.create_reset_token(email)
        if token:
            # Trong thực tế, nên gửi email với token
            # Ở đây chỉ trả về message
            return {
                "success": True,
                "message": f"Token reset đã được tạo. Token của bạn: {token}\n(Lưu ý: Trong production, token sẽ được gửi qua email)"
            }
        else:
            return {"success": False, "message": "Email không tồn tại trong hệ thống"}
    
    def reset_password(self, token: str, new_password: str) -> Dict:
        """
        Reset password với token
        
        Returns:
            Dict với status và message
        """
        if not token or not new_password:
            return {"success": False, "message": "Vui lòng điền đầy đủ thông tin"}
        
        if len(new_password) < 6:
            return {"success": False, "message": "Mật khẩu phải có ít nhất 6 ký tự"}
        
        success = self.db.reset_password(token, new_password)
        if success:
            return {"success": True, "message": "Đặt lại mật khẩu thành công! Vui lòng đăng nhập."}
        else:
            return {"success": False, "message": "Token không hợp lệ hoặc đã hết hạn"}

