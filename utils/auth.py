"""
Module xử lý authentication và session management
"""
import secrets
from typing import Optional, Dict
import logging
import resend
import os

# Cấu hình Resend API Key
resend.api_key = "re_JQk4fB5d_DztySKf3tqBCvx4mEPWp1Sjr"

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
            # Kiểm tra API key
            api_key = "re_JQk4fB5d_DztySKf3tqBCvx4mEPWp1Sjr"
            if not api_key:
                return {
                    "success": False,
                    "message": "Hệ thống email chưa được cấu hình. Vui lòng liên hệ quản trị viên."
                }
            
            if not api_key.startswith("re_"):
                return {
                    "success": False,
                    "message": "API key không hợp lệ. Vui lòng kiểm tra cấu hình."
                }
            
            try:
                # Gửi email với Resend
                params = {
                    "from": "RagVietDocument@gmail.com",
                    "to": [email],
                    "subject": "Mã xác thực đặt lại mật khẩu - RAGViet",
                    "html": f"""
                    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                        <h2 style="color: #333;">Đặt lại mật khẩu</h2>
                        <p>Bạn đã yêu cầu đặt lại mật khẩu cho tài khoản RAGViet.</p>
                        <p>Mã OTP của bạn là:</p>
                        <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 5px; margin: 20px 0;">
                            {token}
                        </div>
                        <p style="color: #666; font-size: 14px;">Mã này sẽ hết hạn sau 15 phút.</p>
                        <p style="color: #666; font-size: 14px;">Nếu bạn không yêu cầu đặt lại mật khẩu, vui lòng bỏ qua email này.</p>
                    </div>
                    """
                }
                
                logger.info(f"Đang gửi email đến {email} với API key: {api_key[:10]}...")
                email_resp = resend.Emails.send(params)
                logger.info(f"✅ Đã gửi email reset password đến {email}. Response: {email_resp}")
                
                return {
                    "success": True,
                    "message": "Mã OTP đã được gửi đến email của bạn. Vui lòng kiểm tra."
                }
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Lỗi khi gửi email đến {email}: {error_msg}")
                
                # Xử lý các lỗi cụ thể
                if "API key is invalid" in error_msg or "invalid" in error_msg.lower():
                    return {
                        "success": False,
                        "message": "API key không hợp lệ. Vui lòng kiểm tra lại cấu hình tại https://resend.com/api-keys"
                    }
                elif "domain" in error_msg.lower():
                    return {
                        "success": False,
                        "message": "Domain email chưa được xác thực. Vui lòng liên hệ quản trị viên."
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Không thể gửi email: {error_msg}"
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

