"""
Module quản lý MongoDB database cho users và chat history
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from typing import Optional, Dict, List
import logging
from datetime import datetime
import hashlib
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """Quản lý kết nối MongoDB và các operations"""
    
    def __init__(self):
        """Khởi tạo kết nối MongoDB"""
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGODB_DB_NAME", "ragviet")
        
        if not self.mongo_uri:
            logger.error("MONGO_URI không được cấu hình trong .env")
            raise ValueError("MONGO_URI không được cấu hình")
        
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Đã kết nối MongoDB: {self.db_name}")
            
            # Tạo indexes
            self._create_indexes()
        except ConnectionFailure as e:
            logger.error(f"Không thể kết nối MongoDB: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Tạo indexes cho collections"""
        # Index cho users collection
        self.db.users.create_index("email", unique=True)
        self.db.users.create_index("username", unique=True)
        
        # Index cho chat_history collection
        self.db.chat_history.create_index("user_id")
        self.db.chat_history.create_index([("user_id", 1), ("timestamp", -1)])
        
        logger.info("Đã tạo indexes")
    
    # ========== USER OPERATIONS ==========
    
    def create_user(self, username: str, email: str, password: str) -> Optional[Dict]:
        """
        Tạo user mới
        
        Args:
            username: Tên đăng nhập
            email: Email
            password: Mật khẩu (sẽ được hash)
            
        Returns:
            User document hoặc None nếu lỗi
        """
        try:
            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user = {
                "username": username,
                "email": email.lower(),
                "password_hash": password_hash,
                "created_at": datetime.utcnow(),
                "is_active": True
            }
            
            result = self.db.users.insert_one(user)
            user["_id"] = result.inserted_id
            user.pop("password_hash")  # Không trả về password hash
            logger.info(f"Đã tạo user: {username}")
            return user
        except DuplicateKeyError as e:
            logger.warning(f"User đã tồn tại: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tạo user: {str(e)}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Lấy user theo email"""
        try:
            user = self.db.users.find_one({"email": email.lower()})
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy user: {str(e)}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Lấy user theo username"""
        try:
            user = self.db.users.find_one({"username": username})
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy user: {str(e)}")
            return None
    
    def verify_password(self, email: str, password: str) -> Optional[Dict]:
        """
        Xác thực mật khẩu
        
        Returns:
            User document nếu đúng, None nếu sai
        """
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user.get("password_hash") == password_hash:
            user.pop("password_hash", None)
            return user
        return None
    
    def create_reset_token(self, email: str) -> Optional[str]:
        """
        Tạo token reset password
        
        Returns:
            Reset token hoặc None
        """
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow().replace(hour=23, minute=59, second=59)
        
        self.db.users.update_one(
            {"email": email.lower()},
            {"$set": {
                "reset_token": token,
                "reset_token_expires": expires_at
            }}
        )
        
        logger.info(f"Đã tạo reset token cho: {email}")
        return token
    
    def verify_reset_token(self, token: str) -> Optional[Dict]:
        """Xác thực reset token"""
        try:
            user = self.db.users.find_one({
                "reset_token": token,
                "reset_token_expires": {"$gt": datetime.utcnow()}
            })
            if user:
                user["_id"] = str(user["_id"])
                return user
            return None
        except Exception as e:
            logger.error(f"Lỗi khi xác thực token: {str(e)}")
            return None
    
    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset mật khẩu"""
        user = self.verify_reset_token(token)
        if not user:
            return False
        
        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
        
        result = self.db.users.update_one(
            {"reset_token": token},
            {"$set": {
                "password_hash": password_hash
            },
             "$unset": {
                "reset_token": "",
                "reset_token_expires": ""
            }}
        )
        
        if result.modified_count > 0:
            logger.info(f"Đã reset password cho user: {user.get('email')}")
            return True
        return False
    
    # ========== CHAT HISTORY OPERATIONS ==========
    
    def save_chat_message(self, user_id: str, message: str, response: str, 
                         selected_file: Optional[str] = None) -> bool:
        """
        Lưu lịch sử chat
        
        Args:
            user_id: ID của user
            message: Câu hỏi
            response: Câu trả lời
            selected_file: File được chọn (nếu có)
        """
        try:
            chat_entry = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "selected_file": selected_file,
                "timestamp": datetime.utcnow()
            }
            
            self.db.chat_history.insert_one(chat_entry)
            logger.info(f"Đã lưu chat message cho user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu chat history: {str(e)}")
            return False
    
    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Lấy lịch sử chat của user
        
        Args:
            user_id: ID của user
            limit: Số lượng message tối đa
            
        Returns:
            List các chat messages
        """
        try:
            messages = list(self.db.chat_history.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit))
            
            for msg in messages:
                msg["_id"] = str(msg["_id"])
                msg["timestamp"] = msg["timestamp"].isoformat()
            
            return list(reversed(messages))  # Trả về theo thứ tự cũ -> mới
        except Exception as e:
            logger.error(f"Lỗi khi lấy chat history: {str(e)}")
            return []
    
    def clear_chat_history(self, user_id: str) -> bool:
        """Xóa lịch sử chat của user"""
        try:
            result = self.db.chat_history.delete_many({"user_id": user_id})
            logger.info(f"Đã xóa {result.deleted_count} messages của user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa chat history: {str(e)}")
            return False

