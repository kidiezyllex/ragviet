"""
Module quản lý MongoDB database cho users và chat history
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime, timezone
import hashlib
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_timestamp(value: Optional[Any]) -> Optional[str]:
    """Đưa timestamp về ISO 8601 (UTC) với hậu tố Z."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return str(value)


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
        
        # Index cho chat_sessions collection
        self.db.chat_sessions.create_index("user_id")
        self.db.chat_sessions.create_index([("user_id", 1), ("created_at", -1)])
        self.db.chat_sessions.create_index("session_id", unique=True)
        
        # Index cho auth_sessions collection
        self.db.auth_sessions.create_index("session_id", unique=True)
        self.db.auth_sessions.create_index("user_id")
        
        # Index cho user_files collection
        self.db.user_files.create_index("user_id")
        self.db.user_files.create_index([("user_id", 1), ("filename", 1)], unique=True)
        self.db.user_files.create_index("filename")
        
        logger.info("Đã tạo indexes")
    
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
        
        # Generate 4-digit OTP
        token = str(secrets.randbelow(10000)).zfill(4)
        expires_at = datetime.utcnow().replace(hour=23, minute=59, second=59)
        
        self.db.users.update_one(
            {"email": email.lower()},
            {"$set": {
                "reset_token": token,
                "reset_token_expires": expires_at
            }}
        )
        
        logger.info(f"Đã tạo reset OTP cho: {email}")
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
    
    def save_chat_message(self, user_id: str, message: str, response: str, 
                         selected_file: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """
        Lưu lịch sử chat
        
        Args:
            user_id: ID của user
            message: Câu hỏi
            response: Câu trả lời
            selected_file: File được chọn (nếu có)
            session_id: ID của chat session (nếu có)
        """
        try:
            chat_entry = {
                "user_id": user_id,
                "message": message,
                "response": response,
                "selected_file": selected_file,
                "session_id": session_id,
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
                msg["timestamp"] = _format_timestamp(msg["timestamp"])
            
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
    
    def create_chat_session(self, user_id: str, title: str = "Cuộc trò chuyện mới") -> Optional[str]:
        """
        Tạo chat session mới
        
        Args:
            user_id: ID của user
            title: Tiêu đề của session
            
        Returns:
            session_id hoặc None
        """
        try:
            session_id = secrets.token_urlsafe(16)
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "title": title,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "message_count": 0
            }
            
            self.db.chat_sessions.insert_one(session)
            logger.info(f"Đã tạo chat session: {session_id} cho user: {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo chat session: {str(e)}")
            return None
    
    def get_chat_sessions(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Lấy danh sách chat sessions của user
        
        Args:
            user_id: ID của user
            limit: Số lượng sessions tối đa
            
        Returns:
            List các chat sessions
        """
        try:
            sessions = list(self.db.chat_sessions.find(
                {"user_id": user_id}
            ).sort("updated_at", -1).limit(limit))
            
            for session in sessions:
                session["_id"] = str(session["_id"])
                session["created_at"] = _format_timestamp(session["created_at"])
                session["updated_at"] = _format_timestamp(session["updated_at"])
            
            return sessions
        except Exception as e:
            logger.error(f"Lỗi khi lấy chat sessions: {str(e)}")
            return []
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Lấy tất cả messages trong một session
        
        Args:
            session_id: ID của session
            
        Returns:
            List các messages
        """
        try:
            messages = list(self.db.chat_history.find(
                {"session_id": session_id}
            ).sort("timestamp", 1))
            
            for msg in messages:
                msg["_id"] = str(msg["_id"])
                msg["timestamp"] = _format_timestamp(msg["timestamp"])
            
            return messages
        except Exception as e:
            logger.error(f"Lỗi khi lấy session messages: {str(e)}")
            return []
    
    def update_session(self, session_id: str, title: Optional[str] = None) -> bool:
        """
        Cập nhật thông tin session
        
        Args:
            session_id: ID của session
            title: Tiêu đề mới (nếu có)
        """
        try:
            update_data = {"updated_at": datetime.utcnow()}
            if title:
                update_data["title"] = title
            
            result = self.db.chat_sessions.update_one(
                {"session_id": session_id},
                {"$set": update_data, "$inc": {"message_count": 1}}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật session: {str(e)}")
            return False
    
    def delete_chat_session(self, session_id: str) -> bool:
        """
        Xóa chat session và tất cả messages trong đó
        
        Args:
            session_id: ID của session
        """
        try:
            self.db.chat_history.delete_many({"session_id": session_id})
            
            # Xóa session
            result = self.db.chat_sessions.delete_one({"session_id": session_id})
            
            logger.info(f"Đã xóa chat session: {session_id}")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Lỗi khi xóa chat session: {str(e)}")
            return False

    def get_last_message_of_session(self, session_id: str) -> Optional[Dict]:
        """
        Lấy message gần nhất của một session

        Args:
            session_id: ID của session
        """
        try:
            message = self.db.chat_history.find_one(
                {"session_id": session_id},
                sort=[("timestamp", -1)]
            )
            if message:
                message["_id"] = str(message["_id"])
                message["timestamp"] = _format_timestamp(message["timestamp"])
            return message
        except Exception as e:
            logger.error(f"Lỗi khi lấy message gần nhất của session: {str(e)}")
            return None
    
    def save_auth_session(self, session_id: str, user_id: str, user_data: Dict) -> bool:
        """
        Lưu auth session vào database
        
        Args:
            session_id: ID của session
            user_id: ID của user
            user_data: Thông tin user (username, email)
        """
        try:
            session_doc = {
                "session_id": session_id,
                "user_id": user_id,
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "created_at": datetime.utcnow(),
                "expires_at": None  # Sessions không hết hạn, có thể thêm logic hết hạn sau
            }
            
            # Update nếu đã tồn tại, insert nếu chưa có
            self.db.auth_sessions.update_one(
                {"session_id": session_id},
                {"$set": session_doc},
                upsert=True
            )
            logger.info(f"Đã lưu auth session: {session_id} cho user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu auth session: {str(e)}")
            return False
    
    def get_auth_session(self, session_id: str) -> Optional[Dict]:
        """
        Lấy auth session từ database
        
        Args:
            session_id: ID của session
            
        Returns:
            User data nếu session hợp lệ, None nếu không
        """
        try:
            session = self.db.auth_sessions.find_one({"session_id": session_id})
            if session:
                return {
                    "user_id": str(session["user_id"]),
                    "username": session.get("username"),
                    "email": session.get("email")
                }
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy auth session: {str(e)}")
            return None
    
    def delete_auth_session(self, session_id: str) -> bool:
        """
        Xóa auth session
        
        Args:
            session_id: ID của session
        """
        try:
            result = self.db.auth_sessions.delete_one({"session_id": session_id})
            logger.info(f"Đã xóa auth session: {session_id}")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Lỗi khi xóa auth session: {str(e)}")
            return False
    
    def load_all_auth_sessions(self) -> Dict[str, Dict]:
        """
        Tải tất cả auth sessions từ database (để restore khi server restart)
        
        Returns:
            Dict {session_id: user_data}
        """
        try:
            sessions = self.db.auth_sessions.find({})
            result = {}
            for session in sessions:
                session_id = session["session_id"]
                result[session_id] = {
                    "user_id": str(session["user_id"]),
                    "username": session.get("username"),
                    "email": session.get("email")
                }
            logger.info(f"Đã tải {len(result)} auth sessions từ database")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi tải auth sessions: {str(e)}")
            return {}

    def get_full_chat_history(self, user_id: str, limit_sessions: int = 50) -> Dict:
        """
        Trả về lịch sử chat đầy đủ của user theo format:
        {
            "user_id": ...,
            "chat_sessions": [
                {
                    "session_id": ...,
                    "title": <câu hỏi user mới nhất trong session>,
                    "created_at": ...,
                    "updated_at": ...,
                    "messages": [
                        {"role": "user" | "assistant", "content": ..., "timestamp": ...},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        try:
            sessions = self.get_chat_sessions(user_id, limit=limit_sessions)
            result_sessions: List[Dict] = []

            for session in sessions:
                session_id = session["session_id"]
                # Lấy toàn bộ messages của session (theo thứ tự thời gian)
                raw_messages = self.get_session_messages(session_id)

                messages: List[Dict] = []
                last_user_question: Optional[str] = None

                for entry in raw_messages:
                    ts = _format_timestamp(entry.get("timestamp"))
                    # Mỗi entry trong chat_history hiện tại là một cặp Q/A
                    user_msg = entry.get("message")
                    if user_msg:
                        messages.append({
                            "role": "user",
                            "content": user_msg,
                            "timestamp": ts,
                        })
                        last_user_question = user_msg
                    
                    assistant_msg = entry.get("response")
                    if assistant_msg:
                        messages.append({
                            "role": "assistant",
                            "content": assistant_msg,
                            "timestamp": ts,
                        })
                
                title = last_user_question or "Cuộc trò chuyện mới"
                last_message_ts = messages[-1]["timestamp"] if messages else session.get("updated_at")

                result_sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "created_at": _format_timestamp(session.get("created_at")),
                    "updated_at": last_message_ts or _format_timestamp(session.get("updated_at")),
                    "messages": messages,
                })

            return {
                "user_id": user_id,
                "chat_sessions": result_sessions,
            }
        except Exception as e:
            logger.error(f"Lỗi khi lấy full chat history cho user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "chat_sessions": [],
            }
    
    def save_user_file(self, user_id: str, filename: str, cloudinary_url: str, cloudinary_public_id: str, 
                       total_chunks: int = 0) -> bool:
        """
        Lưu thông tin file của user
        
        Args:
            user_id: ID của user
            filename: Tên file
            cloudinary_url: URL của file trên Cloudinary
            cloudinary_public_id: Public ID trên Cloudinary
            total_chunks: Số lượng chunks của file
            
        Returns:
            True nếu thành công, False nếu lỗi
        """
        try:
            from datetime import datetime
            file_doc = {
                "user_id": user_id,
                "filename": filename,
                "cloudinary_url": cloudinary_url,
                "cloudinary_public_id": cloudinary_public_id,
                "total_chunks": total_chunks,
                "uploaded_at": datetime.utcnow()
            }
            
            self.db.user_files.update_one(
                {"user_id": user_id, "filename": filename},
                {"$set": file_doc},
                upsert=True
            )
            logger.info(f"Đã lưu thông tin file {filename} cho user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu thông tin file: {str(e)}")
            return False
    
    def get_user_files(self, user_id: str) -> List[Dict]:
        """
        Lấy danh sách files của user
        
        Args:
            user_id: ID của user
            
        Returns:
            List các file documents
        """
        try:
            files = list(self.db.user_files.find({"user_id": user_id}).sort("uploaded_at", -1))
            for file in files:
                file["_id"] = str(file["_id"])
                file["uploaded_at"] = _format_timestamp(file.get("uploaded_at"))
            return files
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách files: {str(e)}")
            return []
    
    def get_user_file(self, user_id: str, filename: str) -> Optional[Dict]:
        """
        Lấy thông tin một file cụ thể của user
        
        Args:
            user_id: ID của user
            filename: Tên file
            
        Returns:
            File document hoặc None
        """
        try:
            file = self.db.user_files.find_one({"user_id": user_id, "filename": filename})
            if file:
                file["_id"] = str(file["_id"])
                file["uploaded_at"] = _format_timestamp(file.get("uploaded_at"))
            return file
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin file: {str(e)}")
            return None
    
    def delete_user_file(self, user_id: str, filename: str) -> bool:
        """
        Xóa thông tin file của user
        
        Args:
            user_id: ID của user
            filename: Tên file
            
        Returns:
            True nếu thành công, False nếu lỗi
        """
        try:
            result = self.db.user_files.delete_one({"user_id": user_id, "filename": filename})
            logger.info(f"Đã xóa thông tin file {filename} của user {user_id}")
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Lỗi khi xóa file: {str(e)}")
            return False
    
    def update_file_chunks(self, user_id: str, filename: str, chunks_count: int) -> bool:
        """
        Cập nhật số lượng chunks của file
        
        Args:
            user_id: ID của user
            filename: Tên file
            chunks_count: Số lượng chunks
            
        Returns:
            True nếu thành công
        """
        try:
            result = self.db.user_files.update_one(
                {"user_id": user_id, "filename": filename},
                {"$set": {"total_chunks": chunks_count}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật chunks: {str(e)}")
            return False

