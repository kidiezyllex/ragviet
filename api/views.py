"""
Django REST Framework API Views
"""
import os
import json
import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.conf import settings

from utils.database import Database
from utils.auth import AuthManager
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.reranker import Reranker
from utils.natural_language import is_natural_question, get_natural_response

logger = logging.getLogger(__name__)

try:
    database = Database()
    auth_manager = AuthManager(database)
    pdf_processor = PDFProcessor(chunk_size=400, overlap=100)
    vector_store = VectorStore()
    reranker = Reranker()
    logger.info("Đã khởi tạo database và các services")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo services: {str(e)}")
    database = None
    auth_manager = None
    pdf_processor = None
    vector_store = None
    reranker = None


def configure_cloudinary():
    """
    Cấu hình Cloudinary và kiểm tra credentials
    Returns: (success: bool, error_message: str)
    """
    if not settings.CLOUDINARY_CLOUD_NAME or not settings.CLOUDINARY_API_KEY or not settings.CLOUDINARY_API_SECRET:
        return False, "Cloudinary credentials chưa được cấu hình. Vui lòng thêm CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, và CLOUDINARY_API_SECRET vào file .env"
    
    try:
        import cloudinary
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET
        )
        return True, None
    except Exception as e:
        return False, f"Lỗi khi cấu hình Cloudinary: {str(e)}"


def get_llm_client():
    """Khởi tạo LLM client (Groq)"""
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info("Đã kết nối Groq API")
            return client, "groq", "llama-3.3-70b-versatile"
        except Exception as e:
            logger.warning(f"Không thể kết nối Groq: {str(e)}")
    
    logger.warning("Chưa cấu hình API key cho LLM")
    return None, None, None


llm_client, llm_provider, llm_model = get_llm_client()


def generate_answer(query: str, context_chunks: list, selected_file: str = None) -> str:
    """Sinh câu trả lời từ LLM dựa trên context"""
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
                    max_tokens=4096
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
                        (answer_clean.endswith(':') and len(answer_clean.split('\n')) < 3)
                    ]
                    
                    if any(incomplete_patterns):
                        logger.warning("Phát hiện câu trả lời có thể bị cắt cụt, thử lại với max_tokens cao hơn...")
                        try:
                            response = llm_client.chat.completions.create(
                                model=llm_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.1,
                                max_tokens=8192
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

COOKIE_NAME = "ragviet_session"
COOKIE_MAX_AGE = 7 * 24 * 3600  # 7 days


def set_auth_cookie(response: Response, session_id: str):
    """
    Đặt HTTP-only cookie cho session id.
    """
    if not session_id:
        return response
    response.set_cookie(
        COOKIE_NAME,
        session_id,
        httponly=True,
        secure=not settings.DEBUG,  # bật secure khi chạy production/https
        samesite="Lax",
        max_age=COOKIE_MAX_AGE,
        path="/",
    )
    return response


def clear_auth_cookie(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


def get_session_id_from_request(request):
    """
    Lấy session_id từ nhiều nguồn: body, query, cookie, Authorization header.
    Thứ tự ưu tiên: body/query -> cookie -> header.
    """
    token = None
    if hasattr(request, "data"):
        token = request.data.get("session_id")
    if not token and hasattr(request, "query_params"):
        token = request.query_params.get("session_id")
    if not token:
        token = request.COOKIES.get(COOKIE_NAME)
    if not token:
        token = request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
    return token


class LoginView(APIView):
    """API endpoint cho đăng nhập"""
    
    def post(self, request):
        if not auth_manager:
            return Response(
                {"success": False, "message": "Hệ thống database chưa được khởi tạo"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        email = request.data.get('email', '').strip()
        password = request.data.get('password', '').strip()
        
        if not email:
            return Response(
                {"success": False, "message": "Vui lòng nhập email của bạn"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if "@" not in email or "." not in email.split("@")[-1]:
            return Response(
                {"success": False, "message": "Email không hợp lệ"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not password:
            return Response(
                {"success": False, "message": "Vui lòng nhập mật khẩu của bạn"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if len(password) < 6:
            return Response(
                {"success": False, "message": "Mật khẩu phải có ít nhất 6 ký tự"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = auth_manager.login(email, password)
        if result["success"]:
            chat_session_id = None
            if database:
                chat_session_id = database.create_chat_session(result["user"]["user_id"])

            resp = Response({
                "success": True,
                "message": result['message'],
                "session_id": result["session_id"],
                "access_token": result.get("access_token", result["session_id"]),
                "user": result["user"],
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
            return set_auth_cookie(resp, result["session_id"])
        else:
            return Response(
                {"success": False, "message": result['message']},
                status=status.HTTP_401_UNAUTHORIZED
            )


class RegisterView(APIView):
    """API endpoint cho đăng ký"""
    
    def post(self, request):
        if not auth_manager:
            return Response(
                {"success": False, "message": "Hệ thống database chưa được khởi tạo"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        username = request.data.get('username', '').strip()
        email = request.data.get('email', '').strip()
        password = request.data.get('password', '').strip()
        confirm_password = request.data.get('confirm_password', '').strip()
        
        if password != confirm_password:
            return Response(
                {"success": False, "message": "Mật khẩu xác nhận không khớp"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = auth_manager.register(username, email, password)
        if result["success"]:
            login_result = auth_manager.login(email, password)
            if login_result["success"]:
                chat_session_id = None
                if database:
                    chat_session_id = database.create_chat_session(login_result["user"]["user_id"])

                resp = Response({
                    "success": True,
                    "message": result['message'] + " Đang tự động đăng nhập...",
                    "session_id": login_result["session_id"],
                    "access_token": login_result.get("access_token", login_result["session_id"]),
                    "user": login_result["user"],
                    "chat_session_id": chat_session_id
                }, status=status.HTTP_200_OK)
                return set_auth_cookie(resp, login_result["session_id"])
            else:
                return Response({
                    "success": True,
                    "message": result['message'],
                    "requires_login": True
                }, status=status.HTTP_200_OK)
        else:
            return Response(
                {"success": False, "message": result['message']},
                status=status.HTTP_400_BAD_REQUEST
            )


class LogoutView(APIView):
    """API endpoint cho đăng xuất"""
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if session_id and auth_manager:
            auth_manager.logout(session_id)

        resp = Response({
            "success": True,
            "message": "Đã đăng xuất"
        }, status=status.HTTP_200_OK)
        return clear_auth_cookie(resp)


class ForgotPasswordView(APIView):
    """API endpoint cho quên mật khẩu"""
    
    def post(self, request):
        if not auth_manager:
            return Response(
                {"success": False, "message": "Hệ thống database chưa được khởi tạo"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        email = request.data.get('email', '').strip()
        result = auth_manager.request_password_reset(email)
        
        if result.get("success"):
            return Response({
                "success": True,
                "message": result["message"]
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {"success": False, "message": result["message"]},
                status=status.HTTP_400_BAD_REQUEST
            )


class ResetPasswordView(APIView):
    """API endpoint cho reset mật khẩu"""
    
    def post(self, request):
        if not auth_manager:
            return Response(
                {"success": False, "message": "Hệ thống database chưa được khởi tạo"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        token = request.data.get('token', '').strip()
        new_password = request.data.get('new_password', '').strip()
        confirm_password = request.data.get('confirm_password', '').strip()
        
        if new_password != confirm_password:
            return Response(
                {"success": False, "message": "Mật khẩu xác nhận không khớp"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result = auth_manager.reset_password(token, new_password)
        
        if result["success"]:
            return Response({
                "success": True,
                "message": result['message']
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {"success": False, "message": result['message']},
                status=status.HTTP_400_BAD_REQUEST
            )


class VerifySessionView(APIView):
    """API endpoint để verify session"""
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response({
                "success": False,
                "valid": False
            }, status=status.HTTP_200_OK)
        
        user = auth_manager.get_user_from_session(session_id)
        if user:
            chat_session_id = None
            if database:
                chat_session_id = database.create_chat_session(user["user_id"])
            
            resp = Response({
                "success": True,
                "valid": True,
                "user": user,
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
            return set_auth_cookie(resp, session_id)
        else:
            return Response({
                "success": False,
                "valid": False
            }, status=status.HTTP_200_OK)

class ChatSendView(APIView):
    """API endpoint để gửi tin nhắn chat"""
    
    def post(self, request):
        message = request.data.get('message', '').strip()
        session_id = get_session_id_from_request(request)
        selected_file = request.data.get('selected_file')
        chat_session_id = request.data.get('chat_session_id')
        
        if not message:
            return Response(
                {"success": False, "message": "Vui lòng nhập câu hỏi"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        natural_response = get_natural_response(message)
        if natural_response:
            if session_id and database and auth_manager:
                user = auth_manager.get_user_from_session(session_id)
                if user:
                    if not chat_session_id:
                        chat_session_id = database.create_chat_session(user["user_id"])
                    database.save_chat_message(user["user_id"], message, natural_response, selected_file, chat_session_id)
                    if chat_session_id:
                        database.update_session(chat_session_id, title=message)
            
            return Response({
                "success": True,
                "response": natural_response,
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
        
        # Lấy user_id nếu có session
        user = None
        user_id = None
        if session_id and auth_manager:
            user = auth_manager.get_user_from_session(session_id)
            if user:
                user_id = user["user_id"]
        
        # Lấy stats theo user_id
        stats = vector_store.get_stats(user_id=user_id)
        if stats["total_chunks"] == 0:
            return Response({
                "success": True,
                "response": "⚠️ Chưa có tài liệu nào được upload. Vui lòng upload file PDF trước khi đặt câu hỏi."
            }, status=status.HTTP_200_OK)
        
        try:
            logger.info(f"Đang tìm kiếm câu trả lời cho: {message} (file: {selected_file}, user: {user_id})")
            
            search_results = vector_store.search(message, top_k=30, filename=selected_file, user_id=user_id)
            
            if not search_results:
                response = "Không tìm thấy thông tin liên quan trong các tài liệu đã upload."
                if selected_file:
                    response += f" (đã tìm trong file: {selected_file})"
                
                if session_id and database and auth_manager:
                    user = auth_manager.get_user_from_session(session_id)
                    if user:
                        if not chat_session_id:
                            chat_session_id = database.create_chat_session(user["user_id"])
                        database.save_chat_message(user["user_id"], message, response, selected_file, chat_session_id)
                        if chat_session_id:
                            database.update_session(chat_session_id, title=message)
                
                return Response({
                    "success": True,
                    "response": response,
                    "chat_session_id": chat_session_id
                }, status=status.HTTP_200_OK)
            
            expanded_results = vector_store.get_adjacent_chunks(search_results, page_range=2)
            reranked_results = reranker.rerank(message, expanded_results, top_k=15)
            answer = generate_answer(message, reranked_results, selected_file)
            
            if session_id and database and auth_manager:
                user = auth_manager.get_user_from_session(session_id)
                if user:
                    if not chat_session_id:
                        chat_session_id = database.create_chat_session(user["user_id"])
                    database.save_chat_message(user["user_id"], message, answer, selected_file, chat_session_id)
                    if chat_session_id:
                        database.update_session(chat_session_id, title=message)
            
            return Response({
                "success": True,
                "response": answer,
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return Response(
                {"success": False, "message": f"Lỗi: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatSessionsView(APIView):
    """API endpoint để lấy danh sách chat sessions"""
    
    def get(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Không thể lấy danh sách chat"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        sessions = database.get_chat_sessions(user["user_id"])
        if not sessions:
            return Response({
                "success": True,
                "sessions": []
            }, status=status.HTTP_200_OK)
        
        from datetime import datetime, timedelta
        
        result_sessions = []
        for session in sessions:
            utc_time = datetime.fromisoformat(session["updated_at"].replace("Z", "+00:00"))
            vn_time = utc_time + timedelta(hours=7)
            updated_time = vn_time.strftime("%d/%m/%Y %H:%M")
            
            last_message = database.get_last_message_of_session(session["session_id"])
            last_question = last_message["message"] if last_message and last_message.get("message") else "Chưa có câu hỏi nào"
            
            result_sessions.append({
                "session_id": session["session_id"],
                "title": session.get("title", "Cuộc trò chuyện mới"),
                "updated_at": updated_time,
                "last_question": last_question[:90] + "..." if len(last_question) > 90 else last_question
            })
        
        return Response({
            "success": True,
            "sessions": result_sessions
        }, status=status.HTTP_200_OK)


class CreateChatSessionView(APIView):
    """API endpoint để tạo chat session mới"""
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Không thể tạo session mới"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        chat_session_id = database.create_chat_session(user["user_id"])
        if chat_session_id:
            return Response({
                "success": True,
                "chat_session_id": chat_session_id,
                "message": "Đã tạo cuộc trò chuyện mới!"
            }, status=status.HTTP_200_OK)
        else:
            return Response(
                {"success": False, "message": "Không thể tạo cuộc trò chuyện mới"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatHistoryView(APIView):
    """API endpoint để lấy lịch sử chat của một session"""
    
    def get(self, request, session_id):
        auth_session_id = get_session_id_from_request(request)
        
        if not auth_session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(auth_session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Không thể lấy lịch sử chat"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        messages = database.get_session_messages(session_id)
        return Response({
            "success": True,
            "messages": messages
        }, status=status.HTTP_200_OK)


class FileUploadView(APIView):
    """API endpoint để upload file PDF lên Cloudinary"""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập để upload file"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user:
            return Response(
                {"success": False, "message": "Session không hợp lệ"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user_id = user["user_id"]
        files = request.FILES.getlist('files')
        if not files:
            return Response(
                {"success": False, "message": "Vui lòng chọn ít nhất một file PDF"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            import cloudinary
            import cloudinary.uploader
            import tempfile
            
            # Cấu hình và kiểm tra Cloudinary credentials
            success, error_msg = configure_cloudinary()
            if not success:
                return Response(
                    {"success": False, "message": error_msg},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            pdf_paths = []
            uploaded_files_info = []
            
            # Upload từng file lên Cloudinary và xử lý
            for file in files:
                if not file.name.endswith('.pdf'):
                    return Response(
                        {"success": False, "message": f"File {file.name} không phải là PDF"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                filename = file.name
                # Tạo public_id với user_id để tránh trùng lặp
                public_id = f"ragviet/{user_id}/{filename.replace('.pdf', '')}"
                
                # Lưu file tạm để xử lý
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    for chunk in file.chunks():
                        tmp_file.write(chunk)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    tmp_path = tmp_file.name
                
                # Kiểm tra file đã được tạo và có nội dung
                if not os.path.exists(tmp_path):
                    return Response(
                        {"success": False, "message": f"Không thể tạo file tạm cho {filename}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                file_size = os.path.getsize(tmp_path)
                if file_size == 0:
                    return Response(
                        {"success": False, "message": f"File {filename} rỗng hoặc bị hỏng"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                logger.info(f"Đã tạo file tạm {tmp_path} cho {filename} (size: {file_size} bytes)")
                
                # Upload lên Cloudinary
                try:
                    upload_result = cloudinary.uploader.upload(
                        tmp_path,
                        resource_type="raw",
                        public_id=public_id,
                        folder=f"ragviet/{user_id}",
                        use_filename=True,
                        unique_filename=False
                    )
                    cloudinary_url = upload_result.get('secure_url') or upload_result.get('url')
                    cloudinary_public_id = upload_result.get('public_id')
                    
                    logger.info(f"Đã upload {filename} lên Cloudinary: {cloudinary_url}")
                    
                    # Lưu file tạm để xử lý PDF
                    pdf_paths.append(tmp_path)
                    uploaded_files_info.append({
                        'filename': filename,
                        'cloudinary_url': cloudinary_url,
                        'cloudinary_public_id': cloudinary_public_id,
                        'tmp_path': tmp_path
                    })
                except Exception as upload_error:
                    logger.error(f"Lỗi khi upload lên Cloudinary: {str(upload_error)}")
                    # Xóa file tạm nếu upload thất bại
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                    return Response(
                        {"success": False, "message": f"Lỗi khi upload file {filename} lên Cloudinary: {str(upload_error)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            
            all_chunks = []
            pages_info = {}
            
            user_files = database.get_user_files(user_id)
            valid_filenames = [f['filename'] for f in user_files] if user_files else []
            for file_info in uploaded_files_info:
                valid_filenames.append(file_info['filename'])
            
            logger.info(f"Đang dọn dẹp các chunks cũ (file tạm) của user {user_id}")
            vector_store.delete_temp_files_by_user(user_id, valid_filenames=valid_filenames)
            
            for file_info in uploaded_files_info:
                filename = file_info['filename']
                tmp_path = file_info['tmp_path']
                
                logger.info(f"Đang xóa chunks cũ của file {filename} (nếu có) cho user {user_id}")
                vector_store.delete_by_filename(filename, user_id=user_id)
                
                # Kiểm tra file tạm có tồn tại và có thể đọc được không
                if not os.path.exists(tmp_path):
                    logger.error(f"File tạm không tồn tại: {tmp_path}")
                    pages_info[filename] = 0
                    continue
                
                # Kiểm tra kích thước file
                file_size = os.path.getsize(tmp_path)
                if file_size == 0:
                    logger.error(f"File tạm rỗng: {tmp_path}")
                    pages_info[filename] = 0
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
                    continue
                
                logger.info(f"Đang xử lý PDF {filename} từ {tmp_path} (size: {file_size} bytes)")
                
                try:
                    chunks, pages_dict = pdf_processor.process_multiple_pdfs([tmp_path], filenames=[filename])
                    if chunks and len(chunks) > 0:
                        for chunk in chunks:
                            chunk['metadata']['user_id'] = user_id
                        all_chunks.extend(chunks)
                        if pages_dict:
                            page_count = list(pages_dict.values())[0] if pages_dict else 0
                            pages_info[filename] = page_count
                            logger.info(f"Đã xử lý {filename}: {len(chunks)} chunks, {page_count} trang")
                        else:
                            pages_info[filename] = 0
                            logger.warning(f"Không có pages_dict cho {filename}")
                    else:
                        logger.warning(f"Không tạo được chunks từ {filename} (0 chunks)")
                        pages_info[filename] = 0
                    
                    # Xóa file tạm sau khi xử lý
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.warning(f"Không thể xóa file tạm {tmp_path}: {e}")
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý PDF {filename}: {str(e)}", exc_info=True)
                    pages_info[filename] = 0
                    # Xóa file tạm
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
            
            for file_info in uploaded_files_info:
                filename = file_info['filename']
                chunks_count = sum(1 for chunk in all_chunks if chunk['metadata']['filename'] == filename)
                save_result = database.save_user_file(
                    user_id=user_id,
                    filename=filename,
                    cloudinary_url=file_info['cloudinary_url'],
                    cloudinary_public_id=file_info['cloudinary_public_id'],
                    total_chunks=chunks_count
                )
                if save_result:
                    logger.info(f"Đã lưu thông tin file {filename} vào database cho user {user_id} ({chunks_count} chunks)")
                else:
                    logger.warning(f"Không thể lưu thông tin file {filename} vào database")
            
            # Thêm vào vector store với user_id (chỉ các file có chunks)
            if all_chunks:
                vector_store.add_documents(all_chunks)
            
            # Kiểm tra và cảnh báo về các file không có text
            files_without_text = [f['filename'] for f in uploaded_files_info if pages_info.get(f['filename'], 0) == 0]
            files_with_text = [f['filename'] for f in uploaded_files_info if pages_info.get(f['filename'], 0) > 0]
            
            # Nếu tất cả files đều không có text, trả về cảnh báo nhưng vẫn thành công (file đã upload lên Cloudinary)
            if not all_chunks and uploaded_files_info:
                warning_msg = f"Đã upload {len(uploaded_files_info)} file(s) lên Cloudinary. "
                if files_without_text:
                    warning_msg += f"Lưu ý: Không thể trích xuất văn bản từ {', '.join(files_without_text)}. "
                    warning_msg += "Có thể file PDF chỉ có hình ảnh, bị mã hóa, hoặc không có nội dung text. "
                    warning_msg += "Bạn vẫn có thể xem file trên Cloudinary nhưng không thể chat với nội dung của file này."
                
                return Response({
                    "success": True,
                    "message": warning_msg,
                    "files_processed": len(uploaded_files_info),
                    "files_with_text": len(files_with_text),
                    "files_without_text": len(files_without_text),
                    "warning": True
                }, status=status.HTTP_200_OK)
            
            total_pages = sum(pages_info.values())
            files_summary = "\n".join([f"  • {name}: {pages} trang" for name, pages in pages_info.items()])
            
            logger.info(f"Upload thành công: {len(uploaded_files_info)} files, {total_pages} pages, {len(all_chunks)} chunks")
            
            return Response({
                "success": True,
                "message": f"Đã xử lý xong {len(uploaded_files_info)} tài liệu, tổng cộng {total_pages} trang",
                "files_processed": len(uploaded_files_info),
                "total_pages": total_pages,
                "files_detail": files_summary
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý PDF: {str(e)}")
            return Response(
                {"success": False, "message": f"Lỗi: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class FileListView(APIView):
    """API endpoint để lấy danh sách file đã upload của user"""
    
    def get(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Session không hợp lệ"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user_id = user["user_id"]
        
        # Lấy files từ database (có Cloudinary URL)
        user_files = database.get_user_files(user_id)
        
        # Lấy stats từ vector store để có số chunks
        stats = vector_store.get_stats(user_id=user_id)
        
        if not user_files:
            return Response({
                "success": True,
                "message": "Chưa có file nào được upload.",
                "files": [],
                "total_files": 0,
                "total_chunks": 0
            }, status=status.HTTP_200_OK)
        
        # Kết hợp thông tin từ database và vector store
        files_list = []
        for file_info in user_files:
            filename = file_info['filename']
            chunks_count = stats['files'].get(filename, 0)
            files_list.append({
                "filename": filename,
                "chunks": chunks_count,
                "cloudinary_url": file_info.get('cloudinary_url'),
                "uploaded_at": file_info.get('uploaded_at')
            })
        
        return Response({
            "success": True,
            "files": files_list,
            "total_files": len(files_list),
            "total_chunks": stats['total_chunks']
        }, status=status.HTTP_200_OK)


class FileDeleteView(APIView):
    """API endpoint để xóa file của user"""
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Session không hợp lệ"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user_id = user["user_id"]
        filename = request.data.get('filename', '').strip()
        
        if not filename:
            return Response(
                {"success": False, "message": "Vui lòng chọn file cần xóa"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Kiểm tra file có thuộc về user không
        file_info = database.get_user_file(user_id, filename)
        if not file_info:
            return Response(
                {"success": False, "message": "File không tồn tại hoặc không thuộc về bạn"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        try:
            import cloudinary
            import cloudinary.uploader
            
            # Cấu hình và kiểm tra Cloudinary credentials
            success, error_msg = configure_cloudinary()
            if not success:
                return Response(
                    {"success": False, "message": error_msg},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Xóa từ Cloudinary
            try:
                cloudinary_public_id = file_info.get('cloudinary_public_id')
                if cloudinary_public_id:
                    cloudinary.uploader.destroy(cloudinary_public_id, resource_type="raw")
                    logger.info(f"Đã xóa file {filename} từ Cloudinary")
            except Exception as cloudinary_error:
                logger.warning(f"Không thể xóa file từ Cloudinary: {str(cloudinary_error)}")
            
            # Xóa từ vector store (chỉ xóa chunks của user này)
            vector_store.delete_by_filename(filename, user_id=user_id)
            
            # Xóa thông tin từ database
            database.delete_user_file(user_id, filename)
            
            return Response({
                "success": True,
                "message": f"Đã xóa file: {filename}"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa file: {str(e)}")
            return Response(
                {"success": False, "message": f"Lỗi: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class FileClearAllView(APIView):
    """API endpoint để xóa toàn bộ tài liệu của user"""
    
    def post(self, request):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Session không hợp lệ"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user_id = user["user_id"]
        
        try:
            import cloudinary
            import cloudinary.uploader
            
            # Cấu hình và kiểm tra Cloudinary credentials
            success, error_msg = configure_cloudinary()
            if not success:
                return Response(
                    {"success": False, "message": error_msg},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Lấy tất cả files của user
            user_files = database.get_user_files(user_id)
            
            # Xóa từng file trên Cloudinary
            for file_info in user_files:
                try:
                    cloudinary_public_id = file_info.get('cloudinary_public_id')
                    if cloudinary_public_id:
                        cloudinary.uploader.destroy(cloudinary_public_id, resource_type="raw")
                except Exception as e:
                    logger.warning(f"Không thể xóa {file_info['filename']} từ Cloudinary: {str(e)}")
                
                # Xóa từ vector store
                vector_store.delete_by_filename(file_info['filename'], user_id=user_id)
            
            # Xóa tất cả records từ database
            for file_info in user_files:
                database.delete_user_file(user_id, file_info['filename'])
            
            return Response({
                "success": True,
                "message": f"Đã xóa {len(user_files)} tài liệu"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa tài liệu: {str(e)}")
            return Response(
                {"success": False, "message": f"Lỗi: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class FileViewView(APIView):
    """API endpoint để lấy URL xem PDF từ Cloudinary"""
    
    def get(self, request, filename):
        session_id = get_session_id_from_request(request)
        
        if not session_id or not auth_manager:
            return Response(
                {"success": False, "message": "Vui lòng đăng nhập"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user = auth_manager.get_user_from_session(session_id)
        if not user or not database:
            return Response(
                {"success": False, "message": "Session không hợp lệ"},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        user_id = user["user_id"]
        
        # Lấy thông tin file
        file_info = database.get_user_file(user_id, filename)
        if not file_info:
            return Response(
                {"success": False, "message": "File không tồn tại hoặc không thuộc về bạn"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        cloudinary_url = file_info.get('cloudinary_url')
        if not cloudinary_url:
            return Response(
                {"success": False, "message": "Không tìm thấy URL của file"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response({
            "success": True,
            "url": cloudinary_url,
            "filename": filename
        }, status=status.HTTP_200_OK)

