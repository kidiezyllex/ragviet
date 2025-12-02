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
            
            return Response({
                "success": True,
                "message": result['message'],
                "session_id": result["session_id"],
                "access_token": result.get("access_token", result["session_id"]),
                "user": result["user"],
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
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
                
                return Response({
                    "success": True,
                    "message": result['message'] + " Đang tự động đăng nhập...",
                    "session_id": login_result["session_id"],
                    "access_token": login_result.get("access_token", login_result["session_id"]),
                    "user": login_result["user"],
                    "chat_session_id": chat_session_id
                }, status=status.HTTP_200_OK)
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
        session_id = request.data.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
        if session_id and auth_manager:
            auth_manager.logout(session_id)
        
        return Response({
            "success": True,
            "message": "Đã đăng xuất"
        }, status=status.HTTP_200_OK)


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
        session_id = request.data.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
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
            
            return Response({
                "success": True,
                "valid": True,
                "user": user,
                "chat_session_id": chat_session_id
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                "success": False,
                "valid": False
            }, status=status.HTTP_200_OK)

class ChatSendView(APIView):
    """API endpoint để gửi tin nhắn chat"""
    
    def post(self, request):
        message = request.data.get('message', '').strip()
        session_id = request.data.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
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
        
        stats = vector_store.get_stats()
        if stats["total_chunks"] == 0:
            return Response({
                "success": True,
                "response": "⚠️ Chưa có tài liệu nào được upload. Vui lòng upload file PDF trước khi đặt câu hỏi."
            }, status=status.HTTP_200_OK)
        
        try:
            logger.info(f"Đang tìm kiếm câu trả lời cho: {message} (file: {selected_file})")
            
            search_results = vector_store.search(message, top_k=30, filename=selected_file)
            
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
        session_id = request.query_params.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
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
        session_id = request.data.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
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
        auth_session_id = request.query_params.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
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
    """API endpoint để upload file PDF"""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        session_id = request.data.get('session_id') or request.META.get('HTTP_AUTHORIZATION', '').replace('Bearer ', '')
        
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
        
        files = request.FILES.getlist('files')
        if not files:
            return Response(
                {"success": False, "message": "Vui lòng chọn ít nhất một file PDF"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            import shutil
            pdf_paths = []
            for file in files:
                if not file.name.endswith('.pdf'):
                    return Response(
                        {"success": False, "message": f"File {file.name} không phải là PDF"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                filename = file.name
                dest_path = os.path.join(settings.MEDIA_ROOT, filename)
                with open(dest_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                pdf_paths.append(dest_path)
            
            all_chunks, pages_info = pdf_processor.process_multiple_pdfs(pdf_paths)
            
            if not all_chunks:
                return Response(
                    {"success": False, "message": "Không thể trích xuất văn bản từ các file PDF"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            vector_store.add_documents(all_chunks)
            
            total_pages = sum(pages_info.values())
            files_summary = "\n".join([f"  • {name}: {pages} trang" for name, pages in pages_info.items()])
            
            return Response({
                "success": True,
                "message": f"Đã xử lý xong {len(pdf_paths)} tài liệu, tổng cộng {total_pages} trang",
                "files_processed": len(pdf_paths),
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
    """API endpoint để lấy danh sách file đã upload"""
    
    def get(self, request):
        stats = vector_store.get_stats()
        
        if stats["total_files"] == 0:
            return Response({
                "success": True,
                "message": "Chưa có file nào được upload.",
                "files": [],
                "total_files": 0,
                "total_chunks": 0
            }, status=status.HTTP_200_OK)
        
        files_list = [{"filename": filename, "chunks": count} for filename, count in stats["files"].items()]
        
        return Response({
            "success": True,
            "files": files_list,
            "total_files": stats['total_files'],
            "total_chunks": stats['total_chunks']
        }, status=status.HTTP_200_OK)


class FileDeleteView(APIView):
    """API endpoint để xóa file"""
    
    def post(self, request):
        filename = request.data.get('filename', '').strip()
        
        if not filename:
            return Response(
                {"success": False, "message": "Vui lòng chọn file cần xóa"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            vector_store.delete_by_filename(filename)
            
            pdf_path = os.path.join(settings.MEDIA_ROOT, filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
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
    """API endpoint để xóa toàn bộ tài liệu"""
    
    def post(self, request):
        try:
            vector_store.clear_all()
            
            for filename in os.listdir(settings.MEDIA_ROOT):
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            return Response({
                "success": True,
                "message": "Đã xóa toàn bộ tài liệu"
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa tài liệu: {str(e)}")
            return Response(
                {"success": False, "message": f"Lỗi: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

