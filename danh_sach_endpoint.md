3.6. DANH SÁCH ENDPOINT ĐƯỢC SỬ DỤNG TRONG HỆ THỐNG

Trong hệ thống, tất cả các API đều được chuẩn hóa trả về theo một cấu trúc thống nhất thông qua đối tượng Response từ Django REST Framework. Điều này giúp cho client (frontend, mobile hoặc hệ thống tích hợp khác) luôn nhận dữ liệu với định dạng đồng nhất, dễ dàng phân tích và xử lý.

Cấu trúc trả về mặc định bao gồm các phần chính:

success: trạng thái kết quả xử lý (true hoặc false). Nếu success = true thì nghĩa là thao tác thành công. Nếu success = false thì đại diện cho lỗi và được kèm theo message mô tả chi tiết.

message: mô tả ngắn gọn về kết quả hoặc lỗi (ví dụ: "Đăng nhập thành công", "Email không hợp lệ", "Vui lòng đăng nhập", "Không tìm thấy thông tin liên quan trong các tài liệu đã upload").

data: dữ liệu trả về, có thể là object, danh sách hoặc null nếu không có dữ liệu. Tùy theo từng endpoint, data có thể chứa các trường như: user, session_id, access_token, chat_session_id, response, files, sessions, messages, v.v.

Hệ thống cũng sử dụng các mã trạng thái HTTP chuẩn để dễ dàng phân loại và truy vết lỗi:

200 OK: thao tác thành công, dữ liệu được trả về đầy đủ.

400 Bad Request: lỗi liên quan đến dữ liệu đầu vào không hợp lệ (ví dụ: email không đúng định dạng, mật khẩu quá ngắn, thiếu thông tin bắt buộc, file không phải PDF).

401 Unauthorized: lỗi liên quan đến xác thực và người dùng (ví dụ: sai mật khẩu, tài khoản không tồn tại, chưa đăng nhập, session không hợp lệ).

500 Internal Server Error: các lỗi chung hoặc ngoại lệ hệ thống (ví dụ: lỗi kết nối cơ sở dữ liệu, lỗi xử lý file PDF, lỗi khi gọi LLM API, lỗi chưa xác định).

Danh sách các Endpoint trong hệ thống

| EndPoint | Phương thức | Dữ liệu nhận | Dữ liệu gửi | Mô tả API |
|----------|-------------|--------------|-------------|-----------|
| /api/auth/login/ | POST | email (string), password (string) | success (boolean), message (string), session_id (string), access_token (string), user (object), chat_session_id (string) | Đăng nhập người dùng vào hệ thống. Trả về thông tin user, session_id và access_token nếu thành công. |
| /api/auth/register/ | POST | username (string), email (string), password (string), confirm_password (string) | success (boolean), message (string), session_id (string), access_token (string), user (object), chat_session_id (string) | Đăng ký tài khoản mới. Sau khi đăng ký thành công, hệ thống tự động đăng nhập và trả về thông tin tương tự như login. |
| /api/auth/logout/ | POST | session_id (string) hoặc Authorization header | success (boolean), message (string) | Đăng xuất người dùng khỏi hệ thống, vô hiệu hóa session hiện tại. |
| /api/auth/forgot-password/ | POST | email (string) | success (boolean), message (string) | Yêu cầu reset mật khẩu. Hệ thống gửi mã OTP đến email đã đăng ký. |
| /api/auth/reset-password/ | POST | token (string), new_password (string), confirm_password (string) | success (boolean), message (string) | Đặt lại mật khẩu mới bằng mã OTP đã nhận. |
| /api/auth/verify-session/ | POST | session_id (string) hoặc Authorization header | success (boolean), valid (boolean), user (object), chat_session_id (string) | Xác thực session hiện tại có còn hợp lệ hay không. Trả về thông tin user nếu session hợp lệ. |
| /api/chat/send/ | POST | message (string), session_id (string, optional), selected_file (string, optional), chat_session_id (string, optional) | success (boolean), response (string), chat_session_id (string) | Gửi câu hỏi đến chatbot. Hệ thống tìm kiếm trong vector store, rerank kết quả và sinh câu trả lời bằng LLM. Trả về câu trả lời dựa trên nội dung tài liệu đã upload. |
| /api/chat/sessions/ | GET | session_id (query param hoặc Authorization header) | success (boolean), sessions (array) | Lấy danh sách tất cả các cuộc trò chuyện (chat sessions) của người dùng đã đăng nhập. Mỗi session bao gồm session_id, title, updated_at, last_question. |
| /api/chat/sessions/create/ | POST | session_id (string) hoặc Authorization header | success (boolean), chat_session_id (string), message (string) | Tạo một cuộc trò chuyện mới (chat session) cho người dùng. |
| /api/chat/history/<session_id>/ | GET | session_id (path param), session_id (query param hoặc Authorization header) | success (boolean), messages (array) | Lấy lịch sử tin nhắn của một cuộc trò chuyện cụ thể. Mỗi message bao gồm message (câu hỏi) và response (câu trả lời). |
| /api/files/upload/ | POST | files (multipart/form-data), session_id (string) | success (boolean), message (string), files_processed (number), total_pages (number), files_detail (string) | Upload một hoặc nhiều file PDF lên hệ thống. Hệ thống sẽ trích xuất văn bản, chia thành chunks và lưu vào vector store. Yêu cầu người dùng phải đăng nhập. |
| /api/files/list/ | GET | - | success (boolean), files (array), total_files (number), total_chunks (number) | Lấy danh sách tất cả các file PDF đã được upload và xử lý. Mỗi file bao gồm filename và số lượng chunks. |
| /api/files/delete/ | POST | filename (string) | success (boolean), message (string) | Xóa một file PDF cụ thể khỏi hệ thống. File sẽ bị xóa khỏi vector store và thư mục lưu trữ. |
| /api/files/clear-all/ | POST | - | success (boolean), message (string) | Xóa toàn bộ tài liệu đã upload khỏi hệ thống. Tất cả files và chunks trong vector store sẽ bị xóa. |

