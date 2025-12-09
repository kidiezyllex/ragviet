CHƯƠNG 4 KẾT QUẢ THỰC NGHIỆM
4.1. GIỚI THIỆU SẢN PHẨM
"Hệ thống RAG (Retrieval-Augmented Generation) cho tài liệu tiếng Việt" là phần mềm được phát triển nhằm hỗ trợ người dùng trong việc tìm kiếm, tra cứu và trả lời câu hỏi dựa trên nội dung các tài liệu PDF đã được upload vào hệ thống. Hệ thống sử dụng công nghệ RAG kết hợp tìm kiếm ngữ nghĩa và mô hình ngôn ngữ lớn (LLM) để cung cấp câu trả lời chính xác, đầy đủ dựa trên nội dung tài liệu.
Hệ thống gồm 4 thành phần chính:
Frontend (giao diện web Gradio): cho phép người dùng đăng nhập, đăng ký, upload tài liệu PDF, đặt câu hỏi và nhận câu trả lời, quản lý lịch sử chat, quản lý tài liệu đã upload.
Backend (máy chủ Django REST Framework): tiếp nhận và xử lý các yêu cầu từ frontend, quản lý xác thực người dùng, xử lý tài liệu PDF, tìm kiếm trong vector store, tạo câu trả lời bằng LLM, lưu trữ lịch sử chat và quản lý dữ liệu.
Database (MongoDB): lưu trữ thông tin người dùng, phiên đăng nhập, lịch sử chat, các cuộc trò chuyện và dữ liệu quản trị của hệ thống.
Vector Store (FAISS): lưu trữ embeddings của các chunks tài liệu, hỗ trợ tìm kiếm ngữ nghĩa nhanh chóng và chính xác, quản lý metadata của tài liệu.
Các chức năng chính của hệ thống:
Người dùng thông thường
Đăng nhập, đăng xuất và đăng ký tài khoản vào hệ thống.
Quản lý thông tin cá nhân và phiên đăng nhập.
Upload tài liệu PDF cho phép tải lên một hoặc nhiều file PDF, hệ thống tự động trích xuất văn bản, chia nhỏ thành chunks và lưu vào vector store.
Quản lý tài liệu cho phép xem danh sách các file đã upload, xóa file cụ thể hoặc xóa toàn bộ tài liệu.
Đặt câu hỏi về nội dung tài liệu cho phép người dùng đặt câu hỏi tự nhiên bằng tiếng Việt, hệ thống sẽ tìm kiếm trong các tài liệu đã upload và trả lời dựa trên nội dung tài liệu.
Quản lý lịch sử chat cho phép xem danh sách các cuộc trò chuyện đã thực hiện, xem lại lịch sử tin nhắn của từng cuộc trò chuyện và tạo cuộc trò chuyện mới.
Nhận câu trả lời tự nhiên cho các câu hỏi chào hỏi, giới thiệu mà không cần tìm kiếm trong tài liệu.
Quản trị viên
Đăng nhập và đăng xuất hệ thống với quyền quản trị.
Tìm kiếm người dùng cho phép tìm kiếm thông tin người dùng trong hệ thống bằng username, email hoặc user_id.
Xem thống kê tần suất sử dụng cho phép xem các thống kê về hoạt động của người dùng bao gồm số lần đăng nhập, số câu hỏi đã gửi, số file đã upload, số chat sessions đã tạo và tần suất sử dụng theo các khoảng thời gian khác nhau (ngày, tuần, tháng).
Hệ thống xử lý tài liệu (Backend Services)
Trích xuất văn bản từ PDF tự động trích xuất văn bản từ các file PDF, giữ nguyên dấu tiếng Việt và tổ chức theo từng trang.
Chia nhỏ tài liệu thành chunks tự động chia văn bản thành các đoạn nhỏ (chunks) với kích thước và overlap phù hợp để tối ưu hóa tìm kiếm.
Tạo embeddings và lưu trữ vector tự động tạo embeddings cho các chunks sử dụng mô hình embedding, lưu trữ vào FAISS index và quản lý metadata.
Tìm kiếm ngữ nghĩa tìm kiếm các chunks liên quan đến câu hỏi trong vector store sử dụng tìm kiếm ngữ nghĩa.
Thêm chunks từ các trang lân cận mở rộng kết quả tìm kiếm bằng cách thêm các chunks từ các trang trước và sau trong cùng file để cung cấp context đầy đủ hơn.
Rerank kết quả tìm kiếm sắp xếp lại các kết quả tìm kiếm theo độ liên quan với câu hỏi để chọn ra các chunks phù hợp nhất.
Tạo câu trả lời bằng LLM sử dụng Groq API với các mô hình LLM (như llama-3.3-70b-versatile) để sinh câu trả lời dựa trên các chunks ngữ cảnh, đảm bảo câu trả lời đầy đủ, chính xác và có định dạng đẹp.
4.2. MÔI TRƯỜNG TRIỂN KHAI THỬ NGHIỆM
4.2.1. Frontend và Backend
Hệ thống website được phát triển và triển khai thử nghiệm trên máy tính cá nhân (Windows 11, Intel Core i5, RAM 16 GB, SSD 256 GB).
Backend được triển khai trên cổng 8000 (chạy trên localhost).
Frontend được triển khai trên cổng 7860 (chạy trên cùng máy phát triển).
Hệ thống chỉ mới triển khai và thử nghiệm trong môi trường local, chưa triển khai trên máy chủ thực tế.
4.2.2. Thiết bị trong môi trường thử nghiệm
Trong phạm vi đề tài, hệ thống được sử dụng để xử lý và tìm kiếm thông tin từ các tài liệu PDF tiếng Việt thông qua công nghệ RAG.
4.2.2.1. Thiết bị Windows
Database MongoDB được cấu hình thông qua biến môi trường MONGO_URI để kết nối đến MongoDB server.
Database lưu trữ các collections: users (thông tin người dùng), auth_sessions (phiên đăng nhập), chat_sessions (các cuộc trò chuyện), chat_history (lịch sử tin nhắn).
Các indexes được tạo tự động cho các trường quan trọng như email, username, session_id, user_id để tối ưu hóa hiệu suất truy vấn.
Vector Store sử dụng FAISS (Facebook AI Similarity Search) để lưu trữ và tìm kiếm embeddings của các chunks tài liệu. Embeddings được tạo bằng mô hình Sentence Transformers hỗ trợ tiếng Việt. Metadata của các chunks (filename, page_number, chunk_id) được lưu trữ trong file JSON kèm theo FAISS index.
4.2.2.2. Phương thức triển khai Agent
Để triển khai hệ thống, các bước chính được thực hiện như sau:
Cài đặt môi trường Python và các thư viện cần thiết từ file requirements.txt (Django, Gradio, PyMuPDF, sentence-transformers, faiss-cpu, pymongo, groq, v.v.).
Cấu hình biến môi trường trong file .env: MONGO_URI (địa chỉ kết nối MongoDB), GROQ_API_KEY (API key cho Groq), DJANGO_SECRET_KEY (secret key cho Django), MONGODB_DB_NAME (tên database).
Khởi động Django backend server trên cổng 8000 bằng lệnh python manage.py runserver hoặc sử dụng WSGI server.
Khởi động Gradio frontend trên cổng 7860 bằng lệnh python main.py, frontend sẽ tự động kết nối đến backend API tại http://localhost:8000/api.
Vector Store và các mô hình embedding được khởi tạo tự động khi backend khởi động, hệ thống sẵn sàng nhận và xử lý các yêu cầu từ người dùng.
