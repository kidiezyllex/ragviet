"""
Module xử lý câu hỏi tự nhiên không liên quan đến tài liệu
"""
import re
from typing import Optional

NATURAL_RESPONSES = {
    # Chào hỏi
    "chào": "Xin chào! Tôi là chatbot trợ lý hành chính Việt Nam. Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu hành chính. Bạn cần hỗ trợ gì?",
    "hello": "Xin chào! Tôi là chatbot trợ lý hành chính Việt Nam. Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu hành chính. Bạn cần hỗ trợ gì?",
    "hi": "Xin chào! Tôi là chatbot trợ lý hành chính Việt Nam. Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu hành chính. Bạn cần hỗ trợ gì?",
    "chào bạn": "Xin chào! Tôi là chatbot trợ lý hành chính Việt Nam. Tôi có thể giúp bạn tìm hiểu thông tin từ các tài liệu hành chính. Bạn cần hỗ trợ gì?",
    
    # Hỏi thăm
    "bạn khỏe không": "Cảm ơn bạn đã hỏi! Tôi là một chatbot nên không có cảm xúc, nhưng tôi luôn sẵn sàng giúp bạn. Bạn có câu hỏi gì về tài liệu hành chính không?",
    "bạn thế nào": "Cảm ơn bạn đã hỏi! Tôi là một chatbot nên không có cảm xúc, nhưng tôi luôn sẵn sàng giúp bạn. Bạn có câu hỏi gì về tài liệu hành chính không?",
    "hôm nay bạn thế nào": "Cảm ơn bạn đã hỏi! Tôi là một chatbot nên không có cảm xúc, nhưng tôi luôn sẵn sàng giúp bạn. Bạn có câu hỏi gì về tài liệu hành chính không?",
    "bạn có khỏe không": "Cảm ơn bạn đã hỏi! Tôi là một chatbot nên không có cảm xúc, nhưng tôi luôn sẵn sàng giúp bạn. Bạn có câu hỏi gì về tài liệu hành chính không?",
    
    # Giới thiệu
    "bạn là ai": "Tôi là chatbot trợ lý hành chính Việt Nam, được xây dựng bằng công nghệ RAG (Retrieval-Augmented Generation). Tôi có thể giúp bạn tìm kiếm và trả lời các câu hỏi về nội dung trong các tài liệu hành chính mà bạn đã upload. Bạn muốn hỏi gì về tài liệu?",
    "giới thiệu về bạn": "Tôi là chatbot trợ lý hành chính Việt Nam, được xây dựng bằng công nghệ RAG (Retrieval-Augmented Generation). Tôi có thể giúp bạn tìm kiếm và trả lời các câu hỏi về nội dung trong các tài liệu hành chính mà bạn đã upload. Bạn muốn hỏi gì về tài liệu?",
    "bạn làm gì": "Tôi là chatbot trợ lý hành chính Việt Nam. Nhiệm vụ của tôi là giúp bạn tìm kiếm và trả lời các câu hỏi về nội dung trong các tài liệu hành chính. Bạn có thể upload file PDF và đặt câu hỏi, tôi sẽ tìm thông tin liên quan và trả lời cho bạn.",
    
    # Cảm ơn
    "cảm ơn": "Không có gì! Rất vui được giúp bạn. Nếu bạn có thêm câu hỏi nào khác về tài liệu, đừng ngần ngại hỏi nhé!",
    "thanks": "Không có gì! Rất vui được giúp bạn. Nếu bạn có thêm câu hỏi nào khác về tài liệu, đừng ngần ngại hỏi nhé!",
    "thank you": "Không có gì! Rất vui được giúp bạn. Nếu bạn có thêm câu hỏi nào khác về tài liệu, đừng ngần ngại hỏi nhé!",
    
    # Tạm biệt
    "tạm biệt": "Tạm biệt! Chúc bạn một ngày tốt lành. Nếu có câu hỏi gì, hãy quay lại nhé!",
    "bye": "Tạm biệt! Chúc bạn một ngày tốt lành. Nếu có câu hỏi gì, hãy quay lại nhé!",
    "goodbye": "Tạm biệt! Chúc bạn một ngày tốt lành. Nếu có câu hỏi gì, hãy quay lại nhé!",
}


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def is_natural_question(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có phải là câu hỏi tự nhiên không
    
    Args:
        query: Câu hỏi
        
    Returns:
        True nếu là câu hỏi tự nhiên
    """
    normalized = normalize_text(query)
    
    greeting_patterns = [
        r'^(chào|hello|hi)',
        r'^(chào|hello|hi)\s+',
        r'^bạn\s+(là|khỏe|thế nào|có khỏe)',
        r'^hôm\s+nay\s+bạn',
        r'^giới\s+thiệu',
        r'^bạn\s+làm\s+gì',
        r'^(cảm\s+ơn|thanks|thank\s+you)',
        r'^(tạm\s+biệt|bye|goodbye)',
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, normalized):
            return True
    
    if normalized in NATURAL_RESPONSES:
        return True
    
    return False


def get_natural_response(query: str) -> Optional[str]:
    """
    Lấy câu trả lời cho câu hỏi tự nhiên
    
    Args:
        query: Câu hỏi
        
    Returns:
        Câu trả lời hoặc None nếu không phải câu hỏi tự nhiên
    """
    normalized = normalize_text(query)
    
    if normalized in NATURAL_RESPONSES:
        return NATURAL_RESPONSES[normalized]
    
    for key, response in NATURAL_RESPONSES.items():
        if key in normalized or normalized in key:
            return response
    
    if re.match(r'^(chào|hello|hi)', normalized):
        return NATURAL_RESPONSES["chào"]
    elif re.match(r'^bạn\s+(là|khỏe|thế nào|có khỏe)', normalized):
        return NATURAL_RESPONSES["bạn là ai"]
    elif re.match(r'^hôm\s+nay\s+bạn', normalized):
        return NATURAL_RESPONSES["hôm nay bạn thế nào"]
    elif re.match(r'^giới\s+thiệu', normalized):
        return NATURAL_RESPONSES["giới thiệu về bạn"]
    elif re.match(r'^(cảm\s+ơn|thanks)', normalized):
        return NATURAL_RESPONSES["cảm ơn"]
    elif re.match(r'^(tạm\s+biệt|bye)', normalized):
        return NATURAL_RESPONSES["tạm biệt"]
    
    return None

