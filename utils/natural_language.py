"""
Module xử lý câu hỏi tự nhiên không liên quan đến tài liệu
"""
import re
from typing import Optional
from collections import Counter

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
    
    if re.match(r'^(chào|hello|hi)(\s+|$)', normalized):
        return NATURAL_RESPONSES["chào"]
    elif re.match(r'^bạn\s+(là|khỏe|thế nào|có khỏe)', normalized):
        return NATURAL_RESPONSES["bạn là ai"]
    elif re.match(r'^hôm\s+nay\s+bạn', normalized):
        return NATURAL_RESPONSES["hôm nay bạn thế nào"]
    elif re.match(r'^giới\s+thiệu', normalized):
        return NATURAL_RESPONSES["giới thiệu về bạn"]
    elif re.match(r'^(cảm\s+ơn|thanks|thank\s+you)(\s+|$)', normalized):
        return NATURAL_RESPONSES["cảm ơn"]
    elif re.match(r'^(tạm\s+biệt|bye|goodbye)(\s+|$)', normalized):
        return NATURAL_RESPONSES["tạm biệt"]
    
    return None


def is_meaningless_query(query: str) -> bool:
    """
    Kiểm tra xem câu hỏi có vô nghĩa không (như "fdfgfgf", "jkjlkjlkjk", ...)
    
    Args:
        query: Câu hỏi cần kiểm tra
        
    Returns:
        True nếu câu hỏi vô nghĩa
    """
    if not query or len(query.strip()) == 0:
        return False
    
    text = query.strip()
    if len(text) < 3:
        return False
    
    if re.match(r'^[\d\s\W]+$', text):
        return True
    
    if re.match(r'^\d+$', text):
        return True
    
    if re.match(r'^[\W\s]+$', text) and not re.search(r'[a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', text):
        return True
    
    clean_text = re.sub(r'[\s\W\d]', '', text.lower())
    
    if len(clean_text) < 3:
        return True
    
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(clean_text)):
        if clean_text[i] == clean_text[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    if max_consecutive >= 3:
        return True
    
    char_counts = Counter(clean_text)
    most_common_count = char_counts.most_common(1)[0][1] if char_counts else 0
    repetition_ratio = most_common_count / len(clean_text) if len(clean_text) > 0 else 0
    
    if repetition_ratio >= 0.5 and len(clean_text) >= 4:
        return True
    
    if repetition_ratio > 0.4 and len(clean_text) >= 6:
        return True
    
    for length in range(2, min(5, len(clean_text) // 2 + 1)):
        for i in range(len(clean_text) - length * 2 + 1):
            substring = clean_text[i:i+length]
            count = clean_text.count(substring)
            if count >= 3 and len(substring) >= 2:
                if count * length / len(clean_text) > 0.6:
                    return True
            elif count >= 2 and len(substring) >= 2:
                if count * length / len(clean_text) >= 0.7:
                    return True
    
    words = re.findall(r'\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+\b', text.lower())
    
    common_words = {
        'của', 'và', 'là', 'có', 'được', 'trong', 'với', 'cho', 'từ', 'về', 'này', 'đó', 'nào',
        'bạn', 'tôi', 'chúng', 'họ', 'mình', 'ta', 'người', 'cái', 'con', 'cây', 'nhà', 'đi',
        'làm', 'nói', 'biết', 'thấy', 'nghe', 'xem', 'học', 'đọc', 'viết', 'nghĩ', 'muốn',
        'gì', 'sao', 'thế', 'nào', 'đâu', 'khi', 'nếu', 'vì', 'nên', 'mà', 'để',
        # Tiếng Anh
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'what', 'when', 'where', 'why', 'how', 'who', 'which',
        'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    meaningful_words = [w for w in words if len(w) >= 2 and (w in common_words or len(w) >= 4)]
    
    if len(meaningful_words) == 0 and len(clean_text) >= 4:
        unique_chars = len(set(clean_text))
        unique_ratio = unique_chars / len(clean_text) if len(clean_text) > 0 else 0
        
        if unique_ratio < 0.3:
            return True
        
        if unique_chars <= 2 and len(clean_text) >= 4:
            return True
        
        if unique_chars == 3 and len(clean_text) >= 8:
            return True
        
        if len(clean_text) >= 10 and unique_chars <= 4:
            return True
    
    if len(clean_text) >= 6:
        for pattern_len in range(2, min(5, len(clean_text) // 2 + 1)):
            if len(clean_text) % pattern_len == 0:
                pattern = clean_text[:pattern_len]
                repeated = pattern * (len(clean_text) // pattern_len)
                if clean_text == repeated:
                    return True
        
        for pattern_len in range(2, min(5, len(clean_text) // 2 + 1)):
            pattern = clean_text[:pattern_len]
            occurrences = 0
            for i in range(0, len(clean_text) - pattern_len + 1, pattern_len):
                if clean_text[i:i+pattern_len] == pattern:
                    occurrences += 1
                else:
                    break
            if occurrences >= 3:
                return True
    
    if len(clean_text) >= 8 and len(meaningful_words) == 0:
        unique_chars = len(set(clean_text))
        if unique_chars <= 4:
            return True
    
    if len(words) >= 3:
        word_counts = Counter(words)
        most_common_word_count = word_counts.most_common(1)[0][1] if word_counts else 0
        if most_common_word_count >= 3 and len(words) < 10:
            return True
        if most_common_word_count / len(words) >= 0.5 and len(words) >= 4:
            return True
    
    if len(meaningful_words) > 0 and len(meaningful_words) <= 3:
        long_words = [w for w in meaningful_words if len(w) >= 4]
        if len(long_words) == 0 and len(clean_text) >= 8:
            if len(set(meaningful_words)) <= 2:
                return True
    
    if re.search(r'\d', text) and len(clean_text) >= 6:
        digit_letter_pattern = re.findall(r'[a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]\d|\d[a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', text)
        if len(digit_letter_pattern) >= 3 and len(meaningful_words) == 0:
            return True
    
    special_char_pattern = re.findall(r'[a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ][\W]|[\W][a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', text)
    if len(special_char_pattern) >= 3 and len(meaningful_words) == 0:
        return True
    
    if len(meaningful_words) >= 4:
        question_words = {'what', 'when', 'where', 'why', 'how', 'who', 'which', 
                         'gì', 'sao', 'thế', 'nào', 'đâu', 'khi', 'ai'}
        has_question_word = any(qw in text.lower() for qw in question_words)
        
        action_words = {'làm', 'nói', 'biết', 'thấy', 'nghe', 'xem', 'học', 'đọc', 'viết', 
                       'nghĩ', 'muốn', 'là', 'có', 'được', 'do', 'does', 'did', 'is', 'are', 
                       'was', 'were', 'have', 'has', 'had', 'can', 'could', 'will', 'would'}
        has_action_word = any(aw in text.lower() for aw in action_words)
        
        if not has_question_word and not has_action_word:
            if len(text.split()) >= 5:
                return True
    
    keyboard_patterns = [
        'qwerty', 'asdfgh', 'zxcvbn', 'qazwsx', 'abcdef', 'ghijkl', 'mnopqr', 'stuvwx', 'yz',
        '123456', 'abcdefgh', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm'
    ]
    clean_lower = clean_text.lower()
    for pattern in keyboard_patterns:
        if pattern in clean_lower and len(clean_text) >= len(pattern):
            if len(meaningful_words) == 0:
                return True
    
    return False


def get_meaningless_response() -> str:
    """
    Trả về câu trả lời cho câu hỏi vô nghĩa
    
    Returns:
        Câu trả lời phù hợp
    """
    return "Xin lỗi, tôi không hiểu câu hỏi của bạn. Vui lòng đặt câu hỏi rõ ràng và có ý nghĩa về nội dung trong các tài liệu đã upload. Ví dụ: 'Quy định về thủ tục hành chính là gì?' hoặc 'Tài liệu này nói về điều gì?'"

