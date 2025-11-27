"""
Module rerank kết quả tìm kiếm để tăng độ chính xác
"""
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """Rerank kết quả tìm kiếm sử dụng cross-encoder"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Khởi tạo reranker
        
        Args:
            model_name: Tên model reranker
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from FlagEmbedding import FlagReranker
            logger.info(f"Đang tải reranker model: {model_name}")
            try:
                self.model = FlagReranker(model_name, use_fp16=True, use_flash_attention_2=True)
                logger.info(f"Đã tải reranker {model_name} với flash_attn_2 thành công")
            except Exception as e_flash:
                logger.info(f"Không thể dùng flash_attn_2, dùng mode thường: {str(e_flash)}")
                self.model = FlagReranker(model_name, use_fp16=True)
                logger.info(f"Đã tải reranker {model_name} thành công")
        except Exception as e:
            logger.warning(f"Không thể tải reranker: {str(e)}")
            logger.warning("Hệ thống sẽ hoạt động mà không có reranker")
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank các documents dựa trên độ liên quan với query
        
        Args:
            query: Câu hỏi
            documents: List các document cần rerank
            top_k: Số lượng kết quả trả về sau khi rerank
            
        Returns:
            List các document đã được rerank
        """
        if not documents:
            return []
        
        if self.model is None:
            logger.info("Reranker không khả dụng, trả về kết quả gốc")
            return documents[:top_k]
        
        try:
            pairs = [[query, doc["text"]] for doc in documents]
            scores = self.model.compute_score(pairs)
            
            if isinstance(scores, float):
                scores = [scores]
            
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])
            
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Đã rerank {len(documents)} documents, lấy top {top_k}")
            return documents[:top_k]
            
        except Exception as e:
            logger.error(f"Lỗi khi rerank: {str(e)}")
            return documents[:top_k]
