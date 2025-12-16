"""
Module xử lý PDF và chia nhỏ văn bản thành chunks
"""
try:
    import fitz
except ImportError:
    fitz = None
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Xử lý file PDF và chia nhỏ văn bản"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 100):
        """
        Khởi tạo PDF processor
        
        Args:
            chunk_size: Kích thước chunk (ký tự) - khuyến nghị 300-500
            overlap: Độ chồng lấp giữa các chunk (ký tự) - khuyến nghị 100
        """
        if chunk_size < 300:
            chunk_size = 300
        elif chunk_size > 500:
            chunk_size = 500
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Trích xuất text từ PDF, giữ nguyên dấu tiếng Việt
        
        Args:
            pdf_path: Đường dẫn đến file PDF
            
        Returns:
            List các dict chứa {page_number, text}
        """
        if fitz is None:
            logger.error("PyMuPDF (fitz) is not installed. Cannot extract text.")
            raise ImportError("PyMuPDF (fitz) missing")

        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text", sort=True)
                
                if text.strip():
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": text.strip()
                    })
            
            doc.close()
            logger.info(f"Đã trích xuất {len(pages_data)} trang từ {os.path.basename(pdf_path)}")
            return pages_data
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý PDF {pdf_path}: {str(e)}")
            raise
    
    def create_chunks(self, text: str, filename: str, page_number: int) -> List[Dict[str, any]]:
        """
        Chia văn bản thành các chunk nhỏ với overlap
        
        Args:
            text: Văn bản cần chia
            filename: Tên file nguồn
            page_number: Số trang
            
        Returns:
            List các chunk với metadata
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_id": chunk_id
                    }
                })
                chunk_id += 1
            
            start += self.chunk_size - self.overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: str, filename: str = None) -> Tuple[List[Dict], int]:
        """
        Xử lý toàn bộ PDF: extract text + chunking
        
        Args:
            pdf_path: Đường dẫn đến file PDF
            filename: Tên file gốc (nếu None thì dùng basename của pdf_path)
            
        Returns:
            Tuple (list chunks, số trang)
        """
        if filename is None:
            filename = os.path.basename(pdf_path)
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        all_chunks = []
        for page_data in pages_data:
            chunks = self.create_chunks(
                page_data["text"],
                filename,
                page_data["page_number"]
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Đã tạo {len(all_chunks)} chunks từ {len(pages_data)} trang của {filename}")
        return all_chunks, len(pages_data)
    
    def process_multiple_pdfs(self, pdf_paths: List[str], filenames: List[str] = None) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Xử lý nhiều file PDF
        
        Args:
            pdf_paths: List đường dẫn đến các file PDF
            filenames: List tên file gốc tương ứng (nếu None thì dùng basename của pdf_paths)
            
        Returns:
            Tuple (list tất cả chunks, dict {filename: số trang})
        """
        all_chunks = []
        pages_info = {}
        
        if filenames is None:
            filenames = [os.path.basename(pdf_path) for pdf_path in pdf_paths]
        
        for pdf_path, filename in zip(pdf_paths, filenames):
            try:
                chunks, num_pages = self.process_pdf(pdf_path, filename=filename)
                all_chunks.extend(chunks)
                pages_info[filename] = num_pages
            except Exception as e:
                logger.error(f"Không thể xử lý {pdf_path}: {str(e)}")
                pages_info[filename] = 0
        
        return all_chunks, pages_info
