"""
Module quản lý FAISS vector store và embeddings
"""
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Quản lý FAISS index và embeddings"""
    
    def __init__(self, 
                 embedding_model_name: str = "keepitreal/vietnamese-sbert",
                 index_path: str = "vector_store/index.faiss",
                 metadata_path: str = "vector_store/metadata.json"):
        """
        Khởi tạo Vector Store
        
        Args:
            embedding_model_name: Tên model embedding
            index_path: Đường dẫn lưu FAISS index
            metadata_path: Đường dẫn lưu metadata
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        
        logger.info(f"Đang tải embedding model: {embedding_model_name}")
        try:
            self.encoder = SentenceTransformer(embedding_model_name)
            logger.info(f"Đã tải model {embedding_model_name} thành công")
        except Exception as e:
            logger.warning(f"Không thể tải {embedding_model_name}: {str(e)}")
            try:
                fallback_model = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
                logger.info(f"Thử tải model dự phòng: {fallback_model}")
                self.encoder = SentenceTransformer(fallback_model)
                logger.info(f"Đã tải model dự phòng {fallback_model} thành công")
            except Exception as e2:
                logger.warning(f"Không thể tải SimCSE-VietNamese: {str(e2)}")
                try:
                    fallback_model2 = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    logger.info(f"Thử tải model dự phòng cuối cùng: {fallback_model2}")
                    self.encoder = SentenceTransformer(fallback_model2)
                    logger.info(f"Đã tải model dự phòng {fallback_model2} thành công")
                except Exception as e3:
                    logger.error(f"Lỗi khi tải tất cả embedding models: {str(e3)}")
                    raise
        
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.load_index()
    
    def load_index(self):
        """Load FAISS index và metadata từ disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Đã load {self.index.ntotal} vectors từ index")
            except Exception as e:
                logger.error(f"Lỗi khi load index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Tạo FAISS index mới"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info(f"Đã tạo FAISS index mới với dimension {self.dimension}")
    
    def save_index(self):
        """Lưu FAISS index và metadata ra disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã lưu index với {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Lỗi khi lưu index: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[Dict]):
        """
        Thêm documents vào vector store
        
        Args:
            chunks: List các chunk với text và metadata
        """
        if not chunks:
            logger.warning("Không có chunk nào để thêm")
            return
        
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Đang tạo embeddings cho {len(texts)} chunks...")
        
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index.add(embeddings)
        
        for chunk in chunks:
            self.metadata.append({
                "text": chunk["text"],
                "filename": chunk["metadata"]["filename"],
                "page_number": chunk["metadata"]["page_number"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            })
        
        self.save_index()
        logger.info(f"Đã thêm {len(chunks)} chunks vào vector store")
    
    def search(self, query: str, top_k: int = 20, filename: Optional[str] = None) -> List[Dict]:
        """
        Tìm kiếm các chunk giống nghĩa nhất
        
        Args:
            query: Câu hỏi
            top_k: Số lượng kết quả trả về
            filename: Tên file cụ thể để tìm kiếm (nếu None thì tìm trong tất cả)
            
        Returns:
            List các chunk tìm được
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store trống, không có dữ liệu để tìm kiếm")
            return []
        
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        search_k = top_k * 3 if filename else top_k
        k = min(search_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                if filename and meta["filename"] != filename:
                    continue
                result = meta.copy()
                result["distance"] = float(distance)
                results.append(result)
                if len(results) >= top_k:
                    break
        
        logger.info(f"Tìm được {len(results)} kết quả cho query: {query[:50]}... (filename filter: {filename})")
        return results
    
    def delete_by_filename(self, filename: str):
        """
        Xóa tất cả chunks của một file
        
        Args:
            filename: Tên file cần xóa
        """
        indices_to_keep = [i for i, meta in enumerate(self.metadata) 
                          if meta["filename"] != filename]
        
        if len(indices_to_keep) == len(self.metadata):
            logger.warning(f"Không tìm thấy file {filename} trong vector store")
            return
        
        if len(indices_to_keep) == 0:
            self._create_new_index()
            self.save_index()
            logger.info(f"Đã xóa tất cả dữ liệu của {filename}, vector store giờ trống")
            return
        
        new_metadata = [self.metadata[i] for i in indices_to_keep]
        
        texts = [meta["text"] for meta in new_metadata]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self._create_new_index()
        self.index.add(embeddings)
        self.metadata = new_metadata
        
        self.save_index()
        logger.info(f"Đã xóa file {filename}, còn lại {len(self.metadata)} chunks")
    
    def clear_all(self):
        """Xóa toàn bộ vector store"""
        self._create_new_index()
        self.save_index()
        logger.info("Đã xóa toàn bộ vector store")
    
    def get_all_chunks_by_filename(self, filename: str) -> List[Dict]:
        """
        Lấy tất cả chunks của một file cụ thể
        
        Args:
            filename: Tên file cần lấy
            
        Returns:
            List các chunk của file đó
        """
        results = []
        for i, meta in enumerate(self.metadata):
            if meta["filename"] == filename:
                result = meta.copy()
                results.append(result)
        
        results.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_id", 0)))
        logger.info(f"Đã lấy {len(results)} chunks từ file {filename}")
        return results
    
    def get_adjacent_chunks(self, chunks: List[Dict], page_range: int = 2) -> List[Dict]:
        """
        Lấy các chunk từ các trang lân cận để liên kết nội dung giữa các trang
        
        Args:
            chunks: List các chunk đã tìm được
            page_range: Số trang trước và sau để lấy thêm (mặc định 2 trang)
            
        Returns:
            List các chunk đã được mở rộng với các trang lân cận
        """
        if not chunks:
            return []
        
        seen_chunks = set()
        expanded_chunks = []
        
        for chunk in chunks:
            chunk_key = (chunk["filename"], chunk.get("page_number", 0), chunk.get("chunk_id", ""))
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                expanded_chunks.append(chunk)
        
        for chunk in chunks:
            filename = chunk["filename"]
            page_num = chunk.get("page_number", 0)
            
            for i, meta in enumerate(self.metadata):
                if meta["filename"] == filename:
                    meta_page = meta.get("page_number", 0)
                    if abs(meta_page - page_num) <= page_range and meta_page != page_num:
                        chunk_key = (meta["filename"], meta.get("page_number", 0), meta.get("chunk_id", ""))
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            expanded_chunks.append(meta.copy())
        
        expanded_chunks.sort(key=lambda x: (
            x.get("filename", ""),
            x.get("page_number", 0),
            x.get("chunk_id", 0)
        ))
        
        logger.info(f"Đã mở rộng từ {len(chunks)} chunks lên {len(expanded_chunks)} chunks (bao gồm {page_range} trang lân cận)")
        return expanded_chunks
    
    def get_stats(self) -> Dict:
        """Lấy thống kê về vector store"""
        files = {}
        for meta in self.metadata:
            filename = meta["filename"]
            if filename not in files:
                files[filename] = 0
            files[filename] += 1
        
        return {
            "total_chunks": len(self.metadata),
            "total_files": len(files),
            "files": files
        }
