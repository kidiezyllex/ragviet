"""
Module quản lý FAISS vector store và embeddings
"""
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import heavy dependencies
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_AI_LIBS = True
except ImportError as e:
    logger.warning(f"Could not import AI libraries (faiss, numpy, sentence-transformers): {e}")
    HAS_AI_LIBS = False
    faiss = None
    np = None
    SentenceTransformer = None



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
        self.metadata_by_file = defaultdict(list)
        
        if HAS_AI_LIBS:
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
                        self.encoder = None
        else:
            self.encoder = None
            logger.warning("AI Libraries missing - VectorStore operating in dummy mode")
        
        if self.encoder:
            self.dimension = self.encoder.get_sentence_embedding_dimension()
        else:
            self.dimension = 768 # Dummy dimension
            
        self.load_index()

    def _build_file_index(self):
        """Rebuild quick lookup map cho metadata theo filename"""
        self.metadata_by_file = defaultdict(list)
        for meta in self.metadata:
            self.metadata_by_file[meta["filename"]].append(meta)
        for entries in self.metadata_by_file.values():
            entries.sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_id", 0)))
    
    def load_index(self):
        """Load FAISS index và metadata từ disk"""
        if HAS_AI_LIBS and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)

                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self._build_file_index()
                logger.info(f"Đã load {self.index.ntotal} vectors từ index")
            except Exception as e:
                logger.error(f"Lỗi khi load index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Tạo FAISS index mới"""
        if HAS_AI_LIBS:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None

        self.metadata = []
        self.metadata_by_file = defaultdict(list)
        logger.info(f"Đã tạo FAISS index mới với dimension {self.dimension}")
    
    def save_index(self):
        """Lưu FAISS index và metadata ra disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            if self.index and HAS_AI_LIBS:
                faiss.write_index(self.index, self.index_path)

            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            
            logger.info(f"Đã lưu index với {self.index.ntotal if self.index else 0} vectors")
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
        
        if self.encoder and self.index:
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            self.index.add(embeddings)
        else:
            logger.warning("AI features disabled. Document text saved but not indexed.")

        
        for chunk in chunks:
            meta_entry = {
                "text": chunk["text"],
                "filename": chunk["metadata"]["filename"],
                "page_number": chunk["metadata"]["page_number"],
                "chunk_id": chunk["metadata"]["chunk_id"],
                "user_id": chunk["metadata"].get("user_id")  # Lưu user_id nếu có
            }
            self.metadata.append(meta_entry)
            filename = meta_entry["filename"]
            self.metadata_by_file[filename].append(meta_entry)
            self.metadata_by_file[filename].sort(key=lambda x: (x.get("page_number", 0), x.get("chunk_id", 0)))
        
        self.save_index()
        logger.info(f"Đã thêm {len(chunks)} chunks vào vector store")
    
    def search(self, query: str, top_k: int = 20, filename: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """
        Tìm kiếm các chunk giống nghĩa nhất
        
        Args:
            query: Câu hỏi
            top_k: Số lượng kết quả trả về
            filename: Tên file cụ thể để tìm kiếm (nếu None thì tìm trong tất cả)
            user_id: ID của user để filter (nếu None thì tìm trong tất cả)
            
        Returns:
            List các chunk tìm được
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("Vector store trống hoặc chưa khởi tạo")
            return []
        
        if not self.encoder:
            return []

        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        search_k = top_k * 3 if filename or user_id else top_k
        k = min(search_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        total_candidates = 0
        filtered_by_filename = 0
        filtered_by_user = 0
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                total_candidates += 1
                meta = self.metadata[idx]
                
                if filename:
                    if meta.get("filename") != filename:
                        filtered_by_filename += 1
                        continue
                if user_id:
                    meta_user_id = meta.get("user_id")
                    if meta_user_id is not None and meta_user_id != user_id:
                        filtered_by_user += 1
                        continue
                
                result = meta.copy()
                result["distance"] = float(distance)
                results.append(result)
                if len(results) >= top_k:
                    break
        
        logger.info(
            f"Tìm được {len(results)}/{total_candidates} kết quả cho query: {query[:50]}... "
            f"(filename filter: {filename}, user_id filter: {user_id}, "
            f"filtered by filename: {filtered_by_filename}, filtered by user: {filtered_by_user})"
        )
        
        if len(results) == 0 and (filename or user_id):
            # Lấy tất cả filenames unique trong vector store
            unique_filenames = set(m.get('filename') for m in self.metadata)
            logger.warning(
                f"Không tìm được kết quả với filename='{filename}', user_id='{user_id}'. "
                f"Các filenames có trong vector store: {sorted(unique_filenames)}"
            )
        
        return results
    
    def delete_by_filename(self, filename: str, user_id: Optional[str] = None):
        """
        Xóa tất cả chunks của một file
        
        Args:
            filename: Tên file cần xóa
            user_id: ID của user (nếu có để đảm bảo chỉ xóa file của user đó)
        """
        if user_id:
            indices_to_keep = [i for i, meta in enumerate(self.metadata) 
                              if not (meta["filename"] == filename and meta.get("user_id") == user_id)]
        else:
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
        
        self._create_new_index()
        if self.encoder and self.index:
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            self.index.add(embeddings)

        self.metadata = new_metadata
        self._build_file_index()
        
        self.save_index()
        logger.info(f"Đã xóa file {filename}, còn lại {len(self.metadata)} chunks")
    
    def delete_temp_files_by_user(self, user_id: str, valid_filenames: List[str] = None):
        """
        Xóa tất cả chunks có tên file tạm (bắt đầu bằng 'tmp') của một user
        và chỉ giữ lại các file có trong valid_filenames (nếu có)
        
        Args:
            user_id: ID của user
            valid_filenames: List các filename hợp lệ (nếu có, chỉ xóa các file không có trong list này)
        """
        import re
        
        # Pattern để nhận diện file tạm: bắt đầu bằng 'tmp' và có thể có số/chữ
        temp_pattern = re.compile(r'^tmp[a-z0-9_]+\.pdf$', re.IGNORECASE)
        
        if valid_filenames:
            valid_set = set(valid_filenames)
            indices_to_keep = [
                i for i, meta in enumerate(self.metadata)
                if not (
                    meta.get("user_id") == user_id and
                    (temp_pattern.match(meta.get("filename", "")) or meta.get("filename") not in valid_set)
                )
            ]
        else:
            # Xóa tất cả file tạm của user
            indices_to_keep = [
                i for i, meta in enumerate(self.metadata)
                if not (
                    meta.get("user_id") == user_id and
                    temp_pattern.match(meta.get("filename", ""))
                )
            ]
        
        deleted_count = len(self.metadata) - len(indices_to_keep)
        
        if deleted_count == 0:
            logger.info(f"Không có file tạm nào để xóa cho user {user_id}")
            return
        
        if len(indices_to_keep) == 0:
            self._create_new_index()
            self.save_index()
            logger.info(f"Đã xóa tất cả chunks của user {user_id}, vector store giờ trống")
            return
        
        new_metadata = [self.metadata[i] for i in indices_to_keep]
        
        texts = [meta["text"] for meta in new_metadata]
        
        self._create_new_index()
        if self.encoder and self.index:
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            self.index.add(embeddings)

        self.metadata = new_metadata
        self._build_file_index()
        
        self.save_index()
        logger.info(f"Đã xóa {deleted_count} chunks (file tạm) của user {user_id}, còn lại {len(self.metadata)} chunks")
    
    def clear_all(self):
        """Xóa toàn bộ vector store"""
        self._create_new_index()
        self.metadata_by_file = defaultdict(list)
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
            file_chunks = self.metadata_by_file.get(filename, [])
            if not file_chunks:
                continue

            for meta in file_chunks:
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
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict:
        """
        Lấy thống kê về vector store
        
        Args:
            user_id: ID của user để filter (nếu None thì lấy tất cả)
        """
        files = {}
        total_chunks = 0
        for meta in self.metadata:
            if user_id and meta.get("user_id") != user_id:
                continue
            filename = meta["filename"]
            if filename not in files:
                files[filename] = 0
            files[filename] += 1
            total_chunks += 1
        
        return {
            "total_chunks": total_chunks,
            "total_files": len(files),
            "files": files
        }
