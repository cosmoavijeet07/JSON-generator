import re
from typing import List, Dict, Any, Optional
import hashlib

class ChunkManager:
    def __init__(self):
        self.default_chunk_size = 2000
        self.overlap_size = 200
        
        # Initialize tokenization
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize = sent_tokenize
            self.nltk_available = True
        except:
            # Fallback sentence splitting
            self.nltk_available = False
            self.sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
    
    def chunk_text(self, 
                  text: str, 
                  method: str = "semantic",
                  chunk_size: Optional[int] = None,
                  overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """Chunk text using specified method"""
        
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.overlap_size
        
        if method == "semantic":
            return self._semantic_chunking(text, chunk_size, overlap)
        elif method == "sentence":
            return self._sentence_chunking(text, chunk_size, overlap)
        elif method == "paragraph":
            return self._paragraph_chunking(text, chunk_size, overlap)
        elif method == "fixed":
            return self._fixed_chunking(text, chunk_size, overlap)
        else:
            return self._semantic_chunking(text, chunk_size, overlap)
    
    def _semantic_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk based on semantic boundaries"""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk),
                        len(chunks)
                    ))
                
                current_start = current_start + len(current_chunk) - overlap
                
                if chunks and overlap > 0:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(),
                current_start,
                len(text),
                len(chunks)
            ))
        
        return chunks
    
    def _sentence_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk based on sentence boundaries"""
        sentences = self.sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk.strip(),
                        current_start,
                        current_start + len(current_chunk),
                        len(chunks)
                    ))
                
                current_start = current_start + len(current_chunk) - overlap
                
                if chunks and overlap > 0:
                    overlap_sentences = current_chunk.split('.')[-3:]
                    overlap_text = '.'.join(overlap_sentences)
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                current_chunk.strip(),
                current_start,
                len(text),
                len(chunks)
            ))
        
        return chunks
    
    def _paragraph_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk based on paragraph boundaries"""
        paragraphs = text.split('\n\n')
        
        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para) > chunk_size:
                sub_chunks = self._fixed_chunking(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                start_idx = text.find(para)
                chunks.append(self._create_chunk_dict(
                    para,
                    start_idx,
                    start_idx + len(para),
                    len(chunks)
                ))
        
        return chunks
    
    def _fixed_chunking(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Fixed-size chunking"""
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            chunks.append(self._create_chunk_dict(
                chunk_text,
                i,
                min(i + chunk_size, len(text)),
                len(chunks)
            ))
            
            if i + chunk_size >= len(text):
                break
        
        return chunks
    
    def _create_chunk_dict(self, content: str, start_idx: int, end_idx: int, chunk_id: int) -> Dict[str, Any]:
        """Create chunk dictionary with metadata"""
        return {
            "chunk_id": chunk_id,
            "content": content,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "length": len(content),
            "hash": hashlib.md5(content.encode()).hexdigest(),
            "context": self._extract_context(content)
        }
    
    def _extract_context(self, content: str) -> str:
        """Extract brief context from chunk"""
        first_sentence = content.split('.')[0] if '.' in content else content[:100]
        return first_sentence[:200] if len(first_sentence) > 200 else first_sentence
    
    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Merge chunks back into text"""
        sorted_chunks = sorted(chunks, key=lambda x: x["chunk_id"])
        merged_text = "\n\n".join(chunk["content"] for chunk in sorted_chunks)
        
        return merged_text