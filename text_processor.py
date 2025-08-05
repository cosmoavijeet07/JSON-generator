import re
import nltk
import spacy
from typing import List, Dict, Tuple, Any
import streamlit as st
from config import Config

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    st.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")

class TextProcessor:
    def __init__(self):
        self.config = Config()
        self.ensure_nltk_data()
    
    def ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep structure
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\\n\r\t]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def detect_structure(self, text: str) -> Dict[str, Any]:
        """Detect document structure"""
        structure = {
            'has_headers': False,
            'has_lists': False,
            'has_tables': False,
            'paragraph_count': 0,
            'estimated_sections': 0
        }
        
        # Detect headers (lines that start with #, are all caps, or are short and followed by content)
        lines = text.split('\n')
        potential_headers = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for markdown headers
            if line.startswith('#'):
                structure['has_headers'] = True
                potential_headers += 1
            
            # Check for all caps headers
            elif len(line) < 100 and line.isupper() and len(line.split()) > 1:
                potential_headers += 1
            
            # Check for lists
            elif re.match(r'^\s*[-\*\+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                structure['has_lists'] = True
            
            # Check for table-like structures
            elif '|' in line and line.count('|') > 2:
                structure['has_tables'] = True
        
        structure['estimated_sections'] = max(1, potential_headers)
        structure['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        return structure
    
    def extract_table_of_contents(self, text: str) -> List[Dict[str, Any]]:
        """Extract table of contents if present"""
        toc = []
        lines = text.split('\n')
        
        in_toc_section = False
        toc_patterns = [
            r'table\s+of\s+contents',
            r'contents',
            r'index',
            r'outline'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if we're entering a TOC section
            if any(re.search(pattern, line_lower) for pattern in toc_patterns):
                in_toc_section = True
                continue
            
            # If in TOC, look for entries
            if in_toc_section:
                # Stop if we hit a clear section break
                if len(line.strip()) == 0:
                    continue
                elif line.startswith('#') or (len(line) > 100):
                    break
                
                # Extract TOC entry
                match = re.match(r'(.+?)\s*\.{2,}\s*(\d+)', line)
                if match:
                    toc.append({
                        'title': match.group(1).strip(),
                        'page': int(match.group(2)),
                        'line_number': i
                    })
        
        return toc
    
    def semantic_chunk(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """Perform semantic chunking using NLP"""
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if overlap is None:
            overlap = self.config.CHUNK_OVERLAP
        
        chunks = []
        
        if nlp:
            # Use spaCy for sentence segmentation
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback to NLTK
            sentences = nltk.sent_tokenize(text)
        
        current_chunk = ""
        current_length = 0
        chunk_num = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append({
                    'chunk_id': chunk_num,
                    'text': current_chunk.strip(),
                    'start_sentence': i - len(current_chunk.split('. ')),
                    'end_sentence': i - 1,
                    'word_count': current_length
                })
                
                # Start new chunk with overlap
                overlap_text = '. '.join(current_chunk.split('. ')[-overlap//20:])
                current_chunk = overlap_text + '. ' + sentence if overlap_text else sentence
                current_length = len(current_chunk.split())
                chunk_num += 1
            else:
                current_chunk += (' ' if current_chunk else '') + sentence
                current_length += sentence_length
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_id': chunk_num,
                'text': current_chunk.strip(),
                'start_sentence': len(sentences) - len(current_chunk.split('. ')),
                'end_sentence': len(sentences) - 1,
                'word_count': current_length
            })
        
        return chunks
    
    def chunk_by_structure(self, text: str, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text based on detected structure"""
        if structure['has_headers']:
            return self._chunk_by_headers(text)
        elif structure['paragraph_count'] > 20:
            return self._chunk_by_paragraphs(text)
        else:
            return self.semantic_chunk(text)
    
    def _chunk_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by headers"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_header = "Introduction"
        chunk_id = 0
        
        for line in lines:
            # Check if line is a header
            if (line.startswith('#') or 
                (len(line) < 100 and line.isupper() and len(line.split()) > 1)):
                
                # Save previous chunk
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if chunk_text.strip():
                        chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text.strip(),
                            'header': current_header,
                            'word_count': len(chunk_text.split())
                        })
                        chunk_id += 1
                
                # Start new chunk
                current_header = line.strip('#').strip()
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text.strip(),
                    'header': current_header,
                    'word_count': len(chunk_text.split())
                })
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            para_word_count = len(paragraph.split())
            
            if current_word_count + para_word_count > self.config.CHUNK_SIZE and current_chunk:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': '\n\n'.join(current_chunk),
                    'paragraph_count': len(current_chunk),
                    'word_count': current_word_count
                })
                
                current_chunk = [paragraph]
                current_word_count = para_word_count
                chunk_id += 1
            else:
                current_chunk.append(paragraph)
                current_word_count += para_word_count
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': '\n\n'.join(current_chunk),
                'paragraph_count': len(current_chunk),
                'word_count': current_word_count
            })
        
        return chunks
