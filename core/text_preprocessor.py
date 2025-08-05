import re
import nltk
from typing import List, Dict, Tuple
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import importlib.util

class TextPreprocessor:
    def __init__(self):
        model_name = "en_core_web_sm"
        if not self._is_spacy_model_installed(model_name):
            import spacy.cli
            print(f"⚠️ Auto-downloading spaCy model '{model_name}'...")
            spacy.cli.download(model_name)

        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            raise RuntimeError(f"❌ Could not load spaCy model '{model_name}': {e}")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _is_spacy_model_installed(self, model_name: str) -> bool:
        """Check if spaCy model is installed."""
        return importlib.util.find_spec(model_name) is not None
    
    def detect_document_structure(self, text: str) -> Dict:
        """Detect headers, sections, lists, tables in the document"""
        structure = {
            'headers': [],
            'sections': [],
            'lists': [],
            'tables': [],
            'paragraphs': []
        }
        
        lines = text.split('\n')
        current_section = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect headers (simple heuristics)
            if self._is_header(line):
                structure['headers'].append({'line': i, 'text': line, 'level': self._get_header_level(line)})
                current_section = line
            
            # Detect lists
            elif self._is_list_item(line):
                if not structure['lists'] or structure['lists'][-1]['end'] < i-2:
                    structure['lists'].append({'start': i, 'end': i, 'items': [line]})
                else:
                    structure['lists'][-1]['end'] = i
                    structure['lists'][-1]['items'].append(line)
            
            # Detect tables (basic)
            elif self._is_table_row(line):
                if not structure['tables'] or structure['tables'][-1]['end'] < i-2:
                    structure['tables'].append({'start': i, 'end': i, 'rows': [line]})
                else:
                    structure['tables'][-1]['end'] = i
                    structure['tables'][-1]['rows'].append(line)
            
            # Regular paragraphs
            else:
                structure['paragraphs'].append({'line': i, 'text': line, 'section': current_section})
        
        return structure
    
    def semantic_chunking(self, text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
        """Split text into semantically coherent chunks"""
        sentences = self._split_into_sentences(text)
        embeddings = self.sentence_model.encode(sentences)
        
        # Cluster sentences by semantic similarity
        n_clusters = max(1, len(sentences) // 10)
        if len(sentences) > 1:
            kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        else:
            clusters = [0]
        
        # Group sentences by clusters and size constraints
        chunks = []
        current_chunk = {"text": "", "sentences": [], "cluster": -1, "start_idx": 0}
        
        for i, (sentence, cluster) in enumerate(zip(sentences, clusters)):
            # Check if we should start a new chunk
            if (len(current_chunk["text"]) + len(sentence) > max_chunk_size and current_chunk["text"]) or \
               (current_chunk["cluster"] != -1 and cluster != current_chunk["cluster"] and len(current_chunk["text"]) > max_chunk_size // 2):
                
                if current_chunk["text"]:
                    chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk["sentences"][-overlap//100:] if current_chunk["sentences"] else []
                current_chunk = {
                    "text": " ".join(overlap_sentences) + " " + sentence if overlap_sentences else sentence,
                    "sentences": overlap_sentences + [sentence],
                    "cluster": cluster,
                    "start_idx": max(0, i - len(overlap_sentences))
                }
            else:
                current_chunk["text"] += " " + sentence
                current_chunk["sentences"].append(sentence)
                if current_chunk["cluster"] == -1:
                    current_chunk["cluster"] = cluster
        
        if current_chunk["text"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def classify_content_type(self, text: str) -> str:
        """Classify the type of content in the text chunk"""
        text_lower = text.lower()
        
        # Personal information indicators
        if any(word in text_lower for word in ['name', 'email', 'phone', 'address', 'contact']):
            return 'personal_info'
        
        # Date/time information
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', text):
            return 'temporal'
        
        # Financial information
        if any(word in text_lower for word in ['$', 'price', 'cost', 'amount', 'payment', 'invoice']):
            return 'financial'
        
        # Technical specifications
        if any(word in text_lower for word in ['specification', 'technical', 'model', 'version']):
            return 'technical'
        
        return 'general'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _is_header(self, line: str) -> bool:
        """Simple heuristic to detect headers"""
        return (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^#+\s', line) or
                 re.match(r'^\d+\.?\s', line) or
                 (not line.endswith('.') and len(line.split()) < 10)))
    
    def _get_header_level(self, line: str) -> int:
        """Determine header level"""
        if re.match(r'^#+\s', line):
            return len(re.match(r'^#+', line).group())
        elif line.isupper():
            return 1
        elif re.match(r'^\d+\.?\s', line):
            return 2
        return 3
    
    def _is_list_item(self, line: str) -> bool:
        """Detect list items"""
        return bool(re.match(r'^[\s]*[-*•]\s|^\s*\d+\.?\s', line))
    
    def _is_table_row(self, line: str) -> bool:
        """Detect table rows"""
        return '|' in line or '\t' in line or len(re.findall(r'\s{2,}', line)) >= 2

