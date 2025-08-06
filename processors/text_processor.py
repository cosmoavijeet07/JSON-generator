import re
from typing import Dict, Any, List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

class TextProcessor:
    class TextProcessor:
        def __init__(self):
            try:
                # Ensure nltk punkt tokenizer is downloaded to correct path
                nltk.data.path.append("/tmp/nltk_data")
                nltk.download("punkt", download_dir="/tmp/nltk_data", quiet=True)

                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"[TextProcessor Init Error] {e}")
                self.nlp = None
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and content"""
        analysis = {
            "total_length": len(text),
            "num_words": len(word_tokenize(text)),
            "num_sentences": len(sent_tokenize(text)),
            "sections": self._identify_sections(text),
            "entities": self._extract_entities(text) if self.nlp else [],
            "structure_type": self._identify_structure_type(text),
            "recommended_chunk_size": self._recommend_chunk_size(text)
        }
        
        return analysis
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections"""
        sections = []
        
        # Look for headers (lines that are shorter and possibly capitalized)
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Heuristics for section detection
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^\d+\.?\s+\w+', line) or
                 re.match(r'^[A-Z][^.!?]*', line))):
                
                sections.append({
                    "line_number": i,
                    "title": line,
                    "start_position": sum(len(l) + 1 for l in lines[:i])
                })
        
        return sections
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text[:1000000])  # Limit for performance
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def _identify_structure_type(self, text: str) -> str:
        """Identify document structure type"""
        # Simple heuristics
        if re.search(r'\{[\s\S]*\}', text):
            return "json_like"
        elif re.search(r'<[^>]+>', text):
            return "xml_like"
        elif re.search(r'\|.*\|.*\|', text):
            return "table_like"
        elif len(re.findall(r'\n\d+\.', text)) > 3:
            return "numbered_list"
        elif len(re.findall(r'\n[-*]', text)) > 3:
            return "bullet_list"
        else:
            return "prose"
    
    def _recommend_chunk_size(self, text: str) -> int:
        """Recommend optimal chunk size"""
        doc_length = len(text)
        
        if doc_length < 5000:
            return doc_length  # No chunking needed
        elif doc_length < 20000:
            return 2000
        elif doc_length < 100000:
            return 5000
        else:
            return 10000
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for extraction"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text.strip()