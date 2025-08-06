import re
from typing import Dict, Any, List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy


class TextProcessor:
    def __init__(self):
        """
        Initialise NLP resources without ever requesting the non-existent
        ‘punkt_tab’ model.
        """
        try:
            # Download only what is really available on the NLTK server
            for pkg in ("punkt", "stopwords", "wordnet"):
                try:
                    nltk.data.find(f"tokenizers/{pkg}")
                except LookupError:
                    nltk.download(pkg, quiet=True)

            # Load spaCy model if present
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
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if (len(line) < 100 and
                (line.isupper() or
                 re.match(r'^\d+\.?\s+\w+', line) or
                 re.match(r'^[A-Z][^.!?]*$', line))):
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
        doc = self.nlp(text[:1_000_000])  # Limit for performance
        return [{
            "text": ent.text,
            "type": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        } for ent in doc.ents]

    def _identify_structure_type(self, text: str) -> str:
        """Identify document structure type"""
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
        length = len(text)
        if length < 5_000:
            return length
        elif length < 20_000:
            return 2_000
        elif length < 100_000:
            return 5_000
        else:
            return 10_000

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for extraction"""
        text = re.sub(r'\s+', ' ', text)
        text = (text.replace("“", '"').replace("”", '"')
                     .replace("‘", "'").replace("’", "'"))
        text = ''.join(c for c in text if ord(c) >= 32 or c == '\n')
        return text.strip()
