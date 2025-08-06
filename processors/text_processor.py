import re
from pathlib import Path
from typing import Dict, Any, List

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    import spacy
except ImportError:  # Allow the rest of the class to function without spaCy
    spacy = None


class TextProcessor:
    """Analyze and preprocess natural-language documents."""

    # ----------------------- initialization helpers ----------------------- #
    def __init__(self) -> None:
        self._ensure_nltk_resources()
        self.nlp = self._load_spacy_model()

    @staticmethod
    def _ensure_nltk_resources() -> None:
        """Download only the NLTK data actually required."""
        needed = ["punkt", "stopwords", "wordnet"]
        for pkg in needed:
            try:
                nltk.data.find(f"tokenizers/{pkg}")  # 'punkt'
            except LookupError:
                nltk.download(pkg, quiet=True)

    @staticmethod
    def _load_spacy_model():
        """Load spaCy small English model if available."""
        if spacy is None:
            return None
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Model not present; tell the caller how to install but keep running.
            print("spaCy model 'en_core_web_sm' not found. "
                  "Install with: python -m spacy download en_core_web_sm")
            return None

    # ----------------------------- public API ----------------------------- #
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Return a dict summarizing structure and content of *text*."""
        cleaned = self.preprocess_text(text)
        return {
            "total_length": len(cleaned),
            "num_words": len(word_tokenize(cleaned)),
            "num_sentences": len(sent_tokenize(cleaned)),
            "sections": self._identify_sections(cleaned),
            "entities": self._extract_entities(cleaned),
            "structure_type": self._identify_structure_type(cleaned),
            "recommended_chunk_size": self._recommend_chunk_size(cleaned),
        }

    def preprocess_text(self, text: str) -> str:
        """Trim whitespace, fix quotes, strip control chars."""
        text = re.sub(r"\s+", " ", text)
        text = (
            text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
        )
        text = ''.join(ch for ch in text if ord(ch) >= 32 or ch == '\n')
        return text.strip()

    # ---------------------------- internals ------------------------------ #
    @staticmethod
    def _identify_sections(text: str) -> List[Dict[str, Any]]:
        """Detect probable section headings by simple heuristics."""
        lines = text.split('\n')
        sections: List[Dict[str, Any]] = []
        char_pos = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                char_pos += len(line) + 1
                continue
            heading_like = (
                len(stripped) < 100
                and (
                    stripped.isupper()
                    or re.match(r"^\d+\.\s+\w+", stripped)
                    or re.match(r"^[A-Z][^.!?]*$", stripped)
                )
            )
            if heading_like:
                sections.append({
                    "line_number": idx,
                    "title": stripped,
                    "start_position": char_pos,
                })
            char_pos += len(line) + 1
        return sections

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Return named-entity spans if spaCy is available, else empty list."""
        if not self.nlp:  # spaCy missing
            return []
        doc = self.nlp(text[:1_000_000])  # hard cap for speed
        return [
            {"text": ent.text, "type": ent.label_,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    @staticmethod
    def _identify_structure_type(text: str) -> str:
        """Classify rough document layout."""
        if re.search(r"\{[\s\S]*\}", text):
            return "json_like"
        if re.search(r"<[^>]+>", text):
            return "xml_like"
        if re.search(r"\|.*\|.*\|", text):
            return "table_like"
        if len(re.findall(r"\n\d+\.", text)) > 3:
            return "numbered_list"
        if len(re.findall(r"\n[-*]", text)) > 3:
            return "bullet_list"
        return "prose"

    @staticmethod
    def _recommend_chunk_size(text: str) -> int:
        """Suggest a chunk size for downstream processing."""
        length = len(text)
        if length < 5000:
            return length
        if length < 20000:
            return 2000
        if length < 100000:
            return 5000
        return 10000