import json
import re
import ast
import textwrap
from typing import List, Dict, Any, Optional, Callable
import tiktoken
import numpy as np
from .llm_interface import call_llm
from .logger_service import log
from .token_estimator import estimate_tokens

class TextAnalyzer:
    """Handles text analysis, semantic structure discovery, and intelligent chunking"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_chunk_tokens = 2000  # Safe limit for processing
        self.generated_functions = {}  # Cache for generated functions
    
    def analyze_semantic_structure(self, text: str, model: str = "gpt-4.1-2025-04-14") -> Dict[str, Any]:
        """Analyze the semantic structure of the text document"""
        
        # Take a sample of text for analysis to avoid token limits
        sample_size = min(3000, len(text))
        text_sample = text[:sample_size]
        
        # Calculate basic statistics locally
        lines = text.split('\n')
        paragraphs = text.split('\n\n')
        sentences = re.split(r'[.!?]+', text)
        
        # Identify patterns locally
        has_headers = bool(re.search(r'^#{1,6}\s+.*$|^[A-Z][A-Z\s]+$', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*•]\s+.*$|^\s*\d+\.\s+.*$', text, re.MULTILINE))
        has_tables = '|' in text and text.count('|') > 10
        
        # Estimate optimal chunk size based on text characteristics
        avg_sentence_length = len(text) / max(len(sentences), 1)
        optimal_chunk_size = min(2000, max(500, int(avg_sentence_length * 10)))
        
        # Create semantic structure without LLM call
        semantic_structure = {
            "document_type": self._detect_document_type(text_sample),
            "sections": self._detect_sections(text),
            "topics": self._extract_topics(text_sample),
            "density_pattern": self._analyze_density(text),
            "recommended_chunk_size": optimal_chunk_size,
            "natural_boundaries": self._find_natural_boundaries(text),
            "statistics": {
                "total_chars": len(text),
                "total_tokens": estimate_tokens(text[:1000], model) * (len(text) / 1000),
                "lines": len(lines),
                "paragraphs": len(paragraphs),
                "sentences": len(sentences),
                "has_headers": has_headers,
                "has_lists": has_lists,
                "has_tables": has_tables
            }
        }
        
        return semantic_structure
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content patterns"""
        if re.search(r'<[^>]+>', text):
            return "html/xml"
        elif re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text):
            return "json/structured"
        elif re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return "markdown"
        elif re.search(r'def\s+\w+\(|class\s+\w+|import\s+\w+', text):
            return "code"
        elif re.search(r'^\d{4}-\d{2}-\d{2}|^\d{2}:\d{2}', text, re.MULTILINE):
            return "log/timeline"
        else:
            return "general_text"
    
    def _detect_sections(self, text: str) -> List[Dict]:
        """Detect logical sections in the text"""
        sections = []
        
        # Look for markdown headers
        header_matches = re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        for match in header_matches:
            sections.append({
                "title": match.group(2),
                "level": len(match.group(1)),
                "position": match.start()
            })
        
        # Look for capital letter headers
        cap_matches = re.finditer(r'^([A-Z][A-Z\s]{3,})$', text, re.MULTILINE)
        for match in cap_matches:
            sections.append({
                "title": match.group(1).strip(),
                "level": 1,
                "position": match.start()
            })
        
        # Sort by position
        sections.sort(key=lambda x: x['position'])
        
        return sections[:20]  # Limit to first 20 sections
    
    def _extract_topics(self, text: str) -> List[Dict]:
        """Extract main topics from text using keyword frequency"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and get top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        topics = []
        for word, freq in sorted_words[:10]:
            topics.append({
                "name": word,
                "frequency": freq,
                "weight": freq / len(words) if words else 0
            })
        
        return topics
    
    def _analyze_density(self, text: str) -> str:
        """Analyze information density pattern"""
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        densities = []
        
        for chunk in chunks[:20]:  # Analyze first 20 chunks
            # Count entities, numbers, special chars as density indicators
            numbers = len(re.findall(r'\d+', chunk))
            capitals = len(re.findall(r'[A-Z]', chunk))
            special = len(re.findall(r'[^a-zA-Z0-9\s]', chunk))
            density = (numbers + capitals + special) / max(len(chunk), 1)
            densities.append(density)
        
        if not densities:
            return "uniform"
        
        avg_density = np.mean(densities)
        std_density = np.std(densities)
        
        if std_density < 0.01:
            return "uniform"
        elif std_density < 0.03:
            return "varied"
        else:
            return "clustered"
    
    def _find_natural_boundaries(self, text: str) -> List[str]:
        """Find natural boundary patterns in text"""
        boundaries = []
        
        # Common boundary patterns
        patterns = [
            r'\n\n+',  # Multiple newlines
            r'\n#{1,6}\s+',  # Markdown headers
            r'\n[A-Z][A-Z\s]+\n',  # Capital headers
            r'\n={3,}\n',  # Horizontal rules
            r'\n-{3,}\n',
            r'\n\*{3,}\n',
            r'\.\s+[A-Z]',  # Sentence boundaries with capital start
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                boundaries.append(pattern)
        
        return boundaries
    
    def generate_chunking_strategy(self, text: str, semantic_structure: Dict, model: str) -> str:
        """Generate Python code for intelligent chunking based on semantic structure"""
        
        # Create a concise prompt focusing on code generation
        prompt = f"""Generate a Python function to chunk text intelligently.

Requirements:
- Function signature: def chunk_text(text: str) -> List[str]
- Target chunk size: {semantic_structure.get('recommended_chunk_size', 1000)} characters
- Use these boundary patterns: {semantic_structure.get('natural_boundaries', [])}
- Include 10-20% overlap between chunks
- Preserve complete sentences
- Document type: {semantic_structure.get('document_type', 'general')}

Return ONLY executable Python code, no explanations:
```python
def chunk_text(text: str) -> List[str]:
    # Your implementation here
    pass
```"""
        
        response = call_llm(prompt, model)
        
        # Extract code from response
        code = self._extract_code_from_response(response)
        
        # Validate and fix common issues
        code = self._validate_and_fix_code(code)
        
        # Store for potential reuse
        self.generated_functions['chunk_text'] = code
        
        return code
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code block
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Try to find function definition
        func_match = re.search(r'(def chunk_text.*?)(?=\n\n|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1)
        
        # If no code found, return a default implementation
        return self._get_default_chunking_code()
    
    def _validate_and_fix_code(self, code: str) -> str:
        """Validate generated code and fix common issues"""
        try:
            # Check if code can be compiled
            compile(code, '<string>', 'exec')
            
            # Ensure required imports are present
            if 'List' not in code and 'def chunk_text(text: str) -> List[str]:' in code:
                code = "from typing import List\n" + code
            
            if 're.' in code and 'import re' not in code:
                code = "import re\n" + code
            
            return code
        
        except SyntaxError:
            # Return default if validation fails
            return self._get_default_chunking_code()
    
    def _get_default_chunking_code(self) -> str:
        """Get default chunking implementation"""
        return '''
from typing import List
import re

def chunk_text(text: str) -> List[str]:
    """Intelligent text chunking with overlap"""
    if not text:
        return []
    
    chunk_size = 1000
    overlap_size = 150
    
    # Split on natural boundaries
    paragraphs = text.split('\\n\\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\\n\\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\\n\\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and len(chunks[i-1]) > overlap_size:
            # Add end of previous chunk as overlap
            overlap = chunks[i-1][-overlap_size:]
            chunk = overlap + " " + chunk
        final_chunks.append(chunk)
    
    return final_chunks
'''
    
    def execute_chunking(self, text: str, chunking_code: str) -> List[str]:
        """Execute the generated chunking code on the text"""
        
        try:
            # Create a safe execution environment
            exec_namespace = {
                '__builtins__': {
                    'len': len,
                    'min': min,
                    'max': max,
                    'str': str,
                    'list': list,
                    'range': range,
                    'enumerate': enumerate,
                    'isinstance': isinstance,
                    'bool': bool,
                    'int': int
                },
                'List': List,
                'textwrap': textwrap,
                're': re
            }
            
            # Execute the code
            exec(chunking_code, exec_namespace)
            
            # Get the chunk_text function
            chunk_function = exec_namespace.get('chunk_text')
            
            if chunk_function and callable(chunk_function):
                chunks = chunk_function(text)
                
                # Validate chunks
                if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks):
                    return chunks
            
            # Fallback if execution fails
            return self._simple_chunk(text)
        
        except Exception as e:
            log("session", "chunking_error", f"Error executing chunking code: {e}", "ERROR")
            return self._simple_chunk(text)
    
    def _simple_chunk(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple fallback chunking method"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + chunk_size * 0.8:
                        end = last_punct + len(punct)
                        break
            
            chunks.append(text[start:end].strip())
            
            # Move start with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def get_chunk_metrics(self, text: str, chunking_code: str) -> Dict[str, Any]:
        """Get metrics about the chunking strategy"""
        
        try:
            chunks = self.execute_chunking(text, chunking_code)
            
            if not chunks:
                return {
                    'chunk_count': 0,
                    'avg_size': 0,
                    'min_size': 0,
                    'max_size': 0,
                    'overlap': 0
                }
            
            sizes = [len(chunk) for chunk in chunks]
            
            # Estimate overlap
            total_chunk_size = sum(sizes)
            original_size = len(text)
            overlap_ratio = ((total_chunk_size - original_size) / original_size * 100) if original_size > 0 else 0
            
            return {
                'chunk_count': len(chunks),
                'avg_size': sum(sizes) / len(sizes),
                'min_size': min(sizes),
                'max_size': max(sizes),
                'overlap': max(0, overlap_ratio)  # Ensure non-negative
            }
        
        except Exception as e:
            print(f"Error getting chunk metrics: {e}")
            return {
                'chunk_count': 'Error',
                'avg_size': 0,
                'min_size': 0,
                'max_size': 0,
                'overlap': 0
            }

# Initialize the global instance
text_analyzer = TextAnalyzer()