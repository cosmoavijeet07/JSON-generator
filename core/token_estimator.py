import tiktoken
from typing import Dict, Any, Optional, List, Tuple
import json
import re

# Token limits for different models
MODEL_TOKEN_LIMITS = {
    # OpenAI models
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4.1-2025-04-14": 128000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "o4-mini-2025-04-16": 128000,
    "o3-2025-04-16": 128000,
    
    # Claude models
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-instant": 100000
}

# Token pricing (per 1K tokens)
MODEL_PRICING = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4.1-2025-04-14": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "o4-mini-2025-04-16": {"prompt": 0.0005, "completion": 0.0015},
    "o3-2025-04-16": {"prompt": 0.015, "completion": 0.06},
    "claude-sonnet-4-20250514": {"prompt": 0.003, "completion": 0.015}
}

class TokenEstimator:
    """Advanced token estimation and management"""
    
    def __init__(self):
        """Initialize token estimator with encodings"""
        self.encodings = {}
        self._load_encodings()
    
    def _load_encodings(self):
        """Load tiktoken encodings for different models"""
        try:
            # GPT-4 and similar models
            self.encodings["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            
            # GPT-3.5 models
            self.encodings["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Claude models use similar tokenization (approximation)
            self.encodings["claude"] = tiktoken.get_encoding("cl100k_base")
            
        except Exception as e:
            print(f"Error loading encodings: {e}")
            # Fallback to cl100k_base for all
            default_encoding = tiktoken.get_encoding("cl100k_base")
            self.encodings = {
                "gpt-4": default_encoding,
                "gpt-3.5-turbo": default_encoding,
                "claude": default_encoding
            }
    
    def estimate_tokens(
        self,
        text: str,
        model: str = "gpt-4"
    ) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Text to estimate
            model: Model name
        
        Returns:
            Estimated token count
        """
        # Get appropriate encoding
        encoding = self._get_encoding(model)
        
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                print(f"Error encoding text: {e}")
        
        # Fallback to character-based estimation
        return self._character_based_estimation(text)
    
    def estimate_json_tokens(
        self,
        json_data: Dict[str, Any],
        model: str = "gpt-4"
    ) -> int:
        """
        Estimate tokens for JSON data
        
        Args:
            json_data: JSON data
            model: Model name
        
        Returns:
            Estimated token count
        """
        json_str = json.dumps(json_data, separators=(',', ':'))
        return self.estimate_tokens(json_str, model)
    
    def estimate_prompt_tokens(
        self,
        system_prompt: Optional[str],
        user_prompt: str,
        model: str = "gpt-4"
    ) -> int:
        """
        Estimate tokens for a complete prompt
        
        Args:
            system_prompt: System prompt (optional)
            user_prompt: User prompt
            model: Model name
        
        Returns:
            Estimated token count
        """
        total_tokens = 0
        
        # Add system prompt tokens
        if system_prompt:
            total_tokens += self.estimate_tokens(system_prompt, model)
            total_tokens += 4  # Overhead for message structure
        
        # Add user prompt tokens
        total_tokens += self.estimate_tokens(user_prompt, model)
        total_tokens += 4  # Overhead for message structure
        
        return total_tokens
    
    def get_model_limit(self, model: str) -> int:
        """
        Get token limit for a model
        
        Args:
            model: Model name
        
        Returns:
            Token limit
        """
        # Check exact match first
        if model in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model]
        
        # Check partial matches
        for key, limit in MODEL_TOKEN_LIMITS.items():
            if key in model or model in key:
                return limit
        
        # Default limits based on provider
        if "claude" in model.lower():
            return 100000  # Conservative Claude limit
        elif "gpt-4" in model.lower():
            return 8192  # Conservative GPT-4 limit
        elif "gpt-3.5" in model.lower():
            return 4096  # Conservative GPT-3.5 limit
        
        # Default conservative limit
        return 4096
    
    def calculate_available_tokens(
        self,
        model: str,
        prompt_tokens: int,
        desired_output_tokens: int = 1000
    ) -> int:
        """
        Calculate available tokens for completion
        
        Args:
            model: Model name
            prompt_tokens: Tokens used in prompt
            desired_output_tokens: Desired output size
        
        Returns:
            Available tokens for completion
        """
        model_limit = self.get_model_limit(model)
        available = model_limit - prompt_tokens - desired_output_tokens
        
        return max(0, available)
    
    def split_text_by_tokens(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-4",
        overlap_tokens: int = 100
    ) -> List[str]:
        """
        Split text into chunks by token count
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            model: Model name
            overlap_tokens: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        encoding = self._get_encoding(model)
        
        if not encoding:
            # Fallback to character-based splitting
            return self._split_by_characters(text, max_tokens * 4, overlap_tokens * 4)
        
        try:
            # Encode entire text
            tokens = encoding.encode(text)
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                # Get chunk tokens
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                
                # Decode chunk
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                # Move start with overlap
                if end < len(tokens):
                    start = end - overlap_tokens
                else:
                    break
            
            return chunks
        
        except Exception as e:
            print(f"Error splitting text: {e}")
            return self._split_by_characters(text, max_tokens * 4, overlap_tokens * 4)
    
    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        model: str = "gpt-4",
        from_end: bool = False
    ) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            model: Model name
            from_end: Whether to truncate from end
        
        Returns:
            Truncated text
        """
        current_tokens = self.estimate_tokens(text, model)
        
        if current_tokens <= max_tokens:
            return text
        
        encoding = self._get_encoding(model)
        
        if encoding:
            try:
                tokens = encoding.encode(text)
                
                if from_end:
                    truncated_tokens = tokens[-max_tokens:]
                else:
                    truncated_tokens = tokens[:max_tokens]
                
                return encoding.decode(truncated_tokens)
            
            except Exception as e:
                print(f"Error truncating text: {e}")
        
        # Fallback to character-based truncation
        estimated_chars = (max_tokens / current_tokens) * len(text)
        estimated_chars = int(estimated_chars * 0.95)  # Conservative estimate
        
        if from_end:
            return text[-estimated_chars:]
        else:
            return text[:estimated_chars]
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """
        Estimate cost for token usage
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
        
        Returns:
            Estimated cost in USD
        """
        # Get pricing for model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING.get("gpt-4"))
        
        # Calculate cost
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def optimize_prompt(
        self,
        prompt: str,
        model: str,
        target_tokens: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Optimize prompt to fit within token limits
        
        Args:
            prompt: Original prompt
            model: Model name
            target_tokens: Target token count
        
        Returns:
            Tuple of (optimized_prompt, token_count)
        """
        if not target_tokens:
            target_tokens = self.get_model_limit(model) // 2  # Use half of limit
        
        current_tokens = self.estimate_tokens(prompt, model)
        
        if current_tokens <= target_tokens:
            return prompt, current_tokens
        
        # Optimization strategies
        optimized = prompt
        
        # 1. Remove extra whitespace
        optimized = re.sub(r'\s+', ' ', optimized)
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # 2. Remove comments
        optimized = re.sub(r'#.*?\n', '\n', optimized)
        optimized = re.sub(r'//.*?\n', '\n', optimized)
        optimized = re.sub(r'/\*.*?\*/', '', optimized, flags=re.DOTALL)
        
        # 3. Shorten repetitive content
        optimized = self._compress_repetitive_content(optimized)
        
        # Check tokens again
        current_tokens = self.estimate_tokens(optimized, model)
        
        if current_tokens <= target_tokens:
            return optimized, current_tokens
        
        # 4. Truncate if still too long
        optimized = self.truncate_to_tokens(optimized, target_tokens, model)
        final_tokens = self.estimate_tokens(optimized, model)
        
        return optimized, final_tokens
    
    def _get_encoding(self, model: str):
        """Get appropriate encoding for model"""
        if "claude" in model.lower():
            return self.encodings.get("claude")
        elif "gpt-4" in model.lower():
            return self.encodings.get("gpt-4")
        elif "gpt-3.5" in model.lower():
            return self.encodings.get("gpt-3.5-turbo")
        else:
            # Default to GPT-4 encoding
            return self.encodings.get("gpt-4")
    
    def _character_based_estimation(self, text: str) -> int:
        """Fallback character-based token estimation"""
        # Rough estimates based on language patterns
        words = len(text.split())
        chars = len(text)
        
        # Average: ~4 characters per token for English
        # Adjust based on content type
        if re.search(r'[\u4e00-\u9fff]', text):
            # Contains Chinese characters (more tokens)
            return chars // 2
        elif re.search(r'[а-яА-Я]', text):
            # Contains Cyrillic (slightly more tokens)
            return chars // 3
        else:
            # Default English/Latin
            return max(words, chars // 4)
    
    def _split_by_characters(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int
    ) -> List[str]:
        """Split text by character count"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary > start + max_chars * 0.8:
                        end = last_boundary + len(boundary)
                        break
            
            chunks.append(text[start:end])
            
            # Move with overlap
            if end < len(text):
                start = end - overlap_chars
            else:
                break
        
        return chunks
    
    def _compress_repetitive_content(self, text: str) -> str:
        """Compress repetitive content in text"""
        # Remove duplicate lines
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                seen.add(line_stripped)
                unique_lines.append(line)
            elif not line_stripped:
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def get_token_statistics(
        self,
        texts: List[str],
        model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Get token statistics for multiple texts
        
        Args:
            texts: List of texts
            model: Model name
        
        Returns:
            Statistics dictionary
        """
        token_counts = [self.estimate_tokens(text, model) for text in texts]
        
        if not token_counts:
            return {
                "total": 0,
                "average": 0,
                "min": 0,
                "max": 0,
                "texts": 0
            }
        
        return {
            "total": sum(token_counts),
            "average": sum(token_counts) / len(token_counts),
            "min": min(token_counts),
            "max": max(token_counts),
            "texts": len(texts),
            "distribution": token_counts
        }

# Initialize global instance
token_estimator_instance = TokenEstimator()

# Convenience functions
def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text
    
    Args:
        text: Text to estimate
        model: Model name
    
    Returns:
        Estimated token count
    """
    return token_estimator_instance.estimate_tokens(text, model)

def get_model_limit(model: str) -> int:
    """
    Get token limit for a model
    
    Args:
        model: Model name
    
    Returns:
        Token limit
    """
    return token_estimator_instance.get_model_limit(model)

def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        model: Model name
    
    Returns:
        Truncated text
    """
    return token_estimator_instance.truncate_to_tokens(text, max_tokens, model)