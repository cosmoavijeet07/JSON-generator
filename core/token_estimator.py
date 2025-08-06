import tiktoken
from typing import Dict, Any, List
import json

class TokenEstimator:
    def __init__(self):
        self.encoders = {}
        self.model_limits = {
            "gpt-4.1-2025-04-14": 128000,
            "o4-mini-2025-04-16": 200000,
            "claude-sonnet-4-20250514": 200000
        }
    
    def get_encoder(self, model: str):
        """Get or create encoder for model"""
        if model not in self.encoders:
            try:
                if "gpt" in model:
                    self.encoders[model] = tiktoken.encoding_for_model(model)
                else:
                    # Use cl100k_base as default for non-OpenAI models
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
            except:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model]
    
    def estimate_tokens(self, text: str, model: str = "gpt-4.1-2025-04-14") -> int:
        """Estimate token count for text"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))
    
    def estimate_json_tokens(self, data: Dict[str, Any], model: str = "gpt-4.1-2025-04-14") -> int:
        """Estimate tokens for JSON data"""
        json_str = json.dumps(data, indent=2)
        return self.estimate_tokens(json_str, model)
    
    def can_fit_in_context(self, text: str, model: str = "gpt-4.1-2025-04-14", buffer: int = 1000) -> bool:
        """Check if text fits in model's context window"""
        tokens = self.estimate_tokens(text, model)
        limit = self.model_limits.get(model, 8192)
        return tokens + buffer < limit
    
    def estimate_chunks_needed(self, text: str, model: str = "gpt-4.1-2025-04-14", chunk_size: int = 2000) -> int:
        """Estimate number of chunks needed"""
        tokens = self.estimate_tokens(text, model)
        return (tokens + chunk_size - 1) // chunk_size
    
    def analyze_content(self, text: str, schema: Dict[str, Any], model: str = "gpt-4.1-2025-04-14") -> Dict[str, Any]:
        """Analyze content and provide token metrics"""
        text_tokens = self.estimate_tokens(text, model)
        schema_tokens = self.estimate_json_tokens(schema, model)
        total_tokens = text_tokens + schema_tokens
        
        return {
            "text_tokens": text_tokens,
            "schema_tokens": schema_tokens,
            "total_tokens": total_tokens,
            "model_limit": self.model_limits.get(model, 8192),
            "utilization_percentage": (total_tokens / self.model_limits.get(model, 8192)) * 100,
            "fits_in_context": self.can_fit_in_context(text + json.dumps(schema), model),
            "estimated_chunks": self.estimate_chunks_needed(text, model),
            "recommended_pipeline": "simple" if total_tokens < self.model_limits.get(model, 8192) * 0.7 else "extensive"
        }