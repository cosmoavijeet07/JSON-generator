import openai
import anthropic
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import time

load_dotenv()

class LLMInterface:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        self.models = {
            "gpt-4.1-2025-04-14": {"provider": "openai", "max_tokens": 128000},
            "o4-mini-2025-04-16": {"provider": "openai", "max_tokens": 200000},
            "claude-sonnet-4-20250514": {"provider": "anthropic", "max_tokens": 200000}
        }
    
    def call_llm(self, 
                 prompt: str, 
                 model:  str = "gpt-4.1-2025-04-14",
                 temperature: float = 0.1,
                 max_retries: int = 3,
                 system_prompt: Optional[str] = None) -> str:
        """Call LLM with retry logic and error handling"""
        
        model_info = self.models.get(model, self.models["gpt-4.1-2025-04-14"])
        
        for attempt in range(max_retries):
            try:
                if model_info["provider"] == "openai":
                    return self._call_openai(prompt, model, temperature, system_prompt)
                elif model_info["provider"] == "anthropic":
                    return self._call_anthropic(prompt, model, temperature, system_prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Max retries exceeded")
    
    def _call_openai(self, prompt: str, model: str, temperature: float, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, model: str, temperature: float, system_prompt: Optional[str]) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        response = self.anthropic_client.messages.create(
            model=model,
            messages=messages,
            system=system_prompt if system_prompt else "",
            temperature=temperature,
            max_tokens=4096
        )
        return response.content[0].text
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost based on token count and model"""
        # Add pricing logic here
        pricing = {
            "gpt-4.1-2025-04-14": 0.01,
            "o4-mini-2025-04-16": 0.002,
            "claude-sonnet-4-20250514": 0.003
        }
        return tokens * pricing.get(model, 0.01) / 1000