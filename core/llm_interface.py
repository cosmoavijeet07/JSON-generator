import openai
import os
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

# Initialize API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize Anthropic client if API key is available
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

# Models that support temperature customization
SUPPORTS_TEMPERATURE = {"gpt-4", "gpt-4o", "gpt-4.1-2025-04-14"}
CLAUDE_MODELS = {"claude-sonnet-4-20250514"}

def call_llm(prompt: str, model: str = "gpt-4.1-2025-04-14", temperature: float = None):
    """
    Call the appropriate LLM based on model selection
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model identifier
        temperature: Optional temperature override
    
    Returns:
        The model's response as a string
    """
    
    if model in CLAUDE_MODELS:
        return _call_claude(prompt, model, temperature)
    else:
        return _call_openai(prompt, model, temperature)

def _call_openai(prompt: str, model: str, temperature: float = None):
    """Call OpenAI models"""
    
    model_base = model.split(":")[-1] if ":" in model else model
    try:
        # Determine temperature
        if temperature is not None and model_base in SUPPORTS_TEMPERATURE:
            use_temp = temperature
            response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=use_temp,
            max_tokens=4096  # Increased for larger outputs
        )
            return response.choices[0].message.content
        elif model_base in SUPPORTS_TEMPERATURE:
            use_temp = 0.1
            response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=use_temp,
            max_tokens=4096  # Increased for larger outputs
        )
            return response.choices[0].message.content
        else:
            use_temp = 1.0
            response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
            return response.choices[0].message.content
    
    
        
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fallback or retry logic
        if "maximum context length" in str(e).lower():
            # Truncate prompt and retry
            truncated_prompt = prompt[:8000] + "\n\n[Truncated due to length...]"
            return _call_openai(truncated_prompt, model, temperature)
        raise e

def _call_claude(prompt: str, model: str, temperature: float = None):
    """Call Anthropic Claude models"""
    
    if not anthropic_client:
        raise ValueError("Anthropic API key not configured. Please set ANTHROPIC_API_KEY in .env file")
    
    # Claude uses temperature range 0-1, default 0.7
    use_temp = temperature if temperature is not None else 0.1
    
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=use_temp,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        # Handle Claude-specific errors
        if "max_tokens" in str(e).lower():
            # Retry with smaller max_tokens
            return _call_claude_with_reduced_tokens(prompt, model, use_temp)
        raise e

def _call_claude_with_reduced_tokens(prompt: str, model: str, temperature: float):
    """Retry Claude with reduced token limit"""
    
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,  # Reduced token limit
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt[:10000]}  # Also truncate prompt
            ]
        )
        return response.content[0].text
    
    except Exception as e:
        print(f"Error in retry with Claude: {e}")
        raise e

def estimate_prompt_tokens(prompt: str, model: str) -> int:
    """
    Estimate the number of tokens in a prompt
    
    Args:
        prompt: The prompt text
        model: The model identifier
    
    Returns:
        Estimated token count
    """
    
    # Simple estimation: ~4 characters per token for English text
    # This is a rough estimate; use tiktoken for more accuracy
    return len(prompt) // 4

def validate_model_availability(model: str) -> bool:
    """
    Check if a model is available and configured
    
    Args:
        model: The model identifier
    
    Returns:
        True if model is available, False otherwise
    """
    
    if model in CLAUDE_MODELS:
        return anthropic_client is not None
    else:
        return openai.api_key is not None

def get_model_info(model: str) -> dict:
    """
    Get information about a model
    
    Args:
        model: The model identifier
    
    Returns:
        Dictionary with model information
    """
    
    info = {
        "name": model,
        "provider": "Anthropic" if model in CLAUDE_MODELS else "OpenAI",
        "available": validate_model_availability(model),
        "supports_temperature": model in SUPPORTS_TEMPERATURE or model in CLAUDE_MODELS,
        "max_tokens": 4096,
        "recommended_for": ""
    }
    
    # Add recommendations
    if model == "gpt-4.1-2025-04-14":
        info["recommended_for"] = "General purpose, balanced performance"
    elif model == "o4-mini-2025-04-16":
        info["recommended_for"] = "Large documents, cost-effective"
    elif model == "o3-2025-04-16":
        info["recommended_for"] = "Complex reasoning, highest quality"
    elif model == "claude-sonnet-4-20250514":
        info["recommended_for"] = "Creative tasks, nuanced understanding"
    
    return info