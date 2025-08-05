import tiktoken

# Model mapping for custom model names to tiktoken-supported encodings
MODEL_ENCODING_MAP = {
    "gpt-4": "cl100k_base",
    "gpt-4.1-2025-04-14": "cl100k_base",  # Use GPT-4 encoding
    "o4-mini-2025-04-16": "cl100k_base",   # Use GPT-4 encoding (most compatible)
    "gpt-3.5-turbo": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
}

def get_encoding_for_model(model: str) -> str:
    """Get the appropriate encoding for a model, including custom model names"""
    # Direct mapping for known models
    if model in MODEL_ENCODING_MAP:
        return MODEL_ENCODING_MAP[model]
    
    # Fallback logic for unknown models
    if "gpt-4" in model.lower():
        return "cl100k_base"
    elif "gpt-3.5" in model.lower() or "turbo" in model.lower():
        return "cl100k_base"
    elif "davinci" in model.lower():
        return "p50k_base"
    else:
        # Default to most common encoding
        return "cl100k_base"

def estimate_tokens(text: str, model: str = "gpt-4"):
    encoding_name = get_encoding_for_model(model)
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def estimate_processing_cost(text: str, schema: str, model: str = "gpt-4") -> dict:
    """
    Estimate processing cost for complex text
    """
    total_text = text + schema
    total_tokens = estimate_tokens(total_text, model)
    
    # Estimate chunks needed
    max_chunk_size = 4000  # Conservative estimate
    estimated_chunks = max(1, len(text) // max_chunk_size)
    
    # Estimate total tokens with processing overhead
    processing_overhead = 1.5  # Account for prompts, retries, etc.
    total_estimated_tokens = int(total_tokens * estimated_chunks * processing_overhead)
    
    # Cost estimation (approximate) - Updated with more realistic rates
    cost_per_1k_tokens = {
        "gpt-4": 0.03,
        "gpt-4.1-2025-04-14": 0.03,
        "o4-mini-2025-04-16": 0.015,  # Assuming mini model is cheaper
        "gpt-3.5-turbo": 0.002,
    }
    
    # Get rate for the model, fallback to GPT-4 rate
    rate = cost_per_1k_tokens.get(model, 0.03)
    estimated_cost = (total_estimated_tokens / 1000) * rate
    
    return {
        'input_tokens': total_tokens,
        'estimated_chunks': estimated_chunks,
        'total_estimated_tokens': total_estimated_tokens,
        'estimated_cost_usd': round(estimated_cost, 4),
        'processing_time_estimate_minutes': estimated_chunks * 0.5,
        'encoding_used': get_encoding_for_model(model)
        }