import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_processing_cost(text: str, schema: str, model: str = "gpt-4") -> dict:
    """
    NEW: Estimate processing cost for complex text
    """
    total_text = text + schema
    total_tokens = estimate_tokens(total_text, model)
    
    # Estimate chunks needed
    max_chunk_size = 4000  # Conservative estimate
    estimated_chunks = max(1, len(text) // max_chunk_size)
    
    # Estimate total tokens with processing overhead
    processing_overhead = 1.5  # Account for prompts, retries, etc.
    total_estimated_tokens = int(total_tokens * estimated_chunks * processing_overhead)
    
    # Cost estimation (approximate)
    cost_per_1k_tokens = {
        "gpt-4": 0.03,
        "gpt-4.1-2025-04-14": 0.03,
        "o4-mini-2025-04-16": 0.015
    }
    
    rate = cost_per_1k_tokens.get(model, 0.03)
    estimated_cost = (total_estimated_tokens / 1000) * rate
    
    return {
        'input_tokens': total_tokens,
        'estimated_chunks': estimated_chunks,
        'total_estimated_tokens': total_estimated_tokens,
        'estimated_cost_usd': round(estimated_cost, 4),
        'processing_time_estimate_minutes': estimated_chunks * 0.5
    }
