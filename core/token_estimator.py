import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
