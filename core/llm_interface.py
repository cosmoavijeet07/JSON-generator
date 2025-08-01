import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Models that allow temperature customization
SUPPORTS_TEMPERATURE = {"gpt-4", "gpt-4o", "gpt-4.1-2025-04-14"}

def call_llm(prompt: str, model="gpt-4.1-2025-04-14"):
    model_base = model.split(":")[-1] if ":" in model else model
    use_temp = 0.1 if model_base in SUPPORTS_TEMPERATURE else 1

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=use_temp
    )
    return response.choices[0].message.content
