def create_prompt(schema: str, text: str, error: str = None):
    return f"""
You are a helpful assistant that extracts structured data from text based on the given JSON schema.
Return only a valid JSON matching the schema.

Schema:
{schema}

Text:
{text}

Validation Errors: {error or 'None'}

Respond with a valid JSON object only.
"""
