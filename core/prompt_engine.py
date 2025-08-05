# core/prompt_engine.py (UPDATED)
import json

def create_prompt(schema: str, text: str, error: str = None):
    """
    Generates a detailed prompt with multi-shot learning, schema logic,
    error feedback, and nested JSON support.
    """
    # Examples: multiple nested schemas
    examples = [
        {
            "schema": {
                "title": "AuthorInfo",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                    "affiliation": {
                        "type": "object",
                        "properties": {
                            "organization": {"type": "string"},
                            "country": {"type": "string"}
                        },
                        "required": ["organization"]
                    }
                },
                "required": ["name", "email", "affiliation"]
            },
            "text": "Dr. Jane Smith is a professor at Stanford University in the US. You can reach her at jane@stanford.edu.",
            "output": {
                "name": "Dr. Jane Smith",
                "email": "jane@stanford.edu",
                "affiliation": {
                    "organization": "Stanford University",
                    "country": "US"
                }
            }
        },
        {
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "orcid": {"type": "string"}
                            },
                            "required": ["name"]
                        }
                    },
                    "year": {"type": "integer"}
                },
                "required": ["title", "authors"]
            },
            "text": "The paper titled 'Quantum Gravity Explained' was authored by Alan Turing and Emmy Noether in 2020. ORCID for Alan is 0000-0001-2345-6789.",
            "output": {
                "title": "Quantum Gravity Explained",
                "authors": [
                    {"name": "Alan Turing", "orcid": "0000-0001-2345-6789"},
                    {"name": "Emmy Noether"}
                ],
                "year": 2020
            }
        }
    ]

    example_section = ""
    for idx, ex in enumerate(examples, 1):
        example_section += f"""
Example {idx}:
Schema:
{json.dumps(ex['schema'], indent=2)}
Text: 
{ex['text']} 
Output: {json.dumps(ex['output'], indent=2)} """

    # Optional error loop section
    error_text = f"""
⚠ Note: The previous output failed validation:
{error}
Please fix field mismatches or types as per schema.
""" if error else ""

    # Prompt core
    prompt = f"""
You are a structured data extraction assistant. Your job is to extract
a valid JSON object from raw text using the provided JSON schema.

Instructions:
- Output only the final JSON (no explanations).
- Match the field names, types, and nesting exactly as defined in the schema.
- Ensure capturing maximum detail from the text.
- If a field is missing in the text, leave it as `null` or an empty array/object.
- If a field is optional and not present in the text, do not include it in the output.
- Do not invent values; omit missing optional fields.
- Ensure all required fields are present with correct types.
- If arrays or nested objects are specified, populate them correctly.

{error_text}

=== INPUT SCHEMA ===
{schema}
=== END SCHEMA ===

=== INPUT TEXT ===
{text}
=== END TEXT ===

{example_section}

 Now, generate the JSON output that exactly follows the schema using the input text above.
Only respond with the raw JSON object.
"""

    return prompt.strip()


def create_focused_prompt(schema: str, text: str, context: dict = None, attempt: int = 0):
    """
    NEW: Creates context-aware prompts for chunk processing
    """
    context_section = ""
    
    if context:
        # Add previous entity information
        if context.get('previous_entities'):
            entities_text = []
            for entity_type, values in context['previous_entities'].items():
                if values:
                    entities_text.append(f"{entity_type}: {', '.join(list(values)[:3])}")
            
            if entities_text:
                context_section += f"""
CONTEXT - Previously extracted entities:
{chr(10).join(entities_text)}
Use this context to maintain consistency and avoid duplicates.
"""
        
        # Add chunk information
        if context.get('chunk_info'):
            chunk_info = context['chunk_info']
            context_section += f"""
CHUNK INFO: Processing chunk {chunk_info['index'] + 1} of {chunk_info['total']}
Content type: {chunk_info['content_type']}
"""
    
    # Attempt-specific instructions
    attempt_section = ""
    if attempt > 0:
        attempt_section = f"""
RETRY ATTEMPT {attempt + 1}: Previous attempts failed. 
Focus on extracting only the most certain information.
If unsure about a field, use null rather than guessing.
"""

    prompt = f"""
You are processing a chunk of a larger document for structured data extraction.
Extract ONLY the information that is clearly present in this specific chunk.

{context_section}

{attempt_section}

=== SCHEMA ===
{schema}
=== END SCHEMA ===

=== TEXT CHUNK ===
{text}
=== END CHUNK ===

Instructions:
- Extract only information that is explicitly stated in this chunk
- Use exact field names and types from the schema
- If information is not in this chunk, use null for that field
- Maintain consistency with previously extracted entities when possible
- Output only valid JSON, no explanations

JSON Output:
"""
    return prompt.strip()