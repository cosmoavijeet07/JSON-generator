import json
from typing import Dict, List, Any, Optional

def create_prompt(schema: str, text: str, error: str = None):
    """
    Original simple prompt creation for backward compatibility
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

Output:
{json.dumps(ex['output'], indent=2)}
"""
    
    # Optional error loop section
    error_text = f"""
⚠️ Note: The previous output failed validation:
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

🟢 Now, generate the JSON output that exactly follows the schema using
the input text above.
Only respond with the raw JSON object.
"""
    
    return prompt.strip()


def create_adaptive_prompt(
    schema: Dict,
    text: str,
    pass_number: int = 1,
    previous_extractions: List[Dict] = None,
    context: Dict[str, Any] = None
) -> str:
    """
    Create an adaptive prompt for multi-pass extraction with context awareness
    
    Args:
        schema: The JSON schema (partition) to extract
        text: The text chunk to extract from
        pass_number: Current pass number (1-5)
        previous_extractions: Results from previous passes
        context: Global context including embeddings and full documents
    
    Returns:
        Adaptive prompt string for the LLM
    """
    
    # Base instructions that evolve with each pass
    pass_instructions = {
        1: "Extract all clearly stated information matching the schema.",
        2: "Refine the extraction by finding additional details and resolving ambiguities.",
        3: "Deep extraction: infer relationships and fill gaps using contextual understanding.",
        4: "Validate and enhance: ensure completeness and logical consistency.",
        5: "Final polish: maximize extraction quality and handle edge cases."
    }
    
    instruction = pass_instructions.get(pass_number, pass_instructions[1])
    
    # Build few-shot examples based on schema complexity
    examples = _generate_adaptive_examples(schema)
    
    # Format previous extractions if available
    previous_section = ""
    if previous_extractions:
        previous_section = "\n=== PREVIOUS EXTRACTION ATTEMPTS ===\n"
        for prev in previous_extractions[-2:]:  # Show last 2 attempts
            previous_section += f"""
Pass {prev['pass']}:
Valid: {prev.get('valid', False)}
{f"Error: {prev.get('error', '')}" if 'error' in prev else ""}
{f"Data: {json.dumps(prev.get('data', {}), indent=2)[:500]}..." if 'data' in prev else ""}
"""
        previous_section += "=== END PREVIOUS ATTEMPTS ===\n"
    
    # Context awareness section
    context_section = ""
    if context:
        context_section = """
=== CONTEXT INFORMATION ===
- Full document contains multiple related sections
- Schema represents part of a larger structure
- Maintain consistency with overall document semantics
- Consider relationships between different data points
=== END CONTEXT ===
"""
    
    prompt = f"""
You are an advanced JSON extraction specialist performing pass {pass_number} of a multi-pass extraction.

CURRENT TASK: {instruction}

{context_section}

=== TARGET SCHEMA ===
{json.dumps(schema, indent=2)}
=== END SCHEMA ===

=== TEXT TO EXTRACT FROM ===
{text[:3000]}{"..." if len(text) > 3000 else ""}
=== END TEXT ===

{previous_section}

=== EXTRACTION GUIDELINES FOR PASS {pass_number} ===
{"1. Focus on explicit information only." if pass_number == 1 else ""}
{"2. Use context to resolve ambiguities from pass 1." if pass_number == 2 else ""}
{"3. Make intelligent inferences based on document patterns." if pass_number == 3 else ""}
{"4. Ensure all required fields have meaningful values." if pass_number == 4 else ""}
{"5. Perfect the extraction with maximum accuracy." if pass_number == 5 else ""}

- Match the schema structure exactly
- Use null for genuinely missing required fields
- Omit optional fields if no data available
- Ensure type compliance (string, number, boolean, array, object)
- NO hallucination - only extract what's supported by the text
- For arrays, extract ALL relevant items, not just examples

=== FEW-SHOT EXAMPLES ===
{examples}
=== END EXAMPLES ===

Now, extract a JSON object that perfectly matches the schema from the given text.
This is pass {pass_number} - {instruction}

Return ONLY the JSON object, no explanations or markdown:
"""
    
    return prompt.strip()


def _generate_adaptive_examples(schema: Dict) -> str:
    """Generate relevant few-shot examples based on schema structure"""
    
    examples = []
    
    # Detect schema patterns and provide relevant examples
    if _has_nested_objects(schema):
        examples.append({
            "pattern": "Nested Objects",
            "example_text": "John Smith, CEO of TechCorp (based in California), announced the merger.",
            "example_output": {
                "person": {
                    "name": "John Smith",
                    "title": "CEO",
                    "company": {
                        "name": "TechCorp",
                        "location": "California"
                    }
                }
            }
        })
    
    if _has_arrays(schema):
        examples.append({
            "pattern": "Arrays",
            "example_text": "The team includes Alice (engineer), Bob (designer), and Carol (manager).",
            "example_output": {
                "team": [
                    {"name": "Alice", "role": "engineer"},
                    {"name": "Bob", "role": "designer"},
                    {"name": "Carol", "role": "manager"}
                ]
            }
        })
    
    if _has_optional_fields(schema):
        examples.append({
            "pattern": "Optional Fields",
            "example_text": "Product X costs $99. No discount available.",
            "example_output": {
                "product": "Product X",
                "price": 99
                # Note: 'discount' field omitted as it's optional and not mentioned
            }
        })
    
    # Format examples into string
    example_str = ""
    for ex in examples:
        example_str += f"""
Pattern: {ex['pattern']}
Text: {ex['example_text']}
Extracted: {json.dumps(ex['example_output'], indent=2)}
---"""
    
    return example_str if example_str else "No specific examples needed for this schema structure."


def _has_nested_objects(schema: Dict) -> bool:
    """Check if schema has nested objects"""
    if 'properties' in schema:
        for prop in schema['properties'].values():
            if prop.get('type') == 'object' and 'properties' in prop:
                return True
    return False


def _has_arrays(schema: Dict) -> bool:
    """Check if schema has array fields"""
    if 'properties' in schema:
        for prop in schema['properties'].values():
            if prop.get('type') == 'array':
                return True
    return False


def _has_optional_fields(schema: Dict) -> bool:
    """Check if schema has optional fields"""
    if 'properties' in schema and 'required' in schema:
        return len(schema['properties']) > len(schema.get('required', []))
    return False


def create_cascade_prompt(
    schema: Dict,
    text: str,
    field_focus: str = None,
    extraction_depth: str = "standard"
) -> str:
    """
    Create cascading prompts for specific field extraction
    
    Args:
        schema: The JSON schema
        text: The text to extract from
        field_focus: Specific field to focus on (optional)
        extraction_depth: "shallow", "standard", or "deep"
    
    Returns:
        Specialized prompt for cascade extraction
    """
    
    depth_instructions = {
        "shallow": "Extract only explicitly stated values.",
        "standard": "Extract stated values and obvious relationships.",
        "deep": "Extract all information including inferred relationships and context."
    }
    
    prompt = f"""
Advanced Extraction Task - Depth: {extraction_depth.upper()}

{depth_instructions.get(extraction_depth, depth_instructions["standard"])}

{"FOCUS FIELD: " + field_focus if field_focus else "Extract all fields equally."}

Schema:
{json.dumps(schema, indent=2)}

Text:
{text[:2000]}...

Instructions:
1. {depth_instructions[extraction_depth]}
2. Pay special attention to {field_focus if field_focus else "all required fields"}
3. Ensure type correctness
4. No hallucination - only what's supported by text

Return ONLY the JSON object:
"""
    
    return prompt.strip()