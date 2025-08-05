import json
from typing import List, Optional

def create_advanced_prompt(
    schema: str,
    text: str,
    error: Optional[str] = None,
    task_fields: Optional[List[str]] = None,
    examples: Optional[List[dict]] = None,
    pass_num: int = 1,
    cascade_mode: bool = False,
    fuzzy_mode: bool = False,
    chunk_id: Optional[int] = None,
    guidance: Optional[str] = None,
) -> str:
    example_section = ""
    if examples:
        example_section = "\n".join([
            f"""Example {i+1}:
Schema:
{json.dumps(e['schema'], indent=2)}

Text:
{e['text']}

Output:
{json.dumps(e['output'], indent=2)}
""" for i, e in enumerate(examples)
        ])
    error_text = ""
    if error:
        error_text = f"""\n⚠️ Prior output failed validation (reason below). Repair only the MISSING/INVALID fields.
Error: {error}
"""
    if task_fields:
        error_text += f"\n⚡ Only extract or fix these fields/subobjects: {', '.join(task_fields)}"

    pass_text = f"\n🔄 Multi-pass extraction: This is refinement pass #{pass_num}.\n" if pass_num > 1 else ""
    chunk_text = f"\n[Chunk ID: {chunk_id}]" if chunk_id else ""
    guidance_text = f"\nUser/system Guidance: {guidance}" if guidance else ""
    fuzzy_instructions = (
        "\nIf any field is ambiguous/missing, return a comment as a field value explaining why, "
        "and suggest how the user or system could resolve the gap."
    ) if fuzzy_mode else ""
    cascade_text = "\nCascade Mode enabled: Focus only on this section/branch; later passes will fill remaining fields. Ignore unrelated information." if cascade_mode else ""

    prompt = f"""
You are an expert structured data extraction assistant. Your job is to extract a valid JSON object from raw text using the provided JSON schema.

Instructions:
- Output ONLY the JSON object (no explanations outside JSON, except as in-field comments if allowed).
- Strictly match field names, types, and nesting as defined in the schema.
- If a field is absent, set it as null or omit if optional.
- NEVER invent details; only use information present in the input.

{error_text}
{pass_text}
{cascade_text}
{fuzzy_instructions}
{chunk_text}
{guidance_text}

=== INPUT SCHEMA ===
{schema}
=== END SCHEMA ===

=== INPUT TEXT ===
{text}
=== END TEXT ===

{example_section}

🟢 Generate only the JSON output, matching the schema, capturing all concrete details from the input above.
"""
    return prompt.strip()
