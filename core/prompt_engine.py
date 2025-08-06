import json
from typing import Dict, Any, Optional, List

class PromptEngine:
    def __init__(self):
        self.templates = {
            "extraction": self._extraction_template,
            "analysis": self._analysis_template,
            "chunking": self._chunking_template,
            "validation": self._validation_template,
            "merge": self._merge_template
        }
    
    def create_prompt(self, 
                     template_type: str,
                     schema: Optional[str] = None,
                     text: Optional[str] = None,
                     error: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt based on template type"""
        
        template_func = self.templates.get(template_type, self._extraction_template)
        return template_func(schema, text, error, context)
    
    def _extraction_template(self, schema: str, text: str, error: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Template for JSON extraction"""
        
        examples = self._get_few_shot_examples(schema)
        error_section = f"\n⚠️ Previous attempt failed with error: {error}\nPlease fix the issue.\n" if error else ""
        
        prompt = f"""You are an expert at extracting structured data from unstructured text.

Your task is to extract information from the provided text and format it according to the JSON schema.

{error_section}

Instructions:
1. Extract ONLY information present in the text
2. Do NOT invent or hallucinate any data
3. Follow the schema structure exactly
4. Use null for missing required fields
5. Omit optional fields if not present in text
6. Ensure all data types match the schema

{examples}

=== SCHEMA ===
{schema}

=== TEXT ===
{text}

=== OUTPUT ===
Provide only the extracted JSON object without any explanation:
"""
        return prompt
    
    def _analysis_template(self, schema: str, text: str, error: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Template for document analysis"""
        
        prompt = f"""Analyze the following document to understand its semantic structure and content organization.

=== DOCUMENT ===
{text[:5000]}... (truncated for analysis)

=== TASK ===
1. Identify the main sections and structure
2. Determine logical boundaries for chunking
3. Find key entities and relationships
4. Suggest optimal chunking strategy

Provide your analysis in the following format:
{{
    "document_type": "type of document",
    "main_sections": ["list", "of", "sections"],
    "suggested_chunk_size": "number of tokens",
    "chunk_boundaries": ["list", "of", "boundary", "markers"],
    "key_entities": ["list", "of", "entities"],
    "semantic_structure": "description of structure"
}}
"""
        return prompt
    
    def _chunking_template(self, schema: str, text: str, error: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Template for intelligent chunking code generation"""
        
        analysis = context.get("analysis", {}) if context else {}
        
        prompt = f"""Generate Python code to intelligently chunk the following document based on the analysis.

=== DOCUMENT ANALYSIS ===
{json.dumps(analysis, indent=2)}

=== REQUIREMENTS ===
1. Preserve semantic coherence
2. Maintain context across chunks
3. Include overlap for continuity
4. Respect natural boundaries

Generate a Python function with this signature:
```python
def chunk_document(text: str, max_chunk_size: int = 2000) -> List[Dict[str, Any]]:
    '''
    Returns list of chunks with metadata:
    [{{
        "chunk_id": int,
        "content": str,
        "start_idx": int,
        "end_idx": int,
        "context": str  # Brief context summary
    }}]
    '''
    # Your implementation here
```

Provide only the complete function code:
"""
        return prompt
    
    def _validation_template(self, schema: str, text: str, error: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Template for validation and correction"""
        
        extracted_json = context.get("extracted_json", {}) if context else {}
        
        prompt = f"""Review and correct the extracted JSON to ensure it matches the schema and source text.

=== SCHEMA ===
{schema}

=== EXTRACTED JSON ===
{json.dumps(extracted_json, indent=2)}

=== VALIDATION ERROR ===
{error}

=== ORIGINAL TEXT (excerpt) ===
{text[:2000]}...

Fix the JSON to:
1. Resolve all validation errors
2. Ensure accuracy to source text
3. Maintain schema compliance

Provide only the corrected JSON:
"""
        return prompt
    
    def _merge_template(self, schema: str, text: str, error: Optional[str], context: Optional[Dict[str, Any]]) -> str:
        """Template for merging multiple JSON outputs"""
        
        outputs = context.get("outputs", []) if context else []
        
        prompt = f"""Intelligently merge the following JSON outputs into a single, coherent result.

=== SCHEMA ===
{schema}

=== JSON OUTPUTS TO MERGE ===
{json.dumps(outputs, indent=2)}

=== MERGING RULES ===
1. Avoid duplicates
2. Prefer non-null values
3. Combine arrays intelligently
4. Resolve conflicts by preferring most complete data
5. Maintain schema compliance

Provide the merged JSON:
"""
        return prompt
    
    def _get_few_shot_examples(self, schema: str) -> str:
        """Generate few-shot examples based on schema complexity"""
        # This would ideally generate dynamic examples based on the schema
        return """
=== EXAMPLES ===

Example 1:
Schema: {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
Text: "John Smith is 35 years old."
Output: {"name": "John Smith", "age": 35}

Example 2:
Schema: {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}}
Text: "The list includes apples, bananas, and oranges."
Output: {"items": ["apples", "bananas", "oranges"]}
"""
