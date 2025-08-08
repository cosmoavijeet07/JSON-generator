import json
import re
from typing import Dict, Any, List, Optional, Tuple
from .llm_interface import call_llm
from .json_extractor import extract_json, validate_against_schema
from .prompt_engine import create_adaptive_prompt
from .logger_service import log
from .token_estimator import estimate_tokens
from .embedding_manager import embedding_manager
from app import log_and_display
        
class ExtractionEngine:
    """Handles multi-pass extraction with context awareness and token management"""
    
    def __init__(self):
        self.max_context_tokens = 10000  # Safe limit for most models
        self.max_output_tokens = 20000   # Expected output size
        self.embedding_manager = embedding_manager
    
    def multi_pass_extract(
        self, 
        text_chunk: str, 
        schema_partition: Dict,
        num_passes: int,
        model: str,
        context: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Perform multi-pass extraction with intelligent token management
        
        Args:
            text_chunk: The text chunk to extract from
            schema_partition: The schema partition to extract
            num_passes: Number of extraction passes (2-5)
            model: The LLM model to use
            context: Global context including embeddings
        
        Returns:
            Extracted JSON matching the schema partition
        """
        
        # Estimate tokens and adjust if needed
        text_tokens = estimate_tokens(text_chunk, model)
        schema_tokens = estimate_tokens(json.dumps(schema_partition), model)
        
        # Truncate text if needed to fit in context window
        if text_tokens + schema_tokens > self.max_context_tokens - 1000:  # Leave room for prompt
            max_text_tokens = self.max_context_tokens - schema_tokens - 1000
            text_chunk = self._truncate_to_tokens(text_chunk, max_text_tokens, model)
            log("session", "text_truncated", f"Text truncated to {max_text_tokens} tokens", "WARNING")
        
        # Check if this is a single extraction (optimization)
        is_single_extraction = (
            context.get('total_chunks', 1) == 1 and 
            context.get('total_partitions', 1) == 1
        )
        
        if is_single_extraction:
            log("session", "optimized_extraction", "Using optimized single extraction", "INFO")
            # For single extraction, we can use fewer passes
            num_passes = min(num_passes, 2)
        
        extracted_data = None
        previous_extractions = []
        
        for pass_num in range(1, num_passes + 1):
            # Use embeddings for context instead of full text
            context_summary = self._create_context_summary(text_chunk, schema_partition, context)
            
            # Create adaptive prompt with token awareness
            prompt = create_adaptive_prompt(
                schema_partition,
                text_chunk,
                pass_number=pass_num,
                previous_extractions=previous_extractions[-2:] if previous_extractions else None,  # Only last 2
                context=context_summary
            )
            
            # Check prompt tokens
            prompt_tokens = estimate_tokens(prompt, model)
            if prompt_tokens > self.max_context_tokens:
                log("session", "prompt_too_long", f"Prompt exceeds token limit: {prompt_tokens}", "WARNING")
                # Reduce prompt size
                prompt = self._reduce_prompt_size(prompt, schema_partition, text_chunk, pass_num)
            
            try:
                # Call LLM for extraction
                response = call_llm(prompt, model)
                log_and_display(response)
                
                # Extract JSON from response
                current_extraction = extract_json(response)
                
                # Validate against schema partition
                is_valid, error = validate_against_schema(schema_partition, current_extraction)
                
                if is_valid:
                    extracted_data = current_extraction
                    previous_extractions.append({
                        'pass': pass_num,
                        'data': current_extraction,
                        'valid': True
                    })
                    
                    # Early exit if we have good extraction
                    if is_single_extraction and is_valid:
                        log("session", "early_exit", f"Valid extraction achieved in pass {pass_num}", "INFO")
                        break
                    elif pass_num >= 2 and is_valid:
                        break
                else:
                    previous_extractions.append({
                        'pass': pass_num,
                        'data': current_extraction,
                        'valid': False,
                        'error': error[:400]  # Limit error message size
                    })
                    
                    # Try to fix on last pass
                    if pass_num == num_passes:
                        extracted_data = self._attempt_fix(
                            current_extraction, 
                            schema_partition, 
                            error, 
                            text_chunk[:1000],  # Use only sample for fix
                            model
                        )
            
            except Exception as e:
                log("session", f"extraction_error_pass_{pass_num}", str(e), "ERROR")
                previous_extractions.append({
                    'pass': pass_num,
                    'error': str(e)[:200],
                    'valid': False
                })
        
        return extracted_data
    
    def _truncate_to_tokens(self, text: str, max_tokens: int, model: str) -> str:
        """Truncate text to fit within token limit"""
        import tiktoken
        
        try:
            if "claude" in model.lower():
                enc = tiktoken.get_encoding("cl100k_base")
            else:
                enc = tiktoken.encoding_for_model("gpt-4")
            
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate and decode
            truncated_tokens = tokens[:max_tokens]
            return enc.decode(truncated_tokens)
        
        except Exception:
            # Fallback to character-based truncation
            estimated_chars = max_tokens * 4  # Rough estimate
            return text[:estimated_chars]
    
    def _create_context_summary(self, text_chunk: str, schema: Dict, context: Dict) -> Dict:
        """Create a summary of context using embeddings"""
        summary = {
            "has_full_document": "full_text" in context,
            "has_embeddings": "text_embeddings" in context or "schema_embeddings" in context,
            "chunk_type": "partial",
            "schema_type": "partition" if "partition" in str(schema) else "full"
        }
        
        # Use embeddings to find relevant context if available
        if "text_embeddings" in context and self.embedding_manager:
            try:
                # Get embedding for current chunk
                chunk_embedding = self.embedding_manager.create_embeddings(text_chunk[:500])
                
                # Find similar content in full document (if available)
                if "full_text" in context:
                    full_text_sample = context["full_text"][:2000]
                    full_embedding = self.embedding_manager.create_embeddings(full_text_sample)
                    similarity = self.embedding_manager.calculate_similarity(chunk_embedding, full_embedding)
                    summary["context_similarity"] = float(similarity)
            except Exception as e:
                log("session", "embedding_error", str(e), "WARNING")
        
        return summary
    
    def _reduce_prompt_size(self, prompt: str, schema: Dict, text: str, pass_num: int) -> str:
        """Reduce prompt size to fit within token limits"""
        # Simplified prompt with minimal examples
        reduced_prompt = f"""Extract JSON from text (Pass {pass_num}).

Schema:
{json.dumps(schema, indent=2)[:1000]}...

Text:
{text[:1500]}...

Rules:
- Match schema structure exactly
- Use null for missing required fields
- Output only valid JSON

JSON Output:"""
        
        return reduced_prompt
    
    def _attempt_fix(self, extraction: Dict, schema: Dict, error: str, text_sample: str, model: str) -> Optional[Dict]:
        """Attempt to fix validation errors with minimal token usage"""
        
        # Parse error to identify specific issues
        error_summary = self._summarize_error(error)
        
        fix_prompt = f"""Fix this JSON to match the schema.

Schema (key fields):
{self._get_schema_summary(schema)}

Current JSON:
{json.dumps(extraction, indent=2)[:1000]}

Error: {error_summary}

Fixed JSON:"""
        
        try:
            response = call_llm(fix_prompt, model)
            fixed_extraction = extract_json(response)
            
            # Quick validation
            is_valid, _ = validate_against_schema(schema, fixed_extraction)
            if is_valid:
                return fixed_extraction
            else:
                return extraction
        
        except Exception as e:
            log("session", "fix_attempt_error", str(e), "WARNING")
            return extraction
    
    def _summarize_error(self, error: str) -> str:
        """Extract key information from validation error"""
        if "required" in error.lower():
            match = re.search(r"'(\w+)'.*required", error)
            if match:
                return f"Missing required field: {match.group(1)}"
        elif "type" in error.lower():
            match = re.search(r"'(\w+)'.*type", error)
            if match:
                return f"Type mismatch for field: {match.group(1)}"
        
        return error[:100] if len(error) > 100 else error
    
    def _get_schema_summary(self, schema: Dict) -> str:
        """Get a concise summary of schema for token efficiency"""
        summary = {
            "type": schema.get("type", "object"),
            "required": schema.get("required", [])[:5],  # First 5 required fields
            "properties": list(schema.get("properties", {}).keys())[:10]  # First 10 properties
        }
        return json.dumps(summary, indent=2)
    
    def fix_validation_errors(
        self,
        json_data: Dict,
        schema: Dict,
        error: str,
        context: Dict,
        model: str
    ) -> Dict:
        """Fix validation errors in final merged JSON with token efficiency"""
        
        # Ensure json_data is a dictionary
        if not isinstance(json_data, dict):
            # If json_data is not a dict, try to convert or create empty dict
            if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                # If it's a list with dict items, take the first one
                json_data = json_data[0]
            else:
                # Create empty dict based on schema
                json_data = {}
        
        # Identify specific issues
        issues = self._identify_validation_issues(json_data, schema, error)
        
        if not issues:
            return json_data
        
        # Fix issues programmatically first
        fixed_data = self._programmatic_fixes(json_data, schema, issues)
        
        # Validate the fixes
        is_valid, remaining_error = validate_against_schema(schema, fixed_data)
        
        if is_valid:
            return fixed_data
        
        # If still invalid, try targeted LLM fix on specific fields only
        problem_fields = self._extract_problem_fields(remaining_error)
        
        if problem_fields:
            fix_prompt = f"""Fix only these fields in the JSON:

Problem fields: {problem_fields}
Schema for these fields:
{self._get_field_schemas(schema, problem_fields)}

Current values:
{self._get_field_values(fixed_data, problem_fields)}

Return only the corrected field values as JSON:"""
            
            try:
                response = call_llm(fix_prompt, model)
                field_fixes = extract_json(response)
                
                # Apply field fixes
                for field, value in field_fixes.items():
                    if field in problem_fields:
                        self._set_field_value(fixed_data, field, value)
                
                return fixed_data
            
            except Exception as e:
                log("session", "field_fix_error", str(e), "WARNING")
        
        return fixed_data
    
    def _identify_validation_issues(self, data: Dict, schema: Dict, error: str) -> List[Dict]:
        """Identify specific validation issues"""
        issues = []
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            # If data is not a dict, all required fields are missing
            required = schema.get("required", [])
            for field in required:
                issues.append({"type": "missing_required", "field": field})
            return issues
        
        # Check for missing required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data or data[field] is None:
                issues.append({"type": "missing_required", "field": field})
        
        # Check for type mismatches
        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in data:
                    expected_type = field_schema.get("type")
                    if expected_type and not self._check_type(data[field], expected_type):
                        issues.append({
                            "type": "type_mismatch",
                            "field": field,
                            "expected": expected_type,
                            "actual": type(data[field]).__name__
                        })
        
        return issues
    
    def _programmatic_fixes(self, data: Dict, schema: Dict, issues: List[Dict]) -> Dict:
        """Apply programmatic fixes for common issues"""
        # Ensure we're working with a dictionary
        if not isinstance(data, dict):
            # If data is not a dict, create a new dict based on schema
            fixed = {}
        else:
            fixed = data.copy()
        
        for issue in issues:
            if issue["type"] == "missing_required":
                field = issue["field"]
                if field in schema.get("properties", {}):
                    field_schema = schema["properties"][field]
                    # Set appropriate default value based on type
                    field_type = field_schema.get("type", "string")
                    if field_type == "string":
                        fixed[field] = ""
                    elif field_type == "number":
                        fixed[field] = 0.0
                    elif field_type == "integer":
                        fixed[field] = 0
                    elif field_type == "boolean":
                        fixed[field] = False
                    elif field_type == "array":
                        fixed[field] = []
                    elif field_type == "object":
                        fixed[field] = {}
                    else:
                        fixed[field] = None
            
            elif issue["type"] == "type_mismatch":
                field = issue["field"]
                expected = issue["expected"]
                
                # Ensure fixed is a dict before trying to access/set fields
                if not isinstance(fixed, dict):
                    fixed = {}
                
                # Try to convert type
                try:
                    if field in data and isinstance(data, dict):
                        if expected == "string":
                            fixed[field] = str(data[field])
                        elif expected == "number":
                            fixed[field] = float(data[field])
                        elif expected == "integer":
                            fixed[field] = int(data[field])
                        elif expected == "boolean":
                            fixed[field] = bool(data[field])
                        elif expected == "array":
                            if not isinstance(data[field], list):
                                fixed[field] = [data[field]] if data[field] else []
                            else:
                                fixed[field] = data[field]
                        elif expected == "object":
                            if not isinstance(data[field], dict):
                                fixed[field] = {}
                            else:
                                fixed[field] = data[field]
                    else:
                        # If field doesn't exist or data is not dict, set default
                        if expected == "string":
                            fixed[field] = ""
                        elif expected == "number":
                            fixed[field] = 0.0
                        elif expected == "integer":
                            fixed[field] = 0
                        elif expected == "boolean":
                            fixed[field] = False
                        elif expected == "array":
                            fixed[field] = []
                        elif expected == "object":
                            fixed[field] = {}
                        else:
                            fixed[field] = None
                except Exception:
                    # If conversion fails, set to default value
                    if expected == "string":
                        fixed[field] = ""
                    elif expected == "number":
                        fixed[field] = 0.0
                    elif expected == "integer":
                        fixed[field] = 0
                    elif expected == "boolean":
                        fixed[field] = False
                    elif expected == "array":
                        fixed[field] = []
                    elif expected == "object":
                        fixed[field] = {}
                    else:
                        fixed[field] = None
        
        return fixed
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        
        return True
    
    def _extract_problem_fields(self, error: str) -> List[str]:
        """Extract field names from error message"""
        fields = []
        
        # Look for field names in quotes
        matches = re.findall(r"'([^']+)'", error)
        for match in matches:
            # Filter to likely field names
            if len(match) < 50 and not match.startswith('/'):
                fields.append(match)
        
        return fields[:5]  # Limit to 5 fields
    
    def _get_field_schemas(self, schema: Dict, fields: List[str]) -> Dict:
        """Get schemas for specific fields"""
        field_schemas = {}
        properties = schema.get("properties", {})
        
        for field in fields:
            if field in properties:
                field_schemas[field] = properties[field]
        
        return field_schemas
    
    def _get_field_values(self, data: Dict, fields: List[str]) -> Dict:
        """Get current values for specific fields"""
        field_values = {}
        
        for field in fields:
            if field in data:
                field_values[field] = data[field]
            else:
                field_values[field] = None
        
        return field_values
    
    def _set_field_value(self, data: Dict, field_path: str, value: Any):
        """Set a field value in nested structure"""
        if '.' not in field_path:
            data[field_path] = value
        else:
            parts = field_path.split('.')
            current = data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

# Initialize global instance
extraction_engine = ExtractionEngine()