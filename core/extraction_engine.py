import json
import re
from typing import Dict, Any, List, Optional
from .llm_interface import call_llm
from .json_extractor import extract_json, validate_against_schema
from .prompt_engine import create_adaptive_prompt
from .logger_service import log

class ExtractionEngine:
    """Handles multi-pass extraction with context awareness"""
    
    def multi_pass_extract(
        self, 
        text_chunk: str, 
        schema_partition: Dict,
        num_passes: int,
        model: str,
        context: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Perform multi-pass extraction on a text chunk with a schema partition
        
        Args:
            text_chunk: The text chunk to extract from
            schema_partition: The schema partition to extract
            num_passes: Number of extraction passes (2-5)
            model: The LLM model to use
            context: Global context including embeddings and full documents
        
        Returns:
            Extracted JSON matching the schema partition, or None if extraction fails
        """
        
        extracted_data = None
        previous_extractions = []
        
        for pass_num in range(1, num_passes + 1):
            # Create adaptive prompt for this pass
            prompt = create_adaptive_prompt(
                schema_partition,
                text_chunk,
                pass_number=pass_num,
                previous_extractions=previous_extractions,
                context=context
            )
            
            try:
                # Call LLM for extraction
                response = call_llm(prompt, model)
                
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
                    
                    # If we have a valid extraction and it's not the last pass,
                    # continue to refine in next passes
                    if pass_num < num_passes:
                        continue
                    else:
                        break
                else:
                    # Store invalid extraction for learning
                    previous_extractions.append({
                        'pass': pass_num,
                        'data': current_extraction,
                        'valid': False,
                        'error': error
                    })
                    
                    # If this is the last pass and we still don't have valid data,
                    # try to fix the extraction
                    if pass_num == num_passes:
                        extracted_data = self._attempt_fix(
                            current_extraction, 
                            schema_partition, 
                            error, 
                            text_chunk,
                            model
                        )
            
            except Exception as e:
                print(f"Error in pass {pass_num}: {e}")
                previous_extractions.append({
                    'pass': pass_num,
                    'error': str(e),
                    'valid': False
                })
        
        # If we have extracted data, enhance it with context
        if extracted_data:
            extracted_data = self._enhance_with_context(
                extracted_data,
                schema_partition,
                text_chunk,
                context,
                model
            )
        
        return extracted_data
    
    def _attempt_fix(
        self,
        extraction: Dict,
        schema: Dict,
        error: str,
        text: str,
        model: str
    ) -> Optional[Dict]:
        """Attempt to fix validation errors in extraction"""
        
        fix_prompt = f"""
        The following JSON extraction has validation errors. Please fix them.
        
        Schema:
        {json.dumps(schema, indent=2)}
        
        Current Extraction:
        {json.dumps(extraction, indent=2)}
        
        Validation Error:
        {error}
        
        Original Text (for reference):
        {text[:1000]}...
        
        Fix the extraction to match the schema exactly. 
        - Ensure all required fields are present
        - Ensure all field types match the schema
        - Remove any fields not in the schema
        - Use null for missing optional fields
        
        Return ONLY the fixed JSON object.
        """
        
        try:
            response = call_llm(fix_prompt, model)
            fixed_extraction = extract_json(response)
            
            # Validate the fix
            is_valid, _ = validate_against_schema(schema, fixed_extraction)
            if is_valid:
                return fixed_extraction
            else:
                return extraction  # Return original if fix didn't work
        
        except Exception as e:
            print(f"Error attempting fix: {e}")
            return extraction
    
    def _enhance_with_context(
        self,
        extraction: Dict,
        schema: Dict,
        text_chunk: str,
        context: Dict,
        model: str
    ) -> Dict:
        """Enhance extraction with global context information"""
        
        # Check if extraction has any null or empty fields that might be filled from context
        empty_fields = self._find_empty_fields(extraction, schema)
        
        if not empty_fields:
            return extraction
        
        enhance_prompt = f"""
        Enhance the following extraction by filling in missing information from the context.
        
        Current Extraction:
        {json.dumps(extraction, indent=2)}
        
        Empty/Missing Fields to Fill:
        {json.dumps(empty_fields, indent=2)}
        
        Current Text Chunk:
        {text_chunk[:500]}...
        
        Full Document Context Available: Yes
        Full Schema Context Available: Yes
        
        Instructions:
        1. Only fill in fields that are currently null or empty
        2. Only use information that is clearly stated in the text
        3. Maintain the same data types as specified in the schema
        4. Do not modify already filled fields
        5. If information is not available, keep the field as null
        
        Return the enhanced JSON object with filled fields where possible.
        Return ONLY the JSON object.
        """
        
        try:
            response = call_llm(enhance_prompt, model)
            enhanced = extract_json(response)
            
            # Validate enhancement
            is_valid, _ = validate_against_schema(schema, enhanced)
            if is_valid:
                return enhanced
            else:
                return extraction  # Return original if enhancement failed
        
        except Exception as e:
            print(f"Error enhancing extraction: {e}")
            return extraction
    
    def _find_empty_fields(self, data: Dict, schema: Dict, path: str = "") -> List[str]:
        """Find empty or null fields in the extraction"""
        empty_fields = []
        
        if 'properties' in schema:
            for field_name, field_schema in schema['properties'].items():
                field_path = f"{path}.{field_name}" if path else field_name
                
                if field_name in data:
                    value = data[field_name]
                    
                    # Check if field is empty
                    if value is None or value == "" or value == [] or value == {}:
                        empty_fields.append(field_path)
                    # Recursively check nested objects
                    elif isinstance(value, dict) and 'properties' in field_schema:
                        nested_empty = self._find_empty_fields(value, field_schema, field_path)
                        empty_fields.extend(nested_empty)
                else:
                    # Field is missing entirely
                    if 'required' not in schema or field_name not in schema.get('required', []):
                        empty_fields.append(field_path)
        
        return empty_fields
    
    def fix_validation_errors(
        self,
        json_data: Dict,
        schema: Dict,
        error: str,
        context: Dict,
        model: str
    ) -> Dict:
        """Fix validation errors in final merged JSON"""
        
        fix_prompt = f"""
        Fix the validation errors in this JSON data to match the schema.
        
        Schema:
        {json.dumps(schema, indent=2)[:3000]}...
        
        Current JSON with Errors:
        {json.dumps(json_data, indent=2)[:3000]}...
        
        Validation Error:
        {error}
        
        Context Information:
        - This is a merged result from multiple extraction passes
        - The original document had {context.get('chunk_count', 'multiple')} chunks
        - The schema had {context.get('partition_count', 'multiple')} partitions
        
        Fix Instructions:
        1. Address the specific validation error mentioned
        2. Ensure all required fields are present
        3. Ensure all data types match the schema
        4. Remove any duplicate or conflicting data
        5. Preserve as much valid extracted information as possible
        
        Return ONLY the complete, fixed JSON object.
        """
        
        try:
            response = call_llm(fix_prompt, model)
            fixed_json = extract_json(response)
            return fixed_json
        
        except Exception as e:
            print(f"Error fixing validation errors: {e}")
            return json_data

# Initialize global instance
extraction_engine = ExtractionEngine()