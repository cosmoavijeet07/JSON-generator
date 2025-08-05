import json
from typing import Tuple, Dict, Any
from core import json_extractor, llm_interface, prompt_engine, logger_service

def validate_and_rectify_final_json(
    final_json: Dict[Any, Any],
    schema: str,
    original_chunks: list,
    llm_fn,
    session_id: str,
    max_rectification_attempts: int = 3
) -> Tuple[Dict[Any, Any], bool, str]:
    """
    Post-processing validator that ensures final merged JSON matches schema.
    If validation fails, attempts to rectify using the original text chunks.
    
    Returns:
        - rectified_json: The corrected JSON
        - success: Whether rectification was successful
        - error_msg: Error message if rectification failed
    """
    
    # First, validate the current final JSON
    try:
        schema_dict = json.loads(schema) if isinstance(schema, str) else schema
        valid, validation_error = json_extractor.validate_against_schema(schema_dict, final_json)
        
        if valid:
            logger_service.log(session_id, "post_validation", "Final JSON passed schema validation")
            return final_json, True, None
            
    except Exception as e:
        validation_error = f"Schema parsing error: {str(e)}"
    
    # If validation failed, attempt rectification
    logger_service.log(session_id, "post_validation_error", f"Validation failed: {validation_error}")
    
    # Combine all chunks for context
    combined_text = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(original_chunks)])
    
    current_json = final_json
    
    for attempt in range(max_rectification_attempts):
        logger_service.log(session_id, f"rectification_attempt_{attempt+1}", f"Starting rectification attempt {attempt+1}")
        
        # Create rectification prompt
        rectification_prompt = create_rectification_prompt(
            schema=schema,
            current_json=current_json,
            validation_error=validation_error,
            source_text=combined_text,
            attempt=attempt + 1
        )
        
        # Get LLM to fix the JSON
        try:
            llm_output = llm_fn(rectification_prompt)
            logger_service.log(session_id, f"rectification_output_{attempt+1}", llm_output)
            
            # Extract and validate the rectified JSON
            rectified_json = json_extractor.extract_json(llm_output)
            valid, new_error = json_extractor.validate_against_schema(schema_dict, rectified_json)
            
            if valid:
                logger_service.log(session_id, "post_rectification_success", f"Rectification successful on attempt {attempt+1}")
                return rectified_json, True, None
            else:
                validation_error = new_error
                current_json = rectified_json
                logger_service.log(session_id, f"rectification_failed_{attempt+1}", f"Attempt {attempt+1} failed: {new_error}")
                
        except Exception as e:
            validation_error = f"Rectification attempt {attempt+1} failed: {str(e)}"
            logger_service.log(session_id, f"rectification_error_{attempt+1}", validation_error)
    
    # If all attempts failed, return the best attempt with error info
    logger_service.log(session_id, "post_rectification_failed", f"All rectification attempts failed. Final error: {validation_error}")
    return current_json, False, validation_error


def create_rectification_prompt(schema: str, current_json: dict, validation_error: str, source_text: str, attempt: int) -> str:
    """
    Creates a specialized prompt for rectifying schema validation errors.
    """
    
    prompt = f"""
You are a JSON rectification specialist. Your task is to fix a JSON object that failed schema validation.

CRITICAL INSTRUCTIONS:
- Fix ONLY the validation errors - do not change correctly formatted fields
- Use ONLY information present in the source text - never invent data
- If data is missing for required fields, set them as null or appropriate empty values
- Maintain all correctly extracted information from the current JSON
- Output ONLY the corrected JSON object (no explanations)

=== SCHEMA TO MATCH ===
{schema}

=== VALIDATION ERROR ===
{validation_error}

=== CURRENT JSON (with errors) ===
{json.dumps(current_json, indent=2)}

=== SOURCE TEXT (for reference) ===
{source_text[:8000]}  # Truncate if too long

=== RECTIFICATION ATTEMPT #{attempt} ===

Analyze the validation error and fix the JSON to strictly comply with the schema.
Focus on:
1. Correcting data types (string vs integer vs boolean)
2. Adding missing required fields
3. Fixing array/object structure mismatches
4. Ensuring nested objects match their schemas

Output the corrected JSON:
"""
    
    return prompt.strip()
