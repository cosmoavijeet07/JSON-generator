import json
import re
from typing import Dict, Any, Optional, Tuple, Union
from jsonschema import validate, ValidationError, Draft7Validator
import ast

def extract_json(output: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM output with multiple strategies
    
    Args:
        output: Raw LLM output that may contain JSON
    
    Returns:
        Extracted JSON as dictionary
    
    Raises:
        ValueError: If no valid JSON found
    """
    
    # Strategy 1: Try to parse the entire output as JSON
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Look for JSON between triple backticks
    code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', output)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON object using regex
    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        try:
            # Balance braces
            json_str = balance_braces(json_match.group())
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Find JSON array
    array_match = re.search(r'\[[\s\S]*\]', output)
    if array_match:
        try:
            return {"data": json.loads(array_match.group())}
        except json.JSONDecodeError:
            pass
    
    # Strategy 5: Try to fix common issues
    cleaned = clean_json_string(output)
    if cleaned:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    # Strategy 6: Extract using AST (for Python dict literals)
    try:
        # Remove any non-dict content
        dict_match = re.search(r'\{[^{}]*\}', output)
        if dict_match:
            result = ast.literal_eval(dict_match.group())
            if isinstance(result, dict):
                return result
    except (ValueError, SyntaxError):
        pass
    
    raise ValueError("No valid JSON found in the output")


def balance_braces(json_str: str) -> str:
    """
    Balance braces in a JSON string
    
    Args:
        json_str: Potentially unbalanced JSON string
    
    Returns:
        Balanced JSON string
    """
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Add missing closing braces
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    # Add missing closing brackets
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    return json_str


def clean_json_string(text: str) -> Optional[str]:
    """
    Clean common issues in JSON strings
    
    Args:
        text: Text that might contain JSON
    
    Returns:
        Cleaned JSON string or None
    """
    # Remove common prefixes
    prefixes = [
        "Here is the JSON:",
        "JSON output:",
        "```json",
        "```",
        "Output:",
        "Result:"
    ]
    
    for prefix in prefixes:
        if prefix in text:
            text = text.split(prefix)[-1]
    
    # Remove common suffixes
    suffixes = ["```", "End of JSON", "---"]
    for suffix in suffixes:
        if suffix in text:
            text = text.split(suffix)[0]
    
    # Strip whitespace
    text = text.strip()
    
    # Fix common issues
    # Replace single quotes with double quotes
    text = re.sub(r"'([^']*)'", r'"\1"', text)
    
    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Add quotes to unquoted keys
    text = re.sub(r'(\w+):', r'"\1":', text)
    
    # Remove comments
    text = re.sub(r'//.*?\n', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text if text else None


def validate_against_schema(
    schema: Dict[str, Any], 
    instance: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON against a schema
    
    Args:
        schema: JSON schema
        instance: JSON instance to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        validate(instance=instance, schema=schema)
        return True, None
    except ValidationError as e:
        # Format error message
        error_msg = str(e.message)
        
        # Add path information if available
        if e.path:
            path = '.'.join(str(p) for p in e.path)
            error_msg = f"At path '{path}': {error_msg}"
        
        # Add schema path if available
        if e.schema_path:
            schema_path = '.'.join(str(p) for p in e.schema_path)
            error_msg += f" (Schema path: {schema_path})"
        
        return False, error_msg
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_validation_errors(
    schema: Dict[str, Any],
    instance: Dict[str, Any]
) -> list:
    """
    Get all validation errors for a JSON instance
    
    Args:
        schema: JSON schema
        instance: JSON instance to validate
    
    Returns:
        List of validation errors
    """
    validator = Draft7Validator(schema)
    errors = []
    
    for error in validator.iter_errors(instance):
        error_dict = {
            'message': error.message,
            'path': list(error.path),
            'schema_path': list(error.schema_path),
            'validator': error.validator,
            'validator_value': error.validator_value
        }
        errors.append(error_dict)
    
    return errors


def fix_common_validation_errors(
    instance: Dict[str, Any],
    schema: Dict[str, Any],
    errors: list
) -> Dict[str, Any]:
    """
    Attempt to fix common validation errors
    
    Args:
        instance: JSON instance with errors
        schema: JSON schema
        errors: List of validation errors
    
    Returns:
        Fixed JSON instance
    """
    fixed = instance.copy()
    
    for error in errors:
        path = error.get('path', [])
        validator = error.get('validator')
        
        if validator == 'required':
            # Add missing required fields
            missing_fields = error.get('validator_value', [])
            current = fixed
            
            # Navigate to the parent object
            for key in path[:-1] if path else []:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Add missing fields with default values
            for field in missing_fields:
                if field not in current:
                    # Get field schema
                    field_schema = schema.get('properties', {}).get(field, {})
                    current[field] = get_default_value(field_schema)
        
        elif validator == 'type':
            # Fix type mismatches
            expected_type = error.get('validator_value')
            current = fixed
            
            # Navigate to the field
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            if path:
                last_key = path[-1]
                if last_key in current:
                    current[last_key] = coerce_type(current[last_key], expected_type)
    
    return fixed


def get_default_value(schema: Dict[str, Any]) -> Any:
    """
    Get default value based on schema type
    
    Args:
        schema: Field schema
    
    Returns:
        Default value for the field
    """
    field_type = schema.get('type', 'null')
    
    if 'default' in schema:
        return schema['default']
    
    defaults = {
        'string': '',
        'number': 0.0,
        'integer': 0,
        'boolean': False,
        'array': [],
        'object': {},
        'null': None
    }
    
    return defaults.get(field_type, None)


def coerce_type(value: Any, target_type: str) -> Any:
    """
    Coerce a value to the target type
    
    Args:
        value: Value to coerce
        target_type: Target type name
    
    Returns:
        Coerced value
    """
    if value is None:
        return get_default_value({'type': target_type})
    
    try:
        if target_type == 'string':
            return str(value)
        elif target_type == 'number':
            return float(value)
        elif target_type == 'integer':
            return int(float(value))
        elif target_type == 'boolean':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        elif target_type == 'array':
            if isinstance(value, list):
                return value
            return [value] if value else []
        elif target_type == 'object':
            if isinstance(value, dict):
                return value
            return {}
        elif target_type == 'null':
            return None
    except (ValueError, TypeError):
        pass
    
    return get_default_value({'type': target_type})


def merge_json_objects(obj1: Dict, obj2: Dict, prefer_non_null: bool = True) -> Dict:
    """
    Merge two JSON objects
    
    Args:
        obj1: First JSON object
        obj2: Second JSON object
        prefer_non_null: Prefer non-null values when merging
    
    Returns:
        Merged JSON object
    """
    merged = obj1.copy()
    
    for key, value in obj2.items():
        if key not in merged:
            merged[key] = value
        elif prefer_non_null:
            # Prefer non-null values
            if merged[key] is None or merged[key] == '' or merged[key] == [] or merged[key] == {}:
                merged[key] = value
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursive merge for nested objects
                merged[key] = merge_json_objects(merged[key], value, prefer_non_null)
            elif isinstance(merged[key], list) and isinstance(value, list):
                # Combine lists
                merged[key] = list(set(merged[key] + value))
    
    return merged